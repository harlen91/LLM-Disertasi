"""
core/fusion_model.py
--------------------
Arsitektur PyTorch: Dual-Stream Hybrid Fusion Model
Menggabungkan Transformer Encoder (IndoBERTweet / XLM-RoBERTa)
dengan Vektor Teacher Annotation Features (SumoPod API 33-Dimensi)
melalui Gated Cross-Modality Fusion & 6 Multi-Task Classification Heads.

Pilar Ilmiah:
1. Gated Cross-Modality Fusion:
   z = sigmoid(W_g * [h_transformer || h_llm] + b_g)
   h_fused = z * h_transformer + (1 - z) * h_llm
2. Multi-Task Learning (MTL) untuk 6 Dimensi Taksonomi Cyberbullying Indonesia
3. Focal Loss (dengan masking ignore_index=-100) & Dynamic Uncertainty Weighting (Kendall et al., 2018)
4. Model Checkpoint, Manifest, & Modality-Masked/Unmasked Inference Support
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from config import INDOBERTWEET_PRETRAINED_MODEL
from core.data_understanding import (
    TAXONOMY_VECTOR_MAP,
    TOTAL_VECTOR_DIM,
    clean_and_normalize_social_text,
    extract_contextual_evidence_vector,
    extract_cue_family_summary,
    format_structured_rationale,
)
from core.evaluation_metrics import load_model_manifest
from core.schema import sanitize_annotation_dict

IGNORE_INDEX = -100


# =============================================================================
# 1. LOSS ENGINES: MASKED FOCAL LOSS & DYNAMIC UNCERTAINTY WEIGHTING
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss dengan dukungan masking `ignore_index=-100` agar label kosong/tidak tersedia
    tidak memicu error atau mengubah gradien secara artifisial.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.60,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != self.ignore_index
        if int(valid_mask.sum().item()) == 0:
            return logits.sum() * 0.0

        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction="none")
        pt = torch.exp(-ce_loss)

        if valid_logits.size(1) == 2:
            alpha_t = torch.where(valid_targets == 1, self.alpha, 1.0 - self.alpha)
        else:
            alpha_t = 1.0

        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: Optional[torch.Tensor],
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[torch.Tensor, bool]:
    """
    Menghitung CrossEntropyLoss hanya pada sampel dengan label valid (`targets != -100`).
    Mengembalikan `(loss_tensor, has_valid_samples)` tanpa menghasilkan NaN jika semua `-100`.
    """
    if targets is None:
        return logits.sum() * 0.0, False
    valid_mask = targets != ignore_index
    if int(valid_mask.sum().item()) == 0:
        return logits.sum() * 0.0, False
    loss = F.cross_entropy(logits[valid_mask], targets[valid_mask])
    return loss, True


class MultiTaskUncertaintyLoss(nn.Module):
    """
    Dynamic Uncertainty Weighting (Kendall et al., CVPR 2018).
    Loss_total = sum_i (0.5 * exp(-s_i) * L_i + 0.5 * s_i), s_i = log(sigma_i^2).
    Task yang tidak memiliki sampel valid pada batch tidak memaksa log-variance runtuh ke -inf.
    """

    def __init__(self, num_tasks: int = 6):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))

    def forward(
        self,
        task_losses: List[torch.Tensor],
        active_masks: Optional[List[bool]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.zeros((), device=self.log_vars.device, dtype=torch.float32)
        weights_info: Dict[str, float] = {}
        for i, loss in enumerate(task_losses):
            is_active = True if active_masks is None else bool(active_masks[i])
            clamped_log_var = torch.clamp(self.log_vars[i], min=-4.0, max=4.0)
            precision = torch.exp(-clamped_log_var)
            if is_active:
                weighted_task_loss = 0.5 * precision * loss + 0.5 * clamped_log_var
                total_loss = total_loss + weighted_task_loss
            weights_info[f"task_{i}_weight"] = float(precision.detach().cpu().item())
            weights_info[f"task_{i}_log_var"] = float(self.log_vars[i].detach().cpu().item())
        return total_loss, weights_info


# =============================================================================
# 2. LAPISAN PROYEKSI & GATED CROSS-MODALITY FUSION
# =============================================================================

class KnowledgeProjection(nn.Module):
    """Proyeksi vektor teacher knowledge 33-dimensi ke ruang laten padat (R^128)."""

    def __init__(self, in_features: int = TOTAL_VECTOR_DIM, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GatedCrossModalityFusion(nn.Module):
    """
    Mekanisme Gated Cross-Modality Fusion:
    z = sigmoid(W_g * [h_trans || h_llm_aligned] + b_g)
    h_fused = z * h_trans + (1 - z) * h_llm_aligned
    """

    def __init__(self, trans_dim: int = 768, llm_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.align_proj = nn.Sequential(
            nn.Linear(llm_dim, trans_dim),
            nn.GELU(),
            nn.LayerNorm(trans_dim),
        )
        self.gate_layer = nn.Sequential(
            nn.Linear(trans_dim * 2, trans_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(trans_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_trans: torch.Tensor, h_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_llm_aligned = self.align_proj(h_prob)
        concat_feat = torch.cat([h_trans, h_llm_aligned], dim=-1)
        gate_z = self.gate_layer(concat_feat)
        h_fused = gate_z * h_trans + (1.0 - gate_z) * h_llm_aligned
        h_fused = self.out_norm(h_fused)
        h_fused = self.dropout(h_fused)
        return h_fused, gate_z


class ConcatCrossModalityFusion(nn.Module):
    """
    Model Ablasi E6 (Hybrid without Gated Fusion):
    Menggabungkan h_trans dan h_llm_aligned melalui proyeksi concatenation linear
    tanpa sigmoid gating adaptif z.
    """

    def __init__(self, trans_dim: int = 768, llm_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.align_proj = nn.Sequential(
            nn.Linear(llm_dim, trans_dim),
            nn.GELU(),
            nn.LayerNorm(trans_dim),
        )
        self.concat_proj = nn.Sequential(
            nn.Linear(trans_dim * 2, trans_dim),
            nn.GELU(),
            nn.LayerNorm(trans_dim),
            nn.Dropout(dropout),
        )

    def forward(self, h_trans: torch.Tensor, h_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_llm_aligned = self.align_proj(h_prob)
        concat_feat = torch.cat([h_trans, h_llm_aligned], dim=-1)
        h_fused = self.concat_proj(concat_feat)
        dummy_gate = torch.full((h_trans.size(0), trans_dim_like := h_trans.size(1)), 0.5, device=h_trans.device, dtype=h_trans.dtype)
        return h_fused, dummy_gate


# =============================================================================
# 3. MODEL UTAMA & MODEL ABLASI STREAM B
# =============================================================================

class LLMKnowledgeOnlyClassifier(nn.Module):
    """
    Model Ablasi Stream B: memprediksi target klasifikasi HANYA dari representasi
    vektor pengetahuan LLM 33-dimensi (tanpa teks Transformer Encoder).
    """

    def __init__(self, in_features: int = TOTAL_VECTOR_DIM, hidden_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.head_binary = nn.Linear(hidden_dim, len(TAXONOMY_VECTOR_MAP["cyberbullying"]))
        self.head_type = nn.Linear(hidden_dim, len(TAXONOMY_VECTOR_MAP["type"]))
        self.head_emotion = nn.Linear(hidden_dim, len(TAXONOMY_VECTOR_MAP["emotion"]))
        self.head_cultural = nn.Linear(hidden_dim, len(TAXONOMY_VECTOR_MAP["cultural_context"]))
        self.head_implicit = nn.Linear(hidden_dim, len(TAXONOMY_VECTOR_MAP["implicit_level"]))
        self.head_severity = nn.Linear(hidden_dim, len(TAXONOMY_VECTOR_MAP["severity"]))
        self.focal_loss_binary = FocalLoss()
        self.uncertainty_loss_engine = MultiTaskUncertaintyLoss(num_tasks=6)

    def forward(
        self,
        llm_vectors: torch.Tensor,
        labels_binary: Optional[torch.Tensor] = None,
        labels_type: Optional[torch.Tensor] = None,
        labels_emotion: Optional[torch.Tensor] = None,
        labels_cultural: Optional[torch.Tensor] = None,
        labels_implicit: Optional[torch.Tensor] = None,
        labels_severity: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        h = self.encoder(llm_vectors)
        logits = {
            "binary": self.head_binary(h),
            "type": self.head_type(h),
            "emotion": self.head_emotion(h),
            "cultural": self.head_cultural(h),
            "implicit": self.head_implicit(h),
            "severity": self.head_severity(h),
        }
        probs = {k: F.softmax(v, dim=-1) for k, v in logits.items()}
        out: Dict[str, Any] = {"logits": logits, "probabilities": probs}

        if labels_binary is not None:
            has_bin = int((labels_binary != IGNORE_INDEX).sum().item()) > 0
            loss_bin = self.focal_loss_binary(logits["binary"], labels_binary)
            loss_typ, has_typ = masked_cross_entropy(logits["type"], labels_type)
            loss_emo, has_emo = masked_cross_entropy(logits["emotion"], labels_emotion)
            loss_cul, has_cul = masked_cross_entropy(logits["cultural"], labels_cultural)
            loss_imp, has_imp = masked_cross_entropy(logits["implicit"], labels_implicit)
            loss_sev, has_sev = masked_cross_entropy(logits["severity"], labels_severity)
            total_loss, weights_info = self.uncertainty_loss_engine(
                [loss_bin, loss_typ, loss_emo, loss_cul, loss_imp, loss_sev],
                active_masks=[has_bin, has_typ, has_emo, has_cul, has_imp, has_sev],
            )
            out["loss"] = total_loss
            out["task_weights"] = weights_info
        return out


class HybridFusionModel(nn.Module):
    """
    Arsitektur Hibrida Paralel:
    1. Stream A: Transformer Encoder (IndoBERTweet / XLM-R)
    2. Stream B: SumoPod LLM Teacher Knowledge Vector (33 Dimensi)
    3. Lapisan Fusi: Gated Cross-Modality Fusion Layer
    4. 6 Multi-Task Classification Heads (binary, type, emotion, cultural, implicit, severity)
    """

    def __init__(
        self,
        pretrained_model_name: str = INDOBERTWEET_PRETRAINED_MODEL,
        llm_input_dim: int = TOTAL_VECTOR_DIM,
        llm_hidden_dim: int = 128,
        use_uncertainty_loss: bool = True,
        focal_alpha: float = 0.60,
        focal_gamma: float = 2.0,
        classifier_dropout: float = 0.20,
        fusion_mode: str = "gated",
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.llm_input_dim = llm_input_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.use_uncertainty_loss = use_uncertainty_loss
        self.fusion_mode = fusion_mode if fusion_mode in {"gated", "concat"} else "gated"

        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.transformer = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        self.trans_hidden_dim = self.config.hidden_size

        self.llm_projection = KnowledgeProjection(
            in_features=llm_input_dim,
            hidden_dim=llm_hidden_dim,
            dropout=0.1,
        )

        if self.fusion_mode == "concat":
            self.fusion: nn.Module = ConcatCrossModalityFusion(
                trans_dim=self.trans_hidden_dim,
                llm_dim=llm_hidden_dim,
                dropout=classifier_dropout,
            )
        else:
            self.fusion = GatedCrossModalityFusion(
                trans_dim=self.trans_hidden_dim,
                llm_dim=llm_hidden_dim,
                dropout=classifier_dropout,
            )

        # 6 Multi-Task Classification Heads
        self.head_binary = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["cyberbullying"]))
        self.head_type = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["type"]))
        self.head_emotion = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["emotion"]))
        self.head_cultural = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["cultural_context"]))
        self.head_implicit = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["implicit_level"]))
        self.head_severity = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["severity"]))

        self.focal_loss_binary = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=IGNORE_INDEX)
        self.uncertainty_loss_engine = MultiTaskUncertaintyLoss(num_tasks=6) if use_uncertainty_loss else None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_vectors: torch.Tensor,
        labels_binary: Optional[torch.Tensor] = None,
        labels_type: Optional[torch.Tensor] = None,
        labels_emotion: Optional[torch.Tensor] = None,
        labels_cultural: Optional[torch.Tensor] = None,
        labels_implicit: Optional[torch.Tensor] = None,
        labels_severity: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        h_trans = outputs.last_hidden_state[:, 0, :]

        h_prob = self.llm_projection(llm_vectors)
        h_fused, gate_z = self.fusion(h_trans, h_prob)

        logits_binary = self.head_binary(h_fused)
        logits_type = self.head_type(h_fused)
        logits_emotion = self.head_emotion(h_fused)
        logits_cultural = self.head_cultural(h_fused)
        logits_implicit = self.head_implicit(h_fused)
        logits_severity = self.head_severity(h_fused)

        result: Dict[str, Any] = {
            "logits": {
                "binary": logits_binary,
                "type": logits_type,
                "emotion": logits_emotion,
                "cultural": logits_cultural,
                "implicit": logits_implicit,
                "severity": logits_severity,
            },
            "probabilities": {
                "binary": F.softmax(logits_binary, dim=-1),
                "type": F.softmax(logits_type, dim=-1),
                "emotion": F.softmax(logits_emotion, dim=-1),
                "cultural": F.softmax(logits_cultural, dim=-1),
                "implicit": F.softmax(logits_implicit, dim=-1),
                "severity": F.softmax(logits_severity, dim=-1),
            },
            "gate_values": gate_z,
            "gate_mean_weight": gate_z.mean(dim=-1),
            "fused_embeddings": h_fused,
            "stream_a_cls": h_trans,
            "stream_b_proj": h_prob,
        }

        has_any_label = any(
            t is not None and int((t != IGNORE_INDEX).sum().item()) > 0
            for t in [labels_binary, labels_type, labels_emotion, labels_cultural, labels_implicit, labels_severity]
        )
        if has_any_label:
            has_bin = labels_binary is not None and int((labels_binary != IGNORE_INDEX).sum().item()) > 0
            loss_bin = (
                self.focal_loss_binary(logits_binary, labels_binary)
                if labels_binary is not None
                else logits_binary.sum() * 0.0
            )
            loss_typ, has_typ = masked_cross_entropy(logits_type, labels_type)
            loss_emo, has_emo = masked_cross_entropy(logits_emotion, labels_emotion)
            loss_cul, has_cul = masked_cross_entropy(logits_cultural, labels_cultural)
            loss_imp, has_imp = masked_cross_entropy(logits_implicit, labels_implicit)
            loss_sev, has_sev = masked_cross_entropy(logits_severity, labels_severity)

            task_losses = [loss_bin, loss_typ, loss_emo, loss_cul, loss_imp, loss_sev]
            active_masks = [has_bin, has_typ, has_emo, has_cul, has_imp, has_sev]

            if self.use_uncertainty_loss and self.uncertainty_loss_engine is not None:
                total_loss, weights_info = self.uncertainty_loss_engine(task_losses, active_masks=active_masks)
                result["task_weights"] = weights_info
            else:
                total_loss = (
                    2.0 * loss_bin
                    + 1.0 * loss_typ
                    + 0.8 * loss_emo
                    + 0.8 * loss_cul
                    + 0.8 * loss_imp
                    + 0.8 * loss_sev
                )

            result["loss"] = total_loss
            result["task_losses"] = {
                "binary": float(loss_bin.detach().cpu().item()),
                "type": float(loss_typ.detach().cpu().item()),
                "emotion": float(loss_emo.detach().cpu().item()),
                "cultural": float(loss_cul.detach().cpu().item()),
                "implicit": float(loss_imp.detach().cpu().item()),
                "severity": float(loss_sev.detach().cpu().item()),
            }

        return result


# =============================================================================
# 4. CHECKPOINT LOADER, MANIFEST RESOLUTION, & LIVE INFERENCE
# =============================================================================

def detect_torch_device(preferred: Optional[str] = None) -> str:
    """Detects best available PyTorch compute device."""
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hybrid_classifier_with_manifest(
    weights_path: Path,
    pretrained_model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[HybridFusionModel, Any, str, Optional[Dict[str, Any]]]:
    """
    Memuat HybridFusionModel beserta tokenizer & model_manifest.json.
    Menolak fallback acak apabila checkpoint `weights_path` tidak ditemukan.
    """
    from transformers import AutoTokenizer

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Checkpoint HybridFusionModel tidak ditemukan di '{weights_path}'. "
            "Jalankan training terlebih dahulu atau sediakan file bobot terbaik (best_hybrid_model.pt)."
        )

    dev = detect_torch_device(device)
    manifest = load_model_manifest(weights_path.parent)
    backbone_name = (
        pretrained_model_name
        or (manifest.get("pretrained_model") if manifest else None)
        or INDOBERTWEET_PRETRAINED_MODEL
    )

    state_dict = torch.load(weights_path, map_location=dev)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    hparams = manifest.get("hyperparameters", {}) if manifest else {}
    proj_w = state_dict.get("llm_projection.proj.0.weight") if isinstance(state_dict, dict) else None
    llm_hidden_dim = int(hparams.get("llm_hidden_dim", proj_w.shape[0] if proj_w is not None else 128))
    llm_input_dim = int(hparams.get("llm_input_dim", proj_w.shape[1] if proj_w is not None else TOTAL_VECTOR_DIM))
    use_uncertainty = (
        "uncertainty_loss_engine.log_vars" in state_dict
        if isinstance(state_dict, dict)
        else True
    )

    fusion_mode = "concat" if (isinstance(state_dict, dict) and any(k.startswith("fusion.concat_proj.") for k in state_dict)) else str(hparams.get("fusion_mode", "gated"))

    model = HybridFusionModel(
        pretrained_model_name=backbone_name,
        llm_input_dim=llm_input_dim,
        llm_hidden_dim=llm_hidden_dim,
        use_uncertainty_loss=use_uncertainty,
        fusion_mode=fusion_mode,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(dev)
    model.eval()

    tokenizer_source = str(weights_path.parent) if (weights_path.parent / "tokenizer_config.json").exists() else backbone_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    return model, tokenizer, dev, manifest


def load_hybrid_classifier(
    weights_path: Path,
    pretrained_model_name: str = INDOBERTWEET_PRETRAINED_MODEL,
    device: Optional[str] = None,
) -> Tuple[HybridFusionModel, Any, str]:
    """Backward-compatible loader returning (model, tokenizer, device)."""
    model, tokenizer, dev, _ = load_hybrid_classifier_with_manifest(
        weights_path=weights_path,
        pretrained_model_name=pretrained_model_name,
        device=device,
    )
    return model, tokenizer, dev


def predict_hybrid_text(
    model: HybridFusionModel,
    tokenizer: Any,
    device: str,
    text: str,
    annotation_dict: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None,
    manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Melakukan inferensi multi-task menggunakan HybridFusionModel:
    - Preprocessing konsisten dengan training (`clean_and_normalize_social_text(text)`).
    - Jika `annotation_dict` tersedia, mengekstrak vektor teacher features 33-dimensi untuk Stream B.
    - Jika `annotation_dict` None, Stream B di-mask ke vektor nol (modality-masked student mode)
      tanpa mengarang anotasi LLM buatan.
    """
    import time

    t_start = time.time()
    norm_text = clean_and_normalize_social_text(text)

    if threshold is None:
        threshold = float(manifest.get("optimal_threshold", 0.50)) if manifest else 0.50

    has_llm_vec = annotation_dict is not None
    if has_llm_vec:
        sanitized = sanitize_annotation_dict(annotation_dict, preserve_missing=True)
        evidence_input = {
            "comment": text,
            "comment_normalized": norm_text,
            "explanation": sanitized.get("explanation", ""),
            "confidence": sanitized.get("confidence", None),
            "contextual_cues": sanitized.get("contextual_cues"),
        }
        llm_vec = extract_contextual_evidence_vector(evidence_input, text=text)
        structured_rat = format_structured_rationale(sanitized)
    else:
        llm_vec = extract_contextual_evidence_vector({"comment": text, "comment_normalized": norm_text}, text=text)
        structured_rat = ""
    cue_summary = extract_cue_family_summary(llm_vec)

    max_len = int(manifest.get("hyperparameters", {}).get("max_length", 128)) if manifest else 128
    trans_cfg = getattr(getattr(model, "transformer", None), "config", None)
    if trans_cfg and hasattr(trans_cfg, "max_position_embeddings"):
        max_len = min(max_len, int(trans_cfg.max_position_embeddings))
    text_mode = (manifest.get("hyperparameters", {}).get("text_input_mode", "text_only")) if manifest else "text_only"

    if text_mode == "text_plus_rationale" and structured_rat:
        encoding = tokenizer(
            norm_text,
            structured_rat,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
    else:
        encoding = tokenizer(
            norm_text,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    encoding = {k: v.to(device) for k, v in encoding.items()}
    llm_vec_tensor = torch.tensor(llm_vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            llm_vectors=llm_vec_tensor,
        )

    probs = outputs["probabilities"]
    bin_probs = probs["binary"][0].detach().cpu().numpy()
    prob_cb = float(bin_probs[1])

    is_cb = prob_cb >= threshold
    verdict = "YES" if is_cb else "NO"

    pred_type = TAXONOMY_VECTOR_MAP["type"][int(torch.argmax(probs["type"][0]).item())]
    pred_emo = TAXONOMY_VECTOR_MAP["emotion"][int(torch.argmax(probs["emotion"][0]).item())]
    pred_cul = TAXONOMY_VECTOR_MAP["cultural_context"][int(torch.argmax(probs["cultural"][0]).item())]
    pred_imp = TAXONOMY_VECTOR_MAP["implicit_level"][int(torch.argmax(probs["implicit"][0]).item())]
    pred_sev = TAXONOMY_VECTOR_MAP["severity"][int(torch.argmax(probs["severity"][0]).item())]

    gate_z = float(outputs["gate_mean_weight"][0].detach().cpu().item())
    latency_ms = round((time.time() - t_start) * 1000.0, 1)

    return {
        "text": text,
        "normalized_text": norm_text,
        "cyberbullying": verdict,
        "prob_cyberbullying": round(prob_cb, 4),
        "confidence": round(prob_cb if is_cb else (1.0 - prob_cb), 4),
        "threshold": round(float(threshold), 4),
        "type": pred_type,
        "emotion": pred_emo,
        "cultural_context": pred_cul,
        "implicit_level": pred_imp,
        "severity": pred_sev,
        "all_head_probabilities": {
            head_name: {
                cls_name: round(float(probs[head_name][0][i].item()), 4)
                for i, cls_name in enumerate(TAXONOMY_VECTOR_MAP[tax_key])
            }
            for head_name, tax_key in [
                ("binary", "cyberbullying"),
                ("type", "type"),
                ("emotion", "emotion"),
                ("cultural", "cultural_context"),
                ("implicit", "implicit_level"),
                ("severity", "severity"),
            ]
        },
        "gate_weight_z": round(gate_z, 4),
        "fusion_mode": getattr(model, "fusion_mode", "gated"),
        "llm_modality_active": bool(np.any(llm_vec > 0.0)),
        "has_llm_context": has_llm_vec,
        "cue_family_summary": cue_summary,
        "stream_b_vector": [round(float(x), 4) for x in llm_vec.tolist()],
        "latency_ms": latency_ms,
        "structured_rationale": structured_rat,
    }
