"""
core/fusion_model.py
--------------------
Arsitektur PyTorch: Dual-Stream Hybrid Fusion Model
Menggabungkan Transformer Encoder (IndoBERTweet / XLM-RoBERTa)
dengan Vektor Representasi Pengetahuan LLM (SumoPod LLaMA-3 33-Dimensi)
melalui Gated Cross-Modality Fusion & 6 Multi-Task Classification Heads.

Pilar Ilmiah:
1. Gated Cross-Modality Fusion: z * h_trans + (1 - z) * h_llm
2. Multi-Task Learning (MTL) untuk 6 Dimensi Taksonomi Cyberbullying Indonesia
3. Focal Loss & Dynamic Uncertainty Weighting (Kendall et al., 2018)
4. Transparansi Komputasi (Gating Analysis untuk Explainability Bab 4 Disertasi)
"""

import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from core.data_understanding import TAXONOMY_VECTOR_MAP, TOTAL_VECTOR_DIM


# =============================================================================
# 1. LOSS ENGINES: FOCAL LOSS & DYNAMIC UNCERTAINTY WEIGHTING
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss untuk menangani ketidakseimbangan kelas dan sampel sulit (hard samples).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.60, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: [batch_size, num_classes], targets: [batch_size]
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        
        # Binary case weighting
        if logits.size(1) == 2:
            alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        else:
            alpha_t = 1.0

        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MultiTaskUncertaintyLoss(nn.Module):
    """
    Dynamic Uncertainty Weighting (Kendall et al., CVPR 2018).
    Mengganti tuning manual lambda_1..lambda_6 dengan parameter varians homoscedastic yang dapat dilatih:
    Loss_total = sum_i (0.5 * exp(-s_i) * L_i + 0.5 * s_i), di mana s_i = log(sigma_i^2)
    """
    def __init__(self, num_tasks: int = 6):
        super().__init__()
        # Inisialisasi log-varians bernilai 0.0 (sigma = 1.0)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))

    def forward(self, task_losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        weights_info = {}
        for i, loss in enumerate(task_losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_task_loss = 0.5 * precision * loss + 0.5 * self.log_vars[i]
            total_loss = total_loss + weighted_task_loss
            weights_info[f"task_{i}_weight"] = float(precision.detach().cpu().item())
        return total_loss, weights_info


# =============================================================================
# 2. LAPISAN PROYEKSI & GATED FUSION
# =============================================================================

class KnowledgeProjection(nn.Module):
    """
    Memproyeksikan vektor probabilitas LLM 33-dimensi ke ruang laten padat (R^128):
    h_prob = LayerNorm(GELU(W_proj * v_llm + b_proj))
    """
    def __init__(self, in_features: int = TOTAL_VECTOR_DIM, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class GatedCrossModalityFusion(nn.Module):
    """
    Mekanisme Gated Cross-Modality Fusion:
    z = sigmoid(W_g * [h_trans || W_e * h_prob] + b_g)
    h_fused = z * h_trans + (1 - z) * (W_e * h_prob)
    
    Nilai z mencerminkan bobot ketergantungan model:
    - z -> 1: Mengutamakan sinyal tekstual leksikal Transformer (cocok untuk ujaran eksplisit E0).
    - z -> 0: Mengutamakan representasi penalaran kultural LLM (cocok untuk sarkasme implisit E1/E2).
    """
    def __init__(self, trans_dim: int = 768, llm_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        # Proyeksikan representasi laten LLM ke dimensi Transformer
        self.align_proj = nn.Sequential(
            nn.Linear(llm_dim, trans_dim),
            nn.GELU(),
            nn.LayerNorm(trans_dim)
        )
        
        # Gerbang penyeimbang adaptif
        self.gate_layer = nn.Sequential(
            nn.Linear(trans_dim * 2, trans_dim),
            nn.Sigmoid()
        )
        
        self.out_norm = nn.LayerNorm(trans_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_trans: torch.Tensor, h_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prob_aligned = self.align_proj(h_prob) # [batch_size, trans_dim]
        
        # Hitung gerbang gating
        concat_feat = torch.cat([h_trans, h_prob_aligned], dim=-1) # [batch_size, trans_dim * 2]
        gate_z = self.gate_layer(concat_feat) # [batch_size, trans_dim]
        
        # Fusi terbobot konveks
        h_fused = gate_z * h_trans + (1.0 - gate_z) * h_prob_aligned
        h_fused = self.out_norm(h_fused)
        h_fused = self.dropout(h_fused)
        
        return h_fused, gate_z


# =============================================================================
# 3. MODEL UTAMA: HYBRID FUSION MULTI-TASK CLASSIFIER
# =============================================================================

class HybridFusionModel(nn.Module):
    """
    Arsitektur Hibrida Paralel:
    1. Stream A: Transformer Encoder (IndoBERTweet / XLM-R)
    2. Stream B: SumoPod LLaMA-3 Probability Vector Extractor (33 Dimensi)
    3. Lapisan Fusi: Gated Cross-Modality Fusion Layer
    4. 6 Multi-Task Classification Heads
    """
    def __init__(
        self,
        pretrained_model_name: str = "indolem/indobertweet-base-uncased",
        llm_input_dim: int = TOTAL_VECTOR_DIM,
        llm_hidden_dim: int = 128,
        use_uncertainty_loss: bool = True,
        focal_alpha: float = 0.60,
        focal_gamma: float = 2.0,
        classifier_dropout: float = 0.20
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.llm_input_dim = llm_input_dim
        self.use_uncertainty_loss = use_uncertainty_loss

        # 1. Backbone Transformer
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.transformer = AutoModel.from_pretrained(pretrained_model_name, config=self.config)
        self.trans_hidden_dim = self.config.hidden_size # 768

        # 2. Modul Proyeksi LLM
        self.llm_projection = KnowledgeProjection(
            in_features=llm_input_dim,
            hidden_dim=llm_hidden_dim,
            dropout=0.1
        )

        # 3. Modul Lapisan Fusi
        self.fusion = GatedCrossModalityFusion(
            trans_dim=self.trans_hidden_dim,
            llm_dim=llm_hidden_dim,
            dropout=classifier_dropout
        )

        # 4. Enam Multi-Task Classification Heads
        # Head 1: Binary Cyberbullying (YES / NO) -> 2 kelas
        self.head_binary = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["cyberbullying"]))
        
        # Head 2: Type (INS, BSH, SHM, SAR, THR, DIS, SEX, EXC, NONE) -> 9 kelas
        self.head_type = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["type"]))
        
        # Head 3: Emotion (ANG, DIS, HUM, MOCK, SAD, NEU) -> 6 kelas
        self.head_emotion = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["emotion"]))
        
        # Head 4: Cultural Context (APP, INT, STA, FAM, EDU, REG, GEN, NONE) -> 8 kelas
        self.head_cultural = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["cultural_context"]))
        
        # Head 5: Implicit Level (E0, E1, E2) -> 3 kelas
        self.head_implicit = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["implicit_level"]))
        
        # Head 6: Severity (LOW, MEDIUM, HIGH, NONE) -> 4 kelas
        self.head_severity = nn.Linear(self.trans_hidden_dim, len(TAXONOMY_VECTOR_MAP["severity"]))

        # 5. Loss Engines
        self.focal_loss_binary = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ce_loss = nn.CrossEntropyLoss()
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
        labels_severity: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward Pass End-to-End:
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        llm_vectors: [batch_size, 33]
        """
        # 1. Ekstraksi Fitur Transformer (Stream A)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Ambil [CLS] token representation (token pertama)
        h_trans = outputs.last_hidden_state[:, 0, :] # [batch_size, 768]

        # 2. Ekstraksi Fitur LLM (Stream B)
        h_prob = self.llm_projection(llm_vectors) # [batch_size, 128]

        # 3. Gated Fusion (Lapisan 3)
        h_fused, gate_z = self.fusion(h_trans, h_prob) # [batch_size, 768], [batch_size, 768]

        # 4. Multi-Task Heads (Lapisan 4)
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
                "severity": logits_severity
            },
            "probabilities": {
                "binary": F.softmax(logits_binary, dim=-1),
                "type": F.softmax(logits_type, dim=-1),
                "emotion": F.softmax(logits_emotion, dim=-1),
                "cultural": F.softmax(logits_cultural, dim=-1),
                "implicit": F.softmax(logits_implicit, dim=-1),
                "severity": F.softmax(logits_severity, dim=-1)
            },
            "gate_values": gate_z,
            "gate_mean_weight": gate_z.mean(dim=-1), # Rata-rata gate per sampel
            "fused_embeddings": h_fused
        }

        # 5. Komputasi Multi-Task Loss (jika labels disediakan)
        if labels_binary is not None:
            loss_bin = self.focal_loss_binary(logits_binary, labels_binary)
            loss_typ = self.ce_loss(logits_type, labels_type) if labels_type is not None else torch.tensor(0.0, device=input_ids.device)
            loss_emo = self.ce_loss(logits_emotion, labels_emotion) if labels_emotion is not None else torch.tensor(0.0, device=input_ids.device)
            loss_cul = self.ce_loss(logits_cultural, labels_cultural) if labels_cultural is not None else torch.tensor(0.0, device=input_ids.device)
            loss_imp = self.ce_loss(logits_implicit, labels_implicit) if labels_implicit is not None else torch.tensor(0.0, device=input_ids.device)
            loss_sev = self.ce_loss(logits_severity, labels_severity) if labels_severity is not None else torch.tensor(0.0, device=input_ids.device)

            task_losses = [loss_bin, loss_typ, loss_emo, loss_cul, loss_imp, loss_sev]

            if self.use_uncertainty_loss and self.uncertainty_loss_engine is not None:
                total_loss, weights_info = self.uncertainty_loss_engine(task_losses)
                result["task_weights"] = weights_info
            else:
                # Default fixed weighting (Utamakan binary cyberbullying)
                total_loss = (
                    2.0 * loss_bin +
                    1.0 * loss_typ +
                    0.8 * loss_emo +
                    0.8 * loss_cul +
                    0.8 * loss_imp +
                    0.8 * loss_sev
                )

            result["loss"] = total_loss
            result["task_losses"] = {
                "binary": float(loss_bin.detach().cpu().item()),
                "type": float(loss_typ.detach().cpu().item()),
                "emotion": float(loss_emo.detach().cpu().item()),
                "cultural": float(loss_cul.detach().cpu().item()),
                "implicit": float(loss_imp.detach().cpu().item()),
                "severity": float(loss_sev.detach().cpu().item())
            }

        return result


# =============================================================================
# 4. FUNGSI INFERENSI LIVE RUNTIME MODEL HIBRIDA
# =============================================================================

def load_hybrid_classifier(
    weights_path: Path,
    pretrained_model_name: str = "indolem/indobertweet-base-uncased",
    device: Optional[str] = None
) -> Tuple[HybridFusionModel, Any, str]:
    """Memuat model HybridFusionModel dan Tokenizer dari bobot terbaik."""
    from transformers import AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = HybridFusionModel(pretrained_model_name=pretrained_model_name)
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    return model, tokenizer, device


def predict_hybrid_text(
    model: HybridFusionModel,
    tokenizer: Any,
    device: str,
    text: str,
    annotation_dict: Optional[Dict[str, Any]] = None,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Melakukan inferensi live multi-task menggunakan HybridFusionModel:
    1. Ekstraksi Vektor Pengetahuan LLM 33-Dimensi & Rationale
    2. Gated Cross-Modality Fusion (IndoBERTweet + SumoPod)
    3. Prediksi ke-6 Dimensi Taksonomi & Nilai Gerbang Fusi (z)
    """
    import time
    from core.data_understanding import (
        clean_and_normalize_social_text,
        format_structured_rationale,
        extract_taxonomy_probability_vector
    )
    from core.schema import sanitize_annotation_dict

    t_start = time.time()
    norm_text = clean_and_normalize_social_text(text)

    # Sanitasi anotasi atau gunakan fallback rasional
    if annotation_dict is None:
        annotation_dict = {
            "cyberbullying": "NO",
            "type": "NONE",
            "emotion": "NEU",
            "cultural_context": "NONE",
            "implicit_level": "E0",
            "severity": "LOW",
            "confidence": 0.85,
            "explanation": "Evaluasi berbasis fusi representasi leksikal IndoBERTweet."
        }

    sanitized = sanitize_annotation_dict(annotation_dict)
    structured_rat = format_structured_rationale(sanitized)
    llm_vec = extract_taxonomy_probability_vector(sanitized)

    # Tokenisasi Dual-Sequence
    encoding = tokenizer(
        norm_text,
        structured_rat,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    llm_vec_tensor = torch.tensor(llm_vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            llm_vectors=llm_vec_tensor
        )

    probs = outputs["probabilities"]
    bin_probs = probs["binary"][0].cpu().numpy()
    prob_cb = float(bin_probs[1])

    is_cb = prob_cb >= threshold
    verdict = "YES" if is_cb else "NO"

    # Multi-task predictions
    type_idx = int(torch.argmax(probs["type"][0]).item())
    emo_idx = int(torch.argmax(probs["emotion"][0]).item())
    cul_idx = int(torch.argmax(probs["cultural"][0]).item())
    imp_idx = int(torch.argmax(probs["implicit"][0]).item())
    sev_idx = int(torch.argmax(probs["severity"][0]).item())

    pred_type = TAXONOMY_VECTOR_MAP["type"][type_idx]
    pred_emo = TAXONOMY_VECTOR_MAP["emotion"][emo_idx]
    pred_cul = TAXONOMY_VECTOR_MAP["cultural_context"][cul_idx]
    pred_imp = TAXONOMY_VECTOR_MAP["implicit_level"][imp_idx]
    pred_sev = TAXONOMY_VECTOR_MAP["severity"][sev_idx]

    gate_z = float(outputs["gate_mean_weight"][0].cpu().item())
    latency_ms = round((time.time() - t_start) * 1000.0, 1)

    return {
        "text": text,
        "normalized_text": norm_text,
        "cyberbullying": verdict,
        "prob_cyberbullying": prob_cb,
        "confidence": prob_cb if is_cb else (1.0 - prob_cb),
        "threshold": threshold,
        "type": pred_type,
        "emotion": pred_emo,
        "cultural_context": pred_cul,
        "implicit_level": pred_imp,
        "severity": pred_sev,
        "gate_weight_z": round(gate_z, 4),
        "latency_ms": latency_ms,
        "structured_rationale": structured_rat
    }
