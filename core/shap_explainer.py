"""
core/shap_explainer.py
----------------------
Modul Keterjelasan Kuantitatif Berbasis Library `shap` Sebenarnya:
1. Local Textual SHAP Attribution (`shap.Explainer` + `shap.maskers.Text`) pada
   HybridFusionModel (maupun baseline IndoBERTweet-Only).
2. Local Numeric Teacher-Vector SHAP Attribution (`shap.Explainer` atas 33-dim
   teacher annotation features) secara terpisah dari modalitas teks, dengan
   dokumentasi ruang lingkup kondisional yang eksplisit.
3. Global Feature Importance (agregasi |SHAP| untuk token & fitur 33D).
4. Dual-XAI Alignment (`align_shap_with_cot`) yang memisahkan secara tegas antara
   metrik diagnostik lexical-overlap vs metrik faithfulness kuantitatif (comprehensiveness).
5. Tidak pernah mengembalikan probabilitas dummy [0.5, 0.5] jika checkpoint tidak ada.
"""

import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import shap

from config import FINAL_DIR, INDOBERTWEET_PRETRAINED_MODEL, MODELS_DIR
from core.data_understanding import (
    TOTAL_VECTOR_DIM,
    clean_and_normalize_social_text,
    extract_teacher_annotation_features,
    get_teacher_feature_names,
)
from core.fusion_model import (
    HybridFusionModel,
    detect_torch_device,
    load_hybrid_classifier_with_manifest,
)


def _merge_subword_tokens(raw_tokens: List[str], values: List[float]) -> Tuple[List[str], List[float]]:
    """Menggabungkan subword WordPiece (##...) menjadi token kata utuh yang mudah dibaca."""
    merged_tokens: List[str] = []
    merged_vals: List[float] = []

    for tok, val in zip(raw_tokens, values):
        tok_str = str(tok).strip()
        if not tok_str or tok_str in {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}:
            continue
        if tok_str.startswith("##") and merged_tokens:
            merged_tokens[-1] = merged_tokens[-1] + tok_str[2:]
            merged_vals[-1] = float(merged_vals[-1] + val)
        else:
            merged_tokens.append(tok_str)
            merged_vals.append(float(val))

    return merged_tokens, [round(v, 4) for v in merged_vals]


def _render_token_heatmap_html(tokens: List[str], shap_values: List[float]) -> str:
    """Menghasilkan HTML heatmap yang aman (HTML-escaped) untuk visualisasi atribusi token."""
    if not tokens:
        return "<p>Tidak ada token valid untuk divisualisasikan.</p>"

    max_abs = max([abs(v) for v in shap_values] + [1e-4])
    html_snippets = []
    for tok, val in zip(tokens, shap_values):
        escaped = html.escape(tok)
        norm_val = min(1.0, abs(val) / max_abs)
        if val > 0:
            bg_color = f"rgba(239, 68, 68, {0.15 + 0.70 * norm_val:.2f})"
            border_color = "rgba(220, 38, 38, 0.45)"
        elif val < 0:
            bg_color = f"rgba(16, 185, 129, {0.15 + 0.70 * norm_val:.2f})"
            border_color = "rgba(5, 150, 105, 0.45)"
        else:
            bg_color = "rgba(100, 116, 139, 0.12)"
            border_color = "transparent"

        html_snippets.append(
            f'<span style="display:inline-block; margin:2px; padding:3px 8px; '
            f'background-color:{bg_color}; border:1px solid {border_color}; '
            f'border-radius:4px; font-weight:500; font-size:14px;" '
            f'title="SHAP Value: {val:+.4f}">{escaped}</span>'
        )

    return (
        '<div style="line-height:2.2; padding:12px; border-radius:8px; '
        'background-color:#0f172a; border:1px solid #334155; color:#f8fafc;">'
        + " ".join(html_snippets)
        + "</div>"
    )


class HybridSHAPExplainer:
    """
    Explainer SHAP sejati (`shap` library) untuk HybridFusionModel dan Baseline IndoBERTweet.
    Menjelaskan modalitas Teks (Stream A) dan modalitas Teacher Vector 33D (Stream B) secara
    terpisah agar atribusi lintas modalitas tidak tercampur secara sembarangan.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_type: str = "hybrid",
        device: Optional[str] = None,
        loaded_model: Optional[Any] = None,
        loaded_tokenizer: Optional[Any] = None,
        manifest: Optional[Dict[str, Any]] = None,
        background_vectors: Optional[np.ndarray] = None,
    ):
        self.device = detect_torch_device(device)
        self.model_type = model_type
        self.manifest = manifest
        self.model: Any = loaded_model
        self.tokenizer: Any = loaded_tokenizer

        if self.model is None or self.tokenizer is None:
            if model_type == "hybrid":
                weights_path = Path(model_path) if model_path else (MODELS_DIR / "hybrid_fusion_indobertweet" / "best_hybrid_model.pt")
                if weights_path.is_dir():
                    weights_path = weights_path / "best_hybrid_model.pt"
                if not weights_path.exists():
                    raise FileNotFoundError(
                        f"Checkpoint HybridFusionModel tidak ditemukan di '{weights_path}'. "
                        "Explainability SHAP menolak prediksi dummy; latih model terlebih dahulu."
                    )
                self.model, self.tokenizer, self.device, self.manifest = load_hybrid_classifier_with_manifest(
                    weights_path=weights_path, device=self.device
                )
            elif model_type == "baseline":
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                base_dir = Path(model_path) if model_path else (MODELS_DIR / "indobertweet_cyberbullying" / "best_model")
                if base_dir.is_dir() and (base_dir / "best_model").exists():
                    base_dir = base_dir / "best_model"
                if not base_dir.exists() or not ((base_dir / "model.safetensors").exists() or (base_dir / "pytorch_model.bin").exists()):
                    raise FileNotFoundError(
                        f"Checkpoint IndoBERTweet baseline tidak ditemukan di '{base_dir}'. "
                        "Explainability SHAP menolak prediksi dummy; latih model baseline terlebih dahulu."
                    )
                self.tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(base_dir)).to(self.device)
                self.model.eval()
            else:
                raise ValueError(f"Tipe model tidak dikenal: '{model_type}'. Gunakan 'hybrid' atau 'baseline'.")

        self.is_hybrid = isinstance(self.model, HybridFusionModel) or hasattr(self.model, "fusion")
        self.max_length = int(self.manifest.get("hyperparameters", {}).get("max_length", 128)) if self.manifest else 128

        # Load background vectors untuk SHAP tabular pada Stream B
        if background_vectors is not None:
            self.background_vectors = np.asarray(background_vectors, dtype=np.float32)
        else:
            default_vec_file = FINAL_DIR / "train_llm_vectors.npy"
            if default_vec_file.exists():
                vecs = np.load(default_vec_file).astype(np.float32)
                self.background_vectors = vecs[: min(32, len(vecs))]
            else:
                self.background_vectors = np.zeros((1, TOTAL_VECTOR_DIM), dtype=np.float32)

    def _prepare_texts(self, texts: Any) -> List[str]:
        if isinstance(texts, str):
            raw_list = [texts]
        else:
            raw_list = list(texts)
        return [clean_and_normalize_social_text(str(t)) for t in raw_list]

    def predict_proba(
        self,
        texts: Any,
        fixed_llm_vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Menghitung probabilitas biner [P(NO), P(YES)] secara konsisten untuk batch teks."""
        clean_texts = self._prepare_texts(texts)
        if not clean_texts:
            return np.zeros((0, 2), dtype=np.float32)

        all_probs: List[np.ndarray] = []
        batch_size = 16
        vec_ref = (
            np.asarray(fixed_llm_vector, dtype=np.float32).reshape(1, TOTAL_VECTOR_DIM)
            if fixed_llm_vector is not None
            else np.zeros((1, TOTAL_VECTOR_DIM), dtype=np.float32)
        )

        with torch.no_grad():
            for start in range(0, len(clean_texts), batch_size):
                chunk = clean_texts[start : start + batch_size]
                enc = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                if self.is_hybrid:
                    vec_batch = np.repeat(vec_ref, len(chunk), axis=0)
                    vec_tensor = torch.tensor(vec_batch, dtype=torch.float32, device=self.device)
                    out = self.model(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        llm_vectors=vec_tensor,
                    )
                    probs = out["probabilities"]["binary"].detach().cpu().numpy()
                else:
                    out = self.model(**enc)
                    probs = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()

                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict_proba_numeric(
        self,
        vectors: np.ndarray,
        fixed_text: str,
    ) -> np.ndarray:
        """Menghitung probabilitas biner [P(NO), P(YES)] untuk batch vektor 33D dengan teks tetap."""
        if not self.is_hybrid:
            raise ValueError("Atribusi vektor numerik 33D hanya berlaku untuk HybridFusionModel.")
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        clean_text = clean_and_normalize_social_text(fixed_text)
        enc = self.tokenizer(
            [clean_text],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        all_probs: List[np.ndarray] = []
        batch_size = 32
        with torch.no_grad():
            for start in range(0, len(vecs), batch_size):
                v_chunk = vecs[start : start + batch_size]
                n_cur = len(v_chunk)
                v_tensor = torch.tensor(v_chunk, dtype=torch.float32, device=self.device)
                out = self.model(
                    input_ids=enc["input_ids"].repeat(n_cur, 1),
                    attention_mask=enc["attention_mask"].repeat(n_cur, 1),
                    llm_vectors=v_tensor,
                )
                probs = out["probabilities"]["binary"].detach().cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def explain_text(
        self,
        text: str,
        llm_vector: Optional[np.ndarray] = None,
        max_evals: int = 120,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        """
        Menghitung nilai Shapley token teks menggunakan `shap.Explainer(predict_fn, shap.maskers.Text)`.
        Apabila model adalah HybridFusionModel, Stream B ditahan tetap pada `llm_vector`
        (atau vektor nol dalam mode student) dan ruang lingkup kondisional didokumentasikan eksplisit.
        """
        norm_text = clean_and_normalize_social_text(text or "")
        if not norm_text:
            return {
                "method": "shap.maskers.Text (Partition Explainer)",
                "tokens": [],
                "shap_values": [],
                "base_value": 0.0,
                "cyberbullying_prob": 0.0,
                "top_positive_tokens": [],
                "top_negative_tokens": [],
                "html_heatmap": "<p>Teks kosong.</p>",
                "conditional_scope": "N/A (empty input)",
            }

        ref_vec = (
            np.asarray(llm_vector, dtype=np.float32)
            if llm_vector is not None
            else np.zeros(TOTAL_VECTOR_DIM, dtype=np.float32)
        )
        cb_prob = float(self.predict_proba([norm_text], fixed_llm_vector=ref_vec)[0][1])

        def _f_text(batch_texts):
            return self.predict_proba(batch_texts, fixed_llm_vector=ref_vec)

        masker = shap.maskers.Text(self.tokenizer)
        explainer = shap.Explainer(_f_text, masker, output_names=["Non-Cyberbullying", "Cyberbullying"])
        explanation = explainer([norm_text], max_evals=max_evals, silent=True)

        raw_vals = np.asarray(explanation.values[0][:, 1], dtype=float).tolist()
        raw_toks = list(explanation.data[0])
        base_vals = explanation.base_values[0]
        base_val = float(base_vals[1]) if hasattr(base_vals, "__len__") and len(base_vals) > 1 else float(base_vals)

        tokens, token_shap = _merge_subword_tokens(raw_toks, raw_vals)
        token_items = [{"token": t, "shap_value": v} for t, v in zip(tokens, token_shap)]

        top_pos = sorted([x for x in token_items if x["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_k]
        top_neg = sorted([x for x in token_items if x["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_k]

        scope_desc = (
            "Conditional Shapley values pada modalitas teks (Stream A) dengan modalitas LLM 33D (Stream B) ditahan tetap pada "
            + ("vektor teacher aktif." if llm_vector is not None else "vektor nol (modality-masked student mode).")
            if self.is_hybrid
            else "Shapley values pada IndoBERTweet Baseline (Text-Only)."
        )

        return {
            "method": "shap.Explainer(shap.maskers.Text)",
            "model_type": "hybrid_fusion" if self.is_hybrid else "indobertweet_baseline",
            "conditional_scope": scope_desc,
            "tokens": tokens,
            "shap_values": token_shap,
            "base_value": round(base_val, 4),
            "cyberbullying_prob": round(cb_prob, 4),
            "top_positive_tokens": top_pos,
            "top_negative_tokens": top_neg,
            "html_heatmap": _render_token_heatmap_html(tokens, token_shap),
        }

    def explain_llm_features(
        self,
        text: str,
        llm_vector: np.ndarray,
        max_evals: int = 160,
        top_k: int = 8,
    ) -> Dict[str, Any]:
        """
        Menghitung atribusi Shapley untuk vektor fitur numerik LLM 33D (`shap.PermutationExplainer` /
        `shap.Explainer` atas background tabular) dengan teks komentar ditahan tetap.
        """
        if not self.is_hybrid:
            return {"available": False, "reason": "Hanya tersedia pada HybridFusionModel."}

        norm_text = clean_and_normalize_social_text(text or "")
        vec = np.asarray(llm_vector, dtype=np.float32).reshape(1, TOTAL_VECTOR_DIM)
        feature_names = get_teacher_feature_names()

        def _f_vec(v_batch):
            return self.predict_proba_numeric(np.asarray(v_batch, dtype=np.float32), fixed_text=norm_text)

        bg = self.background_vectors
        if len(bg) > 16:
            bg = bg[:16]

        explainer = shap.Explainer(_f_vec, bg, feature_names=feature_names)
        shap_exp = explainer(vec, max_evals=max(2 * TOTAL_VECTOR_DIM + 1, max_evals), silent=True)

        vals = np.asarray(shap_exp.values[0, :, 1], dtype=float).tolist()
        base_vals = shap_exp.base_values[0]
        base_val = float(base_vals[1]) if hasattr(base_vals, "__len__") and len(base_vals) > 1 else float(base_vals)

        feat_items = [
            {"feature": name, "value": round(float(vec[0, i]), 4), "shap_value": round(float(vals[i]), 4)}
            for i, name in enumerate(feature_names)
        ]
        top_pos = sorted([x for x in feat_items if x["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_k]
        top_neg = sorted([x for x in feat_items if x["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_k]

        return {
            "available": True,
            "method": "shap.Explainer (Numeric Tabular Permutation)",
            "conditional_scope": "Conditional Shapley values pada Stream B (33D LLM teacher features) dengan teks Stream A ditahan tetap.",
            "feature_names": feature_names,
            "shap_values": [round(float(v), 4) for v in vals],
            "base_value": round(base_val, 4),
            "top_positive_features": top_pos,
            "top_negative_features": top_neg,
        }

    def explain_dual_modalities(
        self,
        text: str,
        annotation_dict: Optional[Dict[str, Any]] = None,
        max_evals_text: int = 120,
        max_evals_vec: int = 160,
    ) -> Dict[str, Any]:
        """Menghasilkan penjelasan SHAP terpisah untuk Stream A (Teks) dan Stream B (LLM 33D)."""
        llm_vec = (
            extract_teacher_annotation_features(annotation_dict)
            if annotation_dict is not None
            else None
        )
        textual_res = self.explain_text(
            text=text,
            llm_vector=llm_vec,
            max_evals=max_evals_text,
        )
        numeric_res = (
            self.explain_llm_features(text=text, llm_vector=llm_vec, max_evals=max_evals_vec)
            if (self.is_hybrid and llm_vec is not None)
            else {"available": False, "reason": "Vector LLM tidak disediakan atau model bukan Hybrid."}
        )
        return {
            "textual_shap": textual_res,
            "llm_vector_shap": numeric_res,
            "modality_separation_note": "Atribusi token teks dan fitur numerik LLM dilaporkan secara terpisah tanpa penggabungan skalar langsung.",
        }

    def compute_global_importance(
        self,
        texts: List[str],
        llm_vectors: Optional[np.ndarray] = None,
        max_samples: int = 20,
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """Menghitung Global Feature Importance (agregasi |SHAP|) untuk token teks dan fitur 33D."""
        sub_texts = texts[: min(max_samples, len(texts))]
        token_abs_map: Dict[str, List[float]] = {}

        for i, txt in enumerate(sub_texts):
            vec = llm_vectors[i] if (llm_vectors is not None and i < len(llm_vectors)) else None
            res = self.explain_text(txt, llm_vector=vec, max_evals=90, top_k=5)
            for tok, val in zip(res.get("tokens", []), res.get("shap_values", [])):
                key = tok.lower()
                token_abs_map.setdefault(key, []).append(abs(float(val)))

        global_tokens = sorted(
            [
                {"token": k, "mean_abs_shap": round(float(np.mean(vs)), 4), "occurrences": len(vs)}
                for k, vs in token_abs_map.items()
            ],
            key=lambda x: x["mean_abs_shap"],
            reverse=True,
        )[:top_k]

        global_llm_features: List[Dict[str, Any]] = []
        if self.is_hybrid and llm_vectors is not None and len(llm_vectors) > 0:
            sub_vecs = np.asarray(llm_vectors[: min(max_samples, len(llm_vectors))], dtype=np.float32)
            all_feat_abs = []
            for i in range(len(sub_texts)):
                nres = self.explain_llm_features(sub_texts[i], sub_vecs[i], max_evals=100)
                if nres.get("available"):
                    all_feat_abs.append(np.abs(nres["shap_values"]))
            if all_feat_abs:
                mean_abs = np.mean(np.vstack(all_feat_abs), axis=0)
                f_names = get_teacher_feature_names()
                global_llm_features = sorted(
                    [{"feature": f_names[j], "mean_abs_shap": round(float(mean_abs[j]), 4)} for j in range(len(f_names))],
                    key=lambda x: x["mean_abs_shap"],
                    reverse=True,
                )[:top_k]

        return {
            "sample_count": len(sub_texts),
            "top_global_tokens": global_tokens,
            "top_global_llm_features": global_llm_features,
        }


def compute_faithfulness_comprehensiveness(
    explainer: HybridSHAPExplainer,
    text: str,
    shap_result: Dict[str, Any],
    llm_vector: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    Metrik kuantitatif ilmiah Faithfulness (Comprehensiveness):
    penurunan probabilitas kelas positif ketika token pemicu SHAP positif teratas dihapus dari teks.
    """
    top_pos = [item["token"] for item in shap_result.get("top_positive_tokens", [])[:3]]
    if not top_pos:
        return None
    orig_prob = float(shap_result.get("cyberbullying_prob", 0.0))
    remove_set = {t.lower() for t in top_pos}
    remaining = [w for w in str(text).split() if w.lower() not in remove_set]
    perturbed_text = " ".join(remaining) if remaining else ""
    if not perturbed_text:
        return None
    perturbed_prob = float(explainer.predict_proba([perturbed_text], fixed_llm_vector=llm_vector)[0][1])
    return round(orig_prob - perturbed_prob, 4)


def align_shap_with_cot(
    shap_result: Dict[str, Any],
    cot_explanation: str,
    comprehensiveness_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Menghitung metrik diagnostik keselarasan leksikal (lexical overlap heuristic) antara
    token atribusi SHAP teratas dan rationale CoT SumoPod LLM.

    CATATAN METODOLOGIS:
    Skor pencocokan leksikal merupakan indikator diagnostik heuristik, BUKAN bukti validitas
    ilmiah keterjelasan kausal. Untuk validasi kuantitatif model, gunakan metrik
    `faithfulness_comprehensiveness` dan evaluasi anotator manusia independen.
    """
    if not cot_explanation or not shap_result.get("top_positive_tokens"):
        return {
            "alignment_type": "diagnostic_lexical_heuristic",
            "aligned_tokens": [],
            "alignment_score": 0.0,
            "is_aligned": False,
            "faithfulness_comprehensiveness": comprehensiveness_score,
            "alignment_summary": "Tidak ada token pemicu positif atau narasi CoT untuk diagnostik leksikal.",
        }

    explanation_lower = cot_explanation.lower()
    aligned = []
    total_pos = len(shap_result["top_positive_tokens"])

    for item in shap_result["top_positive_tokens"]:
        tok = str(item["token"]).lower()
        clean_tok = re.sub(r"[^\w\s]", "", tok)
        if len(clean_tok) >= 3 and clean_tok in explanation_lower:
            aligned.append(item["token"])

    alignment_score = round((len(aligned) / max(1, total_pos)) * 100.0, 1)
    is_aligned = alignment_score >= 50.0

    summary = (
        f"[Diagnostik Leksikal] {len(aligned)} dari {total_pos} token SHAP teratas "
        f"({', '.join(aligned) if aligned else 'tidak ada'}) muncul secara leksikal di rationale SumoPod LLM ({alignment_score}%)."
    )

    return {
        "alignment_type": "diagnostic_lexical_heuristic",
        "aligned_tokens": aligned,
        "alignment_score": alignment_score,
        "is_aligned": is_aligned,
        "faithfulness_comprehensiveness": comprehensiveness_score,
        "alignment_summary": summary,
    }
