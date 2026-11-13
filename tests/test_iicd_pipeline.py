#!/usr/bin/env python3
"""
tests/test_iicd_pipeline.py
---------------------------
Suite pengujian terintegrasi (Unit + Integration + Streamlit Smoke) untuk:
IICD - LLM-Augmented Explainable Annotation & Distillation Studio.

Mencakup seluruh 15 kategori uji wajib:
1. Dataset loading & upload validation
2. Dataset quality audit & curation
3. Taxonomy schema validation & missing-label preservation
4. 33-dim Teacher Feature vector validation
5. Train / Validation / Test stratified separation
6. Cross-split leakage detection & GT vs pseudo-label separation
7. Model forward pass (HybridFusionModel & LLMKnowledgeOnlyClassifier)
8. 6-Head Multi-Task output validation
9. Loss calculation (-100 masking & Dynamic Uncertainty Weighting gradients)
10. Checkpoint save/load, Model Manifest, & Missing checkpoint guard
11. Evaluation metrics (PR-AUC, Calibration, CI, Val-only threshold tuning)
12. Real `shap` Dual-Modality explanation & Faithfulness Comprehensiveness
13. SumoPod API retry & secret masking verification (mocked HTTP)
14. Mock fallback engine & provenance metadata
15. Report generation (no simulation) + Streamlit UI AppTest smoke check
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from core.data_understanding import (  # noqa: E402
    CULTURAL_CUE_SLICE,
    EMOTIONAL_CUE_SLICE,
    TOTAL_VECTOR_DIM,
    clean_and_normalize_social_text,
    extract_contextual_evidence_vector,
    extract_cue_family_summary,
    extract_taxonomy_probability_vector,
    extract_teacher_annotation_features,
    get_teacher_feature_names,
    get_teacher_feature_schema,
)
from core.evaluation_metrics import (  # noqa: E402
    bootstrap_confidence_interval,
    build_model_manifest,
    compute_calibration_metrics,
    compute_comprehensive_classification_metrics,
    find_optimal_threshold_on_val,
    load_model_manifest,
    save_model_manifest,
)
from core.fusion_model import (  # noqa: E402
    IGNORE_INDEX,
    FocalLoss,
    HybridFusionModel,
    LLMKnowledgeOnlyClassifier,
    MultiTaskUncertaintyLoss,
    load_hybrid_classifier_with_manifest,
    masked_cross_entropy,
    predict_hybrid_text,
)
from core.model_engine import MockEngine, SumopodEngine, mask_secret, run_single_annotation  # noqa: E402
from core.parser import extract_json_from_response  # noqa: E402
from core.prompt_templates import PROMPT_VERSION  # noqa: E402
from core.schema import MISSING_LABEL, parse_confidence_score, sanitize_annotation_dict, validate_annotation_record  # noqa: E402
from core.shap_explainer import HybridSHAPExplainer, align_shap_with_cot, compute_faithfulness_comprehensiveness  # noqa: E402
from data_manager.dataset_handler import (  # noqa: E402
    MAX_UPLOAD_SIZE_BYTES,
    detect_ground_truth_column,
    detect_text_column,
    load_dataset_file,
    sanitize_html_text,
    validate_dataset_dataframe,
    validate_upload_file_size,
)
from scripts.curate_dataset import (  # noqa: E402
    audit_dataset_quality,
    curate_and_normalize,
    verify_no_cross_split_leakage,
)
from scripts.evaluate_consistency import (  # noqa: E402
    compute_krippendorff_alpha_nominal,
    measure_llm_annotation_consistency,
)
from scripts.generate_dissertation_report import (  # noqa: E402
    generate_latex_tables,
    generate_markdown_report,
    run_full_evaluation_pipeline,
)


# ---------------------------------------------------------------------------
# Fixtures: Offline Tiny Transformer Backbone & Sample Datasets
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_hf_backbone(tmp_path_factory) -> Path:
    """Membuat backbone BertModel mini 100% offline agar tes cepat tanpa unduhan jaringan."""
    from transformers import BertConfig, BertForSequenceClassification, BertModel, BertTokenizer

    model_dir = tmp_path_factory.mktemp("tiny_bert")
    vocab_tokens = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "dasar", "jelek", "banget", "kamu", "muka", "kayak",
        "ursula", "keren", "bagus", "semangat", "belajar",
        "bodoh", "tolol", "cantik", "hebat", "selamat", "pagi",
    ]
    vocab_file = model_dir / "vocab.txt"
    vocab_file.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")

    tokenizer = BertTokenizer(vocab_file=str(vocab_file), do_lower_case=True)
    tokenizer.save_pretrained(str(model_dir))

    cfg = BertConfig(
        vocab_size=len(vocab_tokens),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    BertModel(cfg).save_pretrained(str(model_dir))
    return model_dir


def _build_synthetic_annotated_df(n_rows: int = 36, include_human_gt: bool = False) -> pd.DataFrame:
    rows = []
    for i in range(n_rows):
        is_cb = i % 2 == 0
        txt = (
            f"dasar kamu jelek banget muka kayak ursula nomor {i}"
            if is_cb
            else f"selamat pagi semangat belajar hari ini nomor {i}"
        )
        row = {
            "comment": txt,
            "sentiment": 1 if i % 3 == 0 else -1,  # intentionally NOT identical to CB to test rejection
            "cyberbullying": "YES" if is_cb else "NO",
            "type": "INSULT" if is_cb else "NONE",
            "emotion": "ANG" if is_cb else "POS",
            "cultural_context": "SLANG_GENZ" if is_cb else "NONE",
            "implicit_level": "E1" if is_cb else "E0",
            "severity": "MOD" if is_cb else "NONE",
            "confidence": 0.91 if is_cb else 0.87,
            "explanation": f"Alasan analisis komentar {i} dengan kata jelek dan ursula." if is_cb else f"Komentar netral ramah {i}.",
            "is_mock": False,
        }
        if include_human_gt:
            row["human_ground_truth"] = 1 if is_cb else 0
            row["human_type"] = "INSULT" if is_cb else "NONE"
            row["human_emotion"] = "ANG" if is_cb else "POS"
            row["human_cultural_context"] = "SLANG_GENZ" if is_cb else "NONE"
            row["human_implicit_level"] = "E1" if is_cb else "E0"
            row["human_severity"] = "MOD" if is_cb else "NONE"
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Dataset Loading, Upload Security & Validation
# ---------------------------------------------------------------------------

def test_dataset_loading_and_validation(tmp_path):
    csv_path = tmp_path / "sample.csv"
    df_in = pd.DataFrame({"comment": ["Komentar bagus sekali", "Dasar jelek banget"], "extra": [1, 2]})
    df_in.to_csv(csv_path, index=False)

    assert validate_upload_file_size(csv_path) is None
    df_loaded, load_err = load_dataset_file(csv_path)
    assert load_err is None and df_loaded is not None
    assert len(df_loaded) == 2
    assert detect_text_column(df_loaded) == "comment"

    is_ok, warnings, errors = validate_dataset_dataframe(df_loaded, text_col="comment")
    assert is_ok and not errors

    # Empty & oversized & bad extension guards
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    assert "kosong" in (validate_upload_file_size(empty_file) or "").lower()

    bad_ext = tmp_path / "data.txt"
    bad_ext.write_text("hello", encoding="utf-8")
    assert "tidak didukung" in (validate_upload_file_size(bad_ext) or "").lower()

    fake_big = SimpleNamespace(name="huge.csv", size=MAX_UPLOAD_SIZE_BYTES + 1024)
    assert "melebihi batas" in (validate_upload_file_size(fake_big) or "").lower()

    # Ground truth detection must reject raw unvalidated 'sentiment'
    assert detect_ground_truth_column(pd.DataFrame({"comment": ["x"], "sentiment": [1]})) is None
    assert detect_ground_truth_column(pd.DataFrame({"comment": ["x"], "human_ground_truth": [1]})) == "human_ground_truth"

    # HTML XSS escaping
    escaped = sanitize_html_text('<script>alert("xss")</script>')
    assert "<script>" not in escaped and "&lt;script&gt;" in escaped


# ---------------------------------------------------------------------------
# 2. Taxonomy Schema, Parser & Missing-Label Preservation
# ---------------------------------------------------------------------------

def test_schema_parser_and_missing_label_preservation():
    # Parser failure must return UNKNOWN and confidence=None (NOT fake "NO" / 0.0)
    failed = extract_json_from_response("Respons tanpa JSON valid sama sekali")
    assert failed["cyberbullying"] == MISSING_LABEL
    assert failed["type"] == MISSING_LABEL
    assert failed["confidence"] is None
    assert failed["error"]

    # Sanitization preserves UNKNOWN instead of forcing missing labels to valid taxonomy labels
    partial = sanitize_annotation_dict({"cyberbullying": "UNKNOWN", "type": "INVALID_CODE"}, preserve_missing=True)
    assert partial["cyberbullying"] == MISSING_LABEL
    assert partial["type"] == MISSING_LABEL
    assert partial["confidence"] is None

    # Consistency rule: NO cannot carry valid bullying subtype (`INS`) or severity (`HIGH`)
    inconsistent = {"cyberbullying": "NO", "type": "INS", "severity": "HIGH", "confidence": 0.9}
    is_valid, errs = validate_annotation_record(inconsistent)
    assert not is_valid and len(errs) >= 1

    fixed = sanitize_annotation_dict(inconsistent, preserve_missing=True)
    assert fixed["type"] == "NONE"
    assert fixed["severity"] == "NONE"

    # Confidence parser handles None, percentage (0-100), & out-of-range fallback cleanly
    assert parse_confidence_score(None, default=None) is None
    assert parse_confidence_score(88) == pytest.approx(0.88)
    assert parse_confidence_score("92%") == pytest.approx(0.92)
    assert parse_confidence_score(150.0, default=None) is None
    assert parse_confidence_score(-0.2, default=None) is None


# ---------------------------------------------------------------------------
# 3 & 4. Non-Leaky 33-Dim Contextual Evidence Vector Validation
# ---------------------------------------------------------------------------

def test_33d_teacher_feature_vector_and_schema():
    names = get_teacher_feature_names()
    schema = get_teacher_feature_schema()
    assert len(names) == TOTAL_VECTOR_DIM == 33
    assert len(schema) == 5  # Linguistic, Cultural, Emotional, Pragmatic, Contextual
    assert sum(block["dim_end"] - block["dim_start"] for block in schema) == TOTAL_VECTOR_DIM == 33

    # ANTI-LEAKAGE INVARIANCE: Mengubah seluruh kolom target TIDAK boleh mengubah vektor 33D!
    base_evidence = {
        "comment": "dasar jancuk muka mirip ursula segentong wkwk??",
        "explanation": "Compares appearance to Ursula, mocking weight with rhetorical sarcasm.",
        "confidence": 0.88,
        "cyberbullying": "YES",
        "type": "BSH",
        "emotion": "MOCK",
        "cultural_context": "APP",
        "implicit_level": "E1",
        "severity": "HIGH",
    }
    flipped_targets = {
        **base_evidence,
        "cyberbullying": "NO",
        "type": "NONE",
        "emotion": "NEU",
        "cultural_context": "NONE",
        "implicit_level": "E0",
        "severity": "NONE",
    }
    vec_a = extract_contextual_evidence_vector(base_evidence)
    vec_b = extract_contextual_evidence_vector(flipped_targets)
    assert vec_a.shape == (33,) and vec_a.dtype == np.float32
    assert np.allclose(vec_a, vec_b)  # Zero target label leakage!
    assert np.any(vec_a[CULTURAL_CUE_SLICE] > 0.0)
    assert np.any(vec_a[EMOTIONAL_CUE_SLICE] > 0.0)
    assert vec_a[32] == pytest.approx(0.88)

    fam_summary = extract_cue_family_summary(vec_a)
    assert set(fam_summary.keys()) == {"Linguistic", "Cultural", "Emotional", "Pragmatic", "Contextual"}

    # Structured SumoPod v3.0 contextual_cues supported natively
    cues_v3 = {
        "comment": "halo",
        "confidence": 0.90,
        "contextual_cues": {
            "cultural": {"appearance_norm": 0.8},
            "emotional": {"derision_ridicule": 0.7},
        },
    }
    vec_v3 = extract_teacher_annotation_features(cues_v3)
    assert vec_v3[8] == pytest.approx(0.8)  # cult_appearance_norm_cue
    assert vec_v3[15] == pytest.approx(0.7)  # emo_derision_ridicule

    # Empty neutral input yields zero vector
    vec_unk = extract_taxonomy_probability_vector({"comment": "", "explanation": "", "confidence": None})
    assert np.allclose(vec_unk, np.zeros(33, dtype=np.float32))


# ---------------------------------------------------------------------------
# 5 & 6. Dataset Curation, Stratified Splits & Cross-Split Leakage Detection
# ---------------------------------------------------------------------------

def test_curation_pseudo_label_and_human_gt_separation(tmp_path):
    # Case A: Pseudo-labeled dataset (only unvalidated `sentiment` column)
    raw_df = _build_synthetic_annotated_df(40, include_human_gt=False)
    # Inject duplicates + mock row + empty row to test quality filtering
    dup_row = raw_df.iloc[0].copy()
    mock_row = raw_df.iloc[1].copy()
    mock_row["is_mock"] = True
    empty_row = raw_df.iloc[2].copy()
    empty_row["comment"] = "  "
    dirty_df = pd.concat([raw_df, pd.DataFrame([dup_row, mock_row, empty_row])], ignore_index=True)

    raw_csv = tmp_path / "raw_checkpoint.csv"
    dirty_df.to_csv(raw_csv, index=False)
    raw_mtime_before = raw_csv.stat().st_mtime

    audit = audit_dataset_quality(dirty_df, text_col="comment")
    assert audit["normalized_duplicates"] >= 1
    assert audit["empty_comments"] >= 1
    assert audit["mock_simulation_rows"] >= 1
    assert audit["label_source"] == "pseudo_labeled_llm"

    out_dir = tmp_path / "final_pseudo"
    out_dir.mkdir()
    curate_and_normalize(raw_csv, out_dir / "cleaned.csv", perform_split=True, random_state=42)

    # Raw dataset must remain untouched
    assert raw_csv.stat().st_mtime == raw_mtime_before
    assert len(pd.read_csv(raw_csv)) == 43

    manifest = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["label_source"] == "pseudo_labeled_llm"
    assert manifest["supervision_policy"] == "llm_generated_pseudo_labels"
    assert manifest["confidence_policy"] == "raw_llm_confidence_uncalibrated"
    assert manifest["has_human_ground_truth"] is False
    assert manifest["failed_annotations_count"] >= 2
    assert (out_dir / "failed_annotations_log.json").exists()
    leak_info = manifest["splits"]["cross_split_leakage_check"]
    assert sum(sum(v.values()) for v in leak_info.values()) == 0
    assert (out_dir / "teacher_feature_names.json").exists()

    # Group-aware split test: verify 0 overlap across user_id groups
    grp_df = _build_synthetic_annotated_df(30, include_human_gt=False)
    grp_df["user_id"] = [f"user_{i // 3}" for i in range(len(grp_df))]
    grp_csv = tmp_path / "grouped_raw.csv"
    grp_df.to_csv(grp_csv, index=False)
    out_grp = tmp_path / "final_grouped"
    out_grp.mkdir()
    curate_and_normalize(grp_csv, out_grp / "cleaned_grp.csv", perform_split=True, random_state=42, group_col="user_id")
    grp_man = json.loads((out_grp / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert "group_aware_stratified" in grp_man["splits"]["strategy"]
    assert sum(sum(v.values()) for v in grp_man["splits"]["cross_split_leakage_check"].values()) == 0

    tr = pd.read_csv(out_dir / "train.csv")
    va = pd.read_csv(out_dir / "val.csv")
    te = pd.read_csv(out_dir / "test.csv")
    assert np.load(out_dir / "train_llm_vectors.npy").shape == (len(tr), 33)
    assert verify_no_cross_split_leakage(tr, va, te)["comment_normalized"]["train_test_overlap"] == 0

    # Leakage detector must raise ValueError if an identical normalized comment leaks across splits
    va_leaked = va.copy()
    va_leaked.loc[0, "comment_normalized"] = tr.loc[0, "comment_normalized"]
    with pytest.raises(ValueError, match="Data leakage terdeteksi"):
        verify_no_cross_split_leakage(tr, va_leaked, te)

    # Case B: True human GT column present -> separated cleanly from LLM teacher columns
    gt_df = _build_synthetic_annotated_df(30, include_human_gt=True)
    gt_csv = tmp_path / "human_gt.csv"
    gt_df.to_csv(gt_csv, index=False)
    out_gt = tmp_path / "final_gt"
    out_gt.mkdir()
    curate_and_normalize(gt_csv, out_gt / "cleaned_gt.csv", perform_split=True, random_state=42)
    gt_manifest = json.loads((out_gt / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert gt_manifest["label_source"] == "human_ground_truth"
    assert gt_manifest["has_human_ground_truth"] is True


# ---------------------------------------------------------------------------
# 7, 8, 9 & 10. Hybrid Model Forward, 6 Heads, Loss Masking & Manifest Save/Load
# ---------------------------------------------------------------------------

def test_hybrid_model_forward_6_heads_loss_masking_and_manifest(tiny_hf_backbone, tmp_path):
    model = HybridFusionModel(
        pretrained_model_name=str(tiny_hf_backbone),
        llm_input_dim=33,
        llm_hidden_dim=16,
        use_uncertainty_loss=True,
    )
    batch = 4
    seq = 12
    input_ids = torch.randint(0, 15, (batch, seq))
    attn = torch.ones((batch, seq), dtype=torch.long)
    llm_vec = torch.rand((batch, 33), dtype=torch.float32)

    # Include -100 missing labels across sub-tasks to verify masked loss & gradient propagation
    y_bin = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    y_type = torch.tensor([0, 2, IGNORE_INDEX, IGNORE_INDEX], dtype=torch.long)
    y_emo = torch.tensor([IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX], dtype=torch.long)
    y_cul = torch.tensor([0, 1, 0, IGNORE_INDEX], dtype=torch.long)
    y_imp = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    y_sev = torch.tensor([0, 2, IGNORE_INDEX, 1], dtype=torch.long)

    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        llm_vectors=llm_vec,
        labels_binary=y_bin,
        labels_type=y_type,
        labels_emotion=y_emo,
        labels_cultural=y_cul,
        labels_implicit=y_imp,
        labels_severity=y_sev,
    )
    assert out["logits"]["binary"].shape == (batch, 2)
    assert out["logits"]["type"].shape == (batch, 9)
    assert out["logits"]["emotion"].shape == (batch, 6)
    assert out["logits"]["cultural"].shape == (batch, 8)
    assert out["logits"]["implicit"].shape == (batch, 3)
    assert out["logits"]["severity"].shape == (batch, 4)
    assert torch.all((out["gate_values"] > 0.0) & (out["gate_values"] < 1.0))
    assert out["gate_mean_weight"].shape == (batch,)

    loss = out["loss"]
    assert loss is not None and not torch.isnan(loss)
    loss.backward()
    assert model.uncertainty_loss_engine.log_vars.grad is not None
    assert float(model.uncertainty_loss_engine.log_vars.grad.abs().sum()) > 0.0

    # All-masked edge cases
    fl = FocalLoss(ignore_index=IGNORE_INDEX)
    assert float(fl(torch.randn(3, 2), torch.full((3,), IGNORE_INDEX, dtype=torch.long))) == 0.0
    mce_loss, mce_active = masked_cross_entropy(torch.randn(3, 4), torch.full((3,), IGNORE_INDEX, dtype=torch.long))
    assert float(mce_loss) == 0.0 and mce_active is False

    # Stream-B only classifier
    llm_clf = LLMKnowledgeOnlyClassifier(in_features=33, hidden_dim=16)
    out_llm = llm_clf(llm_vec, labels_binary=y_bin)
    assert out_llm["logits"]["binary"].shape == (batch, 2) and out_llm["loss"] is not None

    # Save weights + tokenizer + manifest and reload cleanly
    ckpt_dir = tmp_path / "hybrid_ckpt"
    ckpt_dir.mkdir()
    weights_path = ckpt_dir / "best_hybrid_model.pt"
    torch.save(model.state_dict(), weights_path)

    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(str(tiny_hf_backbone)).save_pretrained(str(ckpt_dir))
    manifest = build_model_manifest(
        model_architecture="HybridFusionModel",
        pretrained_model=str(tiny_hf_backbone),
        checkpoint_location=weights_path,
        hyperparameters={"max_length": 32, "llm_hidden_dim": 16},
        evaluation_metrics={"accuracy": 0.75, "macro_f1": 0.74},
        optimal_threshold=0.48,
        dataset_version="test_v1",
        target_source="pseudo_labeled_llm",
        random_seed=42,
    )
    save_model_manifest(ckpt_dir, manifest)

    reloaded_m, reloaded_tok, dev, reloaded_man = load_hybrid_classifier_with_manifest(weights_path)
    assert reloaded_man["optimal_threshold"] == pytest.approx(0.48)
    pred = predict_hybrid_text(reloaded_m, reloaded_tok, dev, "dasar jelek banget", annotation_dict=None, manifest=reloaded_man)
    assert pred["cyberbullying"] in {"YES", "NO"}
    assert pred["has_llm_context"] is False
    assert pred["llm_modality_active"] is True  # surface linguistic cue active without annotation_dict
    assert len(pred["all_head_probabilities"]) == 6

    # E6 Concatenation Fusion variant & Cultural/Emotional cue slicing
    concat_model = HybridFusionModel(
        pretrained_model_name=str(tiny_hf_backbone),
        llm_input_dim=33,
        llm_hidden_dim=16,
        fusion_mode="concat",
    )
    out_concat = concat_model(input_ids=input_ids, attention_mask=attn, llm_vectors=llm_vec, labels_binary=y_bin)
    assert out_concat["logits"]["binary"].shape == (batch, 2) and out_concat["loss"] is not None

    with pytest.raises(FileNotFoundError):
        load_hybrid_classifier_with_manifest(tmp_path / "does_not_exist.pt")


# ---------------------------------------------------------------------------
# 11. Evaluation Metrics & Validation-Only Threshold Selection
# ---------------------------------------------------------------------------

def test_evaluation_metrics_calibration_and_threshold():
    rng = np.random.default_rng(42)
    y_val = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    p_val = np.array([0.1, 0.2, 0.35, 0.65, 0.85, 0.90, 0.30, 0.72, 0.15, 0.80])
    opt_th, val_f1, val_acc = find_optimal_threshold_on_val(p_val, y_val)
    assert 0.30 <= opt_th <= 0.75
    assert val_f1 > 0.8 and val_acc > 0.8

    cal = compute_calibration_metrics(y_val, p_val, n_bins=5)
    assert 0.0 <= cal["ece"] <= 1.0 and 0.0 <= cal["brier_score"] <= 1.0

    df_meta = pd.DataFrame({"target_implicit_level": ["E0"] * 5 + ["E1"] * 5, "target_type": ["INSULT"] * 10})
    res = compute_comprehensive_classification_metrics(y_val, p_val, threshold=opt_th, df_meta=df_meta, n_bootstrap=50)
    assert "pr_auc" in res and "roc_auc" in res and "calibration" in res
    assert "E0" in res["implicitness_breakdown"] and "E1" in res["implicitness_breakdown"]
    assert res["error_analysis"]["false_positive_count"] == 0

    ci = bootstrap_confidence_interval(y_val, p_val, threshold=opt_th, n_bootstrap=30)
    assert ci["accuracy"][0] is not None and ci["accuracy"][0] <= ci["accuracy"][1]
    assert ci["macro_f1"][0] is not None and ci["macro_f1"][0] <= ci["macro_f1"][1]


# ---------------------------------------------------------------------------
# 12. Real `shap` Dual-Modality Explainer & Faithfulness
# ---------------------------------------------------------------------------

def test_real_shap_dual_modality_and_missing_checkpoint(tiny_hf_backbone, tmp_path):
    with pytest.raises(FileNotFoundError):
        HybridSHAPExplainer(model_path=tmp_path / "nonexistent_weights.pt", model_type="hybrid")

    model = HybridFusionModel(
        pretrained_model_name=str(tiny_hf_backbone),
        llm_input_dim=33,
        llm_hidden_dim=16,
    )
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tiny_hf_backbone))

    explainer = HybridSHAPExplainer(
        model_type="hybrid",
        loaded_model=model,
        loaded_tokenizer=tokenizer,
        device="cpu",
        manifest={"hyperparameters": {"max_length": 32}},
        background_vectors=np.zeros((4, 33), dtype=np.float32),
    )

    anno = {
        "llm_cyberbullying": "YES",
        "llm_type": "INSULT",
        "llm_emotion": "ANG",
        "llm_cultural_context": "SLANG_GENZ",
        "llm_implicit_level": "E1",
        "llm_severity": "MOD",
        "llm_confidence": 0.90,
    }
    dual = explainer.explain_dual_modalities(
        "dasar kamu jelek banget <script>x</script>",
        annotation_dict=anno,
        max_evals_text=30,
        max_evals_vec=70,
    )
    text_res = dual["textual_shap"]
    vec_res = dual["llm_vector_shap"]

    assert "shap.Explainer" in text_res["method"]
    assert len(text_res["tokens"]) > 0
    assert "<script>" not in text_res["html_heatmap"]  # HTML escaped in heatmap
    assert vec_res["available"] is True and len(vec_res["shap_values"]) == 33
    assert set(vec_res["cue_family_shap"].keys()) == {"Linguistic", "Cultural", "Emotional", "Pragmatic", "Contextual"}
    cm_contrib = dual["cross_modal_contribution"]
    assert cm_contrib["available"] is True
    assert (cm_contrib["transformer_gate_z"] + cm_contrib["llm_knowledge_share"]) == pytest.approx(1.0, abs=1e-3)

    comp = compute_faithfulness_comprehensiveness(explainer, "dasar kamu jelek banget", text_res)
    assert comp is None or isinstance(comp, float)

    align = align_shap_with_cot(text_res, "Komentar ini mengandung kata jelek.", comprehensiveness_score=comp)
    assert align["alignment_type"] == "diagnostic_lexical_heuristic"


# ---------------------------------------------------------------------------
# 13 & 14. SumoPod API Retries, Secret Masking & Mock Fallback
# ---------------------------------------------------------------------------

def test_sumopod_retries_secret_masking_and_mock_engine():
    secret = "sk-test-secret-1234567890"
    assert secret not in mask_secret(secret)

    eng = SumopodEngine(api_key=secret, max_retries=2, backoff_base=0.01)
    assert secret not in eng._safe_error_text(f"Auth failure using {secret}")

    # Simulate HTTP 429 transient error followed by HTTP 200 JSON response
    resp_429 = SimpleNamespace(status_code=429, text="rate limit", headers={"Retry-After": "0.01"})
    payload_ok = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "cyberbullying": "YES",
                            "type": "INSULT",
                            "emotion": "ANG",
                            "cultural_context": "GENERAL",
                            "implicit_level": "E0",
                            "severity": "MOD",
                            "confidence": 0.92,
                            "explanation": "Mengandung penghinaan langsung.",
                        }
                    )
                },
                "logprobs": {"content": [{"token": "YES", "logprob": -0.05}]},
            }
        ]
    }
    resp_200 = SimpleNamespace(status_code=200, text="ok", headers={}, json=lambda: payload_ok)

    with patch("core.model_engine.requests.post", side_effect=[resp_429, resp_200]) as mocked_post:
        gen = eng.generate_with_metadata([{"role": "user", "content": "test"}], request_logprobs=True)
        assert mocked_post.call_count == 2
        assert gen["logprobs_available"] is True
        assert gen["api_mean_token_prob"] is not None and gen["api_mean_token_prob"] > 0.9

    with patch("core.model_engine.requests.post", return_value=resp_200):
        res = run_single_annotation(eng, "kamu bodoh banget")
        assert res["cyberbullying"] == "YES"
        assert res["annotation_source"] == "sumopod_api"
        assert res["prompt_version"] == PROMPT_VERSION
        assert res["is_mock"] is False

    # Mock engine provenance
    mock_eng = MockEngine()
    assert mock_eng.is_mock is True
    assert mock_eng.backend_type == "mock"
    m_res = run_single_annotation(mock_eng, "dasar jelek banget")
    assert m_res["is_mock"] is True
    assert m_res["annotation_source"] == "mock_simulation"
    assert m_res["prompt_version"] == PROMPT_VERSION


# ---------------------------------------------------------------------------
# 15. Report Generation (No Simulation) & Streamlit UI Smoke Check
# ---------------------------------------------------------------------------

def test_report_generation_no_simulation_and_streamlit_smoke(tmp_path):
    # Must raise FileNotFoundError instead of synthesizing fake probabilities
    fake_csv = tmp_path / "test.csv"
    pd.DataFrame({"comment": ["tes"], "label": [0]}).to_csv(fake_csv, index=False)
    with pytest.raises(FileNotFoundError):
        run_full_evaluation_pipeline(
            model_dir=tmp_path / "missing_model",
            test_csv=fake_csv,
            output_dir=tmp_path / "reports",
        )

    sample_metrics = compute_comprehensive_classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        threshold=0.5,
        n_bootstrap=20,
    )
    out_dir = tmp_path / "report_ok"
    md_path = generate_markdown_report(sample_metrics, "pseudo_labeled_llm", None, out_dir)
    md_txt = md_path.read_text(encoding="utf-8")
    assert "pseudo_labeled_llm" in md_txt and "pseudo-label" in md_txt.lower()
    tex = generate_latex_tables(sample_metrics, None)
    assert "tab:overall_performance" in tex

    # Non-GT Consistency & Krippendorff Alpha metrics verification
    alpha = compute_krippendorff_alpha_nominal([["YES", "YES", "YES"], ["NO", "NO", "NO"], ["YES", "NO", "YES"]])
    assert alpha is not None and 0.0 <= alpha <= 1.0
    cons_mock = measure_llm_annotation_consistency(MockEngine(), ["kamu jelek banget", "semangat belajar ya"], n_repeats=2)
    assert cons_mock["evaluation_scope"] == "llm_internal_consistency_not_human_iaa"
    assert 0.0 <= cons_mock["binary_unanimous_agreement_rate"] <= 1.0

    # Streamlit AppTest smoke test on app.py
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(PROJECT_DIR / "app.py"))
    at.run(timeout=30)
    assert not at.exception


# ===========================================================================
# 15 MANDATORY AUDIT & METHODOLOGICAL REFACTORING TESTS (TEST 1 - TEST 15)
# ===========================================================================

def test_01_contextual_evidence_always_33_dimensions():
    """TEST 1: Contextual evidence harus selalu menghasilkan 33 dimensi."""
    for inp in [
        {},
        {"comment": "hallo"},
        {"comment": "dasar jelek", "confidence": 0.85},
        {"comment": "test", "contextual_cues": {"linguistic": {"derogatory_lexical": 0.5}}},
    ]:
        vec = extract_contextual_evidence_vector(inp)
        assert vec.shape == (33,), f"Expected shape (33,), got {vec.shape}"
        assert vec.dtype == np.float32
        assert np.all(np.isfinite(vec)), "Vector contains non-finite values"
        assert np.all((vec >= 0.0) & (vec <= 1.0)), "Values outside [0.0, 1.0]"


def test_02_target_label_alteration_invariance():
    """TEST 2: Perubahan target label tidak boleh mengubah contextual evidence vector."""
    base_evidence = {
        "comment": "dasar kamu jelek banget muka kayak ursula",
        "confidence": 0.88,
        "contextual_cues": {
            "linguistic": {"derogatory_lexical": 0.9},
            "cultural": {"appearance_norm": 0.8},
        },
        "cyberbullying": "YES",
        "type": "BSH",
        "emotion": "MOCK",
        "cultural_context": "APP",
        "implicit_level": "E1",
        "severity": "HIGH",
    }
    flipped = {
        **base_evidence,
        "cyberbullying": "NO",
        "type": "NONE",
        "emotion": "NEU",
        "cultural_context": "NONE",
        "implicit_level": "E0",
        "severity": "NONE",
    }
    vec1 = extract_contextual_evidence_vector(base_evidence)
    vec2 = extract_contextual_evidence_vector(flipped)
    assert np.allclose(vec1, vec2), "Target labels altered the contextual evidence vector!"


def test_03_explanation_conclusion_independence():
    """TEST 3: Free-form target conclusion dalam explanation tidak boleh memengaruhi Stream B."""
    base_evidence = {
        "comment": "kamu kok terlihat gemukan sekarang",
        "confidence": 0.85,
        "contextual_cues": {"cultural": {"appearance_norm": 0.7}},
    }
    exp_cb = {**base_evidence, "explanation": "The comment is explicit cyberbullying and body shaming insult."}
    exp_neutral = {**base_evidence, "explanation": "This is a polite neutral observation without harmful intent."}
    exp_empty = {**base_evidence, "explanation": ""}

    vec_a = extract_contextual_evidence_vector(exp_cb)
    vec_b = extract_contextual_evidence_vector(exp_neutral)
    vec_c = extract_contextual_evidence_vector(exp_empty)

    assert np.allclose(vec_a, vec_b), "Changing explanation changed the 33D feature vector!"
    assert np.allclose(vec_a, vec_c), "Empty explanation changed the 33D feature vector!"


def test_04_missing_contextual_cues_explicit_policy():
    """TEST 4: Missing contextual cues harus ditangani berdasarkan kebijakan yang eksplisit."""
    vec_missing = extract_contextual_evidence_vector({
        "comment": "dasar bego dan sok pintar lu!",
        "contextual_cues": None,
        "confidence": None,
    })
    assert vec_missing.shape == (33,)
    assert np.all(np.isfinite(vec_missing))
    assert vec_missing[0] > 0.0  # ling_derogatory_lexical from 'bego'
    assert vec_missing[2] > 0.0  # ling_rhetorical_punct from '!'
    assert vec_missing[5] > 0.0  # ling_second_person_target from 'lu'
    assert vec_missing[32] == 0.0  # missing confidence defaults to 0.0 explicitly


def test_05_batch_annotation_persists_contextual_cues(tmp_path):
    """TEST 5: Batch annotation harus mempertahankan contextual_cues."""
    cues_data = {
        "linguistic": {"derogatory_lexical": 0.85},
        "cultural": {"appearance_norm": 0.70},
        "emotional": {"derision_ridicule": 0.60},
        "pragmatic": {"irony_sarcasm": 0.50},
        "contextual": {"direct_addressee": 1.0},
    }
    df = pd.DataFrame([{"comment": "tes komentar", "user_id": "u1"}])
    annotation_cols = [
        "cyberbullying", "type", "emotion", "cultural_context",
        "implicit_level", "severity", "confidence", "explanation", "contextual_cues", "error",
        "llm_model", "prompt_version", "annotated_at", "annotation_source", "is_mock"
    ]
    for col in annotation_cols:
        df[col] = None

    res = {
        "cyberbullying": "YES",
        "type": "INS",
        "emotion": "MOCK",
        "cultural_context": "APP",
        "implicit_level": "E0",
        "severity": "LOW",
        "confidence": 0.90,
        "explanation": "Contoh alasan",
        "contextual_cues": cues_data,
        "error": None,
        "llm_model": "deepseek-v4.1-flash:netra",
        "prompt_version": PROMPT_VERSION,
        "annotated_at": "2026-11-13T00:00:00Z",
        "annotation_source": "sumopod_api",
        "is_mock": False,
    }
    for k in annotation_cols:
        if k in res:
            val = res[k]
            if k == "contextual_cues" and isinstance(val, (dict, list)):
                val = json.dumps(val, ensure_ascii=False)
            df.at[0, k] = val

    saved_cues = df.at[0, "contextual_cues"]
    assert saved_cues is not None
    loaded_dict = json.loads(saved_cues)
    assert loaded_dict["cultural"]["appearance_norm"] == 0.70


def test_06_json_annotation_schema_compliance():
    """TEST 6: JSON annotation harus memenuhi schema."""
    valid_payload = json.dumps({
        "cyberbullying": "YES",
        "type": "BSH",
        "emotion": "MOCK",
        "cultural_context": "APP",
        "implicit_level": "E1",
        "severity": "LOW",
        "confidence": 0.88,
        "explanation": "Sarkasme fisik",
        "contextual_cues": {"cultural": {"appearance_norm": 0.9}},
    })
    rec = extract_json_from_response(valid_payload)
    is_valid, issues = validate_annotation_record(rec)
    assert is_valid is True, f"Validation issues: {issues}"
    assert rec["cyberbullying"] == "YES"
    assert rec["type"] == "BSH"
    assert rec["contextual_cues"]["cultural"]["appearance_norm"] == 0.9


def test_07_mock_annotation_excluded_from_training(tmp_path):
    """TEST 7: Mock annotation tidak boleh masuk ke training dataset."""
    df_raw = _build_synthetic_annotated_df(20, include_human_gt=False)
    df_raw.loc[:9, "is_mock"] = True
    df_raw.loc[:9, "annotation_source"] = "mock_simulation"

    raw_csv = tmp_path / "raw_with_mock.csv"
    df_raw.to_csv(raw_csv, index=False)
    out_dir = tmp_path / "mock_test_out"
    out_dir.mkdir()

    curate_and_normalize(raw_csv, out_dir / "clean.csv", perform_split=True, allow_mock_rows=False)
    clean_df = pd.read_csv(out_dir / "clean.csv")
    assert len(clean_df) <= 10
    if "is_mock" in clean_df.columns:
        assert not any(clean_df["is_mock"].fillna(False).astype(bool))
    if "annotation_source" in clean_df.columns:
        assert not any(clean_df["annotation_source"].fillna("").astype(str) == "mock_simulation")
    failed_log = json.loads((out_dir / "failed_annotations_log.json").read_text(encoding="utf-8"))
    assert any("mock_simulation" in rec["reasons"] for rec in failed_log["records"])


def test_08_group_aware_split_no_overlap(tmp_path):
    """TEST 8: Train, validation, dan test tidak boleh memiliki group overlap."""
    df = _build_synthetic_annotated_df(36, include_human_gt=False)
    df["user_id"] = [f"user_{i // 4}" for i in range(len(df))]
    raw_csv = tmp_path / "groups.csv"
    df.to_csv(raw_csv, index=False)
    out_dir = tmp_path / "out_grp"
    out_dir.mkdir()

    curate_and_normalize(raw_csv, out_dir / "clean.csv", perform_split=True, group_col="user_id")
    tr = pd.read_csv(out_dir / "train.csv")
    va = pd.read_csv(out_dir / "val.csv")
    te = pd.read_csv(out_dir / "test.csv")

    tr_groups = set(tr["user_id"].unique())
    va_groups = set(va["user_id"].unique())
    te_groups = set(te["user_id"].unique())

    assert len(tr_groups.intersection(va_groups)) == 0, "Train-Val group overlap detected!"
    assert len(tr_groups.intersection(te_groups)) == 0, "Train-Test group overlap detected!"
    assert len(va_groups.intersection(te_groups)) == 0, "Val-Test group overlap detected!"


def test_09_duplicate_normalized_comment_cross_split_rejection():
    """TEST 9: Tidak boleh terdapat duplicate normalized comments lintas split."""
    tr = pd.DataFrame({"comment_normalized": ["kamu jelek sekali", "selamat pagi"]})
    va = pd.DataFrame({"comment_normalized": ["kamu jelek sekali"]})
    te = pd.DataFrame({"comment_normalized": ["terima kasih"]})

    with pytest.raises(ValueError, match="Data leakage terdeteksi"):
        verify_no_cross_split_leakage(tr, va, te)


def test_10_model_output_dimensions_and_heads(tiny_hf_backbone):
    """TEST 10: Semua output model harus memiliki dimensi yang sesuai."""
    model = HybridFusionModel(pretrained_model_name=str(tiny_hf_backbone), llm_input_dim=33, llm_hidden_dim=16)
    b, s = 2, 8
    inp = torch.randint(0, 10, (b, s))
    mask = torch.ones((b, s), dtype=torch.long)
    vec = torch.rand((b, 33))

    out = model(input_ids=inp, attention_mask=mask, llm_vectors=vec)
    assert out["logits"]["binary"].shape == (b, 2)
    assert out["logits"]["type"].shape == (b, 9)
    assert out["logits"]["emotion"].shape == (b, 6)
    assert out["logits"]["cultural"].shape == (b, 8)
    assert out["logits"]["implicit"].shape == (b, 3)
    assert out["logits"]["severity"].shape == (b, 4)
    assert out["gate_values"].shape == (b, model.trans_hidden_dim)


def test_11_masked_loss_safety_all_ignore_index():
    """TEST 11: Masked loss harus aman ketika seluruh label dalam satu task bernilai ignore_index."""
    fl = FocalLoss(ignore_index=IGNORE_INDEX)
    logits = torch.randn(4, 2, requires_grad=True)
    all_ignored = torch.full((4,), IGNORE_INDEX, dtype=torch.long)
    loss = fl(logits, all_ignored)
    assert float(loss) == 0.0
    loss.backward()
    assert not torch.isnan(logits.grad).any()

    mce_loss, mce_active = masked_cross_entropy(logits, all_ignored, ignore_index=IGNORE_INDEX)
    assert float(mce_loss) == 0.0 and mce_active is False


def test_12_checkpoint_save_and_reload(tiny_hf_backbone, tmp_path):
    """TEST 12: Checkpoint dapat disimpan dan dimuat kembali."""
    model = HybridFusionModel(pretrained_model_name=str(tiny_hf_backbone), llm_input_dim=33, llm_hidden_dim=16)
    weights_path = tmp_path / "best_hybrid_model.pt"
    torch.save(model.state_dict(), weights_path)

    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(str(tiny_hf_backbone)).save_pretrained(str(tmp_path))

    manifest = build_model_manifest(
        model_architecture="HybridFusionModel",
        pretrained_model=str(tiny_hf_backbone),
        checkpoint_location=weights_path,
        hyperparameters={"llm_hidden_dim": 16},
        evaluation_metrics={"macro_f1": 0.8},
        optimal_threshold=0.52,
    )
    save_model_manifest(tmp_path, manifest)

    reloaded_m, _, _, reloaded_man = load_hybrid_classifier_with_manifest(weights_path)
    assert reloaded_man["optimal_threshold"] == pytest.approx(0.52)
    assert reloaded_m is not None


def test_13_model_inference_consistency_with_manifest(tiny_hf_backbone, tmp_path):
    """TEST 13: Model inference harus konsisten dengan model manifest."""
    model = HybridFusionModel(pretrained_model_name=str(tiny_hf_backbone), llm_input_dim=33, llm_hidden_dim=16)
    weights_path = tmp_path / "best_hybrid_model.pt"
    torch.save(model.state_dict(), weights_path)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tiny_hf_backbone))
    tokenizer.save_pretrained(str(tmp_path))

    manifest = build_model_manifest(
        model_architecture="HybridFusionModel",
        pretrained_model=str(tiny_hf_backbone),
        checkpoint_location=weights_path,
        hyperparameters={"llm_hidden_dim": 16},
        evaluation_metrics={"macro_f1": 0.75},
        optimal_threshold=0.35,
    )
    save_model_manifest(tmp_path, manifest)

    loaded_m, tok, dev, man = load_hybrid_classifier_with_manifest(weights_path)
    pred = predict_hybrid_text(loaded_m, tok, dev, "kamu jelek sekali", manifest=man)
    assert pred["threshold"] == pytest.approx(0.35)
    assert pred["cyberbullying"] in {"YES", "NO"}
    assert len(pred["all_head_probabilities"]) == 6


def test_14_shap_explanation_uses_real_model(tiny_hf_backbone, tmp_path):
    """TEST 14: SHAP explanation harus menggunakan model yang benar-benar terlatih."""
    with pytest.raises(FileNotFoundError):
        HybridSHAPExplainer(model_path=tmp_path / "nonexistent.pt", model_type="hybrid")

    model = HybridFusionModel(pretrained_model_name=str(tiny_hf_backbone), llm_input_dim=33, llm_hidden_dim=16)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tiny_hf_backbone))

    explainer = HybridSHAPExplainer(
        model_type="hybrid",
        loaded_model=model,
        loaded_tokenizer=tokenizer,
        device="cpu",
        manifest={"hyperparameters": {"max_length": 16}},
        background_vectors=np.zeros((2, 33), dtype=np.float32),
    )
    res = explainer.explain_dual_modalities("dasar jelek", max_evals_text=20, max_evals_vec=40)
    assert "shap.Explainer" in res["textual_shap"]["method"]
    assert res["llm_vector_shap"]["available"] is True
    assert len(res["llm_vector_shap"]["shap_values"]) == 33


def test_15_metrics_reproducibility_from_prediction_artifacts():
    """TEST 15: Seluruh metrik harus dapat direproduksi dari prediction artifacts."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15])
    m1 = compute_comprehensive_classification_metrics(y_true, y_prob, threshold=0.5, n_bootstrap=20)
    m2 = compute_comprehensive_classification_metrics(y_true, y_prob, threshold=0.5, n_bootstrap=20)

    assert m1["accuracy"] == m2["accuracy"] == 1.0
    assert m1["macro_f1"] == m2["macro_f1"] == 1.0
    assert m1["roc_auc"] == m2["roc_auc"] == 1.0
    assert m1["calibration"]["brier_score"] == pytest.approx(m2["calibration"]["brier_score"])
