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
    TOTAL_VECTOR_DIM,
    clean_and_normalize_social_text,
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
# 3 & 4. 33-Dim Teacher Feature Vector Validation
# ---------------------------------------------------------------------------

def test_33d_teacher_feature_vector_and_schema():
    names = get_teacher_feature_names()
    schema = get_teacher_feature_schema()
    assert len(names) == TOTAL_VECTOR_DIM == 33
    assert len(schema) == 7
    assert sum(block["dim_end"] - block["dim_start"] for block in schema) == TOTAL_VECTOR_DIM == 33

    # Vector reads `llm_*` columns first (never human target columns)
    record = {
        "cyberbullying": "NO",  # fallback / ignored if llm_* present
        "llm_cyberbullying": "YES",
        "llm_type": "INSULT",
        "llm_emotion": "ANG",
        "llm_cultural_context": "SLANG_GENZ",
        "llm_implicit_level": "E1",
        "llm_severity": "MOD",
        "llm_confidence": 0.88,
    }
    vec = extract_teacher_annotation_features(record)
    assert vec.shape == (33,)
    assert vec.dtype == np.float32
    assert vec[1] == pytest.approx(1.0)  # YES indicator
    assert vec[0] == pytest.approx(0.0)
    assert vec[32] == pytest.approx(0.88)

    # UNKNOWN record produces pure zero vector (no synthetic probability smoothing)
    vec_unk = extract_taxonomy_probability_vector({"cyberbullying": "UNKNOWN", "confidence": None})
    assert np.allclose(vec_unk, np.zeros(33, dtype=np.float32))

    # Actual API probabilities respected when supplied
    probs = {"cyberbullying": {"NO": 0.25, "YES": 0.75}}
    vec_prob = extract_teacher_annotation_features({"cyberbullying": "YES", "confidence": 0.75}, class_probabilities=probs)
    assert vec_prob[0] == pytest.approx(0.25)
    assert vec_prob[1] == pytest.approx(0.75)


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
    assert manifest["has_human_ground_truth"] is False
    leak_info = manifest["splits"]["cross_split_leakage_check"]
    assert sum(sum(v.values()) for v in leak_info.values()) == 0
    assert (out_dir / "teacher_feature_names.json").exists()

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
    assert pred["llm_modality_active"] is False  # zeroed Stream B in text-only inference
    assert len(pred["all_head_probabilities"]) == 6

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
    assert "pseudo_labeled_llm" in md_txt and "gold standard manusia" in md_txt.lower()
    tex = generate_latex_tables(sample_metrics, None)
    assert "tab:overall_performance" in tex

    # Streamlit AppTest smoke test on app.py
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(PROJECT_DIR / "app.py"))
    at.run(timeout=30)
    assert not at.exception
