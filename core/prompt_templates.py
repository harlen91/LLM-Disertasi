import hashlib
from typing import List, Dict, Optional

PROMPT_VERSION = "iicd-v3.0-sumopod"

LABEL_SCHEMA = """
Cyberbullying:
YES / NO

Type:
INS=Insult
BSH=Body Shaming
SHM=Social Humiliation
SAR=Sarcasm
THR=Threat
DIS=Discrimination
SEX=Sexual Harassment
EXC=Social Exclusion
NONE=None (Wajib dipilih jika Cyberbullying adalah NO)

Emotion:
ANG=Anger
DIS=Disgust
HUM=Humiliation
MOCK=Mockery
SAD=Sadness
NEU=Neutral

Cultural Context:
APP=Appearance
INT=Intelligence
STA=Social Status
FAM=Family Background
EDU=Education
REG=Regional Identity
GEN=Gender
NONE=None (Wajib dipilih jika tidak ada konteks kultural spesifik atau Cyberbullying adalah NO)

Implicit:
E0=Explicit
E1=Semi Implicit
E2=Implicit

Severity:
LOW / MEDIUM / HIGH / NONE (NONE jika Cyberbullying adalah NO)

Confidence:
Nilai probabilitas desimal antara 0.00 hingga 1.00 (misalnya 0.95, 0.85, 0.70).
WAJIB angka numerik desimal. DILARANG menggunakan kata seperti "HIGH", "MEDIUM", atau "LOW".

Contextual Evidence Cues (skor bukti independen 0.00-1.00 per sinyal, terpisah dari label target):
- linguistic: derogatory_lexical, metaphor_simile, rhetorical_punct
- cultural: appearance_norm, competence_norm, status_hierarchy, regional_dialect, family_kinship, gender_identity
- emotional: hostility_aggression, derision_ridicule, disgust_contempt, humiliation_shaming, sadness_distress
- pragmatic: irony_sarcasm, presupposition_insinuation, indirect_advice, benign_supportive
- contextual: direct_addressee, public_stage, evaluative_judgment, harm_impact_evidence
"""

DEFAULT_SYSTEM_PROMPT = """You are an expert Indonesian cyberbullying annotation researcher.
Analyze the given text strictly according to the provided multi-dimensional taxonomy and extract independent contextual evidence cues.

CRITICAL RULES:
1. Every taxonomy field is REQUIRED. NEVER leave any field empty (""), null, or missing.
2. If cyberbullying is NO: explicitly set type="NONE", cultural_context="NONE", severity="NONE" (or "LOW"), implicit_level="E0".
3. confidence MUST be a decimal float between 0.00 and 1.00 (e.g. 0.95), NEVER words like "HIGH" or "LOW".
4. explanation MUST be a concise 1-2 sentence chain-of-thought rationale describing linguistic, cultural, emotional, and pragmatic evidence in the comment.
5. contextual_cues MUST provide decimal evidence scores [0.00-1.00] for linguistic, cultural, emotional, pragmatic, and contextual cues observed in the text.
6. Return strictly a single valid JSON object without markdown backticks (```) or any extra commentary."""

def get_prompt_hash(system_prompt: Optional[str] = None) -> str:
    """Returns SHA256 hex digest (first 16 chars) of the prompt template for provenance tracking."""
    content = (system_prompt or DEFAULT_SYSTEM_PROMPT) + "\n" + LABEL_SCHEMA
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

PROMPT_HASH = get_prompt_hash()

def build_user_prompt(text: str) -> str:
    """Constructs the structured user prompt for multi-dimensional cyberbullying annotation and cue extraction."""
    return f"""Text:
{text}

Taxonomy:
{LABEL_SCHEMA}

Format JSON yang wajib diikuti:
{{
  "cyberbullying": "YES or NO",
  "type": "INS/BSH/SHM/SAR/THR/DIS/SEX/EXC/NONE",
  "emotion": "ANG/DIS/HUM/MOCK/SAD/NEU",
  "cultural_context": "APP/INT/STA/FAM/EDU/REG/GEN/NONE",
  "implicit_level": "E0/E1/E2",
  "severity": "LOW/MEDIUM/HIGH/NONE",
  "confidence": 0.95,
  "explanation": "concise rationale",
  "contextual_cues": {{
    "linguistic": {{"derogatory_lexical": 0.0, "metaphor_simile": 0.0, "rhetorical_punct": 0.0}},
    "cultural": {{"appearance_norm": 0.0, "competence_norm": 0.0, "status_hierarchy": 0.0, "regional_dialect": 0.0, "family_kinship": 0.0, "gender_identity": 0.0}},
    "emotional": {{"hostility_aggression": 0.0, "derision_ridicule": 0.0, "disgust_contempt": 0.0, "humiliation_shaming": 0.0, "sadness_distress": 0.0}},
    "pragmatic": {{"irony_sarcasm": 0.0, "presupposition_insinuation": 0.0, "indirect_advice": 0.0, "benign_supportive": 0.0}},
    "contextual": {{"direct_addressee": 0.0, "public_stage": 0.0, "evaluative_judgment": 0.0, "harm_impact_evidence": 0.0}}
  }}
}}"""


def build_chat_messages(text: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Constructs OpenAI-compatible chat messages for the SumoPod AI Gateway API."""
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": build_user_prompt(text)},
    ]
