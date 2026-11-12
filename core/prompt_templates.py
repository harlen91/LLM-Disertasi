from typing import List, Dict, Optional

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
"""

DEFAULT_SYSTEM_PROMPT = """You are an expert Indonesian cyberbullying annotation researcher.
Analyze the given text strictly according to the provided multi-dimensional taxonomy.

CRITICAL RULES:
1. Every field is REQUIRED. NEVER leave any field empty (""), null, or missing.
2. If cyberbullying is NO: explicitly set type="NONE", cultural_context="NONE", severity="NONE" (or "LOW"), implicit_level="E0".
3. confidence MUST be a decimal float between 0.00 and 1.00 (e.g. 0.95), NEVER words like "HIGH" or "LOW".
4. explanation MUST be a concise 1-2 sentence rationale.
5. Return strictly a single valid JSON object without markdown backticks (```) or any extra commentary."""

def build_user_prompt(text: str) -> str:
    """Constructs the structured user prompt for multi-dimensional cyberbullying annotation."""
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
  "explanation": "concise rationale"
}}"""


def build_chat_messages(text: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Constructs OpenAI-compatible chat messages for the SumoPod AI Gateway API."""
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": build_user_prompt(text)},
    ]
