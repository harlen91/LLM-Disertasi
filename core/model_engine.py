import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, List, Dict, Any, Union
import requests

from config import DEFAULT_SUMOPOD_BASE_URL, DEFAULT_SUMOPOD_MODEL
from .prompt_templates import build_chat_messages
from .parser import extract_json_from_response


def _extract_user_text(prompt_or_messages: Union[str, List[Dict[str, str]]]) -> str:
    """Extracts user message text from OpenAI-style messages or plain string prompts."""
    if isinstance(prompt_or_messages, list):
        user_parts = [
            str(m.get("content", ""))
            for m in prompt_or_messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        raw = "\n".join(user_parts) if user_parts else " ".join(str(m.get("content", "")) for m in prompt_or_messages if isinstance(m, dict))
    else:
        raw = str(prompt_or_messages or "")

    if "Text:\n" in raw and "\n\nTaxonomy:" in raw:
        return raw.split("Text:\n", 1)[1].split("\n\nTaxonomy:", 1)[0]
    return raw


class MockEngine:
    """
    Safe fallback simulation engine when SumoPod API Key is not configured yet,
    for quick UI prototyping and offline demonstrations.
    """

    def __init__(self):
        self.device = "mock"
        self.is_mock = True
        self.backend_type = "mock"

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 160,
        temperature: float = 0.1,
        early_stop_json: bool = True,
    ) -> str:
        p_lower = _extract_user_text(prompt).lower()

        bsh_words = [
            "mandi", "jerawat", "ursula", "muka", "ampela", "segentong", "burik",
            "gendut", "gemuk", "pesek", "item", "ireng", "dekil", "kucel", "tepos",
            "ompong", "kurus", "kerempeng", "dower", "behel", "bau", "jelek"
        ]
        ins_words = [
            "bego", "tolol", "goblok", "idiot", "bloon", "bodoh", "otak", "cacat",
            "autis", "babi", "anjing", "monyet", "bangsat", "bajingan", "tai",
            "kontol", "memek", "pantek", "lonte", "jablay", "perek", "jalang",
            "hama", "rusak", "mati", "gagal", "buset"
        ]
        shm_words = [
            "kampungan", "udik", "miskin", "gembel", "pansos", "alay", "norak",
            "belagu", "songong", "sok cantik", "berasa paling cantik", "sok ganteng", "sok kaya"
        ]
        thr_words = ["mati", "bunuh", "habisi", "hajar", "gampar", "pukul", "ancam"]

        is_bsh = any(w in p_lower for w in bsh_words)
        is_ins = any(w in p_lower for w in ins_words)
        is_shm = any(w in p_lower for w in shm_words)
        is_thr = any(w in p_lower for w in thr_words)
        is_negative = is_bsh or is_ins or is_shm or is_thr

        if is_negative:
            if is_thr:
                b_type = "THR"
                cultural = "GEN"
                emotion = "ANG"
                explanation = "Teks mengandung unsur intimidasi dan ancaman kekerasan secara eksplisit."
                severity = "HIGH"
                implicitness = "E0"
            elif is_bsh:
                b_type = "BSH"
                cultural = "APP"
                emotion = "MOCK"
                explanation = "Teks mengandung cercaan fisik (body shaming) yang merendahkan penampilan fisik korban secara langsung."
                severity = "HIGH" if any(w in p_lower for w in ["ampela", "segentong", "burik", "dekil"]) else "MEDIUM"
                implicitness = "E0" if any(w in p_lower for w in ["segentong", "jerawat", "burik", "jelek"]) else "E1"
            elif is_shm:
                b_type = "SHM"
                cultural = "STA"
                emotion = "MOCK"
                explanation = "Teks mengekspresikan ejekan status sosial atau perilaku (social shaming / superiority complex)."
                severity = "MEDIUM"
                implicitness = "E1"
            else:
                b_type = "INS"
                cultural = "INT" if any(w in p_lower for w in ["bego", "tolol", "goblok", "idiot", "bodoh", "otak"]) else "STA"
                emotion = "MOCK"
                explanation = "Teks mengekspresikan ejekan dan penghinaan martabat personal dalam konteks media sosial."
                severity = "MEDIUM"
                implicitness = "E0" if any(w in p_lower for w in ["babi", "anjing", "bangsat", "bajingan"]) else "E1"

            return f"""{{
  "cyberbullying": "YES",
  "type": "{b_type}",
  "emotion": "{emotion}",
  "cultural_context": "{cultural}",
  "implicit_level": "{implicitness}",
  "severity": "{severity}",
  "confidence": 0.92,
  "explanation": "{explanation}"
}}"""
        else:
            return """{
  "cyberbullying": "NO",
  "type": "NONE",
  "emotion": "NEU",
  "cultural_context": "NONE",
  "implicit_level": "E0",
  "severity": "NONE",
  "confidence": 0.98,
  "explanation": "Komentar bernada positif atau netral tanpa unsur ejekan, hinaan, atau pelecehan."
}"""

    def generate_batch(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        max_new_tokens: int = 220,
        temperature: float = 0.1,
        early_stop_json: bool = True,
    ) -> List[str]:
        """Fast simulated batched response."""
        return [self.generate(p, max_new_tokens, temperature, early_stop_json) for p in prompts]


class SumopodEngine:
    """
    OpenAI-compatible AI Gateway Engine for SumoPod (https://api.sumopod.com/v1).
    Primary cloud LLM engine for annotation and Chain-of-Thought reasoning.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_SUMOPOD_BASE_URL,
        model_name: str = DEFAULT_SUMOPOD_MODEL,
    ):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or DEFAULT_SUMOPOD_BASE_URL).strip().rstrip("/")
        self.model_name = (model_name or DEFAULT_SUMOPOD_MODEL).strip()
        self.is_mock = False
        self.backend_type = "sumopod"

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _to_messages(prompt_or_messages: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        if isinstance(prompt_or_messages, list):
            return prompt_or_messages
        return [{"role": "user", "content": str(prompt_or_messages)}]

    def test_connection(self) -> Tuple[bool, str]:
        """Verify API key and endpoint connectivity with a lightweight ping."""
        if not self.api_key:
            return False, "API Key SumoPod belum diisi."
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Ping test"}],
                "max_tokens": 5,
            }
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._build_headers(),
                json=payload,
                timeout=15,
            )
            if resp.status_code == 200:
                return True, f"Koneksi ke SumoPod AI Gateway ({self.model_name}) berhasil terhubung!"
            return False, f"Gagal (HTTP {resp.status_code}): {resp.text[:200]}"
        except Exception as e:
            return False, str(e)

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 220,
        temperature: float = 0.1,
        early_stop_json: bool = False,
    ) -> str:
        if not self.api_key:
            raise ValueError("SumoPod API Key kosong. Masukkan API Key di tab Pengaturan & Model.")

        payload = {
            "model": self.model_name,
            "messages": self._to_messages(prompt),
            "temperature": max(0.0, float(temperature)) if temperature is not None else 0.1,
            "max_tokens": int(max_new_tokens),
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._build_headers(),
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"SumoPod API Error ({resp.status_code}): {resp.text}")

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def generate_batch(
        self,
        prompts: List[Union[str, List[Dict[str, str]]]],
        max_new_tokens: int = 220,
        temperature: float = 0.1,
        early_stop_json: bool = False,
    ) -> List[str]:
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], max_new_tokens, temperature, early_stop_json)]

        max_workers = min(len(prompts), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(
                executor.map(
                    lambda p: self.generate(p, max_new_tokens, temperature, early_stop_json),
                    prompts,
                )
            )


def run_single_annotation(
    engine,
    text: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 220,
    temperature: float = 0.1,
    early_stop_json: bool = False,
) -> Dict[str, Any]:
    """Convenience helper to format chat messages, invoke SumoPod/Mock engine, and extract JSON."""
    messages = build_chat_messages(text, system_prompt=system_prompt)
    start_time = time.time()
    raw_output = engine.generate(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        early_stop_json=early_stop_json,
    )
    latency = time.time() - start_time
    parsed = extract_json_from_response(raw_output)
    parsed["latency_sec"] = round(latency, 2)
    parsed["raw_output"] = raw_output
    return parsed


def run_batch_annotation_step(
    engine,
    texts: List[str],
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 220,
    temperature: float = 0.1,
    early_stop_json: bool = False,
) -> List[Dict[str, Any]]:
    """Runs concurrent SumoPod API (or fallback Mock) inference across a chunk of texts."""
    messages_batch = [build_chat_messages(t, system_prompt=system_prompt) for t in texts]
    start_time = time.time()

    if hasattr(engine, "generate_batch"):
        raw_outputs = engine.generate_batch(
            messages_batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            early_stop_json=early_stop_json,
        )
    else:
        raw_outputs = [
            engine.generate(m, max_new_tokens=max_new_tokens, temperature=temperature)
            for m in messages_batch
        ]

    total_latency = time.time() - start_time
    latency_per_item = round(total_latency / len(texts), 2) if texts else 0

    results = []
    for raw in raw_outputs:
        parsed = extract_json_from_response(raw)
        parsed["latency_sec"] = latency_per_item
        parsed["raw_output"] = raw
        results.append(parsed)

    return results


# Global Cached Engine Getters for Streamlit
try:
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def get_cached_sumopod_engine(
        api_key: str,
        base_url: str,
        model_name: str,
    ) -> SumopodEngine:
        """Globally cached SumoPod AI Gateway engine."""
        return SumopodEngine(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )

    @st.cache_resource(show_spinner=False)
    def get_cached_mock_engine() -> MockEngine:
        """Globally cached MockEngine fallback for prototyping without an API key."""
        return MockEngine()

except ImportError:
    def get_cached_sumopod_engine(api_key: str, base_url: str, model_name: str) -> SumopodEngine:
        return SumopodEngine(api_key=api_key, base_url=base_url, model_name=model_name)

    def get_cached_mock_engine() -> MockEngine:
        return MockEngine()
