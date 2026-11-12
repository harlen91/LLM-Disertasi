from datetime import datetime, timezone
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, List, Dict, Any, Union
import requests

from config import DEFAULT_SUMOPOD_BASE_URL, DEFAULT_SUMOPOD_MODEL
from .prompt_templates import build_chat_messages, PROMPT_VERSION
from .parser import extract_json_from_response

logger = logging.getLogger(__name__)

RECOVERABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def mask_secret(secret: Optional[str]) -> str:
    """Masks sensitive keys before logging or UI display."""
    s = (secret or "").strip()
    if not s:
        return "(kosong)"
    if len(s) <= 8:
        return "****"
    return f"{s[:4]}...{s[-4:]}"


def _extract_user_text(prompt_or_messages: Union[str, List[Dict[str, str]]]) -> str:
    """Extracts user message text from OpenAI-style messages or plain string prompts."""
    if isinstance(prompt_or_messages, list):
        user_parts = [
            str(m.get("content", ""))
            for m in prompt_or_messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        raw = (
            "\n".join(user_parts)
            if user_parts
            else " ".join(str(m.get("content", "")) for m in prompt_or_messages if isinstance(m, dict))
        )
    else:
        raw = str(prompt_or_messages or "")

    if "Text:\n" in raw and "\n\nTaxonomy:" in raw:
        return raw.split("Text:\n", 1)[1].split("\n\nTaxonomy:", 1)[0]
    return raw


class MockEngine:
    """
    Safe fallback simulation engine ONLY for UI prototyping and offline unit tests.
    Outputs produced by MockEngine must NOT be used as final research dataset annotations.
    """

    def __init__(self):
        self.device = "mock"
        self.is_mock = True
        self.backend_type = "mock"
        self.model_name = "mock-heuristic-simulator"

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
            "ompong", "kurus", "kerempeng", "dower", "behel", "bau", "jelek",
        ]
        ins_words = [
            "bego", "tolol", "goblok", "idiot", "bloon", "bodoh", "otak", "cacat",
            "autis", "babi", "anjing", "monyet", "bangsat", "bajingan", "tai",
            "kontol", "memek", "pantek", "lonte", "jablay", "perek", "jalang",
            "hama", "rusak", "mati", "gagal", "buset",
        ]
        shm_words = [
            "kampungan", "udik", "miskin", "gembel", "pansos", "alay", "norak",
            "belagu", "songong", "sok cantik", "berasa paling cantik", "sok ganteng", "sok kaya",
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
                explanation = "[SIMULASI MOCK] Teks mengandung indikasi intimidasi atau ancaman kekerasan."
                severity = "HIGH"
                implicitness = "E0"
            elif is_bsh:
                b_type = "BSH"
                cultural = "APP"
                emotion = "MOCK"
                explanation = "[SIMULASI MOCK] Teks mengandung cercaan fisik (body shaming)."
                severity = "HIGH" if any(w in p_lower for w in ["ampela", "segentong", "burik", "dekil"]) else "MEDIUM"
                implicitness = "E0" if any(w in p_lower for w in ["segentong", "jerawat", "burik", "jelek"]) else "E1"
            elif is_shm:
                b_type = "SHM"
                cultural = "STA"
                emotion = "MOCK"
                explanation = "[SIMULASI MOCK] Teks mengekspresikan ejekan status sosial atau perilaku."
                severity = "MEDIUM"
                implicitness = "E1"
            else:
                b_type = "INS"
                cultural = "INT" if any(w in p_lower for w in ["bego", "tolol", "goblok", "idiot", "bodoh", "otak"]) else "STA"
                emotion = "MOCK"
                explanation = "[SIMULASI MOCK] Teks mengekspresikan ejekan dan penghinaan personal."
                severity = "MEDIUM"
                implicitness = "E0" if any(w in p_lower for w in ["babi", "anjing", "bangsat", "bajingan"]) else "E1"

            return f"""{{
  "cyberbullying": "YES",
  "type": "{b_type}",
  "emotion": "{emotion}",
  "cultural_context": "{cultural}",
  "implicit_level": "{implicitness}",
  "severity": "{severity}",
  "confidence": 0.85,
  "explanation": "{explanation}"
}}"""
        return """{
  "cyberbullying": "NO",
  "type": "NONE",
  "emotion": "NEU",
  "cultural_context": "NONE",
  "implicit_level": "E0",
  "severity": "NONE",
  "confidence": 0.90,
  "explanation": "[SIMULASI MOCK] Komentar bernada netral tanpa indikasi perundungan siber."
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
    Primary cloud LLM engine for annotation and Chain-of-Thought reasoning,
    complete with exponential backoff retry on transient HTTP/network failures.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_SUMOPOD_BASE_URL,
        model_name: str = DEFAULT_SUMOPOD_MODEL,
        max_retries: int = 3,
        backoff_base: float = 1.5,
        request_timeout: int = 60,
    ):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or DEFAULT_SUMOPOD_BASE_URL).strip().rstrip("/")
        self.model_name = (model_name or DEFAULT_SUMOPOD_MODEL).strip()
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = float(backoff_base)
        self.request_timeout = int(request_timeout)
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

    def _safe_error_text(self, text: str) -> str:
        """Sanitizes error text to guarantee API keys never appear in logs or UI messages."""
        safe = str(text or "")
        if self.api_key and self.api_key in safe:
            safe = safe.replace(self.api_key, mask_secret(self.api_key))
        return safe[:300]

    def _post_with_retry(self, payload: Dict[str, Any], timeout: Optional[int] = None) -> Dict[str, Any]:
        """Executes HTTP POST with exponential backoff for recoverable transient errors."""
        if not self.api_key:
            raise ValueError("SumoPod API Key kosong. Konfigurasikan melalui .streamlit/secrets.toml atau env SUMOPOD_API_KEY.")

        url = f"{self.base_url}/chat/completions"
        req_timeout = timeout or self.request_timeout
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers=self._build_headers(),
                    json=payload,
                    timeout=req_timeout,
                )
                if resp.status_code == 200:
                    return resp.json()

                err_snippet = self._safe_error_text(resp.text)
                if resp.status_code in RECOVERABLE_STATUS_CODES and attempt < self.max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    wait_sec = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else (self.backoff_base ** attempt)
                    logger.warning(
                        "SumoPod API transient HTTP %s (model=%s, attempt=%s/%s). Retrying in %.1fs.",
                        resp.status_code,
                        self.model_name,
                        attempt + 1,
                        self.max_retries + 1,
                        wait_sec,
                    )
                    time.sleep(min(wait_sec, 15.0))
                    continue

                raise RuntimeError(f"SumoPod API Error (HTTP {resp.status_code}): {err_snippet}")

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    wait_sec = self.backoff_base ** attempt
                    logger.warning(
                        "SumoPod connection issue (attempt=%s/%s): %s. Retrying in %.1fs.",
                        attempt + 1,
                        self.max_retries + 1,
                        type(exc).__name__,
                        wait_sec,
                    )
                    time.sleep(min(wait_sec, 15.0))
                    continue
                raise RuntimeError(f"SumoPod API koneksi gagal setelah {self.max_retries + 1} percobaan: {type(exc).__name__}") from exc

        raise RuntimeError(f"SumoPod API gagal: {last_error}")

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
            return False, f"Gagal (HTTP {resp.status_code}): {self._safe_error_text(resp.text)}"
        except Exception as e:
            return False, f"Koneksi gagal: {type(e).__name__}: {self._safe_error_text(str(e))}"

    def generate_with_metadata(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 220,
        temperature: float = 0.1,
        request_logprobs: bool = False,
    ) -> Dict[str, Any]:
        """Calls SumoPod API and returns text alongside any API-supplied token logprobs/usage."""
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._to_messages(prompt),
            "temperature": max(0.0, float(temperature)) if temperature is not None else 0.1,
            "max_tokens": int(max_new_tokens),
        }
        if request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        data = self._post_with_retry(payload)
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError("Respons SumoPod API tidak memiliki field 'choices' yang valid.")
        choice0 = choices[0]
        content = str(choice0.get("message", {}).get("content", "")).strip()

        # Extract token probabilities if returned by provider
        logprobs_info = choice0.get("logprobs")
        mean_token_prob = None
        if isinstance(logprobs_info, dict) and isinstance(logprobs_info.get("content"), list):
            tok_lps = [
                float(t["logprob"])
                for t in logprobs_info["content"]
                if isinstance(t, dict) and t.get("logprob") is not None
            ]
            if tok_lps:
                mean_token_prob = round(math.exp(sum(tok_lps) / len(tok_lps)), 4)

        return {
            "text": content,
            "api_mean_token_prob": mean_token_prob,
            "logprobs_available": logprobs_info is not None,
            "usage": data.get("usage", {}),
        }

    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 220,
        temperature: float = 0.1,
        early_stop_json: bool = False,
    ) -> str:
        res = self.generate_with_metadata(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            request_logprobs=False,
        )
        return res["text"]

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


def _attach_annotation_metadata(
    parsed: Dict[str, Any],
    engine: Any,
    latency_sec: float,
    prompt_version: str = PROMPT_VERSION,
    raw_output: str = "",
) -> Dict[str, Any]:
    """Attaches standardized provenance metadata to each annotation record."""
    is_mock = bool(getattr(engine, "is_mock", False))
    parsed["latency_sec"] = round(latency_sec, 2)
    parsed["raw_output"] = raw_output
    parsed["llm_model"] = getattr(engine, "model_name", "mock-heuristic-simulator" if is_mock else DEFAULT_SUMOPOD_MODEL)
    parsed["prompt_version"] = prompt_version
    parsed["annotated_at"] = datetime.now(timezone.utc).isoformat()
    parsed["annotation_source"] = "mock_simulation" if is_mock else "sumopod_api"
    parsed["is_mock"] = is_mock
    return parsed


def run_single_annotation(
    engine,
    text: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 220,
    temperature: float = 0.1,
    early_stop_json: bool = False,
) -> Dict[str, Any]:
    """Convenience helper to format chat messages, invoke SumoPod/Mock engine, and extract JSON + metadata."""
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
    return _attach_annotation_metadata(parsed, engine=engine, latency_sec=latency, raw_output=raw_output)


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
    latency_per_item = round(total_latency / len(texts), 2) if texts else 0.0

    results = []
    for raw in raw_outputs:
        parsed = extract_json_from_response(raw)
        _attach_annotation_metadata(parsed, engine=engine, latency_sec=latency_per_item, raw_output=raw)
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
