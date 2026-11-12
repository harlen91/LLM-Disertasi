import os
import time
import torch
from typing import Tuple, Any, Optional, List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

from .prompt_templates import build_llama3_prompt
from .parser import extract_json_from_llama


class JsonStopCriteria(StoppingCriteria):
    """Stops generation as soon as JSON closing brace '}' is emitted."""
    def __init__(self, tokenizer, prompt_lens: List[int]):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_lens = prompt_lens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if all sequences have emitted closing brace '}'
        all_closed = True
        for i, seq in enumerate(input_ids):
            start = self.prompt_lens[i] if i < len(self.prompt_lens) else self.prompt_lens[0]
            gen_text = self.tokenizer.decode(seq[start:], skip_special_tokens=True).strip()
            if not ("}" in gen_text and gen_text.endswith("}")):
                all_closed = False
                break
        return all_closed


class MockEngine:
    """
    Simulation engine for quick testing, development, and UI demonstrations
    without loading 8B parameters into memory.
    """
    def __init__(self):
        self.device = "mock"
        self.is_mock = True

    def generate(self, prompt: str, max_new_tokens: int = 160, temperature: float = 0.1, early_stop_json: bool = True) -> str:
        p_lower = prompt.lower()
        
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

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 220, temperature: float = 0.1, early_stop_json: bool = True) -> List[str]:
        # Fast simulated batched response
        return [self.generate(p, max_new_tokens, temperature, early_stop_json) for p in prompts]


class LlamaEngine:
    """
    Optimized HuggingFace Transformers loader for Meta-Llama-3-8B-Instruct
    with Stop Token tuning, Batched Generation, and Metal (MPS) support.
    """
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        cache_dir: Optional[str] = None,
        device_mode: str = "auto",
        quantization: str = "4-bit"
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface/hub")
        self.device_mode = device_mode
        self.quantization = quantization
        self.is_mock = False
        
        self.tokenizer = None
        self.model = None
        self.terminators = []
        self._load_model()

    def _load_model(self):
        # 1. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        )
        
        # Left-padding is mandatory for decoder batched inference
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define all LLaMA-3 EOS and EOT stop token IDs
        self.terminators = [
            self.tokenizer.eos_token_id,
            128001,  # <|end_of_text|>
            128009   # <|eot_id|>
        ]

        # 2. Configure Hardware Device & Quantization
        model_kwargs = {
            "cache_dir": self.cache_dir,
        }

        if self.quantization == "4-bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        elif self.device_mode == "mps" and torch.backends.mps.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "mps"
        elif self.device_mode == "cuda" and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["device_map"] = "cpu"

        # 3. Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 160,
        temperature: float = 0.1,
        early_stop_json: bool = True
    ) -> str:
        results = self.generate_batch([prompt], max_new_tokens=max_new_tokens, temperature=temperature, early_stop_json=early_stop_json)
        return results[0] if results else ""

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 160,
        temperature: float = 0.1,
        early_stop_json: bool = True
    ) -> List[str]:
        """
        Batched parallel inference for high throughput.
        """
        if not prompts:
            return []

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 0.01) if temperature > 0 else None,
            "do_sample": True if temperature > 0 else False,
            "eos_token_id": self.terminators,
            "pad_token_id": self.tokenizer.eos_token_id
        }

        if early_stop_json:
            prompt_lens = [len(inputs.input_ids[i]) for i in range(len(prompts))]
            stop_criteria = StoppingCriteriaList([JsonStopCriteria(self.tokenizer, prompt_lens)])
            gen_kwargs["stopping_criteria"] = stop_criteria

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )

        decoded_results = []
        input_len = inputs.input_ids.shape[1]
        
        for i in range(len(prompts)):
            gen_tokens = outputs[i][input_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            if not text:
                text = self.tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            decoded_results.append(text)

        return decoded_results


def run_single_annotation(
    engine,
    text: str,
    system_prompt: str = None,
    max_new_tokens: int = 220,
    temperature: float = 0.1,
    early_stop_json: bool = True
) -> Dict[str, Any]:
    """Convenience helper to format prompt, generate, and extract JSON for a single text."""
    prompt = build_llama3_prompt(text, system_prompt=system_prompt)
    start_time = time.time()
    raw_output = engine.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        early_stop_json=early_stop_json
    )
    latency = time.time() - start_time
    parsed = extract_json_from_llama(raw_output)
    parsed["latency_sec"] = round(latency, 2)
    parsed["raw_output"] = raw_output
    return parsed


def run_batch_annotation_step(
    engine,
    texts: List[str],
    system_prompt: str = None,
    max_new_tokens: int = 220,
    temperature: float = 0.1,
    early_stop_json: bool = True
) -> List[Dict[str, Any]]:
    """Runs batched inference across a chunk of texts for maximum throughput."""
    prompts = [build_llama3_prompt(t, system_prompt=system_prompt) for t in texts]
    start_time = time.time()
    
    if hasattr(engine, "generate_batch"):
        raw_outputs = engine.generate_batch(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            early_stop_json=early_stop_json
        )
    else:
        raw_outputs = [
            engine.generate(p, max_new_tokens=max_new_tokens, temperature=temperature)
            for p in prompts
        ]
        
    total_latency = time.time() - start_time
    latency_per_item = round(total_latency / len(texts), 2) if texts else 0

    results = []
    for raw in raw_outputs:
        parsed = extract_json_from_llama(raw)
        parsed["latency_sec"] = latency_per_item
        parsed["raw_output"] = raw
        results.append(parsed)

    return results


class SumopodEngine:
    """
    OpenAI-compatible AI Gateway Engine for SumoPod (https://api.sumopod.com/v1).
    Enables low-latency, cloud GPU execution with IDR payment and minimal RAM on Streamlit Cloud.
    """
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.sumopod.com/v1",
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    ):
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "https://api.sumopod.com/v1").strip().rstrip("/")
        self.model_name = (model_name or "meta-llama/Meta-Llama-3-8B-Instruct").strip()
        self.is_mock = False
        self.backend_type = "sumopod"

    def test_connection(self) -> Tuple[bool, str]:
        """Verify API key and endpoint connectivity with a lightweight ping."""
        import requests
        if not self.api_key:
            return False, "API Key SumoPod belum diisi."
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Ping test"}],
                "max_tokens": 5
            }
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            if resp.status_code == 200:
                return True, "Koneksi ke SumoPod AI Gateway berhasil terhubung!"
            else:
                return False, f"Gagal (HTTP {resp.status_code}): {resp.text[:200]}"
        except Exception as e:
            return False, str(e)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 220,
        temperature: float = 0.1,
        early_stop_json: bool = False
    ) -> str:
        import requests
        if not self.api_key:
            raise ValueError("SumoPod API Key kosong. Masukkan API Key di tab Pengaturan & Model.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Convert prompt with LLaMA chat special tokens into standard OpenAI messages if present
        if "<|start_header_id|>system<|end_header_id|>" in prompt and "<|start_header_id|>user<|end_header_id|>" in prompt:
            parts = prompt.split("<|start_header_id|>user<|end_header_id|>")
            sys_part = parts[0].replace("<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "").replace("<|eot_id|>", "").strip()
            user_part = parts[1].replace("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", "").strip()
            messages = [
                {"role": "system", "content": sys_part},
                {"role": "user", "content": user_part}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": max(temperature, 0.01) if temperature > 0 else 0.1,
            "max_tokens": max_new_tokens
        }

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        if resp.status_code != 200:
            raise RuntimeError(f"SumoPod API Error ({resp.status_code}): {resp.text}")

        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 220,
        temperature: float = 0.1,
        early_stop_json: bool = False
    ) -> List[str]:
        return [self.generate(p, max_new_tokens, temperature, early_stop_json) for p in prompts]


# Global Cached Engine Getters for Streamlit
try:
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def get_cached_llama_engine(
        model_name: str,
        cache_dir: str,
        device_mode: str,
        quantization: str
    ) -> LlamaEngine:
        """Globally cached LLaMA-3 engine across all Streamlit sessions and reruns."""
        return LlamaEngine(
            model_name=model_name,
            cache_dir=cache_dir,
            device_mode=device_mode,
            quantization=quantization
        )

    @st.cache_resource(show_spinner=False)
    def get_cached_sumopod_engine(
        api_key: str,
        base_url: str,
        model_name: str
    ) -> SumopodEngine:
        """Globally cached SumoPod AI Gateway engine."""
        return SumopodEngine(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name
        )

    @st.cache_resource(show_spinner=False)
    def get_cached_mock_engine() -> MockEngine:
        """Globally cached MockEngine for fast prototyping."""
        return MockEngine()

except ImportError:
    def get_cached_llama_engine(model_name: str, cache_dir: str, device_mode: str, quantization: str) -> LlamaEngine:
        return LlamaEngine(model_name=model_name, cache_dir=cache_dir, device_mode=device_mode, quantization=quantization)

    def get_cached_sumopod_engine(api_key: str, base_url: str, model_name: str) -> SumopodEngine:
        return SumopodEngine(api_key=api_key, base_url=base_url, model_name=model_name)

    def get_cached_mock_engine() -> MockEngine:
        return MockEngine()


