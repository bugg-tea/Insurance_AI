"""
Groq LLM Client
===============
- Uses Groq free tier (llama-3.1-8b-instant — fastest, lowest latency)
- Fallback chain: llama-3.1-8b-instant → llama3-8b-8192 → gemma2-9b-it
- Simple in-memory response cache to avoid redundant API calls
- Exponential backoff retry (max 2 retries — keep it fast)
- JSON extraction helper used by all agents

Set env:  GROQ_API_KEY=your_key
"""

from __future__ import annotations
from dotenv import load_dotenv
from pathlib import Path


import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional


load_dotenv()

from openai import OpenAI

_cerebras = OpenAI(
    api_key=os.getenv("CEREBRAS_API_KEY"),
    base_url="https://api.cerebras.ai/v1",
)

_samba = OpenAI(
    api_key=os.getenv("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)


try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# ── CONFIG ───────────────────────────────────────────────────────────────────
PRIMARY_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
)

FALLBACK_MODELS = [

    "llama3-8b-8192",
    
]
MAX_TOKENS      = 1024      # keep small — agents use structured JSON output
TEMPERATURE     = 0.1       # low = deterministic, good for insurance QA
MAX_RETRIES     = 2
RETRY_DELAY     = 1.0       # seconds (doubles each retry)

# ── CACHE ────────────────────────────────────────────────────────────────────
# Simple dict cache keyed by (model, messages_hash)
# Survives one process lifetime — good enough for dev + saves Groq quota

_cache: Dict[str, str] = {}
_cache_hits: int = 0
_cache_misses: int = 0


def _cache_key(model: str, messages: List[Dict]) -> str:
    raw = model + json.dumps(messages, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


# ── CLIENT ───────────────────────────────────────────────────────────────────

class GroqClient:
    def __init__(self):
        if not _GROQ_AVAILABLE:
            raise ImportError("groq package not installed. Run: pip install groq")

        api_key = os.getenv("GROQ_API_KEY")
        
        print("=" * 60)
        print("Groq Model :", PRIMARY_MODEL)
        print("API Key Loaded :", "YES" if api_key else "NO")
        print("=" * 60)
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set")

        self.client = Groq(api_key=api_key)

    # ── CORE CALL ────────────────────────────────────────────────────────────

    def call(
        self,
        messages: List[Dict[str, str]],
        model: str = PRIMARY_MODEL,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        use_cache: bool = True,
    ) -> str:
        global _cache_hits, _cache_misses

        # Cache check
        key = _cache_key(model, messages)
        if use_cache and key in _cache:
            _cache_hits += 1
            return _cache[key]

        _cache_misses += 1
        models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]

        last_error = None
        for attempt_model in models_to_try:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = self.client.chat.completions.create(
                        model=attempt_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    text = resp.choices[0].message.content or ""
                    if use_cache:
                        _cache[key] = text
                    return text

                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()

                    # Rate limit → wait and retry
                    if "rate_limit" in err_str or "429" in err_str:
                        wait = RETRY_DELAY * (2 ** attempt)
                        time.sleep(wait)
                        continue

                    # Model not available → try next model immediately
                    if "model" in err_str or "404" in err_str:
                        break

                    # Other errors → retry with backoff
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY * (2 ** attempt))

        
        print("\nGroq exhausted all retries.")

        return call_fallback_chain(
            messages=messages,
            groq_error=last_error,
)
        

    # ── JSON HELPER ──────────────────────────────────────────────────────────
    
    def call_json(
        self,
        messages: List[Dict[str, str]],
        model: str = PRIMARY_MODEL,
        max_tokens: int = MAX_TOKENS,
    ) -> Dict[str, Any]:
        """
    Robust JSON LLM wrapper.
    - backward compatible
    - prevents pipeline crashes
    - handles malformed / empty / markdown JSON
    """

        text = self.call(messages, model=model, max_tokens=max_tokens)

        if not text:
            return {}

    # 1️⃣ normal extraction attempt
        try:
            return extract_json(text) or {}
        except Exception:
            pass

    # 2️⃣ fallback: extract JSON block manually
        import re
        import json

        try:
        # grab first JSON object in text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

    # 3️⃣ fallback: try cleaning markdown fences
        try:
            cleaned = (
                text.replace("```json", "")
                    .replace("```", "")
                    .strip()
        )
            return json.loads(cleaned)
        except Exception:
            pass

    # 4️⃣ SAFE FAIL (IMPORTANT: never crash pipeline)
        return {}

    # ── STATS ────────────────────────────────────────────────────────────────

    @staticmethod
    def cache_stats() -> Dict[str, int]:
        return {"hits": _cache_hits, "misses": _cache_misses}


# ── JSON EXTRACTOR ────────────────────────────────────────────────────────────

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response.
    Handles: raw JSON, ```json ... ```, ``` ... ```, partial wraps.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip("` \n")

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Return empty dict on complete failure (agents handle gracefully)
    return {}


# ── SINGLETON ────────────────────────────────────────────────────────────────

_client: Optional[GroqClient] = None


def get_client() -> GroqClient:
    global _client
    if _client is None:
        _client = GroqClient()
    return _client

# ─────────────────────────────────────────────────────────────
# 🔥 HUGGINGFACE FALLBACK CLIENT (NEW)
# ─────────────────────────────────────────────────────────────


    
    
# ─────────────────────────────────────────────────────────────
# 🧪 TEST MAIN FUNCTION
# ─────────────────────────────────────────────────────────────

def call_cerebras(messages):

    response = _cerebras.chat.completions.create(

        model="llama-3.3-70b",

        messages=messages,

        temperature=0.1,

        max_tokens=1024,

    )

    return response.choices[0].message.content




def call_sambanova(messages):

    response = _samba.chat.completions.create(

        model="Meta-Llama-3.3-70B-Instruct",

        messages=messages,

        temperature=0.1,

        max_tokens=1024,

    )

    return response.choices[0].message.content

# ---------------------------------------------------------
# LOCAL FALLBACK (Lazy Loaded)
# ---------------------------------------------------------

LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = None
model = None


def load_local_model():
    """
    Loads the local model only once.
    It is NOT downloaded until every cloud provider fails.
    """

    global tokenizer, model

    if tokenizer is not None and model is not None:
        return

    print("\nDownloading / Loading local model...")

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )
    import torch

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL
    )

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL,
        device_map="auto",
        torch_dtype=(
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        ),
    )


def call_local(messages):

    load_local_model()

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
    )

    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    
def call_fallback_chain(messages, groq_error):

    # --------------------------------------------------
    # Cerebras
    # --------------------------------------------------

    try:
        print("\n⚠ Groq failed")
        print("→ Trying Cerebras...\n")

        return call_cerebras(messages)

    except Exception as cerebras_error:

        # ----------------------------------------------
        # SambaNova
        # ----------------------------------------------

        try:
            print("⚠ Cerebras failed")
            print("→ Trying SambaNova...\n")

            return call_sambanova(messages)

        except Exception as samba_error:

            # ------------------------------------------
            # Local model
            # ------------------------------------------

            try:
                print("⚠ SambaNova failed")
                print("→ Loading local model...\n")

                return call_local(messages)

            except Exception as local_error:

                raise RuntimeError(
f"""
================ ALL LLM PROVIDERS FAILED ================

Groq
-----
{groq_error}

Cerebras
---------
{cerebras_error}

SambaNova
----------
{samba_error}

Local
-----
{local_error}

==========================================================
"""
                )
                
if __name__ == "__main__":

    messages = [
        {
            "role": "user",
            "content": "Explain insurance in one short line."
        }
    ]

    client = get_client()

    print("\n==============================")
    print("TEST 1 : GROQ")
    print("==============================")

    try:
        print(client.call(messages, use_cache=False))
    except Exception as e:
        print(e)

    print("\n==============================")
    print("TEST 2 : CEREBRAS")
    print("==============================")

    try:
        print(call_cerebras(messages))
    except Exception as e:
        print(e)

    print("\n==============================")
    print("TEST 3 : SAMBANOVA")
    print("==============================")

    try:
        print(call_sambanova(messages))
    except Exception as e:
        print(e)

    print("\n==============================")
    print("TEST 4 : LOCAL")
    print("==============================")

    try:
        print(call_local(messages))
    except Exception as e:
        print(e)

    print("\n==============================")
    print("TEST 5 : COMPLETE FALLBACK")
    print("==============================")

    try:
        response = client.call(
            messages,
            model="invalid-model",
            use_cache=False,
        )

        print(response)

    except Exception as e:
        print(e)