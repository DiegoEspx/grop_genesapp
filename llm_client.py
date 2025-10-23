from __future__ import annotations
import os, threading
from typing import List, Dict, Any

_LLM_LOCK = threading.Lock()

class GroqLLM:
    def __init__(self, api_key: str, model: str):
        from groq import Groq
        self.cli = Groq(api_key=api_key)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **opts) -> Dict[str, Any]:
        temperature = float(opts.get("temperature", 0.1))
        max_tokens  = int(opts.get("num_predict", 220))
        top_p       = float(opts.get("top_p", 0.9))
        with _LLM_LOCK:
            r = self.cli.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
        content = (r.choices[0].message.content if r.choices else "") or ""
        return {"message": {"content": content}}

def build_llm_from_env():
    api_key = os.getenv("GROQ_API_KEY")
    model   = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    if not api_key:
        raise RuntimeError("Falta GROQ_API_KEY.")
    return GroqLLM(api_key, model), "groq", model
