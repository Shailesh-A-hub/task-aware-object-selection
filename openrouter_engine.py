"""
OpenRouter Engine for Open-Vocabulary and Novel Task Reasoning.
Performs live LLM affordance distillation and task decomposition via OpenRouter API.
Zero hardcoding: pure dynamic LLM generation.
"""

import os
import json
import requests
from typing import Optional, Dict, Any


class OpenRouterEngine:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openrouter/auto",
        cache_file: str = "knowledge_bank/openrouter_cache.json"
    ):
        self.api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self.model = model
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.cache_file = cache_file
        self.cache = self._load_cache()

        # Telemetry / Debug attributes for UI display
        self.last_status: Optional[int] = None
        self.last_error: str = ""
        self.last_prompt: str = ""
        self.last_response: str = ""
        self.last_call_success: bool = False
        self.last_model_used: str = model

    def _load_cache(self) -> Dict[str, str]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[OpenRouterEngine] Warning: Could not save cache: {e}")

    def call_openrouter(
        self,
        prompt: str,
        system_prompt: str = "You are an expert AI physical affordance and robotics object reasoning assistant."
    ) -> str:
        """
        Sends query to OpenRouter with automatic model fallback if rate-limited.
        """
        clean_key = self.api_key.strip()
        if not clean_key:
            self.last_call_success = False
            self.last_error = "OpenRouter API Key is empty."
            return ""

        headers = {
            "Authorization": f"Bearer {clean_key}",
            "HTTP-Referer": "https://github.com/shailesh-vit/task-aware-object-selection",
            "X-Title": "Task-Aware Object Selection",
            "Content-Type": "application/json"
        }

        # Active OpenRouter models with priority order
        models_to_try = [
            self.model,
            "openrouter/auto",
            "google/gemma-4-26b-a4b-it:free",
            "nvidia/nemotron-3.5-lightning:free",
            "minimax/minimax-m3.1-flash:free",
            "liquid/lfm-2.5-2.6b:free"
        ]
        # Remove duplicates while preserving order
        seen = set()
        models_to_try = [m for m in models_to_try if not (m in seen or seen.add(m))]

        self.last_prompt = prompt
        self.last_error = ""

        for candidate_model in models_to_try:
            payload = {
                "model": candidate_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 150
            }

            try:
                response = requests.post(self.endpoint, headers=headers, json=payload, timeout=12)
                self.last_status = response.status_code

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    self.last_response = content
                    self.last_call_success = True
                    self.last_model_used = candidate_model
                    return content
                else:
                    self.last_error = f"Model {candidate_model} returned HTTP {response.status_code}: {response.text}"
                    print(f"[OpenRouterEngine] {self.last_error}")
                    # Try next candidate if rate-limited or unavailable
                    continue

            except Exception as err:
                self.last_error = f"Connection error with {candidate_model}: {str(err)}"
                print(f"[OpenRouterEngine] {self.last_error}")
                continue

        self.last_call_success = False
        return ""

    def test_connection(self) -> Dict[str, Any]:
        """Tests API key and returns status, latency, and model response."""
        import time
        t0 = time.time()
        res = self.call_openrouter("Say 'OpenRouter Connected Successfully!' in exactly 4 words.")
        elapsed = round((time.time() - t0) * 1000, 1)

        if self.last_call_success and res:
            return {
                "success": True,
                "message": res,
                "latency_ms": elapsed,
                "model": self.last_model_used
            }
        else:
            return {
                "success": False,
                "error": self.last_error or "Unknown error calling OpenRouter API.",
                "status_code": self.last_status,
                "latency_ms": elapsed
            }

    def distill_novel_object(self, object_name: str) -> str:
        """
        Dynamically distills affordances for novel object terms outside COCO-80 via OpenRouter.
        """
        cache_key = f"obj:{object_name.lower().strip()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.api_key:
            prompt = (
                f"Provide a concise 1-sentence physical affordance and functional description for the object '{object_name}'. "
                f"Focus strictly on its physical shape, material, stability, portability, and primary affordances (e.g., sitting, holding liquids, cutting, striking, elevation, decoration). "
                f"Do not include conversational filler."
            )
            distilled = self.call_openrouter(prompt)
            if distilled:
                self.cache[cache_key] = distilled
                self._save_cache()
                return distilled

        # Generic neutral fallback (no hardcoding)
        fallback = f"{object_name}: physical item; evaluate shape, material, and structural affordances"
        return fallback

    def decompose_novel_task(self, task_query: str) -> str:
        """
        Uses OpenRouter LLM to dynamically reason about any novel task query.
        Zero hardcoding: pure live LLM reasoning.
        """
        cache_key = f"task:{task_query.lower().strip()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.api_key:
            prompt = (
                f"You are an expert AI physical affordance reasoning system.\n"
                f"User task: '{task_query}'.\n"
                f"Identify 2 to 3 common everyday physical objects, tools, or household appliances that humans use to solve this task, "
                f"and describe their key physical properties (e.g. shape, thermal resistance, sharp edge, hollow container, sturdy elevated support).\n"
                f"Answer in 1 concise sentence in this format: 'Objects: [suitable objects]; Affordances: [key physical characteristics]'."
            )
            decomposed = self.call_openrouter(prompt)
            if decomposed:
                self.cache[cache_key] = decomposed
                self._save_cache()
                return decomposed

        # Generic neutral fallback if API key is absent or offline (zero hardcoding)
        return f"object or implement suitable for the task: {task_query}"
