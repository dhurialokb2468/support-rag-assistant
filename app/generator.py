import json
import time
from typing import Any

import requests

from app.config import settings
from app.logger import get_logger
from app.models import SupportAnswer
from app.validators import (
    clean_markdown_fences,
    create_fallback_support_answer,
    validate_support_answer,
)

logger = get_logger("generator")


class OllamaGenerator:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    def is_available(self) -> bool:
        try:
            logger.debug(f"Checking Ollama connectivity at {self.base_url}...")
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            available = response.ok
            if not available:
                logger.warning(f"Ollama server returned status {response.status_code}.")
            return available
        except requests.RequestException as exc:
            logger.warning(f"Ollama server connection failed ({self.base_url}): {exc}")
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
    ) -> tuple[str, float]:
        started = time.perf_counter()
        logger.debug(f"Sending prompt to Ollama model '{self.model}' (temp={temperature})...")

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                    },
                },
                timeout=180,
            )
            response.raise_for_status()
            elapsed = time.perf_counter() - started
            logger.debug(f"Ollama response received in {elapsed:.4f}s.")
            return response.json()["response"], elapsed

        except requests.RequestException as exc:
            logger.error(f"Ollama API request failed to endpoint '{self.base_url}': {exc}")
            raise

    def generate_json(
        self,
        prompt: str,
    ) -> tuple[dict, float]:
        response_text, elapsed = self.generate(prompt)
        cleaned = clean_markdown_fences(response_text)
        return json.loads(cleaned), elapsed

    def generate_support_answer(
        self,
        question: str,
        context: str,
    ) -> tuple[SupportAnswer, dict[str, Any]]:
        started = time.perf_counter()

        prompt = f"""
You are an expert product support assistant for InsightFlow.

Answer the user's question using ONLY the supplied context.

Rules:
1. Output ONLY a single valid JSON object adhering strictly to the JSON schema below. Do NOT output explanations, conversational preamble, or markdown formatting outside the JSON object.
2. Use ONLY the supplied context. Do NOT use outside knowledge or invent product settings, behaviors, or policies.
3. If the context is insufficient to answer the question, clearly state "Insufficient information in supplied context" in the "answer" field, set "confidence" to "low", "confidence_score" to 0.0, and "escalation_required" to true.
4. Cite evidence using ONLY the supplied source IDs (e.g. "S1", "S2") in the "citations" array.
5. Separate the root cause in "likely_cause" from step-by-step resolution actions in "resolution_steps".

Required JSON Schema:
{{
  "answer": "<string>",
  "likely_cause": "<string or null>",
  "resolution_steps": ["<string>"],
  "citations": ["<string>"],
  "confidence": "high" | "medium" | "low",
  "confidence_score": <float between 0.0 and 1.0>,
  "escalation_required": <boolean>,
  "escalation_reason": "<string or null>",
  "conflicts_detected": <boolean>
}}

Question:
{question}

Supplied Context:
{context}
"""

        trace: dict[str, Any] = {
            "attempts": 1,
            "repaired": False,
            "fallback_used": False,
            "initial_error": None,
            "repair_error": None,
        }

        try:
            raw_response, elapsed = self.generate(prompt, temperature=0.1)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.warning(f"Ollama generation failed: {exc}. Using fallback support answer.")
            trace["initial_error"] = str(exc)
            trace["fallback_used"] = True
            trace["generation_time"] = elapsed
            fallback_ans = create_fallback_support_answer(str(exc), question)
            return fallback_ans, trace

        support_ans, val_err = validate_support_answer(raw_response)

        if support_ans is not None:
            logger.info("Generated support answer parsed and validated successfully.")
            trace["generation_time"] = elapsed
            return support_ans, trace

        # First validation attempt failed -> Make 1 repair request to Ollama
        logger.warning(f"First JSON validation failed: {val_err}. Triggering 1 repair request.")
        trace["initial_error"] = val_err
        trace["attempts"] = 2

        repair_prompt = f"""
Your previous output failed JSON/schema validation.

Validation Error:
{val_err}

Previous Response:
{raw_response}

Please return ONLY a valid, corrected JSON object conforming strictly to the required schema:
{{
  "answer": "<string>",
  "likely_cause": "<string or null>",
  "resolution_steps": ["<string>"],
  "citations": ["<string>"],
  "confidence": "high" | "medium" | "low",
  "confidence_score": <float 0.0-1.0>,
  "escalation_required": <boolean>,
  "escalation_reason": "<string or null>",
  "conflicts_detected": <boolean>
}}

Original Question:
{question}

Supplied Context:
{context}
"""

        try:
            repair_response, repair_elapsed = self.generate(repair_prompt, temperature=0.0)
            elapsed += repair_elapsed
            repaired_ans, repair_val_err = validate_support_answer(repair_response)

            if repaired_ans is not None:
                logger.info("Repair request succeeded. Support answer validated.")
                trace["repaired"] = True
                trace["generation_time"] = elapsed
                return repaired_ans, trace
            else:
                logger.error(f"Repair request validation also failed: {repair_val_err}")
                trace["repair_error"] = repair_val_err
        except Exception as repair_exc:
            logger.error(f"Repair request exception: {repair_exc}")
            trace["repair_error"] = str(repair_exc)

        # Repair also failed -> return safe fallback SupportAnswer requiring escalation
        logger.warning("Using fallback support answer after validation failure.")
        trace["fallback_used"] = True
        trace["generation_time"] = elapsed
        final_err = trace["repair_error"] or trace["initial_error"] or "Unknown validation error"
        fallback_ans = create_fallback_support_answer(final_err, question)
        return fallback_ans, trace