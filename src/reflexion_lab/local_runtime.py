from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer


DEFAULT_MODEL_NAME = os.getenv("REFLEXION_LOCAL_MODEL", "google/flan-t5-small")
MAX_INPUT_TOKENS = int(os.getenv("REFLEXION_MAX_INPUT_TOKENS", "1024"))
MAX_ACTOR_TOKENS = int(os.getenv("REFLEXION_MAX_ACTOR_TOKENS", "32"))
MAX_EVALUATOR_TOKENS = int(os.getenv("REFLEXION_MAX_EVALUATOR_TOKENS", "96"))
MAX_REFLECTOR_TOKENS = int(os.getenv("REFLEXION_MAX_REFLECTOR_TOKENS", "96"))
DEFAULT_CACHE_DIR = Path(os.getenv("REFLEXION_HF_CACHE_DIR", Path(__file__).resolve().parents[2] / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


@dataclass
class LLMCallMetrics:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: int


def _extract_json_block(text: str) -> dict | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    candidate = match.group(0) if match else cleaned
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _context_text(example: QAExample) -> str:
    return "\n\n".join(f"Title: {chunk.title}\nText: {chunk.text}" for chunk in example.context)


class LocalTextRuntime:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self.cache_dir = str(DEFAULT_CACHE_DIR)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int) -> tuple[str, LLMCallMetrics]:
        prompt = f"System:\n{system_prompt.strip()}\n\nUser:\n{user_prompt.strip()}\n\nAssistant:\n"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        started = perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        latency_ms = max(1, round((perf_counter() - started) * 1000))
        completion_tokens = int(outputs.shape[-1])
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        metrics = LLMCallMetrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
        )
        return text, metrics

    def actor_answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> tuple[str, LLMCallMetrics]:
        reflection_block = (
            "\n".join(f"- {item}" for item in reflection_memory)
            if reflection_memory
            else "None"
        )
        user_prompt = f"""
Question:
{example.question}

Attempt:
{attempt_id}

Agent type:
{agent_type}

Reflection memory:
{reflection_block}

Context:
{_context_text(example)}
""".strip()
        return self.generate(ACTOR_SYSTEM, user_prompt, max_new_tokens=MAX_ACTOR_TOKENS)

    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, LLMCallMetrics]:
        user_prompt = f"""
Question:
{example.question}

Gold answer:
{example.gold_answer}

Predicted answer:
{answer}

Context:
{_context_text(example)}
""".strip()
        raw_text, metrics = self.generate(
            EVALUATOR_SYSTEM,
            user_prompt,
            max_new_tokens=MAX_EVALUATOR_TOKENS,
        )
        payload = _extract_json_block(raw_text)
        if payload is None:
            payload = self._fallback_evaluator(example, answer).model_dump()
        return JudgeResult.model_validate(payload), metrics

    def reflector(
        self,
        example: QAExample,
        attempt_id: int,
        answer: str,
        judge: JudgeResult,
    ) -> tuple[ReflectionEntry, LLMCallMetrics]:
        user_prompt = f"""
Attempt ID:
{attempt_id}

Question:
{example.question}

Incorrect answer:
{answer}

Evaluator failure reason:
{judge.reason}

Context:
{_context_text(example)}
""".strip()
        raw_text, metrics = self.generate(
            REFLECTOR_SYSTEM,
            user_prompt,
            max_new_tokens=MAX_REFLECTOR_TOKENS,
        )
        payload = _extract_json_block(raw_text)
        if payload is None:
            payload = self._fallback_reflector(example, attempt_id, judge).model_dump()
        payload.setdefault("attempt_id", attempt_id)
        return ReflectionEntry.model_validate(payload), metrics

    @staticmethod
    def _fallback_evaluator(example: QAExample, answer: str) -> JudgeResult:
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(
                score=1,
                reason="Fallback exact-match evaluator marked the answer correct.",
                missing_evidence=[],
                spurious_claims=[],
            )
        return JudgeResult(
            score=0,
            reason="Fallback exact-match evaluator marked the answer incorrect.",
            missing_evidence=["The final answer does not match the gold answer after normalization."],
            spurious_claims=[answer] if answer else [],
        )

    @staticmethod
    def _fallback_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
        return ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Double-check each reasoning hop before committing to the final answer.",
            next_strategy="Re-read the context and verify the final entity named by the question.",
        )


@lru_cache(maxsize=1)
def get_local_runtime(model_name: str = DEFAULT_MODEL_NAME) -> LocalTextRuntime:
    return LocalTextRuntime(model_name=model_name)
