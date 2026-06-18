from __future__ import annotations
import os
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer
from .local_runtime import LLMCallMetrics, get_local_runtime

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}
RUNTIME_MODE = os.getenv("REFLEXION_RUNTIME_MODE", "mock").strip().lower()


def get_runtime_mode() -> str:
    return RUNTIME_MODE


def _text_metrics(*parts: str) -> LLMCallMetrics:
    combined = " ".join(part for part in parts if part)
    total_tokens = max(1, (len(combined) + 3) // 4) if combined else 0
    return LLMCallMetrics(
        prompt_tokens=max(0, total_tokens - 8),
        completion_tokens=min(8, total_tokens),
        total_tokens=total_tokens,
        latency_ms=1,
    )


def _mock_actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer


def _mock_evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
    return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])


def _mock_reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, LLMCallMetrics]:
    if RUNTIME_MODE == "local":
        return get_local_runtime().actor_answer(example, attempt_id, agent_type, reflection_memory)
    answer = _mock_actor_answer(example, attempt_id, agent_type, reflection_memory)
    return answer, _text_metrics(example.question, answer, " ".join(reflection_memory))


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, LLMCallMetrics]:
    if RUNTIME_MODE == "local":
        return get_local_runtime().evaluator(example, answer)
    judge = _mock_evaluator(example, answer)
    return judge, _text_metrics(example.question, example.gold_answer, answer, judge.reason)


def reflector(
    example: QAExample,
    attempt_id: int,
    answer: str,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, LLMCallMetrics]:
    if RUNTIME_MODE == "local":
        return get_local_runtime().reflector(example, attempt_id, answer, judge)
    reflection = _mock_reflector(example, attempt_id, judge)
    return reflection, _text_metrics(example.question, answer, judge.reason, reflection.next_strategy)
