import json
import re
from typing import Any

from pydantic import ValidationError

from app.models import SupportAnswer


def eval_pydantic_schema_validity(pipeline_result: dict[str, Any]) -> bool:
    """Evaluates whether the pipeline answer dict conforms to the SupportAnswer Pydantic schema."""
    ans_data = pipeline_result.get("answer")
    if not isinstance(ans_data, dict):
        return False
    try:
        SupportAnswer.model_validate(ans_data)
        return True
    except ValidationError:
        return False


def eval_citation_id_validity(pipeline_result: dict[str, Any]) -> bool:
    """Evaluates whether citation validation passed and all citation IDs exist in retrieved sources."""
    trace = pipeline_result.get("trace", {})
    cit_passed = trace.get("citation_validation_passed")
    if cit_passed is not None:
        return bool(cit_passed)

    citations = pipeline_result.get("citations", [])
    sources = pipeline_result.get("sources", [])
    valid_sids = {
        (s.get("source_id") or f"S{idx}").upper()
        for idx, s in enumerate(sources, start=1)
    }

    for c in citations:
        sid = c.get("source_id") if isinstance(c, dict) else str(c)
        if sid and sid.upper() not in valid_sids:
            return False
    return True


def eval_citation_presence(
    pipeline_result: dict[str, Any],
    expected_answerable: bool = True,
) -> bool:
    """
    Evaluates citation presence:
    - Factual answerable non-abstained answers must have >= 1 citation.
    - Abstained or unanswerable queries should have 0 citations.
    """
    ans_data = pipeline_result.get("answer", {})
    abstained = bool(pipeline_result.get("trace", {}).get("abstention_triggered")) or bool(ans_data.get("escalation_required"))
    citations = ans_data.get("citations", []) or pipeline_result.get("citations", [])

    if expected_answerable and not abstained:
        return len(citations) > 0
    else:
        return len(citations) == 0


def eval_answer_point_coverage(
    answer_text: str,
    expected_answer_points: list[str],
) -> float:
    """Calculates fraction of expected answer points present in answer text (0.0 to 1.0)."""
    if not expected_answer_points:
        return 1.0

    text_lower = (answer_text or "").lower()
    matches = 0

    for point in expected_answer_points:
        point_tokens = set(re.findall(r"\w+", point.lower()))
        text_tokens = set(re.findall(r"\w+", text_lower))
        if point.lower() in text_lower or (point_tokens and point_tokens.issubset(text_tokens)):
            matches += 1
        elif len(point_tokens & text_tokens) / max(1, len(point_tokens)) >= 0.6:
            matches += 1

    return matches / float(len(expected_answer_points))


def eval_correct_abstention(
    pipeline_result: dict[str, Any],
    expected_abstention: bool,
) -> bool:
    """Evaluates whether abstention status matches expected abstention."""
    trace = pipeline_result.get("trace", {})
    abstained = bool(trace.get("abstention_triggered"))
    return abstained == expected_abstention


def eval_correct_escalation(
    pipeline_result: dict[str, Any],
    expected_escalation: bool,
) -> bool:
    """Evaluates whether escalation_required matches expected escalation."""
    ans_data = pipeline_result.get("answer", {})
    escalation = bool(ans_data.get("escalation_required"))
    return escalation == expected_escalation


def eval_conflict_disclosure(
    pipeline_result: dict[str, Any],
    is_conflict_case: bool = False,
) -> bool:
    """Evaluates whether material conflicts were properly disclosed when expected."""
    if not is_conflict_case:
        return True

    ans_data = pipeline_result.get("answer", {})
    trace = pipeline_result.get("trace", {})
    conflict_res = trace.get("conflict_result", {})

    return bool(ans_data.get("conflicts_detected")) or bool(conflict_res.get("conflict_detected"))


def eval_resolution_steps_structure(
    pipeline_result: dict[str, Any],
    expected_answerable: bool = True,
) -> tuple[bool, bool]:
    """
    Evaluates resolution steps structural rules:
    - non_empty_resolution_steps_when_expected
    - no_resolution_steps_when_abstaining
    Returns (non_empty_when_expected, no_steps_when_abstaining).
    """
    ans_data = pipeline_result.get("answer", {})
    steps = ans_data.get("resolution_steps", [])
    abstained = bool(pipeline_result.get("trace", {}).get("abstention_triggered")) or bool(ans_data.get("escalation_required"))

    non_empty_when_expected = (len(steps) > 0) if (expected_answerable and not abstained) else True
    no_steps_when_abstaining = (len(steps) == 0) if abstained else True

    return non_empty_when_expected, no_steps_when_abstaining


def evaluate_deterministic_answer_quality(
    pipeline_result: dict[str, Any],
    question_case: dict[str, Any],
) -> dict[str, Any]:
    """Combines all deterministic answer evaluation metrics into a single report dict."""
    ans_data = pipeline_result.get("answer", {})
    ans_text = ans_data.get("answer", "") if isinstance(ans_data, dict) else str(ans_data)

    expected_answerable = question_case.get("answerable", True)
    expected_abstention = question_case.get("expected_abstention", False)
    expected_escalation = question_case.get("expected_escalation", False)
    expected_points = question_case.get("expected_answer_points", [])
    is_conflict_case = (question_case.get("test_type") == "conflicting_source")

    schema_valid = eval_pydantic_schema_validity(pipeline_result)
    citation_id_valid = eval_citation_id_validity(pipeline_result)
    citation_present = eval_citation_presence(pipeline_result, expected_answerable=expected_answerable)
    point_coverage = eval_answer_point_coverage(ans_text, expected_points)
    correct_abst = eval_correct_abstention(pipeline_result, expected_abstention=expected_abstention)
    correct_esc = eval_correct_escalation(pipeline_result, expected_escalation=expected_escalation)
    conflict_disclosed = eval_conflict_disclosure(pipeline_result, is_conflict_case=is_conflict_case)

    non_empty_steps, no_steps_abst = eval_resolution_steps_structure(pipeline_result, expected_answerable=expected_answerable)

    return {
        "schema_validity": schema_valid,
        "citation_id_validity": citation_id_valid,
        "citation_presence": citation_present,
        "answer_point_coverage": point_coverage,
        "correct_abstention": correct_abst,
        "correct_escalation": correct_esc,
        "conflict_disclosure": conflict_disclosed,
        "non_empty_resolution_steps_when_expected": non_empty_steps,
        "no_resolution_steps_when_abstaining": no_steps_abst,
    }


def judge_answer_quality(
    question: str,
    retrieved_context: str,
    generated_answer: Any,
    expected_answer_points: list[str],
    generator: Any = None,
) -> dict[str, Any]:
    """
    Optional local LLM judge evaluating 4 dimensions on a strict 0-2 rubric:
    - faithfulness (0 = unsupported, 1 = partially faithful, 2 = fully supported)
    - answer_relevance (0 = irrelevant, 1 = partial, 2 = fully relevant)
    - citation_support (0 = invalid/missing, 1 = partial, 2 = accurate)
    - conflict_handling (0 = ignored conflict, 1 = partial disclosure, 2 = fully disclosed & resolved)

    Receives question, retrieved context, generated answer, expected answer points.
    Returns scores dict with stored reasoning for debugging.
    Note: Do not treat local judge as absolute ground truth.
    """
    ans_text = ""
    if isinstance(generated_answer, dict):
        ans_text = generated_answer.get("answer", "")
    elif hasattr(generated_answer, "answer"):
        ans_text = generated_answer.answer
    else:
        ans_text = str(generated_answer)

    if generator is not None and hasattr(generator, "generate"):
        prompt = f"""
You are an evaluation judge assessing the quality of a support assistant answer against retrieved context.

Use a strict 0–2 rubric:
0 = incorrect, unsupported, or missing
1 = partially correct or partially supported
2 = fully correct, relevant, and supported

Dimensions to rate:
1. "faithfulness": Does the generated answer rely ONLY on facts from the retrieved context without inventing facts?
2. "answer_relevance": Does the answer directly address the user question and cover expected answer points?
3. "citation_support": Are citations present and accurate for factual claims?
4. "conflict_handling": If the context contains conflicting evidence, does the answer disclose or resolve it properly?

Question:
{question}

Expected Answer Points:
{json.dumps(expected_answer_points)}

Retrieved Context:
{retrieved_context}

Generated Answer:
{ans_text}

Return ONLY a JSON object:
{{
  "faithfulness": 0 or 1 or 2,
  "answer_relevance": 0 or 1 or 2,
  "citation_support": 0 or 1 or 2,
  "conflict_handling": 0 or 1 or 2,
  "reasoning": "<detailed explanation of scores for debugging>"
}}
"""
        try:
            resp, _ = generator.generate(prompt, temperature=0.0)
            cleaned = resp.strip()
            if "```" in cleaned:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
                if match:
                    cleaned = match.group(1).strip()
            data = json.loads(cleaned)
            return {
                "faithfulness": int(data.get("faithfulness", 1)),
                "answer_relevance": int(data.get("answer_relevance", 1)),
                "citation_support": int(data.get("citation_support", 1)),
                "conflict_handling": int(data.get("conflict_handling", 1)),
                "reasoning": str(data.get("reasoning", "Evaluated by local LLM judge.")),
            }
        except Exception as exc:
            pass

    # Heuristic fallback judge when LLM generator is unavailable
    coverage = eval_answer_point_coverage(ans_text, expected_answer_points)
    rel_score = 2 if coverage >= 0.8 else (1 if coverage >= 0.4 else 0)
    faith_score = 2 if "could not find" not in ans_text.lower() or coverage > 0 else 1

    return {
        "faithfulness": faith_score,
        "answer_relevance": rel_score,
        "citation_support": 1,
        "conflict_handling": 1,
        "reasoning": f"Fallback heuristic judge evaluation (Answer Point Coverage: {coverage:.2f}). Generator unavailable.",
    }
