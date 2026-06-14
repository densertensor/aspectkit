"""Prompt and JSON-schema construction for the LLM backend.

The prompts follow the structured-extraction pattern validated in
production ABSA systems: a fixed system prompt defining the task and the
JSON contract, optional few-shot exemplars rendered exactly like the
expected output, and a final input block.  The same element definitions
drive both the prose instructions and the machine-readable JSON schema,
so providers with native structured output enforce precisely what the
prompt describes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from aspectkit.llm.base import Message
from aspectkit.schema import ABSAExample, SentimentTuple
from aspectkit.tasks import Task

__all__ = [
    "build_classification_schema",
    "build_extraction_schema",
    "classification_messages",
    "extraction_messages",
    "tuple_to_payload",
]

_ELEMENT_DEFINITIONS = {
    "aspect": (
        '"aspect": the exact words from the text naming the opinion target. '
        "Copy them verbatim; do not paraphrase. Use null when the target is "
        "implicit (clearly evaluated but not named with explicit words)."
    ),
    "category": ('"category": the aspect category of the opinion target{inventory}.'),
    "opinion": (
        '"opinion": the exact words from the text expressing the evaluation, '
        "copied verbatim. Use null when the opinion is implicit."
    ),
    "polarity": '"polarity": the sentiment polarity, one of: {polarities}.',
}


def _category_inventory(categories: Sequence[str] | None) -> str:
    if categories:
        return ", one of: " + ", ".join(categories)
    return ' in the "ENTITY#ATTRIBUTE" format (e.g. "FOOD#QUALITY")'


def _element_lines(
    elements: Sequence[str],
    categories: Sequence[str] | None,
    polarities: Sequence[str],
) -> list[str]:
    lines = []
    for element in elements:
        template = _ELEMENT_DEFINITIONS[element]
        lines.append(
            "- "
            + template.format(
                inventory=_category_inventory(categories),
                polarities=", ".join(polarities),
            )
        )
    return lines


def tuple_to_payload(t: SentimentTuple, elements: Sequence[str]) -> dict[str, Any]:
    """Render a tuple as the JSON object the model is asked to produce."""
    payload: dict[str, Any] = {}
    for element in elements:
        if element == "aspect":
            payload["aspect"] = t.aspect_text
        elif element == "opinion":
            payload["opinion"] = t.opinion_text
        elif element == "category":
            payload["category"] = t.category
        elif element == "polarity":
            payload["polarity"] = t.polarity
    return payload


def build_extraction_schema(
    task: Task,
    categories: Sequence[str] | None,
    polarities: Sequence[str],
) -> dict[str, Any]:
    """JSON schema for extraction outputs: ``{"tuples": [...]}``.

    The root is an object (a requirement of several structured-output
    implementations), with one property per predicted element and
    ``additionalProperties: false`` throughout (required for strict
    enforcement on OpenAI and Anthropic).
    """
    properties: dict[str, Any] = {}
    for element in task.ordered_elements(task.predicted):
        if element in ("aspect", "opinion"):
            properties[element] = {"type": ["string", "null"]}
        elif element == "category":
            if categories:
                properties[element] = {"type": "string", "enum": list(categories)}
            else:
                properties[element] = {"type": "string"}
        elif element == "polarity":
            properties[element] = {"type": "string", "enum": list(polarities)}
    return {
        "type": "object",
        "properties": {
            "tuples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["tuples"],
        "additionalProperties": False,
    }


def build_classification_schema(polarities: Sequence[str]) -> dict[str, Any]:
    """JSON schema for classification outputs: ``{"polarity": "..."}``."""
    return {
        "type": "object",
        "properties": {"polarity": {"type": "string", "enum": list(polarities)}},
        "required": ["polarity"],
        "additionalProperties": False,
    }


def _extraction_system_prompt(
    task: Task,
    categories: Sequence[str] | None,
    polarities: Sequence[str],
) -> str:
    elements = task.ordered_elements(task.predicted)
    keys = ", ".join(f'"{e}"' for e in elements)
    lines = [
        "You are an expert annotator for aspect-based sentiment analysis.",
        "",
        f"Task: {task.description}.",
        f"From the input text, extract every distinct opinion as an object with the keys {keys}.",
        "",
        "Element definitions:",
        *_element_lines(elements, categories, polarities),
        "",
        "Rules:",
        "- Extract one object per opinion; a sentence may express several.",
        '- If the text expresses no opinion, return {"tuples": []}.',
        "- Respond with a single JSON object of the form "
        '{"tuples": [...]} and nothing else: no prose, no code fences.',
    ]
    return "\n".join(lines)


def _classification_system_prompt(polarities: Sequence[str]) -> str:
    lines = [
        "You are an expert annotator for aspect-based sentiment analysis.",
        "",
        "Task: classify the sentiment polarity that the input text expresses "
        "towards the given aspect (not the overall sentiment of the text).",
        "",
        "Element definitions:",
        *_element_lines(("polarity",), None, polarities),
        "",
        "Rules:",
        "- Judge only the sentiment directed at the given aspect.",
        "- Respond with a single JSON object of the form "
        '{"polarity": "..."} and nothing else: no prose, no code fences.',
    ]
    return "\n".join(lines)


def _render_extraction_input(text: str) -> str:
    return f"Text: {text}"


def _render_classification_input(text: str, aspect: str | None) -> str:
    target = aspect if aspect is not None else "(implicit / the text overall)"
    return f"Text: {text}\nAspect: {target}"


def extraction_messages(
    text: str,
    task: Task,
    categories: Sequence[str] | None,
    polarities: Sequence[str],
    exemplars: Sequence[ABSAExample] = (),
) -> list[Message]:
    """Build the chat messages for one extraction call.

    Few-shot exemplars are rendered as alternating user/assistant turns,
    each assistant turn being exactly the JSON the model should emit —
    the strongest format signal available to chat models.
    """
    elements = task.ordered_elements(task.predicted)
    messages: list[Message] = [
        {"role": "system", "content": _extraction_system_prompt(task, categories, polarities)}
    ]
    for exemplar in exemplars:
        payload = {"tuples": [tuple_to_payload(t, elements) for t in exemplar.tuples]}
        messages.append({"role": "user", "content": _render_extraction_input(exemplar.text)})
        messages.append({"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)})
    messages.append({"role": "user", "content": _render_extraction_input(text)})
    return messages


def classification_messages(
    text: str,
    aspect: str | None,
    polarities: Sequence[str],
    exemplars: Sequence[tuple[str, str | None, str]] = (),
) -> list[Message]:
    """Build the chat messages for one (text, aspect) polarity call.

    Args:
        text: The input sentence.
        aspect: Surface text of the target aspect, or ``None`` if implicit.
        polarities: Allowed polarity labels.
        exemplars: Few-shot triples ``(text, aspect_or_None, polarity)``.
    """
    messages: list[Message] = [
        {"role": "system", "content": _classification_system_prompt(polarities)}
    ]
    for ex_text, ex_aspect, ex_polarity in exemplars:
        messages.append(
            {"role": "user", "content": _render_classification_input(ex_text, ex_aspect)}
        )
        messages.append({"role": "assistant", "content": json.dumps({"polarity": ex_polarity})})
    messages.append({"role": "user", "content": _render_classification_input(text, aspect)})
    return messages
