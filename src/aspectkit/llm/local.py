"""Connector for locally loaded Hugging Face ``transformers`` models.

Designed for the notebook workflow: a model already loaded in the
session — as a ``pipeline("text-generation", ...)`` or as a
model/tokenizer pair — plugs into aspectkit directly, with no
serialisation round-trip::

    from transformers import pipeline
    from aspectkit import ABSA

    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct")
    absa = ABSA(task="acos", backend="llm", model=pipe)

Generation is greedy (``do_sample=False``) by default for reproducible
extraction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from aspectkit.exceptions import LLMError, MissingDependencyError
from aspectkit.llm.base import ChatLLM, Message

__all__ = ["TransformersChat", "looks_like_hf_model", "looks_like_hf_pipeline"]


def looks_like_hf_pipeline(obj: Any) -> bool:
    """Duck-typed check for a ``transformers`` text-generation pipeline."""
    return (
        callable(obj)
        and not isinstance(obj, type)
        and hasattr(obj, "tokenizer")
        and hasattr(obj, "model")
    )


def looks_like_hf_model(obj: Any) -> bool:
    """Duck-typed check for a ``transformers`` causal-LM model object."""
    return hasattr(obj, "generate") and hasattr(obj, "config")


class TransformersChat(ChatLLM):
    """Chat connector for in-process ``transformers`` models.

    Accepts, in order of convenience:

    * a Hugging Face Hub model id (``str``) — a ``text-generation``
      pipeline is created lazily on first use;
    * a ready ``transformers.Pipeline`` object;
    * a ``(model, tokenizer)`` pair of ``PreTrainedModel`` and
      ``PreTrainedTokenizer`` objects.

    The tokenizer's chat template is applied when available; otherwise a
    plain ``System/User/Assistant`` transcript is rendered, which keeps
    base (non-chat) models usable.

    Args:
        model: Hub id, pipeline object, or model object.
        tokenizer: Required when ``model`` is a bare model object.
        device: Device hint forwarded to ``pipeline(...)`` when loading
            from a hub id (ignored for already-loaded objects).
        **generate_kwargs: Extra generation arguments forwarded to every
            call (e.g. ``repetition_penalty=1.1``).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any | None = None,
        *,
        device: Any | None = None,
        **generate_kwargs: Any,
    ) -> None:
        self._generate_kwargs = generate_kwargs
        self._pipeline: Any | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._hub_id: str | None = None
        self._device = device

        if isinstance(model, str):
            self._hub_id = model
            self.model = model
        elif looks_like_hf_pipeline(model):
            self._pipeline = model
            self.model = getattr(getattr(model, "model", None), "name_or_path", "pipeline")
        elif tokenizer is not None:
            self._model = model
            self._tokenizer = tokenizer
            name = getattr(model, "name_or_path", None) or getattr(
                getattr(model, "config", None), "name_or_path", None
            )
            self.model = str(name) if name else "model"
        else:
            raise TypeError(
                "TransformersChat expects a hub id, a transformers pipeline, or a "
                "(model, tokenizer) pair; got "
                f"{type(model).__name__} without a tokenizer"
            )

    def _load_pipeline(self) -> Any:
        if self._pipeline is None and self._hub_id is not None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise MissingDependencyError(
                    "transformers", "transformers", "TransformersChat"
                ) from exc
            kwargs: dict[str, Any] = {"model": self._hub_id}
            if self._device is not None:
                kwargs["device"] = self._device
            self._pipeline = pipeline("text-generation", **kwargs)
        return self._pipeline

    def _render(self, messages: Sequence[Message], tokenizer: Any) -> str:
        """Render messages to a prompt string via the chat template,
        falling back to a plain transcript for base models."""
        template = getattr(tokenizer, "chat_template", None)
        if template and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        role_names = {"system": "System", "user": "User", "assistant": "Assistant"}
        lines = [f"{role_names.get(m['role'], m['role'])}: {m['content']}" for m in messages]
        lines.append("Assistant:")
        return "\n\n".join(lines)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        # json_schema is advisory here: open models have no constrained
        # decoding hook at this level, and the backends embed the schema
        # in the prompt and parse defensively.
        del json_schema
        if self._model is not None:
            return self._complete_with_model(messages, max_tokens)
        return self._complete_with_pipeline(messages, max_tokens)

    def _complete_with_pipeline(self, messages: Sequence[Message], max_tokens: int) -> str:
        pipe = self._load_pipeline()
        prompt = self._render(messages, pipe.tokenizer)
        outputs = pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_full_text=False,
            **self._generate_kwargs,
        )
        if not outputs or "generated_text" not in outputs[0]:
            raise LLMError(f"{self.name} returned no generation output")
        text = outputs[0]["generated_text"]
        if not isinstance(text, str) or not text.strip():
            raise LLMError(f"{self.name} returned an empty completion")
        return text

    def _complete_with_model(self, messages: Sequence[Message], max_tokens: int) -> str:
        model, tokenizer = self._model, self._tokenizer
        assert model is not None and tokenizer is not None  # guaranteed by complete()
        prompt = self._render(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            **self._generate_kwargs,
        }
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        output_ids = model.generate(**inputs, **generate_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        new_token_ids = output_ids[0][prompt_length:]
        text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        if not text.strip():
            raise LLMError(f"{self.name} returned an empty completion")
        return text
