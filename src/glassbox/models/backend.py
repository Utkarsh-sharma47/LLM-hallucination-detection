"""Load one causal LM and reuse it for every question in the session."""

from __future__ import annotations

import atexit
import fcntl
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.activations.extract import (
    feature_index,
    mean_answer_vector,
    module_device,
    resolve_layer,
    vectors_at_position,
)
from glassbox.analysis.capture import (
    select_layers,
    slice_generated_spans,
    spans_within_budget,
)
from glassbox.guards import reject_download
from glassbox.models.distributions import (
    TokenAlternative,
    entropy_nats,
    normalized_entropy,
    other_candidates,
    probabilities_from_logits,
    scale_logits,
)
from glassbox.models.resources import (
    ModelMemoryError,
    assert_weights_fit,
    available_ram_bytes,
    resolve_dtype,
)

POSITIONS = ("last_prompt_token", "last_generated_token")
_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")


def acquire_model_lock(path: Path | None = None):
    """Hold an exclusive lock for as long as this process keeps the model.

    A second Glass Box process fails immediately instead of loading another
    copy of Qwen. The lock is released when the process exits.
    """
    destination = path or _default_lock_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = destination.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(
            "Another Glass Box process already has Qwen loaded. "
            "Quit that session before training or starting a second one."
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    atexit.register(handle.close)
    return handle


def _default_lock_path() -> Path:
    from glassbox.config import repo_root

    return repo_root() / "artifacts" / "runtime" / "qwen.lock"


def cached_commit(repo_id: str, revision: str) -> str | None:
    """Return the cached snapshot commit for a branch revision, without downloading."""
    if _COMMIT_HASH.fullmatch(revision or ""):
        return revision
    try:
        from huggingface_hub.file_download import try_to_load_from_cache
    except ImportError:
        return None
    located = try_to_load_from_cache(repo_id, "config.json", revision=revision)
    if not isinstance(located, (str, Path)):
        return None
    commit = Path(located).parent.name
    if _COMMIT_HASH.fullmatch(commit):
        return commit
    return None


@dataclass(frozen=True)
class ModelSpec:
    """How to load a Hugging Face causal language model."""

    name: str
    revision: str = "main"
    device: str = "auto"
    dtype: str = "auto"
    allow_download: bool = False
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("model name must be a non-empty string")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ValueError("revision must be a non-empty string")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if not self.allow_download and not self.local_files_only:
            reject_download(False, resource=f"model {self.name}@{self.revision}")


@dataclass(frozen=True)
class GenerationSettings:
    """Decoding and representation settings for one session."""

    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 5
    layer_index: int = -1
    position: str = "last_prompt_token"
    capture_all_layers: bool = False
    capture_mode: str = "final"
    distribution_export: str = "topk"

    def __post_init__(self) -> None:
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise ValueError("max_new_tokens must be an integer")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError("temperature must be a number")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be an integer >= 1")
        if self.position not in POSITIONS:
            raise ValueError(f"position must be one of {', '.join(POSITIONS)}")
        if self.distribution_export not in {"topk", "full"}:
            raise ValueError("distribution_export must be topk or full")
        if isinstance(self.layer_index, bool) or not isinstance(self.layer_index, int):
            raise ValueError("layer_index must be an integer")
        if self.capture_mode not in {"final", "generated", "selected", "all"}:
            raise ValueError("capture_mode must be final, generated, selected, or all")


@dataclass
class LoadedModel:
    """A model and tokenizer kept in memory for the session."""

    spec: ModelSpec
    model: Any
    tokenizer: Any
    device: str
    dtype: str
    resolved_revision: str


@dataclass(frozen=True)
class GeneratedToken:
    """The token that was emitted and the distribution it was drawn from."""

    token_id: int
    text: str
    probability: float
    top_alternatives: tuple[TokenAlternative, ...]
    is_argmax: bool
    entropy_nats: float = 0.0
    normalized_entropy: float = 0.0
    margin: float = 0.0
    rank: int = 1
    step: int = 1
    conditioned_on: str = "prompt"
    decoding: str = "greedy"
    temperature: float = 0.0
    vocabulary_probabilities: np.ndarray | None = None


@dataclass
class GenerationResult:
    """One model answer plus the representations used to produce it."""

    question: str
    prompt_text: str
    prompt_mode: str
    answer: str
    model_name: str
    model_revision: str
    device: str
    dtype: str
    latency_seconds: float
    prompt_tokens: int
    generated_tokens: int
    tokens: tuple[GeneratedToken, ...]
    hidden_states_available: bool
    logits_available: bool
    feature: np.ndarray | None
    feature_layer: int | None
    hidden_state_count: int | None
    position: str
    layer_vectors: dict[int, np.ndarray]
    load_count: int
    note: str = ""
    decoding: str = "greedy"
    temperature: float = 0.0
    answer_mean_feature: np.ndarray | None = None
    token_spans: dict[int, np.ndarray] | None = None
    capture_mode: str = "final"
    capture_bytes: int = 0
    capture_note: str = ""


class HuggingFaceBackend:
    """Load weights once and generate with the cached module."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._loaded: LoadedModel | None = None
        self.load_count = 0

    @property
    def model_name(self) -> str:
        return self.spec.name

    @property
    def device(self) -> str:
        if self._loaded is None:
            return self.spec.device
        return self._loaded.device

    def ensure_loaded(self) -> LoadedModel:
        """Return the cached weights, loading them on the first call."""
        if self._loaded is None:
            self._lock = acquire_model_lock()
            self._loaded = load_hf_model(self.spec)
            self.load_count += 1
        return self._loaded

    def generate(
        self,
        question: str,
        settings: GenerationSettings,
        on_token: Any = None,
    ) -> GenerationResult:
        """Generate one answer without reloading the weights."""
        loaded = self.ensure_loaded()
        result = run_generation(loaded, question, settings, on_token=on_token)
        result.load_count = self.load_count
        return result


def load_hf_model(spec: ModelSpec) -> LoadedModel:
    """Download or read a cached causal LM and freeze it."""
    if not spec.allow_download and not spec.local_files_only:
        reject_download(False, resource=f"model {spec.name}@{spec.revision}")
    if spec.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch = _import_torch()
    transformers = _import_transformers()
    available = available_ram_bytes()
    dtype_name = resolve_dtype(spec.name, spec.dtype, available)
    assert_weights_fit(spec.name, dtype_name, available)
    dtype = getattr(torch, dtype_name)
    try:
        model = _from_pretrained(transformers, spec, dtype)
    except (MemoryError, RuntimeError, OSError) as exc:
        if not _is_memory_error(exc) or dtype_name == "bfloat16":
            if _is_memory_error(exc):
                raise ModelMemoryError(f"Loading {spec.name} ran out of memory.") from exc
            raise
        dtype_name = "bfloat16"
        assert_weights_fit(spec.name, dtype_name, available_ram_bytes())
        model = _from_pretrained(transformers, spec, torch.bfloat16)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec.name,
        revision=spec.revision,
        local_files_only=spec.local_files_only,
    )
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    device = _resolve_device(spec.device, torch)
    try:
        model.to(device)
    except (MemoryError, RuntimeError) as exc:
        if _is_memory_error(exc):
            raise ModelMemoryError(f"Moving {spec.name} to {device} ran out of memory.") from exc
        raise
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen parameters were not frozen")
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if not isinstance(revision, str) or _COMMIT_HASH.fullmatch(revision) is None:
        revision = cached_commit(spec.name, spec.revision) or spec.revision
    return LoadedModel(
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device=str(device),
        dtype=dtype_name,
        resolved_revision=str(revision),
    )


def run_generation(
    loaded: LoadedModel,
    question: str,
    settings: GenerationSettings,
    on_token: Any = None,
) -> GenerationResult:
    """Generate with the loaded model and read its logits and hidden states."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    torch = _import_torch()
    model = loaded.model
    tokenizer = loaded.tokenizer
    prompt_text, prompt_mode = render_prompt(tokenizer, question.strip())
    encoded = tokenizer(prompt_text, return_tensors="pt")
    device = module_device(model)
    model_inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if key in {"input_ids", "attention_mask"}
    }
    prompt_tokens = int(model_inputs["input_ids"].shape[1])
    started = time.perf_counter()
    new_ids, tokens = _decode_tokens(
        model,
        tokenizer,
        model_inputs["input_ids"],
        model_inputs.get("attention_mask"),
        settings,
        on_token=on_token,
    )
    latency = time.perf_counter() - started
    if new_ids:
        continuation = torch.tensor(
            [new_ids],
            device=model_inputs["input_ids"].device,
            dtype=model_inputs["input_ids"].dtype,
        )
        sequences = torch.cat([model_inputs["input_ids"], continuation], dim=1)
    else:
        sequences = model_inputs["input_ids"]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip() if new_ids else ""
    feature, layer_vectors, hidden_count, feature_layer, hidden_ok, note, answer_mean, capture = (
        _read_hidden_states(
            model,
            sequences,
            prompt_tokens=prompt_tokens,
            generated_count=len(new_ids),
            settings=settings,
        )
    )
    return GenerationResult(
        question=question.strip(),
        prompt_text=prompt_text,
        prompt_mode=prompt_mode,
        answer=answer,
        model_name=loaded.spec.name,
        model_revision=loaded.resolved_revision,
        device=str(device),
        dtype=loaded.dtype,
        latency_seconds=latency,
        prompt_tokens=prompt_tokens,
        generated_tokens=len(tokens),
        tokens=tuple(tokens),
        hidden_states_available=hidden_ok,
        logits_available=bool(tokens),
        feature=feature,
        feature_layer=feature_layer,
        hidden_state_count=hidden_count,
        position=settings.position,
        layer_vectors=layer_vectors,
        load_count=0,
        note=note,
        decoding="greedy" if settings.temperature <= 0 else "temperature",
        temperature=float(settings.temperature),
        answer_mean_feature=answer_mean,
        token_spans=capture["spans"],
        capture_mode=str(capture["mode"]),
        capture_bytes=int(capture["nbytes"]),
        capture_note=str(capture["note"]),
    )


def completion_hidden_states(
    loaded: LoadedModel,
    question: str,
    completion: str,
    settings: GenerationSettings,
) -> tuple[dict[int, np.ndarray], int | None, int | None, str, str]:
    """Read hidden states for a supplied completion using the demo prompt.

    The prompt is tokenized by the same call as ``run_generation``. The
    completion is appended with ``add_special_tokens=False`` so no extra
    beginning-of-sequence token is inserted between the prompt and the answer.
    The returned layer index is the resolved value of ``settings.layer_index``.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(completion, str) or not completion.strip():
        raise ValueError("completion must be a non-empty string")
    torch = _import_torch()
    tokenizer = loaded.tokenizer
    prompt_text, prompt_mode = render_prompt(tokenizer, question.strip())
    encoded = tokenizer(prompt_text, return_tensors="pt")
    device = module_device(loaded.model)
    input_ids = encoded["input_ids"].to(device)
    answer_ids = tokenizer.encode(completion.strip(), add_special_tokens=False)
    if not answer_ids:
        raise ValueError("completion produced no tokens")
    if int(input_ids.shape[1]) + len(answer_ids) > 256:
        raise ValueError("prompt plus completion is longer than 256 tokens")
    continuation = torch.tensor([answer_ids], device=device, dtype=input_ids.dtype)
    sequences = torch.cat([input_ids, continuation], dim=1)
    _feature, vectors, hidden_count, feature_layer, ok, note, _answer_mean, _capture = (
        _read_hidden_states(
            loaded.model,
            sequences,
            prompt_tokens=int(input_ids.shape[1]),
            generated_count=len(answer_ids),
            settings=settings,
        )
    )
    if not ok or hidden_count is None or feature_layer is None:
        raise RuntimeError(note or "Hidden states were unavailable")
    return vectors, hidden_count, feature_layer, prompt_mode, prompt_text


def render_prompt(tokenizer: Any, question: str) -> tuple[str, str]:
    """Apply the model chat template, or a plain question format when it has none."""
    template = getattr(tokenizer, "chat_template", None)
    apply = getattr(tokenizer, "apply_chat_template", None)
    instruction = (
        "Answer directly and concisely. If the question assumes something false, point that out."
    )
    if template and callable(apply):
        try:
            rendered = apply(
                [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            rendered = apply(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return str(rendered), "chat_template"
    return f"{instruction}\n\nQuestion: {question}\nAnswer:", "plain"


def _decode_tokens(
    model: Any,
    tokenizer: Any,
    input_ids: Any,
    attention_mask: Any,
    settings: GenerationSettings,
    on_token: Any = None,
) -> tuple[list[int], list[GeneratedToken]]:
    """Greedy or temperature sampling with the actual next-token logits."""
    torch = _import_torch()
    device = input_ids.device
    running = input_ids
    attention = torch.ones_like(input_ids) if attention_mask is None else attention_mask
    past = None
    new_ids: list[int] = []
    tokens: list[GeneratedToken] = []
    stop_ids = _stop_ids(tokenizer, model)
    for _step in range(settings.max_new_tokens):
        feed = running if past is None else running[:, -1:]
        kwargs: dict[str, Any] = {
            "input_ids": feed,
            "attention_mask": attention,
            "use_cache": True,
        }
        if past is not None:
            kwargs["past_key_values"] = past
        try:
            with torch.inference_mode():
                outputs = model(**kwargs)
        except RuntimeError as exc:
            if _is_memory_error(exc):
                raise ModelMemoryError("Generation ran out of memory.") from exc
            raise
        past = outputs.past_key_values
        logits = outputs.logits[0, -1].detach().float().cpu().numpy()
        del outputs
        chosen = _choose_id(logits, settings.temperature)
        if chosen in stop_ids or _token_text(tokenizer, chosen).strip() in {
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
        }:
            break
        token = _token_from_logits(
            tokenizer,
            chosen,
            logits,
            top_k=settings.top_k,
            temperature=settings.temperature,
            step=len(tokens) + 1,
            retain_full=settings.distribution_export == "full",
        )
        new_ids.append(chosen)
        tokens.append(token)
        nxt = torch.tensor([[chosen]], device=device, dtype=running.dtype)
        running = torch.cat([running, nxt], dim=1)
        attention = torch.cat(
            [attention, torch.ones((1, 1), device=device, dtype=attention.dtype)],
            dim=1,
        )
        if on_token is not None:
            on_token(token)
    return new_ids, tokens


def _choose_id(logits: np.ndarray, temperature: float) -> int:
    """Pick the next id from the same distribution the report will display."""
    scaled = scale_logits(logits, temperature)
    if temperature <= 0:
        return int(np.argmax(scaled))
    probabilities = probabilities_from_logits(scaled)
    return int(np.random.choice(probabilities.shape[0], p=probabilities))


def _token_from_logits(
    tokenizer: Any,
    token_id: int,
    logits: np.ndarray,
    *,
    top_k: int,
    temperature: float = 0.0,
    step: int = 1,
    retain_full: bool = False,
) -> GeneratedToken:
    """Record the written token under the decoding distribution.

    ``top_k`` counts alternative candidates and does not include the written
    token. Step 1 is conditioned only on the prompt. Later steps are
    conditioned on the prompt plus output tokens already written.
    """
    scaled = scale_logits(logits, temperature)
    probabilities = probabilities_from_logits(scaled)
    probability = float(probabilities[int(token_id)])
    alternatives = other_candidates(
        probabilities,
        int(token_id),
        count=top_k,
        text_for_id=lambda index: _token_text(tokenizer, index),
    )
    order = np.argsort(-probabilities)
    top_probability = float(probabilities[int(order[0])])
    second = float(probabilities[int(order[1])]) if probabilities.size > 1 else 0.0
    greedy = temperature <= 0
    return GeneratedToken(
        token_id=int(token_id),
        text=_token_text(tokenizer, int(token_id)),
        probability=probability,
        top_alternatives=alternatives,
        is_argmax=bool(probability == top_probability),
        entropy_nats=entropy_nats(probabilities),
        normalized_entropy=normalized_entropy(probabilities),
        margin=float(top_probability - second),
        rank=int(np.sum(probabilities > probability)) + 1,
        step=int(step),
        conditioned_on="prompt" if step <= 1 else "prompt_plus_generated",
        decoding="greedy" if greedy else "temperature",
        temperature=float(temperature),
        vocabulary_probabilities=probabilities.astype(np.float32) if retain_full else None,
    )


def _stop_ids(tokenizer: Any, model: Any) -> set[int]:
    found: set[int] = set()
    sources = [getattr(tokenizer, "eos_token_id", None)]
    generation = getattr(model, "generation_config", None)
    if generation is not None:
        sources.append(getattr(generation, "eos_token_id", None))
    for source in sources:
        if source is None:
            continue
        if isinstance(source, (list, tuple, set)):
            found.update(int(item) for item in source if item is not None)
        else:
            found.add(int(source))
    return found


def _token_text(tokenizer: Any, token_id: int) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        try:
            return str(
                decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            )
        except TypeError:
            return str(decode([token_id]))
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        return str(convert(token_id))
    return str(token_id)


def _read_hidden_states(
    model: Any,
    sequences: Any,
    *,
    prompt_tokens: int,
    generated_count: int,
    settings: GenerationSettings,
) -> tuple[
    np.ndarray | None,
    dict[int, np.ndarray],
    int | None,
    int | None,
    bool,
    str,
    np.ndarray | None,
    dict[str, object],
]:
    torch = _import_torch()
    try:
        position = feature_index(prompt_tokens, generated_count, settings.position)
    except ValueError as exc:
        return (
            None,
            {},
            None,
            None,
            False,
            str(exc),
            None,
            {"mode": settings.capture_mode, "spans": {}, "nbytes": 0, "note": str(exc)},
        )
    try:
        with torch.inference_mode():
            forward = model(sequences, output_hidden_states=True, use_cache=False)
    except (MemoryError, RuntimeError) as exc:
        if _is_memory_error(exc):
            return (
                None,
                {},
                None,
                None,
                False,
                "Hidden-state forward pass ran out of memory.",
                None,
                {
                    "mode": settings.capture_mode,
                    "spans": {},
                    "nbytes": 0,
                    "note": "Hidden-state forward pass ran out of memory.",
                },
            )
        return (
            None,
            {},
            None,
            None,
            False,
            f"Hidden-state forward pass failed: {exc}",
            None,
            {
                "mode": settings.capture_mode,
                "spans": {},
                "nbytes": 0,
                "note": f"Hidden-state forward pass failed: {exc}",
            },
        )
    hidden_states = forward.hidden_states
    if not hidden_states:
        return (
            None,
            {},
            None,
            None,
            False,
            "The model returned no hidden states.",
            None,
            {
                "mode": settings.capture_mode,
                "spans": {},
                "nbytes": 0,
                "note": "The model returned no hidden states.",
            },
        )
    feature_layer = resolve_layer(len(hidden_states), settings.layer_index)
    layers = None if settings.capture_all_layers else (feature_layer,)
    vectors = vectors_at_position(hidden_states, position, layers=layers)
    feature = vectors.get(feature_layer)
    answer_mean = None
    if generated_count >= 1:
        try:
            answer_mean = mean_answer_vector(
                hidden_states, prompt_tokens, generated_count, feature_layer
            )
        except ValueError:
            answer_mean = None

    # Capture token spans based on capture_mode
    capture_layers = select_layers(len(hidden_states), settings.capture_mode, feature_layer)
    token_spans = {}
    capture_note = ""
    if generated_count >= 1 and capture_layers:
        try:
            selected, note = spans_within_budget(
                generated_count, capture_layers, int(hidden_states[0].shape[2])
            )
            if note:
                capture_note = note
            if selected:
                token_spans = slice_generated_spans(
                    hidden_states,
                    prompt_tokens=prompt_tokens,
                    generated_count=generated_count,
                    layers=selected,
                )
        except (ValueError, RuntimeError) as exc:
            capture_note = f"Token span capture failed: {exc}"

    capture_bytes = sum(span.nbytes for span in token_spans.values())

    return (
        feature,
        vectors,
        len(hidden_states),
        feature_layer,
        feature is not None,
        "",
        answer_mean,
        {
            "mode": settings.capture_mode,
            "spans": token_spans,
            "nbytes": capture_bytes,
            "note": capture_note or "Capture completed.",
        },
    )


def _from_pretrained(transformers: Any, spec: ModelSpec, dtype: Any) -> Any:
    kwargs = {
        "revision": spec.revision,
        "local_files_only": spec.local_files_only,
        "dtype": dtype,
        "attn_implementation": "eager",
    }
    try:
        import accelerate  # noqa: F401

        kwargs["low_cpu_mem_usage"] = True
    except ImportError:
        pass
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(spec.name, **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        kwargs.pop("low_cpu_mem_usage", None)
        return transformers.AutoModelForCausalLM.from_pretrained(spec.name, **kwargs)


def _resolve_device(requested: str, torch: Any) -> Any:
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise ModelMemoryError("CUDA was requested, but this PyTorch build has no CUDA device.")
        return torch.device("cuda")
    return torch.device("cpu")


def _is_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "not enough memory" in message


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Use the CPU wheel: "
            "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch


def _import_transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "transformers is not installed. Install the models extra: "
            'python -m pip install -e ".[models]"'
        ) from exc
    return transformers
