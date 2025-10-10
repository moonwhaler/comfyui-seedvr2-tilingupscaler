"""Compatibility helpers for interacting with SeedVR2 stable and nightly builds."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from inspect import Signature, signature
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

FALLBACK_DEFAULT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
MODELS: Tuple[str, ...] = (
    "seedvr2_ema_3b_fp16.safetensors",
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_7b_fp16.safetensors",
    "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_7b_sharp_fp16.safetensors",
    "seedvr2_ema_7b_sharp_fp8_e4m3fn.safetensors",
)


@dataclass(frozen=True)
class SeedVR2Defaults:
    """Default configuration applied when nightly parameters are supported."""

    color_correction: str = "wavelet"
    input_noise_scale: float = 0.0
    latent_noise_scale: float = 0.0


DEFAULTS = SeedVR2Defaults()


def resolve_model_choices() -> Tuple[List[str], str]:
    """Return the available SeedVR2 models and preferred default.

    Introspects the installed SeedVR2 node to obtain dynamic model choices and default,
    falling back to the bundled list to preserve compatibility when the node is not available.
    """

    models: List[str] = list(MODELS)
    default_model: str = FALLBACK_DEFAULT_MODEL

    nightly_models: Optional[Iterable[str]] = None
    nightly_default: Optional[str] = None

    # Try introspect the installed SeedVR2 ComfyUI node to obtain
    # its own dynamic model list and default.
    try:
        import nodes  # type: ignore

        seedvr2_cls = None
        for node_class in nodes.NODE_CLASS_MAPPINGS.values():
            if getattr(node_class, "__name__", "") == "SeedVR2":
                seedvr2_cls = node_class
                break

        if seedvr2_cls and hasattr(seedvr2_cls, "INPUT_TYPES"):
            info = seedvr2_cls.INPUT_TYPES()  # type: ignore[attr-defined]
            required = info.get("required", {}) if isinstance(info, dict) else {}
            model_spec = required.get("model") if isinstance(required, dict) else None

            if isinstance(model_spec, tuple) and len(model_spec) >= 1:
                choices_src = model_spec[0]
                if isinstance(choices_src, (list, tuple)):
                    extracted = [m for m in choices_src if isinstance(m, str) and m]
                    if extracted:
                        nightly_models = extracted
                        logger.warning(
                            "Resolved SeedVR2 models via node registry: %d models.",
                            len(extracted),
                        )

                if len(model_spec) >= 2 and isinstance(model_spec[1], dict):
                    dflt = model_spec[1].get("default")
                    if isinstance(dflt, str) and dflt:
                        nightly_default = dflt
                        logger.warning("Resolved SeedVR2 default via node registry: %s", nightly_default)
    except Exception:
        logger.exception("Failed to resolve models via SeedVR2 node registry introspection.")

    # No secondary import mechanism; if introspection fails, use bundled list

    sanitized_nightly: Optional[List[str]] = None
    if nightly_models:
        sanitized_nightly = [model for model in nightly_models if isinstance(model, str) and model]

    if sanitized_nightly:
        models = sanitized_nightly
    else:
        models = list(MODELS)
        if nightly_models is None:
            logger.warning("Using bundled SeedVR2 model list; node introspection unavailable or empty.")

    sanitized_default = nightly_default if isinstance(nightly_default, str) and nightly_default else None
    if sanitized_default:
        default_model = sanitized_default
    else:
        logger.warning("Using fallback SeedVR2 default model: %s", default_model)

    deduped: List[str] = []
    seen = set()
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        deduped.append(model)

    if default_model not in seen and default_model:
        deduped.insert(0, default_model)

    return deduped, default_model


def build_execute_kwargs(
        instance: Any,
        *,
        images: Any,
        model: str,
        seed: int,
        new_resolution: int,
        batch_size: int,
        preserve_vram: bool,
        block_swap_config: Optional[Any] = None,
        color_correction: Optional[str] = None,
        input_noise_scale: Optional[float] = None,
        latent_noise_scale: Optional[float] = None,
        extra_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct keyword arguments compatible with the current SeedVR2 build.

    Supports both stable and nightly builds:
    - Stable: preserve_vram is a top-level parameter
    - Nightly: preserve_vram is moved into extra_args

    The signature introspection automatically filters parameters based on
    what the installed version supports.
    """

    sig = _get_execute_signature(instance)

    effective_color_correction = color_correction if color_correction is not None else DEFAULTS.color_correction
    effective_input_noise = input_noise_scale if input_noise_scale is not None else DEFAULTS.input_noise_scale
    effective_latent_noise = latent_noise_scale if latent_noise_scale is not None else DEFAULTS.latent_noise_scale

    # Build base payload
    payload: Dict[str, Any] = {
        "images": images,
        "model": model,
        "seed": seed,
        "new_resolution": new_resolution,
        "batch_size": batch_size,
        "color_correction": effective_color_correction,
        "input_noise_scale": effective_input_noise,
        "latent_noise_scale": effective_latent_noise,
    }

    # Add block_swap_config if provided
    if block_swap_config is not None:
        payload["block_swap_config"] = block_swap_config

    # Build extra_args for nightly support
    # Start with preserve_vram which is now in extra_args in nightly
    extra_payload: Dict[str, Any] = {"preserve_vram": preserve_vram}

    # Merge any additional extra_args provided by caller
    if extra_args:
        extra_payload.update(extra_args)

    # Add extra_args to payload (will be filtered out if not supported)
    payload["extra_args"] = extra_payload

    # Also add preserve_vram to top level for backward compatibility with stable
    # (will be filtered out if not supported)
    payload["preserve_vram"] = preserve_vram

    return _filter_supported_kwargs(payload, sig)


def _filter_supported_kwargs(payload: Dict[str, Any], sig: Optional[Signature]) -> Dict[str, Any]:
    """Filter payload keys so they match the `execute` signature of the instance."""

    cleaned = {key: value for key, value in payload.items() if value is not None}

    if sig is None:
        return cleaned

    supported = sig.parameters.keys()
    filtered: Dict[str, Any] = {}

    for key, value in cleaned.items():
        if key == "extra_args" and "extra_args" not in supported:
            continue
        if key == "block_swap_config" and "block_swap_config" not in supported:
            continue
        if key in supported:
            filtered[key] = value

    return filtered


@lru_cache(maxsize=16)
def _cached_signature_for_type(cls: type) -> Optional[Signature]:
    try:
        return signature(cls.execute)  # type: ignore[attr-defined]
    except (AttributeError, TypeError, ValueError):
        return None


def _get_execute_signature(instance: Any) -> Optional[Signature]:
    return _cached_signature_for_type(type(instance))
