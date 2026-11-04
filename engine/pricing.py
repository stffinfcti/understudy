"""List-price resolution and monthly cost projection. Hermes-free: the
models.dev catalog is injected as a plain dict."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Optional

LOCAL_PROVIDERS = frozenset({"ollama", "lmstudio", "llamacpp", "vllm"})
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_VENDOR_PREFIX = {
    "moonshotai": "moonshotai", "deepseek-ai": "deepseek", "deepseek": "deepseek",
    "zai-org": "zai", "z-ai": "zai", "zai": "zai", "qwen": "alibaba",
    "nvidia": "nvidia", "anthropic": "anthropic", "openai": "openai",
    "google": "google", "meta": "meta", "meta-llama": "meta",
    "mistralai": "mistral", "x-ai": "xai", "xai": "xai",
}

_TAG_RE = re.compile(
    r"(?:-\d{8}|-\d{6}|-\d{4}|-preview|-latest|-fp8|-nvfp4|-int4|-int8|-awq|-gguf|-bf16)+$")


@dataclass
class Price:
    input_per_m: float
    output_per_m: float
    cache_read_per_m: Optional[float]
    source: str


def _normalize_model(model: str) -> str:
    model = str(model or "").lower()
    if "/" in model:
        model = model.rsplit("/", 1)[-1]
    return _TAG_RE.sub("", model)


def _vendor_of(model: str) -> str:
    prefix = str(model or "").lower().split("/", 1)[0]
    return _VENDOR_PREFIX.get(prefix, "")


def _host(base_url: Optional[str]) -> str:
    if not base_url:
        return ""
    host = re.sub(r"^https?://", "", str(base_url).strip()).split("/")[0].split(":")[0]
    return host.lower()


def _zero(source: str) -> Price:
    return Price(0.0, 0.0, 0.0, source)


def _settings_price(settings_prices: Optional[dict], provider: Optional[str],
                    model: str) -> Optional[Price]:
    if not isinstance(settings_prices, dict):
        return None
    keys = [f"{provider}:{model}" if provider else "", str(model)]
    lowered = {str(k).lower(): v for k, v in settings_prices.items()}
    for key in keys:
        if not key:
            continue
        entry = lowered.get(key.lower())
        if isinstance(entry, dict) and "input" in entry and "output" in entry:
            try:
                return Price(float(entry["input"]), float(entry["output"]),
                             float(entry["cache_read"]) if entry.get("cache_read") is not None else None,
                             "settings")
            except (TypeError, ValueError):
                continue
    return None


def _catalog_price(provider: Optional[str], model: str,
                   catalog: Optional[dict]) -> Optional[Price]:
    if not isinstance(catalog, dict) or not catalog:
        return None
    target = _normalize_model(model)
    if not target:
        return None
    vendor = _vendor_of(model)
    matches: list[tuple[str, dict]] = []  # (provider_id, cost dict)
    for pid, pdata in catalog.items():
        models = pdata.get("models") if isinstance(pdata, dict) else None
        if not isinstance(models, dict):
            continue
        for mid, entry in models.items():
            if _normalize_model(mid) != target or not isinstance(entry, dict):
                continue
            cost = entry.get("cost")
            if not isinstance(cost, dict):
                continue
            try:
                inp, out = float(cost.get("input") or 0), float(cost.get("output") or 0)
            except (TypeError, ValueError):
                continue
            if inp == 0 and out == 0:
                continue
            cr = cost.get("cache_read")
            matches.append((str(pid).lower(), {
                "input": inp, "output": out,
                "cache_read": float(cr) if isinstance(cr, (int, float)) else None}))
    if not matches:
        return None
    if vendor:
        own = [m for pid, m in matches if pid == vendor]
        if own:
            m = own[0]
            return Price(m["input"], m["output"], m["cache_read"], f"models.dev ({vendor})")
    inp = median(m["input"] for _, m in matches)
    out = median(m["output"] for _, m in matches)
    crs = [m["cache_read"] for _, m in matches if m["cache_read"] is not None]
    cr = median(crs) if crs else None
    return Price(inp, out, cr, f"models.dev (median of {len(matches)})")


def resolve_price(provider: Optional[str], model: str, base_url: Optional[str] = None,
                  settings_prices: Optional[dict] = None,
                  catalog: Optional[dict] = None) -> Optional[Price]:
    """USD per 1M tokens for a route, or None when nothing can price it."""
    price = _settings_price(settings_prices, provider, model)
    if price is not None:
        return price
    host = _host(base_url)
    if (str(provider or "").lower() in LOCAL_PROVIDERS
            or host in LOCAL_HOSTS or host.endswith(".local")):
        return _zero("local")
    return _catalog_price(provider, model, catalog)


def call_cost(usage: Any, price: Optional[Price]) -> tuple[Optional[float], str]:
    """``(cost_usd, source)`` for one LLM call's usage object."""
    cost = getattr(usage, "cost_usd", None)
    if isinstance(cost, (int, float)):
        return float(cost), "metered"
    if price is None:
        return None, ""
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    cr = getattr(usage, "cache_read_tokens", 0) or 0
    total = (inp * price.input_per_m + out * price.output_per_m
             + (cr * (price.cache_read_per_m or 0.0))) / 1_000_000
    return total, price.source


def usage_cost(tokens: dict, price: Optional[Price]) -> Optional[float]:
    """Price a {input,output,cache_read} token bundle."""
    if price is None:
        return None
    return ((tokens.get("input_tokens", 0) * price.input_per_m)
            + (tokens.get("output_tokens", 0) * price.output_per_m)
            + (tokens.get("cache_read_tokens", 0) * (price.cache_read_per_m or 0.0))) / 1_000_000


def load_catalog() -> Optional[dict]:
    """models.dev cache from the active home, then the default root home."""
    from hermes_constants import get_hermes_home
    homes = [get_hermes_home()]
    try:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root()
        if root != homes[0]:
            homes.append(root)
    except Exception:
        pass
    for home in homes:
        path = home / "models_dev_cache.json"
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except (OSError, ValueError):
            continue
    return None
