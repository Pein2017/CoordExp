from __future__ import annotations

import tomllib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from .model import Usage


@dataclass(frozen=True)
class PriceRate:
    provider: str
    model: str
    currency: str
    input_per_million: Decimal
    cached_input_per_million: Decimal = Decimal("0")
    cache_write_per_million: Decimal = Decimal("0")
    output_per_million: Decimal = Decimal("0")
    reasoning_per_million: Decimal | None = None
    cache_write_is_in_input: bool = False
    reasoning_is_in_output: bool = True
    source: str | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


def _decimal(value: Any, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (ArithmeticError, ValueError):
        raise ValueError(f"price field {field!r} must be numeric") from None
    if result < 0:
        raise ValueError(f"price field {field!r} must be non-negative")
    return result


def load_rates(path: str | Path | None) -> dict[tuple[str, str], PriceRate]:
    if path is None:
        return {}
    price_path = Path(path)
    with price_path.open("rb") as stream:
        data = tomllib.load(stream)
    entries = data.get("rates", [])
    if not isinstance(entries, list):
        raise ValueError("prices TOML must contain [[rates]] entries")
    rates: dict[tuple[str, str], PriceRate] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each [[rates]] entry must be a table")
        provider = entry.get("provider")
        model = entry.get("model")
        currency = entry.get("currency", "USD")
        if not isinstance(provider, str) or not isinstance(model, str):
            raise ValueError("each price rate requires string provider and model")
        if not isinstance(currency, str):
            raise ValueError("price currency must be a string")
        rate = PriceRate(
            provider=provider,
            model=model,
            currency=currency,
            input_per_million=_decimal(
                entry.get("input_per_million", 0), field="input_per_million"
            ),
            cached_input_per_million=_decimal(
                entry.get("cached_input_per_million", 0),
                field="cached_input_per_million",
            ),
            cache_write_per_million=_decimal(
                entry.get("cache_write_per_million", 0), field="cache_write_per_million"
            ),
            output_per_million=_decimal(
                entry.get("output_per_million", 0), field="output_per_million"
            ),
            reasoning_per_million=(
                _decimal(entry["reasoning_per_million"], field="reasoning_per_million")
                if "reasoning_per_million" in entry
                else None
            ),
            cache_write_is_in_input=bool(entry.get("cache_write_is_in_input", False)),
            reasoning_is_in_output=bool(entry.get("reasoning_is_in_output", True)),
            source=entry.get("source")
            if isinstance(entry.get("source"), str)
            else None,
        )
        rates[(provider, model)] = rate
    return rates


def lookup_rate(
    rates: dict[tuple[str, str], PriceRate], provider: str | None, model: str | None
) -> PriceRate | None:
    if not provider or not model:
        return None
    return rates.get((provider, model)) or rates.get((provider, "*"))


def estimate_cost(
    usage: Usage, rate: PriceRate
) -> tuple[Decimal | None, dict[str, Any]]:
    """Estimate cost while making cache/reasoning accounting assumptions explicit."""

    input_tokens = max(usage.input_tokens, 0)
    cached_tokens = max(usage.cached_input_tokens, 0)
    cache_write_tokens = max(usage.cache_write_input_tokens, 0)
    if rate.cache_write_is_in_input:
        uncached_tokens = max(input_tokens - cached_tokens - cache_write_tokens, 0)
    else:
        uncached_tokens = max(input_tokens - cached_tokens, 0)

    output_tokens = max(usage.output_tokens, 0)
    reasoning_tokens = max(usage.reasoning_output_tokens, 0)
    if rate.reasoning_is_in_output:
        visible_output_tokens = output_tokens
        billed_reasoning_tokens = 0
    else:
        if rate.reasoning_per_million is None and reasoning_tokens:
            return None, {"status": "missing_reasoning_rate"}
        visible_output_tokens = max(output_tokens - reasoning_tokens, 0)
        billed_reasoning_tokens = reasoning_tokens

    million = Decimal(1_000_000)
    cost = (
        Decimal(uncached_tokens) * rate.input_per_million
        + Decimal(cached_tokens) * rate.cached_input_per_million
        + Decimal(cache_write_tokens) * rate.cache_write_per_million
        + Decimal(visible_output_tokens) * rate.output_per_million
        + Decimal(billed_reasoning_tokens) * (rate.reasoning_per_million or Decimal(0))
    ) / million
    return cost, {
        "status": "ok",
        "rate_key": rate.key,
        "currency": rate.currency,
        "source": rate.source,
        "assumptions": {
            "cache_write_is_in_input": rate.cache_write_is_in_input,
            "reasoning_is_in_output": rate.reasoning_is_in_output,
        },
        "billable_tokens": {
            "uncached_input_tokens": uncached_tokens,
            "cached_input_tokens": cached_tokens,
            "cache_write_input_tokens": cache_write_tokens,
            "output_tokens": visible_output_tokens,
            "reasoning_output_tokens": billed_reasoning_tokens,
        },
    }
