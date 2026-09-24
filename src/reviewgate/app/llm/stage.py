"""Hosted LLM enrichment stage orchestration (issues #59–#64)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from reviewgate.core.config import ReviewGateConfig
from reviewgate.core.schemas import ReviewabilityReport

from reviewgate.app.analysis.pipeline import PipelineAnalysisArtifacts
from reviewgate.app.llm.budgets import (
    _HARD_MAX_USD_PER_ANALYSIS,
    _unrounded_cost_usd,
    estimated_prompt_cost_within_hard_cap,
    llm_input_packaging_mode,
    resolve_model_token_prices,
    rough_token_estimate,
)
from reviewgate.app.llm.client import LlmCallUsage, complete_reviewability_json
from reviewgate.app.llm.input_pack import build_llm_user_message
from reviewgate.app.llm.merge_report import apply_llm_to_deterministic_report, zero_llm_cost_fields
from reviewgate.app.llm.prompts import load_reviewability_v1_prompt
from reviewgate.app.settings import AppSettings

logger = logging.getLogger(__name__)

_ASSUMED_COMPLETION_TOKENS: Final[int] = 900


@dataclass(frozen=True, slots=True)
class HostedLlmStageOutcome:
    """Result of :func:`maybe_apply_hosted_llm_stage`."""

    report: ReviewabilityReport
    llm_used: bool
    llm_provider: str | None
    input_tokens: int | None
    output_tokens: int | None
    estimated_cost_usd: Decimal | None


def _usage_cost_fields(
    usage: LlmCallUsage | None,
    *,
    input_per_million: Decimal,
    output_per_million: Decimal,
) -> tuple[str | None, int | None, int | None, Decimal | None]:
    """Derive §11.4 provider, token and cost fields from a billed LLM call.

    Applies the post-hoc hard cap check so that spend is observable on every
    path where the provider billed tokens, including parse failures.

    Args:
        usage: Token usage reported by the provider, or ``None`` when the
            call did not report any.

    Returns:
        A tuple of ``(provider, input_tokens, output_tokens, estimated_cost_usd)``,
        matching the corresponding :class:`HostedLlmStageOutcome` fields.
    """

    in_tok = usage.input_tokens if usage is not None else None
    out_tok = usage.output_tokens if usage is not None else None
    provider = usage.provider if usage is not None else None
    cost: Decimal | None = None
    if in_tok is not None and out_tok is not None:
        raw_cost = _unrounded_cost_usd(
            input_tokens=in_tok,
            output_tokens=out_tok,
            input_per_million=input_per_million,
            output_per_million=output_per_million,
        )
        cost = raw_cost.quantize(Decimal("0.0001"))
        if raw_cost > _HARD_MAX_USD_PER_ANALYSIS:
            logger.warning(
                "hosted_llm_post_hoc_cost_over_cap",
                extra={"estimated_cost_usd": str(raw_cost)},
            )
    return provider, in_tok, out_tok, cost


def maybe_apply_hosted_llm_stage(
    settings: AppSettings,
    *,
    deterministic_report: ReviewabilityReport,
    effective_config: ReviewGateConfig,
    artifacts: PipelineAnalysisArtifacts | None,
) -> HostedLlmStageOutcome:
    """§21.3 / §11: optionally call the LLM and merge into the published report.

    Skips entirely when ``llm_reports`` is false, when fail-fast left no
    artifacts, when no API key is configured, or when the pre-flight cost/token
    estimate exceeds §11.4 budgets.
    """

    if not effective_config.llm_reports:
        u1, u2, u3, u4, u5 = zero_llm_cost_fields()
        return HostedLlmStageOutcome(
            report=deterministic_report,
            llm_used=u1,
            llm_provider=u2,
            input_tokens=u3,
            output_tokens=u4,
            estimated_cost_usd=u5,
        )

    if artifacts is None:
        logger.info("hosted_llm_skipped_fail_fast_tier")
        u1, u2, u3, u4, u5 = zero_llm_cost_fields()
        return HostedLlmStageOutcome(
            report=deterministic_report,
            llm_used=u1,
            llm_provider=u2,
            input_tokens=u3,
            output_tokens=u4,
            estimated_cost_usd=u5,
        )

    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value().strip():
        logger.info("hosted_llm_skipped_missing_openai_key")
        u1, u2, u3, u4, u5 = zero_llm_cost_fields()
        return HostedLlmStageOutcome(
            report=deterministic_report,
            llm_used=u1,
            llm_provider=u2,
            input_tokens=u3,
            output_tokens=u4,
            estimated_cost_usd=u5,
        )

    prices = resolve_model_token_prices(
        settings.llm_model,
        input_per_million=settings.llm_input_usd_per_million,
        output_per_million=settings.llm_output_usd_per_million,
    )
    if prices is None:
        logger.warning(
            "hosted_llm_skipped_unknown_model_pricing",
            extra={"model": settings.llm_model},
        )
        u1, u2, u3, u4, u5 = zero_llm_cost_fields()
        return HostedLlmStageOutcome(
            report=deterministic_report,
            llm_used=u1,
            llm_provider=u2,
            input_tokens=u3,
            output_tokens=u4,
            estimated_cost_usd=u5,
        )

    input_price, output_price = prices
    packaging = llm_input_packaging_mode(artifacts.changed_files_count)
    user_message = build_llm_user_message(
        pr=artifacts.pr,
        report=deterministic_report,
        files=artifacts.files,
        mode=packaging,
    )
    est_in = rough_token_estimate(load_reviewability_v1_prompt()) + rough_token_estimate(
        user_message,
    )
    if not estimated_prompt_cost_within_hard_cap(
        estimated_input_tokens=est_in,
        assumed_output_tokens=_ASSUMED_COMPLETION_TOKENS,
        input_per_million=input_price,
        output_per_million=output_price,
    ):
        logger.warning(
            "hosted_llm_skipped_preflight_budget",
            extra={"estimated_input_tokens": est_in},
        )
        u1, u2, u3, u4, u5 = zero_llm_cost_fields()
        return HostedLlmStageOutcome(
            report=deterministic_report,
            llm_used=u1,
            llm_provider=u2,
            input_tokens=u3,
            output_tokens=u4,
            estimated_cost_usd=u5,
        )

    try:
        result = complete_reviewability_json(
            settings,
            system_prompt=load_reviewability_v1_prompt(),
            user_prompt=user_message,
        )
    except Exception:
        logger.exception("hosted_llm_unexpected_failure")
        u1, u2, u3, u4, u5 = zero_llm_cost_fields()
        return HostedLlmStageOutcome(
            report=deterministic_report,
            llm_used=u1,
            llm_provider=u2,
            input_tokens=u3,
            output_tokens=u4,
            estimated_cost_usd=u5,
        )

    if result.parsed is None:
        logger.info("hosted_llm_parse_failed_using_deterministic_only")
        provider, in_tok, out_tok, cost = _usage_cost_fields(
            result.usage,
            input_per_million=input_price,
            output_per_million=output_price,
        )
        return HostedLlmStageOutcome(
            report=deterministic_report,
            llm_used=False,
            llm_provider=provider,
            input_tokens=in_tok,
            output_tokens=out_tok,
            estimated_cost_usd=cost,
        )

    merged = apply_llm_to_deterministic_report(
        deterministic_report,
        result.parsed,
        labels=effective_config.labels,
    )
    provider, in_tok, out_tok, cost = _usage_cost_fields(
        result.usage,
        input_per_million=input_price,
        output_per_million=output_price,
    )

    return HostedLlmStageOutcome(
        report=merged,
        llm_used=True,
        llm_provider=provider,
        input_tokens=in_tok,
        output_tokens=out_tok,
        estimated_cost_usd=cost,
    )
