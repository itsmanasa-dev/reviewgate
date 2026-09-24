"""Regression tests for model-aware hosted LLM pricing (issue #150)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from reviewgate.app.analysis.pipeline import PipelineAnalysisArtifacts
from reviewgate.app.llm.budgets import (
    estimate_cost_usd,
    estimated_prompt_cost_within_hard_cap,
    resolve_model_token_prices,
)
from reviewgate.app.llm.client import LlmCallResult, LlmCallUsage
from reviewgate.app.llm.stage import (
    _usage_cost_fields,
    maybe_apply_hosted_llm_stage,
)
from reviewgate.app.settings import AppSettings
from reviewgate.core.config import ReviewGateConfig
from reviewgate.core.schemas import ChangedFile, PRRecord, ReviewabilityReport


@pytest.fixture(autouse=True)
def clear_price_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REVIEWGATE_LLM_INPUT_USD_PER_MILLION", raising=False)
    monkeypatch.delenv("REVIEWGATE_LLM_OUTPUT_USD_PER_MILLION", raising=False)


def _stage_inputs() -> tuple[ReviewabilityReport, PipelineAnalysisArtifacts]:
    report = ReviewabilityReport(
        reviewability="PASS",
        stats={},
        warnings=[],
        suggested_labels=[],
        file_categories=[],
        split_hints=[],
        reviewer_checklist=[],
    )
    artifacts = PipelineAnalysisArtifacts(
        pr=PRRecord(
            title="Fixes #150",
            body="Improve model cost accounting.",
            author="octocat",
            base_branch="main",
            head_branch="fix/pricing",
            additions=1,
            deletions=0,
            changed_files=1,
        ),
        files=[
            ChangedFile(
                filename="README.md",
                status="modified",
                additions=1,
                deletions=0,
                changes=1,
            ),
        ],
        changed_files_count=1,
    )
    return report, artifacts


def _run_stage(settings: AppSettings) -> object:
    report, artifacts = _stage_inputs()
    return maybe_apply_hosted_llm_stage(
        settings,
        deterministic_report=report,
        effective_config=ReviewGateConfig(llm_reports=True),
        artifacts=artifacts,
    )


def test_only_exact_default_model_uses_bundled_prices() -> None:
    assert resolve_model_token_prices("gpt-4o-mini") == (
        Decimal("0.150"),
        Decimal("0.600"),
    )
    assert resolve_model_token_prices("gpt-4") is None
    assert resolve_model_token_prices("gpt-4o-mini-new-version") is None


def test_explicit_prices_override_even_the_default_model() -> None:
    assert resolve_model_token_prices(
        "gpt-4o-mini",
        input_per_million=Decimal("4"),
        output_per_million=Decimal("8"),
    ) == (Decimal("4"), Decimal("8"))
    assert resolve_model_token_prices(
        "custom-model",
        input_per_million=Decimal("1"),
        output_per_million=Decimal("2"),
    ) == (Decimal("1"), Decimal("2"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"llm_input_usd_per_million": Decimal("2")},
        {"llm_output_usd_per_million": Decimal("3")},
        {
            "llm_input_usd_per_million": Decimal("-1"),
            "llm_output_usd_per_million": Decimal("2"),
        },
    ],
)
def test_partial_or_negative_settings_are_rejected(
    kwargs: dict[str, Decimal],
) -> None:
    with pytest.raises(ValidationError):
        AppSettings(**kwargs)


def test_explicit_prices_load_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REVIEWGATE_LLM_INPUT_USD_PER_MILLION", "12.50")
    monkeypatch.setenv("REVIEWGATE_LLM_OUTPUT_USD_PER_MILLION", "30")
    settings = AppSettings(llm_model="custom-model")
    assert settings.llm_input_usd_per_million == Decimal("12.50")
    assert settings.llm_output_usd_per_million == Decimal("30")


def test_preflight_uses_custom_prices_instead_of_mini_rates() -> None:
    baseline = estimated_prompt_cost_within_hard_cap(
        estimated_input_tokens=50_000,
        assumed_output_tokens=900,
    )
    expensive = estimated_prompt_cost_within_hard_cap(
        estimated_input_tokens=50_000,
        assumed_output_tokens=900,
        input_per_million=Decimal("10"),
        output_per_million=Decimal("20"),
    )
    assert baseline is True
    assert expensive is False


def test_unknown_model_skips_provider_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with patch(
        "reviewgate.app.llm.stage.complete_reviewability_json",
        side_effect=AssertionError("unknown-price model must not be called"),
    ) as provider:
        outcome = _run_stage(
            AppSettings(
                llm_model="custom-unpriced-model",
                openai_api_key=SecretStr("sk-test"),
            )
        )
    provider.assert_not_called()
    assert outcome.llm_used is False
    assert outcome.estimated_cost_usd is None
    assert "hosted_llm_skipped_unknown_model_pricing" in caplog.text


def test_expensive_model_fails_preflight_without_provider_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = AppSettings(
        llm_model="custom-expensive-model",
        llm_input_usd_per_million=Decimal("1000"),
        llm_output_usd_per_million=Decimal("1000"),
        openai_api_key=SecretStr("sk-test"),
    )
    with patch(
        "reviewgate.app.llm.stage.complete_reviewability_json",
        side_effect=AssertionError("preflight must skip this request"),
    ) as provider:
        outcome = _run_stage(settings)
    provider.assert_not_called()
    assert outcome.llm_used is False
    assert "hosted_llm_skipped_preflight_budget" in caplog.text


def test_custom_prices_apply_when_json_parse_fails() -> None:
    settings = AppSettings(
        llm_model="custom-priced-model",
        llm_input_usd_per_million=Decimal("1"),
        llm_output_usd_per_million=Decimal("2"),
        openai_api_key=SecretStr("sk-test"),
    )
    billed = LlmCallResult(
        parsed=None,
        usage=LlmCallUsage(
            input_tokens=1000,
            output_tokens=1000,
            provider="openai",
        ),
    )
    with patch(
        "reviewgate.app.llm.stage.complete_reviewability_json",
        return_value=billed,
    ) as provider:
        outcome = _run_stage(settings)
    provider.assert_called_once()
    assert outcome.llm_used is False
    assert outcome.input_tokens == 1000
    assert outcome.output_tokens == 1000
    assert outcome.estimated_cost_usd == Decimal("0.0030")
    assert outcome.estimated_cost_usd == estimate_cost_usd(
        input_tokens=1000,
        output_tokens=1000,
        input_per_million=Decimal("1"),
        output_per_million=Decimal("2"),
    )


def test_post_hoc_cap_checks_custom_prices_on_parse_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = AppSettings(
        llm_model="custom-priced-model",
        llm_input_usd_per_million=Decimal("1"),
        llm_output_usd_per_million=Decimal("1"),
        openai_api_key=SecretStr("sk-test"),
    )
    billed = LlmCallResult(
        parsed=None,
        usage=LlmCallUsage(
            input_tokens=1000,
            output_tokens=250_000,
            provider="openai",
        ),
    )
    with patch(
        "reviewgate.app.llm.stage.complete_reviewability_json",
        return_value=billed,
    ):
        outcome = _run_stage(settings)
    assert outcome.estimated_cost_usd == Decimal("0.2510")
    assert "hosted_llm_post_hoc_cost_over_cap" in caplog.text


def test_preflight_detects_subcent_overrun() -> None:
    """Display rounding must not hide a cost above the cap."""

    prices = {
        "input_per_million": Decimal("1"),
        "output_per_million": Decimal("0"),
    }
    assert estimate_cost_usd(
        input_tokens=200_001,
        output_tokens=0,
        **prices,
    ) == Decimal("0.2000")

    assert estimated_prompt_cost_within_hard_cap(
        estimated_input_tokens=200_000,
        assumed_output_tokens=0,
        **prices,
    )
    assert not estimated_prompt_cost_within_hard_cap(
        estimated_input_tokens=200_001,
        assumed_output_tokens=0,
        **prices,
    )


def test_posthoc_detects_subcent_overrun(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The post-hoc warning compares unrounded provider usage."""

    usage = LlmCallUsage(
        input_tokens=200_001,
        output_tokens=0,
        provider="openai",
    )
    provider, in_tok, out_tok, cost = _usage_cost_fields(
        usage,
        input_per_million=Decimal("1"),
        output_per_million=Decimal("0"),
    )

    assert provider == "openai"
    assert in_tok == 200_001
    assert out_tok == 0
    assert cost == Decimal("0.2000")
    assert "hosted_llm_post_hoc_cost_over_cap" in caplog.text
