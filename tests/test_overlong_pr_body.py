"""Regression tests for configurable PR-body upper bounds (issue #142)."""

from __future__ import annotations

import pytest

from reviewgate.core.config import ReviewGateConfig, load_config
from reviewgate.core.engine import analyze
from reviewgate.core.pr_body import (
    WARN_CODE_OVERLONG_BODY,
    overlong_body_warning,
)
from reviewgate.core.schemas import ChangedFile, EngineInput, PRRecord


@pytest.mark.parametrize(
    ("count", "severity"),
    [
        (79, None),
        (80, None),
        (3000, None),
        (3001, "medium"),
        (8000, "medium"),
        (8001, "high"),
    ],
)
def test_upper_boundaries(count: int, severity: str | None) -> None:
    warning = overlong_body_warning(
        "x" * count,
        warn_threshold=3000,
        fail_threshold=8000,
    )
    assert (warning.severity if warning else None) == severity
    if warning is not None:
        assert warning.code == WARN_CODE_OVERLONG_BODY
        assert warning.evidence == {
            "meaningful_chars": count,
            "warn_threshold": 3000,
            "fail_threshold": 8000,
        }


def test_template_noise_and_whitespace_do_not_inflate_count() -> None:
    body = "x" * 90 + " " * 4000 + "<!-- " + "y" * 9000 + " -->"
    assert overlong_body_warning(body, warn_threshold=3000, fail_threshold=8000) is None


def test_disabled_limits_are_silent() -> None:
    config = ReviewGateConfig.model_validate(
        {"thresholds": {"warn": {"pr_body_chars": 0}, "fail": {"pr_body_chars": 0}}}
    )
    assert config.thresholds.warn.pr_body_chars == 0
    assert overlong_body_warning("x" * 20000, warn_threshold=0, fail_threshold=0) is None


@pytest.mark.parametrize(
    "yaml_text",
    [
        "thresholds:\\n  warn:\\n    pr_body_chars: -1\\n",
        "thresholds:\\n  warn:\\n    pr_body_chars: 79\\n",
        "thresholds:\\n  warn:\\n    pr_body_chars: 9000\\n  fail:\\n    pr_body_chars: 8000\\n",
        "thresholds:\\n  warn:\\n    pr_body_chars: 0\\n",
    ],
)
def test_invalid_bounds_use_existing_config_recovery(yaml_text: str) -> None:
    outcome = load_config(yaml_text)
    assert [warning.code for warning in outcome.warnings] == ["config_invalid"]
    assert outcome.config.thresholds.warn.pr_body_chars == 3000
    assert outcome.config.thresholds.fail.pr_body_chars == 8000


def test_custom_thresholds_flow_through_engine_and_report() -> None:
    report = analyze(
        EngineInput(
            pr=PRRecord(
                title="Fixes #142",
                body="x" * 121,
                author="octocat",
                base_branch="main",
                head_branch="feature",
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
                )
            ],
            config={
                "thresholds": {
                    "warn": {"pr_body_chars": 100},
                    "fail": {"pr_body_chars": 120},
                }
            },
        )
    )
    warnings = [warning for warning in report.warnings if warning.code == WARN_CODE_OVERLONG_BODY]
    assert len(warnings) == 1
    assert warnings[0].severity == "high"
    assert warnings[0].evidence["meaningful_chars"] == 121
    assert report.reviewability == "WARN"
    assert any(
        warning["code"] == WARN_CODE_OVERLONG_BODY
        for warning in report.model_dump(mode="json")["warnings"]
    )
