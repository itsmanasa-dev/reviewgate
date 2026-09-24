"""Tests for per-file LOC thresholds and per-check exemptions (issue #171)."""

from __future__ import annotations

import pytest

from reviewgate.core.config import ReviewGateConfig, load_config
from reviewgate.core.engine import analyze
from reviewgate.core.schemas import ChangedFile, EngineInput, FileCategoryRow, PRRecord
from reviewgate.core.size import (
    WARN_CODE_FILE_TOO_LARGE,
    WARN_CODE_TOO_LARGE_HUMAN_LOC,
    per_file_loc_warnings,
)


def _row(
    filename: str,
    changes: int,
    *,
    human_authored: bool = True,
) -> FileCategoryRow:
    return FileCategoryRow(
        filename=filename,
        categories=["source"] if human_authored else ["lockfile"],
        risky=False,
        human_authored=human_authored,
        changes=changes,
    )


@pytest.mark.parametrize(
    ("changes", "expected_severity", "expected_threshold"),
    [
        (299, None, None),
        (300, None, None),
        (301, "medium", 300),
        (800, "medium", 300),
        (801, "high", 800),
    ],
)
def test_per_file_boundaries(
    changes: int,
    expected_severity: str | None,
    expected_threshold: int | None,
) -> None:
    warnings = per_file_loc_warnings(
        [_row("src/large.py", changes)],
        warn_per_file_human_loc=300,
        fail_per_file_human_loc=800,
    )
    if expected_severity is None:
        assert warnings == []
        return

    assert len(warnings) == 1
    warning = warnings[0]
    assert warning.code == WARN_CODE_FILE_TOO_LARGE
    assert warning.severity == expected_severity
    assert warning.evidence == {
        "filename": "src/large.py",
        "human_loc_changed": changes,
        "threshold": expected_threshold,
    }


def test_default_zero_limits_do_not_emit_warnings() -> None:
    cfg = ReviewGateConfig()
    assert cfg.thresholds.warn.per_file_human_loc == 0
    assert cfg.thresholds.fail.per_file_human_loc == 0
    assert cfg.thresholds.per_file_loc_exempt_paths == []
    assert (
        per_file_loc_warnings(
            [_row("src/huge.py", 10000)],
            warn_per_file_human_loc=0,
            fail_per_file_human_loc=0,
        )
        == []
    )


def test_individual_tiers_can_be_disabled() -> None:
    big_file = [_row("src/large.py", 801)]
    fail_only = per_file_loc_warnings(
        big_file,
        warn_per_file_human_loc=0,
        fail_per_file_human_loc=800,
    )
    warn_only = per_file_loc_warnings(
        big_file,
        warn_per_file_human_loc=300,
        fail_per_file_human_loc=0,
    )
    assert [w.severity for w in fail_only] == ["high"]
    assert [w.severity for w in warn_only] == ["medium"]


def test_non_human_rows_and_matching_globs_skip_only_this_check() -> None:
    warnings = per_file_loc_warnings(
        [
            _row("src/large.py", 801),
            _row("testdata/fixture.py", 900),
            _row("package-lock.json", 2000, human_authored=False),
            _row("src/second.py", 301),
        ],
        warn_per_file_human_loc=300,
        fail_per_file_human_loc=800,
        exempt_paths=["testdata/**"],
    )
    assert [w.evidence["filename"] for w in warnings] == [
        "src/large.py",
        "src/second.py",
    ]
    assert [w.severity for w in warnings] == ["high", "medium"]


@pytest.mark.parametrize(
    "yaml_text",
    [
        "thresholds:\n  warn:\n    per_file_human_loc: -1\n",
        "thresholds:\n  fail:\n    per_file_human_loc: -2\n",
        "thresholds:\n  warn:\n    per_file_human_loc: 900\n  fail:\n    per_file_human_loc: 300\n",
        "thresholds:\n  per_file_loc_exempt_paths: 42\n",
    ],
)
def test_invalid_config_recovers_with_defaults(yaml_text: str) -> None:
    result = load_config(yaml_text)
    assert [warning.code for warning in result.warnings] == ["config_invalid"]
    assert result.config.thresholds.warn.per_file_human_loc == 0
    assert result.config.thresholds.fail.per_file_human_loc == 0


def _engine_input(
    changes: int,
    *,
    body: str,
    thresholds: dict[str, object],
    filename: str = "README.md",
) -> EngineInput:
    return EngineInput(
        pr=PRRecord(
            title="Fixes #171",
            body=body,
            author="octocat",
            base_branch="main",
            head_branch="feature",
            additions=changes,
            deletions=0,
            changed_files=1,
        ),
        files=[
            ChangedFile(
                filename=filename,
                status="modified",
                additions=changes,
                deletions=0,
                changes=changes,
            )
        ],
        config={"thresholds": thresholds},
    )


@pytest.mark.parametrize(
    ("changes", "expected_severity", "expected_verdict"),
    [
        (301, "medium", "WARN"),
        (801, "high", "FAIL"),
    ],
)
def test_engine_warning_uses_normal_verdict_aggregation(
    changes: int,
    expected_severity: str,
    expected_verdict: str,
) -> None:
    # Short body adds an independent medium weak_pr_body warning.
    # This tests the existing medium+medium and high+medium verdict rules.
    report = analyze(
        _engine_input(
            changes,
            body="Fixes #171.",
            thresholds={
                "warn": {"per_file_human_loc": 300},
                "fail": {"per_file_human_loc": 800},
            },
        )
    )
    warnings = [w for w in report.warnings if w.code == WARN_CODE_FILE_TOO_LARGE]
    assert len(warnings) == 1
    assert warnings[0].severity == expected_severity
    assert report.reviewability == expected_verdict
    assert "too-large" in report.suggested_labels
    assert any(
        warning["code"] == WARN_CODE_FILE_TOO_LARGE
        for warning in report.model_dump(mode="json")["warnings"]
    )


def test_exemption_preserves_aggregate_size_and_file_categories() -> None:
    # The file is exempt only from file_too_large, not aggregate size.
    report = analyze(
        _engine_input(
            450,
            body=(
                "Fixes #171. This change updates documentation and includes "
                "clear review context, implementation notes, and verification."
            ),
            thresholds={
                "warn": {
                    "per_file_human_loc": 300,
                    "human_loc_changed": 300,
                },
                "fail": {
                    "per_file_human_loc": 800,
                    "human_loc_changed": 1000,
                },
                "per_file_loc_exempt_paths": ["README.md"],
            },
        )
    )
    codes = [w.code for w in report.warnings]
    assert WARN_CODE_FILE_TOO_LARGE not in codes
    assert WARN_CODE_TOO_LARGE_HUMAN_LOC in codes
    assert report.stats["human_loc_changed"] == 450
    assert [row.filename for row in report.file_categories] == ["README.md"]


def test_disabled_per_file_thresholds_keep_engine_behavior() -> None:
    report = analyze(
        _engine_input(
            400,
            body=(
                "Fixes #171. This change includes a clear explanation of the "
                "implementation and describes the verification performed."
            ),
            thresholds={},
        )
    )
    assert WARN_CODE_FILE_TOO_LARGE not in [w.code for w in report.warnings]
