"""
Reporting wrapper for post-processing and date pseudonym substitution.

The normal ``post_process_spans`` entrypoint intentionally returns only spans.
This module keeps that behavior intact and adds an opt-in wrapper that returns
the same processed spans plus an audit overview of regex corrections and date
substitution coverage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from date_pseudonyms import pseudonymize_date_text
from post_process import post_process_spans


DATE_SUBSTITUTION_LABELS = {"Date", "Age_Birthdate"}
DOCUMENT_CREATION_DATE_METADATA_KEY = "document_creation_date"


@dataclass(frozen=True)
class PostProcessOverviewResult:
    spans: list[dict]
    overview: dict[str, Any]


class RegexCorrectionCollector:
    def __init__(self, *, max_examples_per_rule: int) -> None:
        self.max_examples_per_rule = max_examples_per_rule
        self._rules: dict[tuple[str, str], dict[str, Any]] = {}

    def record(
        self,
        label: str,
        rule_name: str,
        regex: re.Pattern,
        before: dict | None,
        after: dict | None,
    ) -> None:
        if before is None or after is None:
            return

        key = (label, rule_name)
        rule = self._rules.setdefault(
            key,
            {
                "label": label,
                "rule": rule_name,
                "pattern": compact_pattern(regex),
                "count": 0,
                "examples": [],
            },
        )
        rule["count"] += 1
        if len(rule["examples"]) < self.max_examples_per_rule:
            rule["examples"].append(
                {
                    "before": before.get("text", ""),
                    "after": after.get("text", ""),
                    "begin": after.get("begin"),
                    "end": after.get("end"),
                }
            )

    def overview(self) -> dict[str, Any]:
        rules = sorted(
            self._rules.values(),
            key=lambda rule: (rule["label"], rule["rule"]),
        )
        return {
            "total_corrections": sum(rule["count"] for rule in rules),
            "rules": rules,
        }


def post_process_spans_with_overview(
    spans: list[dict],
    text: str,
    metadata: dict | None = None,
    *,
    date_shift_days: int | None = None,
    max_examples_per_rule: int = 3,
) -> PostProcessOverviewResult:
    """
    Run the standard post-process pipeline and collect a compact audit overview.

    Date substitution is reported for final ``Date`` and ``Age_Birthdate`` spans
    only. The wrapper reports substitution values, but it does not rewrite the
    document text or mutate span text.
    """
    regex_collector = RegexCorrectionCollector(
        max_examples_per_rule=max_examples_per_rule
    )
    processed_spans = post_process_spans(
        spans,
        text,
        metadata=metadata,
        regex_observer=regex_collector.record,
    )
    overview = {
        "regex_corrections": regex_collector.overview(),
        "date_substitutions": build_date_substitution_overview(
            processed_spans,
            text,
            metadata=metadata,
            date_shift_days=date_shift_days,
            max_examples=max_examples_per_rule,
        ),
    }
    return PostProcessOverviewResult(spans=processed_spans, overview=overview)


def build_date_substitution_overview(
    spans: list[dict],
    text: str,
    *,
    metadata: dict | None = None,
    date_shift_days: int | None,
    max_examples: int,
) -> dict[str, Any]:
    date_spans = [span for span in spans if span.get("label") in DATE_SUBSTITUTION_LABELS]
    total = len(date_spans)
    substituted: list[dict[str, Any]] = []
    not_substituted: list[dict[str, Any]] = []
    document_creation_date = extract_document_creation_date(metadata)
    substituted_count = 0

    for span in date_spans:
        source = span.get("text", "")
        label = span.get("label", "")
        substitute = pseudonymize_date_text(
            source,
            label=label,
            date_shift_days=date_shift_days,
            context_before=context_before(text, span),
            context_after=context_after(text, span),
            document_creation_date=document_creation_date,
        )
        example = {
            "label": label,
            "begin": span.get("begin"),
            "end": span.get("end"),
            "source": source,
        }
        if substitute is None:
            if len(not_substituted) < max_examples:
                not_substituted.append(example)
            continue

        substituted_count += 1
        if len(substituted) < max_examples:
            substituted.append({**example, "substitute": substitute})

    not_substituted_count = total - substituted_count
    return {
        "date_shift_days": date_shift_days,
        "total": total,
        "substituted": {
            "count": substituted_count,
            "fraction": fraction(substituted_count, total),
            "examples": substituted,
        },
        "not_substituted": {
            "count": not_substituted_count,
            "fraction": fraction(not_substituted_count, total),
            "examples": not_substituted,
        },
    }


def format_post_process_overview(overview: dict[str, Any]) -> str:
    """Render a human-readable Markdown summary of an overview dict."""
    lines: list[str] = []
    regex_report = overview.get("regex_corrections", {})
    rules = regex_report.get("rules", [])

    lines.append("## Regex corrections")
    if not rules:
        lines.append("No regex-backed corrections were triggered.")
    for rule in rules:
        lines.append(f"- {rule['label']} / {rule['rule']}: {rule['count']}")
        for example in rule.get("examples", []):
            lines.append(
                "  - "
                f"{example['before']!r} -> {example['after']!r} "
                f"({example['begin']}:{example['end']})"
            )

    date_report = overview.get("date_substitutions", {})
    total = date_report.get("total", 0)
    shift = date_report.get("date_shift_days")
    substituted = date_report.get("substituted", {})
    not_substituted = date_report.get("not_substituted", {})

    lines.append("")
    lines.append(f"## Date substitutions (shift={shift})")
    lines.append(
        "- Substituted: "
        f"{substituted.get('count', 0)}/{total} "
        f"({format_percent(substituted.get('fraction', 0.0))})"
    )
    for example in substituted.get("examples", []):
        lines.append(
            "  - "
            f"{example['source']!r} -> {example['substitute']!r} "
            f"[{example['label']}]"
        )
    lines.append(
        "- No substitute: "
        f"{not_substituted.get('count', 0)}/{total} "
        f"({format_percent(not_substituted.get('fraction', 0.0))})"
    )
    for example in not_substituted.get("examples", []):
        lines.append(
            "  - "
            f"{example['source']!r} [{example['label']}] "
            f"({example['begin']}:{example['end']})"
        )

    return "\n".join(lines)


def compact_pattern(regex: re.Pattern) -> str:
    return " ".join(regex.pattern.split())


def fraction(count: int, total: int) -> float:
    return count / total if total else 0.0


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def extract_document_creation_date(metadata: dict | None) -> str | None:
    if not metadata:
        return None
    value = metadata.get(DOCUMENT_CREATION_DATE_METADATA_KEY)
    return str(value) if value else None


def context_before(text: str, span: dict) -> str:
    begin = span.get("begin")
    if not isinstance(begin, int):
        return ""
    return text[max(0, begin - 80) : begin]


def context_after(text: str, span: dict) -> str:
    end = span.get("end")
    if not isinstance(end, int):
        return ""
    return text[end : min(len(text), end + 80)]
