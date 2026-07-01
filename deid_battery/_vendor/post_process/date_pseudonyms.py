from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal


DateGranularity = Literal[
    "day",
    "day_month",
    "month",
    "month_name",
    "month_phase",
    "month_range",
    "season",
    "year",
    "range",
]
REFERENCE_YEAR_FOR_YEARLESS_DATES = 2000

DUTCH_MONTHS_FULL = (
    "januari",
    "februari",
    "maart",
    "april",
    "mei",
    "juni",
    "juli",
    "augustus",
    "september",
    "oktober",
    "november",
    "december",
)
DUTCH_MONTHS_ABBR = (
    "jan",
    "feb",
    "mrt",
    "apr",
    "mei",
    "jun",
    "jul",
    "aug",
    "sep",
    "okt",
    "nov",
    "dec",
)
DUTCH_WEEKDAYS_FULL = (
    "maandag",
    "dinsdag",
    "woensdag",
    "donderdag",
    "vrijdag",
    "zaterdag",
    "zondag",
)
DUTCH_WEEKDAYS_ABBR = ("ma", "di", "wo", "do", "vr", "za", "zo")
DUTCH_SEASONS = ("lente", "zomer", "herfst", "winter")

MONTH_TOKEN_TO_VALUE: dict[str, tuple[int, str]] = {
    token: (idx + 1, "full") for idx, token in enumerate(DUTCH_MONTHS_FULL)
}
MONTH_TOKEN_TO_VALUE.update(
    {token: (idx + 1, "abbr") for idx, token in enumerate(DUTCH_MONTHS_ABBR)}
)

SEASON_INTERVALS = {
    "lente": (3, 1, 5, 31),
    "zomer": (6, 1, 8, 31),
    "herfst": (9, 1, 11, 30),
    "winter": (12, 1, 2, 0),
}

WEEKDAY_PREFIX_RE = re.compile(
    r"^(?P<weekday>maandag|dinsdag|woensdag|donderdag|vrijdag|zaterdag|zondag|"
    r"ma|di|wo|do|vr|za|zo)(?P<sep>\s*,?\s+)(?P<body>.+)$",
    re.IGNORECASE,
)
NUMERIC_DMY_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<sep1>\s*[/-]\s*)(?P<month>\d{1,2})(?P<sep2>\s*[/-]\s*)(?P<year>\d{2}|\d{4})$"
)
NUMERIC_DMY_LOOSE_SHORT_YEAR_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<sep1>\s*[/-]\s*)(?P<month>\d{1,2})(?P<sep2>\s+)(?P<year>\d)$"
)
NUMERIC_DMY_LONG_YEAR_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<sep1>\s*[/-]\s*)(?P<month>\d{1,2})(?P<sep2>\s*[/-]\s*)(?P<year>[12]\d{4})$"
)
NUMERIC_DMY_DOT_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<sep1>\.)(?P<month>\d{1,2})(?P<sep2>\.)(?P<year>\d{2}|\d{4})$"
)
NUMERIC_DAY_MONTH_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<sep>\s*[/-]\s*)(?P<month>\d{1,2})$"
)
NUMERIC_DAY_YEAR_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<sep>\s*/\s*)(?P<year>\d{2,4})$"
)
NUMERIC_SHARED_MONTH_RANGE_RE = re.compile(
    r"^(?P<start_day>\d{1,2})(?P<range_sep>\s*[-–—]\s*)"
    r"(?P<end_day>\d{1,2})(?P<date_sep>[/-])(?P<month>\d{1,2})$"
)
NUMERIC_YMD_RE = re.compile(
    r"^(?P<year>\d{4})(?P<sep1>[/-])(?P<month>\d{1,2})(?P<sep2>[/-])(?P<day>\d{1,2})$"
)
NUMERIC_DMY_RANGE_RE = re.compile(
    r"^(?P<start_day>\d{1,2})(?P<date_sep>[/.])(?P<start_month>\d{1,2})"
    r"(?:(?P=date_sep)(?P<start_year>\d{2}|\d{4}))?"
    r"(?P<range_sep>\s*[-–—]\s*)"
    r"(?P<end_day>\d{1,2})(?P=date_sep)(?P<end_month>\d{1,2})"
    r"(?:(?P=date_sep)(?P<end_year>\d{2}|\d{4}))?$"
)
TEXTUAL_DMY_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<sep1>\s+|\s*[-/]\s*)(?P<month>[A-Za-zÀ-ÿ]+)"
    r"(?P<dot>\.?)(?P<sep2>\s+|\s*[-/]\s*)(?P<year>\d{2}|\d{4})$",
    re.IGNORECASE,
)
TEXTUAL_DAY_MONTH_RE = re.compile(
    r"^(?P<day>\d{1,2})(?P<sep>\s+|\s*[-/]\s*)(?P<month>[A-Za-zÀ-ÿ]+)"
    r"(?P<dot>\.?)$",
    re.IGNORECASE,
)
TEXTUAL_SHARED_MONTH_RANGE_RE = re.compile(
    r"^(?P<start_day>\d{1,2})(?P<range_sep>\s*[-–—]\s*)"
    r"(?P<end_day>\d{1,2})(?P<sep1>\s+)(?P<month>[A-Za-zÀ-ÿ]+)"
    r"(?P<dot>\.?)(?:(?P<sep2>\s+)(?P<year>\d{2}|\d{4}))?$",
    re.IGNORECASE,
)
TEXTUAL_MONTH_YEAR_RE = re.compile(
    r"^(?P<month>[A-Za-zÀ-ÿ]+)(?P<dot>\.?)(?P<sep>\s+)(?P<year>\d{4})$",
    re.IGNORECASE,
)
TEXTUAL_MONTH_RE = re.compile(
    r"^(?P<month>[A-Za-zÀ-ÿ]+)(?P<dot>\.?)$",
    re.IGNORECASE,
)
MONTH_PHASE_RE = re.compile(
    r"^(?P<phase>begin|midden|half|eind)(?P<phase_tail>(?:/(?:begin|midden|half|eind))*)"
    r"(?P<sep>\s+)(?P<month>[A-Za-zÀ-ÿ]+)(?P<dot>\.?)$",
    re.IGNORECASE,
)
TEXTUAL_MONTH_YEAR_RANGE_RE = re.compile(
    r"^(?P<start_month>[A-Za-zÀ-ÿ]+)(?P<start_dot>\.?)"
    r"(?P<start_sep>\s+)(?P<start_year>\d{4})"
    r"(?P<range_sep>\s*[-–—]\s*)"
    r"(?P<end_month>[A-Za-zÀ-ÿ]+)(?P<end_dot>\.?)"
    r"(?P<end_sep>\s+)(?P<end_year>\d{4})$",
    re.IGNORECASE,
)
NUMERIC_MONTH_YEAR_RE = re.compile(
    r"^(?P<month>\d{1,2})(?P<sep>[/-])(?P<year>\d{4})$"
)
NUMERIC_MONTH_YEAR_RANGE_RE = re.compile(
    r"^(?P<start_month>\d{1,2})(?P<start_sep>[/-])(?P<start_year>\d{4})"
    r"(?P<range_sep>\s*[-–—]\s*)"
    r"(?P<end_month>\d{1,2})(?P<end_sep>[/-])(?P<end_year>\d{4})$"
)
SEASON_YEAR_RE = re.compile(
    r"^(?P<season>lente|zomer|herfst|winter)(?P<sep>\s+)(?P<year>\d{4})$",
    re.IGNORECASE,
)
YEAR_ONLY_RE = re.compile(r"^(?P<year>\d{4})$")
APPROX_YEAR_RE = re.compile(
    r"^(?P<prefix>rond|circa|ca\.?|ongeveer)(?P<sep>\s+)(?P<year>\d{4})$",
    re.IGNORECASE,
)
TRAILING_YEAR_RE = re.compile(r"(?<!\d)(?P<year>[12]\d{3})\s*$")
AGE_GRANULAR_RE = re.compile(
    r"^\s*\d{1,3}(?:[.,]\d+)?\s*(?:-|–|—)?\s*"
    r"(?:jaar|jaren|jarige|jarig|jr|j|year|years|yr|yrs|"
    r"maand|maanden|mnd|month|months|mo|mos|m|"
    r"week|weken|wk|w|dag|dagen|day|days|d)\b"
    r"(?:\s+(?:jongere|ouder))?\s*$",
    re.IGNORECASE,
)
AGE_COMPOSITE_RE = re.compile(
    r"^\s*"
    r"\d{1,3}(?:[.,]\d+)?\s*"
    r"(?:jaar|jaren|year|years|yr|yrs|jr|j|maand|maanden|month|months|mo|mos|m|dag|dagen|day|days|d)\b"
    r"(?:\s*,?\s+"
    r"\d{1,3}(?:[.,]\d+)?\s*"
    r"(?:jaar|jaren|year|years|yr|yrs|jr|j|maand|maanden|month|months|mo|mos|m|dag|dagen|day|days|d)\b"
    r")+\s*$",
    re.IGNORECASE,
)
AGE_STANDALONE_RE = re.compile(r"^\s*(?P<age>\d{1,3})(?:[.,]\d+)?\s*$")
AGE_BIRTHDATE_CONTEXT_RE = re.compile(
    r"(?:"
    r"\b(?:birth|birty|sample|geboorte)[\w\s:._/-]{0,24}?"
    r"(?:year|month|day|jaar|maand|dag)\b"
    r"|"
    r"\b(?:leeftijd|age|jaren|maanden|dagen|jaar|maand|dag|"
    r"years|months|days|year|month|day)\b"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedDate:
    start: date
    end: date
    granularity: DateGranularity
    style: dict[str, str | int | bool]
    prefix: tuple[str, str] | None = None


def pseudonymize_date_text(
    text: str,
    *,
    label: str,
    date_shift_days: int | None,
    context_before: str = "",
    context_after: str = "",
    document_creation_date: str | None = None,
) -> str | None:
    if date_shift_days is None:
        return None

    document_date = parse_document_creation_date(document_creation_date)

    if label == "Age_Birthdate":
        if AGE_GRANULAR_RE.fullmatch(text) or AGE_COMPOSITE_RE.fullmatch(text):
            return text
        standalone_substitute = pseudonymize_standalone_age_birthdate(
            text,
            date_shift_days=date_shift_days,
            context_before=context_before,
            context_after=context_after,
            document_date=document_date,
        )
        if standalone_substitute is not None:
            return standalone_substitute
    elif label == "Date":
        standalone_substitute = pseudonymize_standalone_date_component(
            text,
            date_shift_days=date_shift_days,
            context_before=context_before,
            context_after=context_after,
            document_date=document_date,
        )
        if standalone_substitute is not None:
            return standalone_substitute

    try:
        parsed = parse_date_text(text, document_date=document_date)
    except ValueError:
        return None

    shifted_start = parsed.start + timedelta(days=date_shift_days)
    shifted_end = parsed.end + timedelta(days=date_shift_days)

    if label == "Age_Birthdate":
        return render_year_interval(shifted_start.year, shifted_end.year)
    if label == "Date":
        return render_shifted_date(parsed, shifted_start, shifted_end, text)
    return None


def parse_date_text(text: str, *, document_date: date | None = None) -> ParsedDate:
    leading, body, trailing = split_outer_whitespace(text)
    prefix: tuple[str, str] | None = None
    prefix_match = WEEKDAY_PREFIX_RE.fullmatch(body)
    if prefix_match:
        prefix = (prefix_match.group("weekday"), prefix_match.group("sep"))
        body = prefix_match.group("body")

    parsed = parse_date_body(body, document_date=document_date)
    if leading or trailing:
        style = dict(parsed.style)
        style["leading_ws"] = leading
        style["trailing_ws"] = trailing
        return ParsedDate(
            start=parsed.start,
            end=parsed.end,
            granularity=parsed.granularity,
            style=style,
            prefix=prefix or parsed.prefix,
        )
    if prefix:
        return ParsedDate(
            start=parsed.start,
            end=parsed.end,
            granularity=parsed.granularity,
            style=parsed.style,
            prefix=prefix,
        )
    return parsed


def parse_date_body(body: str, *, document_date: date | None = None) -> ParsedDate:
    numeric_month_year_range = NUMERIC_MONTH_YEAR_RANGE_RE.fullmatch(body)
    if numeric_month_year_range:
        start_month = int(numeric_month_year_range.group("start_month"))
        end_month = int(numeric_month_year_range.group("end_month"))
        if not 1 <= start_month <= 12 or not 1 <= end_month <= 12:
            raise ValueError("invalid numeric month-year range")
        start_year = int(numeric_month_year_range.group("start_year"))
        end_year = int(numeric_month_year_range.group("end_year"))
        start_date = date(start_year, start_month, 1)
        end_date = end_of_month(end_year, end_month)
        if end_date < start_date:
            raise ValueError("month-year range ends before it starts")
        return ParsedDate(
            start=start_date,
            end=end_date,
            granularity="month_range",
            style={
                "kind": "numeric_month_year_range",
                "start_month_width": len(numeric_month_year_range.group("start_month")),
                "end_month_width": len(numeric_month_year_range.group("end_month")),
                "start_sep": numeric_month_year_range.group("start_sep"),
                "end_sep": numeric_month_year_range.group("end_sep"),
                "range_sep": numeric_month_year_range.group("range_sep"),
            },
        )

    textual_month_year_range = TEXTUAL_MONTH_YEAR_RANGE_RE.fullmatch(body)
    if textual_month_year_range:
        try:
            start_month, start_month_style = parse_month_token(
                textual_month_year_range.group("start_month")
            )
            end_month, end_month_style = parse_month_token(
                textual_month_year_range.group("end_month")
            )
        except ValueError as exc:
            raise ValueError("unsupported month-year range month") from exc
        start_year = int(textual_month_year_range.group("start_year"))
        end_year = int(textual_month_year_range.group("end_year"))
        start_date = date(start_year, start_month, 1)
        end_date = end_of_month(end_year, end_month)
        if end_date < start_date:
            raise ValueError("month-year range ends before it starts")
        return ParsedDate(
            start=start_date,
            end=end_date,
            granularity="month_range",
            style={
                "kind": "textual_month_year_range",
                "start_month_style": start_month_style,
                "end_month_style": end_month_style,
                "start_month_source": textual_month_year_range.group("start_month"),
                "end_month_source": textual_month_year_range.group("end_month"),
                "start_dot": bool(textual_month_year_range.group("start_dot")),
                "end_dot": bool(textual_month_year_range.group("end_dot")),
                "start_sep": textual_month_year_range.group("start_sep"),
                "end_sep": textual_month_year_range.group("end_sep"),
                "range_sep": textual_month_year_range.group("range_sep"),
            },
        )

    numeric_range = NUMERIC_DMY_RANGE_RE.fullmatch(body)
    if numeric_range:
        start_year_token = numeric_range.group("start_year")
        end_year_token = numeric_range.group("end_year")
        if bool(start_year_token) != bool(end_year_token):
            raise ValueError("partial year range")
        has_year = start_year_token is not None
        start_year = (
            expand_year(start_year_token, document_date=document_date)
            if start_year_token
            else REFERENCE_YEAR_FOR_YEARLESS_DATES
        )
        end_year = (
            expand_year(end_year_token, document_date=document_date)
            if end_year_token
            else REFERENCE_YEAR_FOR_YEARLESS_DATES
        )
        try:
            start_date = date(
                start_year,
                int(numeric_range.group("start_month")),
                int(numeric_range.group("start_day")),
            )
            end_date = date(
                end_year,
                int(numeric_range.group("end_month")),
                int(numeric_range.group("end_day")),
            )
        except ValueError as exc:
            raise ValueError("invalid numeric range") from exc
        if end_date < start_date and not has_year:
            end_date = date(
                end_date.year + 1,
                end_date.month,
                end_date.day,
            )
        if end_date < start_date:
            raise ValueError("date range ends before it starts")
        return ParsedDate(
            start=start_date,
            end=end_date,
            granularity="range",
            style={
                "kind": "numeric_range",
                "has_year": has_year,
                "start_day_width": range_component_width(
                    numeric_range.group("start_day")
                ),
                "end_day_width": range_component_width(
                    numeric_range.group("end_day")
                ),
                "start_month_width": range_component_width(
                    numeric_range.group("start_month")
                ),
                "end_month_width": range_component_width(
                    numeric_range.group("end_month")
                ),
                "year_width": len(start_year_token or ""),
                "date_sep": numeric_range.group("date_sep"),
                "range_sep": numeric_range.group("range_sep"),
            },
        )

    exact_numeric = parse_exact_numeric_date_body(body, document_date=document_date)
    if exact_numeric:
        return exact_numeric

    numeric_shared_month_range = NUMERIC_SHARED_MONTH_RANGE_RE.fullmatch(body)
    if numeric_shared_month_range:
        year = REFERENCE_YEAR_FOR_YEARLESS_DATES
        month_value = int(numeric_shared_month_range.group("month"))
        try:
            start_date = date(
                year,
                month_value,
                int(numeric_shared_month_range.group("start_day")),
            )
            end_date = date(
                year,
                month_value,
                int(numeric_shared_month_range.group("end_day")),
            )
        except ValueError as exc:
            raise ValueError("invalid numeric shared-month range") from exc
        if end_date < start_date:
            raise ValueError("date range ends before it starts")
        return ParsedDate(
            start=start_date,
            end=end_date,
            granularity="range",
            style={
                "kind": "numeric_shared_month_range",
                "has_year": False,
                "start_day_width": range_component_width(
                    numeric_shared_month_range.group("start_day")
                ),
                "end_day_width": range_component_width(
                    numeric_shared_month_range.group("end_day")
                ),
                "month_width": range_component_width(
                    numeric_shared_month_range.group("month")
                ),
                "range_sep": numeric_shared_month_range.group("range_sep"),
                "date_sep": numeric_shared_month_range.group("date_sep"),
            },
        )

    textual_range = TEXTUAL_SHARED_MONTH_RANGE_RE.fullmatch(body)
    if textual_range:
        try:
            month_value, month_style = parse_month_token(textual_range.group("month"))
        except ValueError as exc:
            raise ValueError("unsupported range month") from exc
        year_token = textual_range.group("year")
        year = (
            expand_year(year_token, document_date=document_date)
            if year_token
            else REFERENCE_YEAR_FOR_YEARLESS_DATES
        )
        try:
            start_date = date(year, month_value, int(textual_range.group("start_day")))
            end_date = date(year, month_value, int(textual_range.group("end_day")))
        except ValueError as exc:
            raise ValueError("invalid textual range") from exc
        if end_date < start_date:
            raise ValueError("date range ends before it starts")
        return ParsedDate(
            start=start_date,
            end=end_date,
            granularity="range",
            style={
                "kind": "textual_shared_month_range",
                "has_year": year_token is not None,
                "start_day_width": 1,
                "end_day_width": 1,
                "year_width": len(year_token or ""),
                "range_sep": textual_range.group("range_sep"),
                "sep1": textual_range.group("sep1"),
                "sep2": textual_range.group("sep2") or "",
                "month_style": month_style,
                "month_source": textual_range.group("month"),
                "month_dot": bool(textual_range.group("dot")),
            },
        )

    textual = TEXTUAL_DMY_RE.fullmatch(body)
    if textual:
        try:
            month_value, month_style = parse_month_token(textual.group("month"))
        except ValueError:
            month_value = None
            month_style = ""
        if month_value is None:
            textual = None
        else:
            year_token = textual.group("year")
            try:
                parsed_date = date(
                    expand_year(year_token, document_date=document_date),
                    month_value,
                    int(textual.group("day")),
                )
            except ValueError as exc:
                raise ValueError("invalid textual date") from exc
            return ParsedDate(
                start=parsed_date,
                end=parsed_date,
                granularity="day",
                style={
                    "kind": "textual",
                    "order": "dmy",
                    "day_width": 1,
                    "year_width": len(year_token),
                    "sep1": textual.group("sep1"),
                    "sep2": textual.group("sep2"),
                    "month_style": month_style,
                    "month_source": textual.group("month"),
                    "month_dot": bool(textual.group("dot")),
                },
            )

    textual_day_month = TEXTUAL_DAY_MONTH_RE.fullmatch(body)
    if textual_day_month:
        try:
            month_value, month_style = parse_month_token(
                textual_day_month.group("month")
            )
        except ValueError:
            month_value = None
            month_style = ""
        if month_value is not None:
            try:
                parsed_date = date(
                    REFERENCE_YEAR_FOR_YEARLESS_DATES,
                    month_value,
                    int(textual_day_month.group("day")),
                )
            except ValueError as exc:
                raise ValueError("invalid textual day-month date") from exc
            return ParsedDate(
                start=parsed_date,
                end=parsed_date,
                granularity="day_month",
                style={
                    "kind": "textual_day_month",
                    "day_width": range_component_width(textual_day_month.group("day")),
                    "sep": textual_day_month.group("sep"),
                    "month_style": month_style,
                    "month_source": textual_day_month.group("month"),
                    "month_dot": bool(textual_day_month.group("dot")),
                },
            )

    numeric_month_year = NUMERIC_MONTH_YEAR_RE.fullmatch(body)
    if numeric_month_year:
        month_value = int(numeric_month_year.group("month"))
        if not 1 <= month_value <= 12:
            raise ValueError("invalid month")
        year = int(numeric_month_year.group("year"))
        return ParsedDate(
            start=date(year, month_value, 1),
            end=end_of_month(year, month_value),
            granularity="month",
            style={
                "kind": "numeric_month_year",
                "month_width": len(numeric_month_year.group("month")),
                "sep": numeric_month_year.group("sep"),
            },
        )

    numeric_day_month = NUMERIC_DAY_MONTH_RE.fullmatch(body)
    if numeric_day_month:
        day_token = numeric_day_month.group("day")
        month_token = numeric_day_month.group("month")
        try:
            parsed_date = date(
                REFERENCE_YEAR_FOR_YEARLESS_DATES,
                int(month_token),
                int(day_token),
            )
        except ValueError:
            parsed_date = None
        if parsed_date is not None:
            return ParsedDate(
                start=parsed_date,
                end=parsed_date,
                granularity="day_month",
                style={
                    "kind": "numeric_day_month",
                    "day_width": range_component_width(day_token),
                    "month_width": range_component_width(month_token),
                    "sep": numeric_day_month.group("sep"),
                },
            )

    numeric_day_year = NUMERIC_DAY_YEAR_RE.fullmatch(body)
    if numeric_day_year:
        day_token = numeric_day_year.group("day")
        year_token = numeric_day_year.group("year")
        year = expand_year(year_token, document_date=document_date)
        try:
            parsed_date = date(year, 1, int(day_token))
        except ValueError as exc:
            raise ValueError("invalid day-year date") from exc
        return ParsedDate(
            start=parsed_date,
            end=parsed_date,
            granularity="day",
            style={
                "kind": "numeric_day_year",
                "order": "dy",
                "day_width": range_component_width(day_token),
                "year_width": len(year_token),
                "sep1": numeric_day_year.group("sep"),
                "sep2": "",
            },
        )

    month_year = TEXTUAL_MONTH_YEAR_RE.fullmatch(body)
    if month_year:
        try:
            month_value, month_style = parse_month_token(month_year.group("month"))
        except ValueError:
            month_value = None
            month_style = ""
        if month_value is not None:
            year = int(month_year.group("year"))
            return ParsedDate(
                start=date(year, month_value, 1),
                end=end_of_month(year, month_value),
                granularity="month",
                style={
                    "kind": "textual_month_year",
                    "month_style": month_style,
                    "month_source": month_year.group("month"),
                    "month_dot": bool(month_year.group("dot")),
                    "sep": month_year.group("sep"),
                },
            )

    month_phase = MONTH_PHASE_RE.fullmatch(body)
    if month_phase:
        try:
            month_value, month_style = parse_month_token(month_phase.group("month"))
        except ValueError:
            month_value = None
            month_style = ""
        if month_value is not None:
            phases = month_phase_tokens(
                month_phase.group("phase"),
                month_phase.group("phase_tail"),
            )
            start_day, end_day = month_phase_day_bounds(
                phases,
                REFERENCE_YEAR_FOR_YEARLESS_DATES,
                month_value,
            )
            return ParsedDate(
                start=date(REFERENCE_YEAR_FOR_YEARLESS_DATES, month_value, start_day),
                end=date(REFERENCE_YEAR_FOR_YEARLESS_DATES, month_value, end_day),
                granularity="month_phase",
                style={
                    "kind": "month_phase",
                    "phases": "/".join(phases),
                    "sep": month_phase.group("sep"),
                    "month_style": month_style,
                    "month_source": month_phase.group("month"),
                    "month_dot": bool(month_phase.group("dot")),
                },
            )

    textual_month = TEXTUAL_MONTH_RE.fullmatch(body)
    if textual_month:
        try:
            month_value, month_style = parse_month_token(textual_month.group("month"))
        except ValueError:
            month_value = None
            month_style = ""
        if month_value is not None:
            return ParsedDate(
                start=date(REFERENCE_YEAR_FOR_YEARLESS_DATES, month_value, 1),
                end=end_of_month(REFERENCE_YEAR_FOR_YEARLESS_DATES, month_value),
                granularity="month_name",
                style={
                    "kind": "textual_month",
                    "month_style": month_style,
                    "month_source": textual_month.group("month"),
                    "month_dot": bool(textual_month.group("dot")),
                },
            )

    season_year = SEASON_YEAR_RE.fullmatch(body)
    if season_year:
        season = season_year.group("season")
        year = int(season_year.group("year"))
        start, end = season_interval(normalize_token(season), year)
        return ParsedDate(
            start=start,
            end=end,
            granularity="season",
            style={
                "kind": "season_year",
                "season_source": season,
                "sep": season_year.group("sep"),
            },
        )

    approx_year = APPROX_YEAR_RE.fullmatch(body)
    if approx_year:
        year = int(approx_year.group("year"))
        return ParsedDate(
            start=date(year, 1, 1),
            end=date(year, 12, 31),
            granularity="year",
            style={
                "kind": "year",
                "prefix": approx_year.group("prefix"),
                "prefix_sep": approx_year.group("sep"),
            },
        )

    year_only = YEAR_ONLY_RE.fullmatch(body)
    if year_only:
        year = int(year_only.group("year"))
        return ParsedDate(
            start=date(year, 1, 1),
            end=date(year, 12, 31),
            granularity="year",
            style={"kind": "year"},
        )

    trailing_year = TRAILING_YEAR_RE.search(body)
    if trailing_year:
        year = int(trailing_year.group("year"))
        return ParsedDate(
            start=date(year, 1, 1),
            end=date(year, 12, 31),
            granularity="year",
            style={"kind": "year"},
        )

    raise ValueError("unsupported date")


def parse_exact_numeric_date_body(
    body: str,
    *,
    document_date: date | None,
) -> ParsedDate | None:
    for regex, order in (
        (NUMERIC_DMY_LONG_YEAR_RE, "dmy"),
        (NUMERIC_DMY_RE, "dmy"),
        (NUMERIC_DMY_DOT_RE, "dmy"),
        (NUMERIC_DMY_LOOSE_SHORT_YEAR_RE, "dmy"),
        (NUMERIC_YMD_RE, "ymd"),
    ):
        match = regex.fullmatch(body)
        if not match:
            continue

        day_token = match.group("day")
        month_token = match.group("month")
        year_token = match.group("year")
        normalized_year_token = normalize_year_token(year_token)
        year = expand_year(normalized_year_token, document_date=document_date)
        try:
            parsed_date = date(year, int(month_token), int(day_token))
        except ValueError:
            continue
        return ParsedDate(
            start=parsed_date,
            end=parsed_date,
            granularity="day",
            style={
                "kind": "numeric",
                "order": order,
                "day_width": len(day_token),
                "month_width": len(month_token),
                "year_width": len(normalized_year_token),
                "sep1": match.group("sep1"),
                "sep2": match.group("sep2"),
            },
        )
    return None


def render_shifted_date(
    parsed: ParsedDate,
    shifted_start: date,
    shifted_end: date,
    original_text: str,
) -> str:
    if parsed.granularity == "day":
        rendered = render_exact_date(shifted_start, parsed)
    elif parsed.granularity == "day_month":
        rendered = render_day_month(shifted_start, parsed.style)
    elif parsed.granularity == "month":
        rendered = render_month_interval(shifted_start, shifted_end, parsed.style)
    elif parsed.granularity == "month_name":
        rendered = render_month_name_interval(
            shifted_start,
            shifted_end,
            parsed.style,
        )
    elif parsed.granularity == "month_phase":
        rendered = render_month_phase_interval(
            shifted_start,
            shifted_end,
            parsed.style,
        )
    elif parsed.granularity == "month_range":
        rendered = render_month_range(shifted_start, shifted_end, parsed.style)
    elif parsed.granularity == "season":
        rendered = render_season_interval(shifted_start, shifted_end, parsed.style)
    elif parsed.granularity == "year":
        rendered = render_year_like_interval(
            shifted_start.year,
            shifted_end.year,
            parsed.style,
        )
    elif parsed.granularity == "range":
        rendered = render_date_range(shifted_start, shifted_end, parsed.style)
    else:
        raise ValueError("unsupported granularity")

    leading = str(parsed.style.get("leading_ws", ""))
    trailing = str(parsed.style.get("trailing_ws", ""))
    return f"{leading}{rendered}{trailing}" if leading or trailing else rendered


def render_exact_date(shifted: date, parsed: ParsedDate) -> str:
    style = parsed.style
    year_width = int(style["year_width"])
    day = render_int(shifted.day, int(style["day_width"]))
    year = render_year(shifted.year, year_width)

    if style["kind"] == "numeric":
        month = render_int(shifted.month, int(style["month_width"]))
        if style["order"] == "ymd":
            body = f"{year}{style['sep1']}{month}{style['sep2']}{day}"
        else:
            body = f"{day}{style['sep1']}{month}{style['sep2']}{year}"
    elif style["kind"] == "numeric_day_year":
        body = f"{day}{style['sep1']}{year}"
    else:
        month = render_month(
            shifted.month,
            str(style["month_style"]),
            str(style["month_source"]),
        )
        if style.get("month_dot"):
            month = f"{month}."
        body = f"{day}{style['sep1']}{month}{style['sep2']}{year}"

    if parsed.prefix:
        weekday_source, separator = parsed.prefix
        weekday = render_weekday(shifted.weekday(), weekday_source)
        return f"{weekday}{separator}{body}"
    return body


def render_day_month(shifted: date, style: dict[str, str | int | bool]) -> str:
    day = render_int(shifted.day, int(style["day_width"]))
    if style["kind"] == "textual_day_month":
        month = render_month(
            shifted.month,
            str(style["month_style"]),
            str(style["month_source"]),
        )
        if style.get("month_dot"):
            month = f"{month}."
        return f"{day}{style['sep']}{month}"

    month = render_int(shifted.month, int(style["month_width"]))
    return f"{day}{style['sep']}{month}"


def render_month_interval(
    shifted_start: date,
    shifted_end: date,
    style: dict[str, str | int | bool],
) -> str:
    start_value = render_month_year(shifted_start.year, shifted_start.month, style)
    end_value = render_month_year(shifted_end.year, shifted_end.month, style)
    if start_value == end_value:
        return start_value

    if style["kind"] == "textual_month_year" and shifted_start.year == shifted_end.year:
        start_month = render_month(
            shifted_start.month,
            str(style["month_style"]),
            str(style["month_source"]),
        )
        end_month = render_month(
            shifted_end.month,
            str(style["month_style"]),
            str(style["month_source"]),
        )
        if style.get("month_dot"):
            start_month = f"{start_month}."
            end_month = f"{end_month}."
        return f"{start_month}/{end_month}{style['sep']}{shifted_start.year:04d}"

    if style["kind"] == "numeric_month_year":
        return f"{start_value}-{end_value}"
    return f"{start_value}/{end_value}"


def render_month_name_interval(
    shifted_start: date,
    shifted_end: date,
    style: dict[str, str | int | bool],
) -> str:
    start_month = render_month(
        shifted_start.month,
        str(style["month_style"]),
        str(style["month_source"]),
    )
    end_month = render_month(
        shifted_end.month,
        str(style["month_style"]),
        str(style["month_source"]),
    )
    if style.get("month_dot"):
        start_month = f"{start_month}."
        end_month = f"{end_month}."
    if start_month == end_month:
        return start_month
    return f"{start_month}/{end_month}"


def render_month_phase_interval(
    shifted_start: date,
    shifted_end: date,
    style: dict[str, str | int | bool],
) -> str:
    start_phase = phase_for_day(shifted_start.day)
    end_phase = phase_for_day(shifted_end.day)
    start_month = render_month(
        shifted_start.month,
        str(style["month_style"]),
        str(style["month_source"]),
    )
    end_month = render_month(
        shifted_end.month,
        str(style["month_style"]),
        str(style["month_source"]),
    )
    if style.get("month_dot"):
        start_month = f"{start_month}."
        end_month = f"{end_month}."

    if shifted_start.year == shifted_end.year and shifted_start.month == shifted_end.month:
        if start_phase == end_phase:
            phase_text = match_case(start_phase, str(style["phases"]))
        else:
            phase_text = match_case(f"{start_phase}/{end_phase}", str(style["phases"]))
        return f"{phase_text}{style['sep']}{start_month}"

    start_text = (
        f"{match_case(start_phase, str(style['phases']))}{style['sep']}{start_month}"
    )
    end_text = f"{match_case(end_phase, str(style['phases']))}{style['sep']}{end_month}"
    return f"{start_text}/{end_text}"


def month_phase_tokens(phase: str, phase_tail: str) -> list[str]:
    return [phase.casefold(), *[part for part in phase_tail.casefold().split("/") if part]]


def month_phase_day_bounds(phases: list[str], year: int, month: int) -> tuple[int, int]:
    return phase_start_day(phases[0]), phase_end_day(phases[-1], year, month)


def phase_start_day(phase: str) -> int:
    if phase == "begin":
        return 1
    if phase in {"midden", "half"}:
        return 11
    return 21


def phase_end_day(phase: str, year: int, month: int) -> int:
    if phase == "begin":
        return 10
    if phase in {"midden", "half"}:
        return 20
    return calendar.monthrange(year, month)[1]


def phase_for_day(day: int) -> str:
    if day <= 10:
        return "begin"
    if day <= 20:
        return "midden"
    return "eind"


def render_month_range(
    shifted_start: date,
    shifted_end: date,
    style: dict[str, str | int | bool],
) -> str:
    if style["kind"] == "numeric_month_year_range":
        start_value = render_numeric_month_year_range_part(
            shifted_start.year,
            shifted_start.month,
            int(style["start_month_width"]),
            str(style["start_sep"]),
        )
        end_value = render_numeric_month_year_range_part(
            shifted_end.year,
            shifted_end.month,
            int(style["end_month_width"]),
            str(style["end_sep"]),
        )
        return f"{start_value}{style['range_sep']}{end_value}"

    start_value = render_textual_month_year_range_part(
        shifted_start.year,
        shifted_start.month,
        str(style["start_month_style"]),
        str(style["start_month_source"]),
        bool(style["start_dot"]),
        str(style["start_sep"]),
    )
    end_value = render_textual_month_year_range_part(
        shifted_end.year,
        shifted_end.month,
        str(style["end_month_style"]),
        str(style["end_month_source"]),
        bool(style["end_dot"]),
        str(style["end_sep"]),
    )
    return f"{start_value}{style['range_sep']}{end_value}"


def render_numeric_month_year_range_part(
    year: int,
    month: int,
    month_width: int,
    separator: str,
) -> str:
    return f"{render_int(month, month_width)}{separator}{year:04d}"


def render_textual_month_year_range_part(
    year: int,
    month: int,
    month_style: str,
    month_source: str,
    month_dot: bool,
    separator: str,
) -> str:
    month_text = render_month(month, month_style, month_source)
    if month_dot:
        month_text = f"{month_text}."
    return f"{month_text}{separator}{year:04d}"


def render_date_range(
    shifted_start: date,
    shifted_end: date,
    style: dict[str, str | int | bool],
) -> str:
    if style["kind"] == "numeric_range":
        start_value = render_numeric_range_date(
            shifted_start,
            style,
            day_width_key="start_day_width",
            month_width_key="start_month_width",
        )
        end_value = render_numeric_range_date(
            shifted_end,
            style,
            day_width_key="end_day_width",
            month_width_key="end_month_width",
        )
        return f"{start_value}{style['range_sep']}{end_value}"

    if style["kind"] == "numeric_shared_month_range":
        return render_numeric_shared_month_range(shifted_start, shifted_end, style)

    return render_textual_shared_month_range(shifted_start, shifted_end, style)


def render_numeric_shared_month_range(
    shifted_start: date,
    shifted_end: date,
    style: dict[str, str | int | bool],
) -> str:
    start_day = render_int(shifted_start.day, int(style["start_day_width"]))
    end_day = render_int(shifted_end.day, int(style["end_day_width"]))
    start_month = render_int(shifted_start.month, int(style["month_width"]))
    end_month = render_int(shifted_end.month, int(style["month_width"]))
    if shifted_start.month == shifted_end.month:
        return f"{start_day}{style['range_sep']}{end_day}{style['date_sep']}{end_month}"
    return (
        f"{start_day}{style['date_sep']}{start_month}"
        f"{style['range_sep']}{end_day}{style['date_sep']}{end_month}"
    )


def render_numeric_range_date(
    value: date,
    style: dict[str, str | int | bool],
    *,
    day_width_key: str,
    month_width_key: str,
) -> str:
    day = render_int(value.day, int(style[day_width_key]))
    month = render_int(value.month, int(style[month_width_key]))
    if not style.get("has_year"):
        return f"{day}{style['date_sep']}{month}"
    year = render_year(value.year, int(style["year_width"]))
    return f"{day}{style['date_sep']}{month}{style['date_sep']}{year}"


def render_textual_shared_month_range(
    shifted_start: date,
    shifted_end: date,
    style: dict[str, str | int | bool],
) -> str:
    start_day = render_int(shifted_start.day, int(style["start_day_width"]))
    end_day = render_int(shifted_end.day, int(style["end_day_width"]))
    start_month = render_month(
        shifted_start.month,
        str(style["month_style"]),
        str(style["month_source"]),
    )
    end_month = render_month(
        shifted_end.month,
        str(style["month_style"]),
        str(style["month_source"]),
    )
    if style.get("month_dot"):
        start_month = f"{start_month}."
        end_month = f"{end_month}."

    same_month = shifted_start.year == shifted_end.year and shifted_start.month == shifted_end.month
    if same_month:
        body = (
            f"{start_day}{style['range_sep']}{end_day}"
            f"{style['sep1']}{start_month}"
        )
    else:
        body = (
            f"{start_day}{style['sep1']}{start_month}"
            f"{style['range_sep']}{end_day}{style['sep1']}{end_month}"
        )

    if not style.get("has_year"):
        return body

    if same_month or shifted_start.year == shifted_end.year:
        return f"{body}{style['sep2']}{shifted_end.year:04d}"

    start_year = render_year(shifted_start.year, int(style["year_width"]))
    end_year = render_year(shifted_end.year, int(style["year_width"]))
    return (
        f"{start_day}{style['sep1']}{start_month}{style['sep2']}{start_year}"
        f"{style['range_sep']}{end_day}{style['sep1']}{end_month}{style['sep2']}{end_year}"
    )


def render_month_year(year: int, month: int, style: dict[str, str | int | bool]) -> str:
    if style["kind"] == "numeric_month_year":
        month_text = render_int(month, int(style["month_width"]))
        return f"{month_text}{style['sep']}{year:04d}"

    month_text = render_month(
        month,
        str(style["month_style"]),
        str(style["month_source"]),
    )
    if style.get("month_dot"):
        month_text = f"{month_text}."
    return f"{month_text}{style['sep']}{year:04d}"


def render_season_interval(
    shifted_start: date,
    shifted_end: date,
    style: dict[str, str | int | bool],
) -> str:
    start_season, start_year = season_for_date(shifted_start)
    end_season, end_year = season_for_date(shifted_end)
    start_text = render_season_year(start_season, start_year, style)
    end_text = render_season_year(end_season, end_year, style)
    if start_text == end_text:
        return start_text
    if start_year == end_year:
        start_label = match_case(start_season, str(style["season_source"]))
        end_label = match_case(end_season, str(style["season_source"]))
        return f"{start_label}/{end_label}{style['sep']}{start_year:04d}"
    return f"{start_text}/{end_text}"


def render_season_year(
    season: str,
    year: int,
    style: dict[str, str | int | bool],
) -> str:
    return f"{match_case(season, str(style['season_source']))}{style['sep']}{year:04d}"


def render_year_interval(start_year: int, end_year: int) -> str:
    if start_year == end_year:
        return f"{start_year:04d}"
    return f"{start_year:04d}/{end_year:04d}"


def render_year_like_interval(
    start_year: int,
    end_year: int,
    style: dict[str, str | int | bool],
) -> str:
    rendered = render_year_interval(start_year, end_year)
    prefix = style.get("prefix")
    if not prefix:
        return rendered
    return f"{prefix}{style.get('prefix_sep', ' ')}{rendered}"


def pseudonymize_standalone_age_birthdate(
    text: str,
    *,
    date_shift_days: int,
    context_before: str,
    context_after: str,
    document_date: date | None,
) -> str | None:
    match = AGE_STANDALONE_RE.fullmatch(text)
    if not match:
        return None

    value = int(match.group("age"))
    token = match.group("age")
    context_kind = nearest_age_birthdate_context(context_before, context_after)
    if context_kind == "age":
        return text if 0 <= value <= 120 else None

    if context_kind not in {"jaar", "maand", "dag", "year", "month", "day"}:
        return text if 0 <= value <= 120 else None
    if text.strip() != token:
        return None

    if context_kind in {"jaar", "year"}:
        year = expand_year(token, document_date=document_date) if value < 100 else value
        shifted_start = date(year, 1, 1) + timedelta(days=date_shift_days)
        shifted_end = date(year, 12, 31) + timedelta(days=date_shift_days)
        return render_year_interval(shifted_start.year, shifted_end.year)

    if context_kind in {"maand", "month"}:
        if not 1 <= value <= 12:
            return None
        shifted_start = date(REFERENCE_YEAR_FOR_YEARLESS_DATES, value, 1) + timedelta(
            days=date_shift_days
        )
        shifted_end = end_of_month(
            REFERENCE_YEAR_FOR_YEARLESS_DATES,
            value,
        ) + timedelta(days=date_shift_days)
        return render_numeric_component_interval(
            shifted_start.month,
            shifted_end.month,
            range_component_width(token),
        )

    if not 1 <= value <= 31:
        return None
    try:
        shifted = date(REFERENCE_YEAR_FOR_YEARLESS_DATES, 1, value) + timedelta(
            days=date_shift_days
        )
    except ValueError:
        return None
    return render_int(shifted.day, range_component_width(token))


def pseudonymize_standalone_date_component(
    text: str,
    *,
    date_shift_days: int,
    context_before: str,
    context_after: str,
    document_date: date | None,
) -> str | None:
    match = AGE_STANDALONE_RE.fullmatch(text)
    if not match or text.strip() != match.group("age"):
        return None

    value = int(match.group("age"))
    token = match.group("age")
    context_kind = nearest_age_birthdate_context(context_before, context_after)
    if context_kind in {"year", "jaar"}:
        year = expand_year(token, document_date=document_date) if value < 100 else value
        shifted_start = date(year, 1, 1) + timedelta(days=date_shift_days)
        shifted_end = date(year, 12, 31) + timedelta(days=date_shift_days)
        return render_year_interval(shifted_start.year, shifted_end.year)

    if context_kind in {"month", "maand"}:
        if not 1 <= value <= 12:
            return None
        shifted_start = date(REFERENCE_YEAR_FOR_YEARLESS_DATES, value, 1) + timedelta(
            days=date_shift_days
        )
        shifted_end = end_of_month(
            REFERENCE_YEAR_FOR_YEARLESS_DATES,
            value,
        ) + timedelta(days=date_shift_days)
        return render_numeric_component_interval(
            shifted_start.month,
            shifted_end.month,
            range_component_width(token),
        )

    if context_kind not in {"day", "dag"}:
        return text if 0 <= value <= 120 else None
    if not 1 <= value <= 31:
        return None
    try:
        shifted = date(REFERENCE_YEAR_FOR_YEARLESS_DATES, 1, value) + timedelta(
            days=date_shift_days
        )
    except ValueError:
        return None
    return render_int(shifted.day, range_component_width(token))


def nearest_age_birthdate_context(
    context_before: str,
    context_after: str,
) -> str | None:
    matches: list[tuple[int, str]] = []
    before = context_before[-50:]
    after = context_after[:50]

    for match in AGE_BIRTHDATE_CONTEXT_RE.finditer(before):
        matches.append(
            (
                len(before) - match.end(),
                age_birthdate_context_kind(match.group(0)),
            )
        )
    for match in AGE_BIRTHDATE_CONTEXT_RE.finditer(after):
        matches.append((match.start(), age_birthdate_context_kind(match.group(0))))

    if not matches:
        return None
    return min(matches, key=lambda item: item[0])[1]


def age_birthdate_context_kind(token: str) -> str:
    normalized = token.casefold()
    normalized_words = re.sub(r"[^a-z]+", " ", normalized).split()
    word_set = set(normalized_words)
    if normalized in {"leeftijd", "age", "jaren", "maanden", "dagen"}:
        return "age"
    if word_set & {"leeftijd", "age", "jaren", "maanden", "dagen"}:
        return "age"
    if normalized in {"years", "months", "days"}:
        return "age"
    if word_set & {"year", "jaar"} or component_suffix_context(normalized, "year"):
        return "year"
    if word_set & {"month", "maand"} or component_suffix_context(normalized, "month"):
        return "month"
    if word_set & {"day", "dag"} or component_suffix_context(normalized, "day"):
        return "day"
    if normalized == "jaar":
        return "jaar"
    if normalized == "maand":
        return "maand"
    if normalized == "dag":
        return "dag"
    return normalized


def component_suffix_context(normalized: str, component: str) -> bool:
    prefixes = ("birth", "birty", "sample", "geboorte")
    return any(prefix in normalized for prefix in prefixes) and normalized.endswith(component)


def render_numeric_component_interval(
    start_value: int,
    end_value: int,
    width: int,
) -> str:
    start_text = render_int(start_value, width)
    if start_value == end_value:
        return start_text
    return f"{start_text}/{render_int(end_value, width)}"


def parse_month_token(token: str) -> tuple[int, str]:
    normalized = normalize_token(token)
    value = MONTH_TOKEN_TO_VALUE.get(normalized)
    if value is None:
        raise ValueError("unsupported month")
    return value


def normalize_year_token(token: str) -> str:
    if len(token) == 5 and token.startswith(("19", "20")):
        return token[:4]
    return token


def expand_year(token: str, *, document_date: date | None = None) -> int:
    if len(token) == 4:
        return int(token)
    suffix = int(token)
    if document_date is None:
        return 2000 + suffix if suffix <= 30 else 1900 + suffix

    modulo = 10 ** len(token)
    base = (document_date.year // modulo) * modulo
    candidates = [base + suffix, base - modulo + suffix, base + modulo + suffix]
    return min(candidates, key=lambda year: (abs(year - document_date.year), year))


def render_year(year: int, width: int) -> str:
    if width == 1:
        return str(year % 10)
    if width == 2:
        return f"{year % 100:02d}"
    return f"{year:04d}"


def render_int(value: int, width: int) -> str:
    return f"{value:0{width}d}" if width > 1 else str(value)


def range_component_width(token: str) -> int:
    return len(token) if token.startswith("0") else 1


def render_month(month: int, style: str, source: str) -> str:
    if style == "abbr":
        token = DUTCH_MONTHS_ABBR[month - 1]
    else:
        token = DUTCH_MONTHS_FULL[month - 1]
    return match_case(token, source)


def render_weekday(weekday: int, source: str) -> str:
    token = (
        DUTCH_WEEKDAYS_ABBR[weekday]
        if normalize_token(source) in DUTCH_WEEKDAYS_ABBR
        else DUTCH_WEEKDAYS_FULL[weekday]
    )
    return match_case(token, source)


def match_case(value: str, source: str) -> str:
    if source.isupper():
        return value.upper()
    if source.istitle():
        return value.title()
    return value


def normalize_token(token: str) -> str:
    return token.strip().rstrip(".").casefold()


def end_of_month(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def season_interval(season: str, year: int) -> tuple[date, date]:
    start_month, start_day, end_month, end_day = SEASON_INTERVALS[season]
    end_year = year
    if season == "winter":
        end_year = year + 1
        end_day = calendar.monthrange(end_year, end_month)[1]
    return date(year, start_month, start_day), date(end_year, end_month, end_day)


def season_for_date(value: date) -> tuple[str, int]:
    if 3 <= value.month <= 5:
        return "lente", value.year
    if 6 <= value.month <= 8:
        return "zomer", value.year
    if 9 <= value.month <= 11:
        return "herfst", value.year
    if value.month == 12:
        return "winter", value.year
    return "winter", value.year - 1


def split_outer_whitespace(text: str) -> tuple[str, str, str]:
    match = re.match(r"^(\s*)(.*?)(\s*)$", text, flags=re.DOTALL)
    if not match:
        return "", text, ""
    return match.group(1), match.group(2), match.group(3)


def parse_document_creation_date(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value)
    match = re.search(
        r"(?<!\d)(?P<a>\d{1,4})[/-](?P<b>\d{1,2})[/-](?P<c>\d{1,4})(?!\d)",
        text,
    )
    if not match:
        return None

    first = match.group("a")
    middle = match.group("b")
    last = match.group("c")
    try:
        if len(first) == 4:
            return date(int(first), int(middle), int(last))
        return date(int(last), int(middle), int(first))
    except ValueError:
        return None
