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
MIN_RECOMMENDED_ABS_DATE_SHIFT_DAYS = 366


def is_weak_date_shift(date_shift_days: int | None) -> bool:
    """Return whether an offset falls within the literature-informed warning range."""
    return (
        date_shift_days is not None
        and abs(date_shift_days) < MIN_RECOMMENDED_ABS_DATE_SHIFT_DAYS
    )

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
APOSTROPHE_SHORT_YEAR_RE = re.compile(
    r"(?P<prefix>['’])(?P<year>\d{2})$"
)
NUMERIC_DAY_MONTH_TRAILING_DOT_RE = re.compile(
    r"^(?P<body>\d{1,2}\s*[/-]\s*\d{1,2})(?P<punct>\.)$"
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
WORD_NUMERIC_DMY_RANGE_RE = re.compile(
    r"^(?P<start_day>\d{1,2})(?P<start_sep1>[/-])"
    r"(?P<start_month>\d{1,2})(?P<start_sep2>[/-])"
    r"(?P<start_year>\d{2}|\d{4})"
    r"(?P<range_sep>\s+(?:tot|t/m)\s+)"
    r"(?P<end_day>\d{1,2})(?P<end_sep1>[/-])"
    r"(?P<end_month>\d{1,2})(?P<end_sep2>[/-])"
    r"(?P<end_year>\d{2}|\d{4})$",
    re.IGNORECASE,
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
AGE_UNIT_ALIASES: dict[str, tuple[str, str]] = {
    "jaar": ("year", "nl"),
    "jaren": ("year", "nl"),
    "jarig": ("year", "nl"),
    "jarige": ("year", "nl"),
    "jr": ("year", "nl"),
    "j": ("year", "nl"),
    "year": ("year", "en"),
    "years": ("year", "en"),
    "yr": ("year", "en"),
    "yrs": ("year", "en"),
    "maand": ("month", "nl"),
    "maanden": ("month", "nl"),
    "mnd": ("month", "nl"),
    "m": ("month", "nl"),
    "month": ("month", "en"),
    "months": ("month", "en"),
    "mo": ("month", "en"),
    "mos": ("month", "en"),
    "week": ("week", "nl"),
    "weken": ("week", "nl"),
    "wk": ("week", "nl"),
    "w": ("week", "nl"),
    "weeks": ("week", "en"),
    "wks": ("week", "en"),
    "dag": ("day", "nl"),
    "dagen": ("day", "nl"),
    "d": ("day", "nl"),
    "day": ("day", "en"),
    "days": ("day", "en"),
}
AGE_UNIT_WORDS: dict[str, dict[str, tuple[str, str]]] = {
    "nl": {
        "year": ("jaar", "jaar"),
        "month": ("maand", "maanden"),
        "week": ("week", "weken"),
        "day": ("dag", "dagen"),
    },
    "en": {
        "year": ("year", "years"),
        "month": ("month", "months"),
        "week": ("week", "weeks"),
        "day": ("day", "days"),
    },
}
AGE_UNIT_ABBREVIATIONS = frozenset(
    {"jr", "j", "yr", "yrs", "mnd", "m", "mo", "mos", "wk", "w", "wks", "d"}
)
AGE_ADJECTIVAL_TOKENS = frozenset({"jarig", "jarige"})
AGE_SUFFIX_RE = re.compile(r"\s+(?:jongere|ouder)$", re.IGNORECASE)
AGE_APPROX_PREFIX_RE = re.compile(
    r"^(?:ca\.?|circa|rond|bijna|ongeveer|\+\s*/\s*-|±)\s*",
    re.IGNORECASE,
)
AGE_PART_RE = re.compile(
    r"(?P<value>\d{1,3})(?:[.,](?P<fraction>\d+))?"
    r"(?P<gap>\s*(?:-|–|—)?\s*)"
    r"(?P<unit>[A-Za-z]+)"
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
BIRTHDATE_COMPONENT_CONTEXT_RE = re.compile(
    r"\b(?:birth|birty|geboorte)[\w\s:._/-]{0,24}?"
    r"(?P<component>year|month|day|jaar|maand|dag)\b",
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
    birthdate_replacement_mode: str = "age",
) -> str | None:
    substitute = pseudonymize_date_text_body(
        text,
        label=label,
        date_shift_days=date_shift_days,
        context_before=context_before,
        context_after=context_after,
        document_creation_date=document_creation_date,
        birthdate_replacement_mode=birthdate_replacement_mode,
    )
    if substitute is None:
        return None
    return bracket_substitute(substitute)


def pseudonymize_date_text_body(
    text: str,
    *,
    label: str,
    date_shift_days: int | None,
    context_before: str = "",
    context_after: str = "",
    document_creation_date: str | None = None,
    birthdate_replacement_mode: str = "age",
) -> str | None:
    if date_shift_days is None:
        return None

    document_date = parse_document_creation_date(document_creation_date)

    if label == "Age_Birthdate":
        age_expression = parse_age_expression(text)
        if age_expression is not None:
            # An age is already a duration, so unlike a birthdate it needs no
            # document creation date to be interpreted. The bands are still
            # derived by anchoring the duration to a reference date: when the
            # caller supplied one we use it, so an age and the equivalent
            # birthdate read the same. Otherwise we anchor to a fixed internal
            # date, which keeps the output deterministic.
            #
            # The anchor is not entirely neutral, because calendar months vary
            # in length: an expression sitting exactly on a band edge can fall
            # either side of it depending on where it is anchored. "3 maanden"
            # is the case in practice — it is 89-92 days, and the weeks band
            # ends at 90. January 1st is chosen because counting back over the
            # 31-day months keeps such expressions in the coarser band, which
            # is the granularity the source already used.
            reference_date = (
                document_date + timedelta(days=date_shift_days)
                if document_date is not None
                else date(REFERENCE_YEAR_FOR_YEARLESS_DATES, 1, 1)
            )
            return render_age_expression(age_expression, reference_date)
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
        shifted_document_date = (
            document_date + timedelta(days=date_shift_days) if document_date else None
        )
        if (
            birthdate_replacement_mode != "year_fallback"
            and shifted_document_date is not None
        ):
            age_substitute = render_birthdate_age_interval(
                shifted_start,
                shifted_end,
                shifted_document_date,
            )
            if age_substitute is not None:
                return age_substitute
        return render_year_interval(shifted_start.year, shifted_end.year)
    if label == "Date":
        return render_shifted_date(parsed, shifted_start, shifted_end, text)
    return None


def bracket_substitute(substitute: str) -> str:
    stripped = substitute.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped
    return f"[{stripped}]"


def parse_date_text(text: str, *, document_date: date | None = None) -> ParsedDate:
    leading, body, trailing = split_outer_whitespace(text)
    prefix: tuple[str, str] | None = None
    prefix_match = WEEKDAY_PREFIX_RE.fullmatch(body)
    if prefix_match:
        prefix = (prefix_match.group("weekday"), prefix_match.group("sep"))
        body = prefix_match.group("body")

    year_prefix = ""
    apostrophe_year = APOSTROPHE_SHORT_YEAR_RE.search(body)
    if apostrophe_year:
        year_prefix = apostrophe_year.group("prefix")
        body = f"{body[:apostrophe_year.start()]}{apostrophe_year.group('year')}"

    terminal_punctuation = ""
    trailing_dot = NUMERIC_DAY_MONTH_TRAILING_DOT_RE.fullmatch(body)
    if trailing_dot:
        body = trailing_dot.group("body")
        terminal_punctuation = trailing_dot.group("punct")

    parsed = parse_date_body(body, document_date=document_date)
    if leading or trailing or year_prefix or terminal_punctuation:
        style = dict(parsed.style)
        style["leading_ws"] = leading
        style["trailing_ws"] = trailing
        style["year_prefix"] = year_prefix
        style["terminal_punctuation"] = terminal_punctuation
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

    word_numeric_range = WORD_NUMERIC_DMY_RANGE_RE.fullmatch(body)
    if word_numeric_range:
        start_year_token = word_numeric_range.group("start_year")
        end_year_token = word_numeric_range.group("end_year")
        try:
            start_date = date(
                expand_year(start_year_token, document_date=document_date),
                int(word_numeric_range.group("start_month")),
                int(word_numeric_range.group("start_day")),
            )
            end_date = date(
                expand_year(end_year_token, document_date=document_date),
                int(word_numeric_range.group("end_month")),
                int(word_numeric_range.group("end_day")),
            )
        except ValueError as exc:
            raise ValueError("invalid word-separated numeric range") from exc
        if end_date < start_date:
            raise ValueError("date range ends before it starts")
        return ParsedDate(
            start=start_date,
            end=end_date,
            granularity="range",
            style={
                "kind": "word_numeric_range",
                "start_day_width": range_component_width(
                    word_numeric_range.group("start_day")
                ),
                "start_month_width": range_component_width(
                    word_numeric_range.group("start_month")
                ),
                "start_year_width": len(start_year_token),
                "start_sep1": word_numeric_range.group("start_sep1"),
                "start_sep2": word_numeric_range.group("start_sep2"),
                "end_day_width": range_component_width(
                    word_numeric_range.group("end_day")
                ),
                "end_month_width": range_component_width(
                    word_numeric_range.group("end_month")
                ),
                "end_year_width": len(end_year_token),
                "end_sep1": word_numeric_range.group("end_sep1"),
                "end_sep2": word_numeric_range.group("end_sep2"),
                "range_sep": word_numeric_range.group("range_sep"),
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
    terminal_punctuation = str(parsed.style.get("terminal_punctuation", ""))
    rendered = f"{rendered}{terminal_punctuation}"
    return f"{leading}{rendered}{trailing}" if leading or trailing else rendered


def render_exact_date(shifted: date, parsed: ParsedDate) -> str:
    style = parsed.style
    year_width = int(style["year_width"])
    day = render_int(shifted.day, int(style["day_width"]))
    year = f"{style.get('year_prefix', '')}{render_year(shifted.year, year_width)}"

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
    if style["kind"] == "word_numeric_range":
        start_value = render_word_numeric_range_date(shifted_start, style, "start")
        end_value = render_word_numeric_range_date(shifted_end, style, "end")
        return f"{start_value}{style['range_sep']}{end_value}"

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


def render_word_numeric_range_date(
    value: date,
    style: dict[str, str | int | bool],
    side: str,
) -> str:
    day = render_int(value.day, int(style[f"{side}_day_width"]))
    month = render_int(value.month, int(style[f"{side}_month_width"]))
    year = render_year(value.year, int(style[f"{side}_year_width"]))
    return (
        f"{day}{style[f'{side}_sep1']}{month}"
        f"{style[f'{side}_sep2']}{year}"
    )


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


def render_birthdate_age_interval(
    shifted_start: date,
    shifted_end: date,
    shifted_document_date: date,
) -> str | None:
    start_age = render_birthdate_age(shifted_start, shifted_document_date)
    end_age = render_birthdate_age(shifted_end, shifted_document_date)
    if start_age is None or end_age is None:
        return None
    if start_age == end_age:
        return start_age
    return f"{start_age}/{end_age}"


def render_birthdate_age(birthdate: date, reference_date: date) -> str | None:
    parts = age_band_parts(birthdate, reference_date)
    if parts is None:
        return None
    rendered = ", ".join(AGE_PART_TEXT[unit](value) for value, unit in parts)
    return f"{rendered} oud"


def age_band_parts(
    birthdate: date,
    reference_date: date,
) -> tuple[tuple[int, str], ...] | None:
    """Return the age as ``(value, unit)`` parts at the band granularity.

    The bands are the single source of truth for age precision: they are used
    both for birthdate spans and for age expressions that are already written
    as a duration in the source text.
    """
    if birthdate > reference_date:
        return None

    total_days = (reference_date - birthdate).days
    years, months, days = age_calendar_parts(birthdate, reference_date)
    total_months = years * 12 + months

    if total_days <= 28:
        return ((total_days, "day"),)
    if total_days <= 90:
        weeks, remaining_days = divmod(total_days, 7)
        if remaining_days == 0:
            return ((weeks, "week"),)
        return ((weeks, "week"), (remaining_days, "day"))
    if total_months < 6:
        weeks = days // 7
        if weeks == 0:
            return ((total_months, "month"),)
        return ((total_months, "month"), (weeks, "week"))
    if total_months < 24:
        return ((total_months, "month"),)
    if years < 12:
        if months == 0:
            return ((years, "year"),)
        return ((years, "year"), (months, "month"))
    return ((years, "year"),)


def age_calendar_parts(birthdate: date, reference_date: date) -> tuple[int, int, int]:
    years = reference_date.year - birthdate.year
    if (reference_date.month, reference_date.day) < (birthdate.month, birthdate.day):
        years -= 1

    anniversary = add_years_clamped(birthdate, years)
    months = 0
    while add_months_clamped(anniversary, months + 1) <= reference_date:
        months += 1

    month_anniversary = add_months_clamped(anniversary, months)
    days = (reference_date - month_anniversary).days
    return years, months, days


def add_years_clamped(value: date, years: int) -> date:
    year = value.year + years
    day = min(value.day, calendar.monthrange(year, value.month)[1])
    return date(year, value.month, day)


def add_months_clamped(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def years_text(value: int) -> str:
    return f"{value} jaar"


def months_text(value: int) -> str:
    unit = "maand" if value == 1 else "maanden"
    return f"{value} {unit}"


def weeks_text(value: int) -> str:
    unit = "week" if value == 1 else "weken"
    return f"{value} {unit}"


def days_text(value: int) -> str:
    unit = "dag" if value == 1 else "dagen"
    return f"{value} {unit}"


AGE_PART_TEXT = {
    "year": years_text,
    "month": months_text,
    "week": weeks_text,
    "day": days_text,
}


@dataclass(frozen=True)
class AgeExpression:
    """An age already written as a duration in the source text."""

    source: str
    parts: tuple[tuple[float, str], ...]
    tokens: dict[str, str]
    lang: str
    adjectival: bool
    suffix: str
    approximation_prefix: str


def parse_age_expression(text: str) -> AgeExpression | None:
    """Parse ``18 weken``/``6-jarig``/``7 jaar 7 maand`` into its parts.

    Returns ``None`` when the text is not an age duration, which lets the
    caller fall through to birthdate parsing.
    """
    body = text.strip()
    if not body:
        return None

    approximation_prefix = ""
    approximation_match = AGE_APPROX_PREFIX_RE.match(body)
    if approximation_match:
        approximation_prefix = approximation_match.group(0)
        body = body[approximation_match.end() :]

    suffix = ""
    suffix_match = AGE_SUFFIX_RE.search(body)
    if suffix_match:
        suffix = body[suffix_match.start() :]
        body = body[: suffix_match.start()]

    parts: list[tuple[float, str]] = []
    tokens: dict[str, str] = {}
    lang = "nl"
    adjectival = False
    position = 0

    for match in AGE_PART_RE.finditer(body):
        if body[position : match.start()].strip(" \t,"):
            return None
        unit_token = match.group("unit").casefold()
        alias = AGE_UNIT_ALIASES.get(unit_token)
        if alias is None:
            return None
        unit, token_lang = alias
        fraction = match.group("fraction")
        value = float(f"{match.group('value')}.{fraction}" if fraction else match.group("value"))
        parts.append((value, unit))
        tokens.setdefault(unit, match.group("unit"))
        if token_lang == "en":
            lang = "en"
        if unit_token in AGE_ADJECTIVAL_TOKENS:
            adjectival = True
        position = match.end()

    if not parts or body[position:].strip(" \t,"):
        return None

    return AgeExpression(
        source=text,
        parts=tuple(parts),
        tokens=tokens,
        lang=lang,
        adjectival=adjectival,
        suffix=suffix,
        approximation_prefix=approximation_prefix,
    )


def render_age_expression(
    expression: AgeExpression,
    reference_date: date,
) -> str | None:
    """Re-render an age expression at the band granularity.

    The value and unit follow the same bands as birthdate-derived ages, so the
    same real age reads the same whether the source wrote a birthdate or an
    age. The source phrasing is kept: text that already matches its band is
    returned untouched, and adjectival forms stay adjectival.
    """
    birthdate = age_expression_birthdate(expression, reference_date)
    parts = age_band_parts(birthdate, reference_date)
    if parts is None:
        return None
    if tuple((float(value), unit) for value, unit in parts) == expression.parts:
        return expression.source
    return (
        f"{expression.approximation_prefix}"
        f"{format_age_parts(parts, expression)}{expression.suffix}"
    )


def age_expression_birthdate(
    expression: AgeExpression,
    reference_date: date,
) -> date:
    """Anchor an age duration to a birthdate, counting back from the reference."""
    years = 0
    months = 0
    days = 0
    for value, unit in expression.parts:
        whole = int(value)
        fraction = value - whole
        if unit == "year":
            years += whole
            days += round(fraction * 365.25)
        elif unit == "month":
            months += whole
            days += round(fraction * 30.44)
        elif unit == "week":
            days += whole * 7 + round(fraction * 7)
        else:
            days += whole + round(fraction)

    anchored = add_years_clamped(reference_date, -years)
    anchored = add_months_clamped(anchored, -months)
    return anchored - timedelta(days=days)


def format_age_parts(
    parts: tuple[tuple[int, str], ...],
    expression: AgeExpression,
) -> str:
    if expression.adjectival and len(parts) == 1 and parts[0][1] == "year":
        token = expression.tokens.get("year", "jarige")
        return f"{parts[0][0]}-{token}"

    rendered = []
    for value, unit in parts:
        token = expression.tokens.get(unit)
        if token is None or token.casefold() not in AGE_UNIT_ABBREVIATIONS:
            singular, plural = AGE_UNIT_WORDS[expression.lang][unit]
            token = singular if value == 1 else plural
        rendered.append(f"{value} {token}")
    return ", ".join(rendered)


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
        return "Age_Birthdate" if 1 <= value <= 12 else None

    if not 1 <= value <= 31:
        return None
    return "Age_Birthdate"


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
    birthdate_component = nearest_birthdate_component_context(
        context_before,
        context_after,
    )
    if birthdate_component in {"month", "maand"}:
        return "Age_Birthdate" if 1 <= value <= 12 else None
    if birthdate_component in {"day", "dag"}:
        return "Age_Birthdate" if 1 <= value <= 31 else None

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


def nearest_birthdate_component_context(
    context_before: str,
    context_after: str,
) -> str | None:
    matches: list[tuple[int, str]] = []
    before = context_before[-50:]
    after = context_after[:50]

    for match in BIRTHDATE_COMPONENT_CONTEXT_RE.finditer(before):
        matches.append((len(before) - match.end(), match.group("component").casefold()))
    for match in BIRTHDATE_COMPONENT_CONTEXT_RE.finditer(after):
        matches.append((match.start(), match.group("component").casefold()))

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
