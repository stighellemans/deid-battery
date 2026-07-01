"""
Span post-processing for de-identification output.

This module takes raw model spans and normalizes them into cleaner, more
consistent annotations. The core use case is that an upstream recognizer often
produces spans that are slightly too short, include stray punctuation, miss
context such as weekdays, or omit metadata-derived patient names.

The pipeline in ``post_process_spans`` is intentionally phase-based:
1. Normalize each incoming span independently.
2. Drop spans that are only punctuation / symbols.
3. Add metadata-derived patient name spans and merge adjacent patient-name
   fragments.
4. Extend spans using label-specific regular expressions.
5. Extend dates with leading weekdays while preventing overlaps.
6. Relabel degree-prefixed dates as birthdates.
7. Remove exact duplicates that may arise during extension.

The code is written around simple span dictionaries with at least these fields:
``label``, ``begin``, ``end``, and ``text``.
"""

import logging
import re

logger = logging.getLogger(__name__)

# --- NEW: doc_id context handling -----------------------------------------
CURRENT_DOC_ID: str | None = None


def set_current_doc_id(doc_id: str | None):
    """
    Set the current document id for logging context.
    Call this once per document before running post-process.
    """
    global CURRENT_DOC_ID
    CURRENT_DOC_ID = doc_id


BLACKLIST_NAME_TOKENS = {
    "de",
    "van",
    "op",
}
"""Tokens that should not become standalone patient-name spans."""


# ---------- Shared weekday ----------
WEEKDAY = (
    r"(?:ma|di|wo|do|vr|za|zo|"
    r"maandag|dinsdag|woensdag|donderdag|vrijdag|zaterdag|zondag)"
)

MONTH_WORD_FULL = (
    r"(?:januari|februari|maart|april|mei|juni|juli|augustus|"
    r"september|oktober|november|december)"
)
MONTH_WORD_ABBR = r"(?:jan|feb|mrt|apr|jun|jul|aug|sep|okt|nov|dec)"
MONTH_WORD = r"(?:" + MONTH_WORD_FULL + r"|" + MONTH_WORD_ABBR + r")"
MONTH_WORD_WITH_SUFFIX = (
    r"(?:" + MONTH_WORD_FULL + r"(?![A-Za-z])|"
    + MONTH_WORD_ABBR + r"\.?(?![A-Za-z]))"
)

SEASON_WORD = r"(?:lente|zomer|herfst|winter)"
MONTH_PHASE_WORD = r"(?:begin|midden|half|eind)"

# ---------- For / and - (digits + . + * allowed) ----------

DAY_SD = (
    r"(?:"
    r"0?[1-9]|[12]\d|3[01]"  # valid numeric day
    r"|(?=[0-9.*]*[.*])[0-9.*]{1,2}"  # 1–2 chars, contains . or * as wildcard
    r")"
)

MONTH_SD = (
    r"(?:"
    r"0?[1-9]|1[0-2]"  # valid numeric month
    r"|(?=[0-9.*]*[.*])[0-9.*]{1,2}"  # 1–2 chars, contains . or *
    r")"
)

YEAR_SD = (
    r"(?:"
    r"[0-9.*]{2}"  # 2 chars: digits / . / *
    r"|"
    r"[0-9.*]{4}"  # 4 chars: digits / . / *
    r")"
)

SLASH_DASH_DATE = DAY_SD + r"([/-])" + MONTH_SD + r"(?:\1" + YEAR_SD + r")?"

MONTH_YEAR_SD = r"(?<!\d)(?<!/)" + MONTH_SD + r"([/-])" + YEAR_SD + r"(?!\d)"

# ---------- For dot dates (digits + * only in components) ----------

DAY_DOT = (
    r"(?:"
    r"0?[1-9]|[12]\d|3[01]"  # valid numeric day
    r"|(?=[0-9*]*\*)[0-9*]{1,2}"  # 1–2 chars, contains * as wildcard
    r")"
)

MONTH_DOT = (
    r"(?:"
    r"0?[1-9]|1[0-2]"  # valid numeric month
    r"|(?=[0-9*]*\*)[0-9*]{1,2}"  # 1–2 chars, contains *
    r")"
)

YEAR_DOT = (
    r"(?:"
    r"[0-9*]{2}"  # 2 chars: digits / *
    r"|"
    r"[0-9*]{4}"  # 4 chars: digits / *
    r")"
)

DOT_DATE = DAY_DOT + r"\." + MONTH_DOT + r"\." + YEAR_DOT

RANGE_SEP = r"\s*[-–—]\s*"
EXPLICIT_HYPHEN_DATE_RANGE_SEP = r"(?:\s*[–—]\s*|\s+-\s*|\s*-\s+)"

TEXTUAL_DMY_DATE = DAY_SD + r"\s+" + MONTH_WORD_WITH_SUFFIX + r"\s+" + YEAR_SD
TEXTUAL_DMY_NO_YEAR = DAY_SD + r"\s+" + MONTH_WORD_WITH_SUFFIX
TEXTUAL_DMY_HYPHEN_DATE = (
    DAY_SD + r"\s*-\s*" + MONTH_WORD_WITH_SUFFIX + r"\s*-\s*" + YEAR_SD
)
TEXTUAL_DAY_MONTH = DAY_SD + r"\s+" + MONTH_WORD_WITH_SUFFIX
MONTH_PHASE_WITH_MONTH = (
    MONTH_PHASE_WORD
    + r"(?:/"
    + MONTH_PHASE_WORD
    + r")*"
    + r"\s+"
    + MONTH_WORD_WITH_SUFFIX
)
TEXTUAL_DMY_RANGE = (
    # Expand only when one textual date supplies a year missing from the other.
    # Fully specified textual dates should stay separate.
    r"(?:"
    + TEXTUAL_DMY_DATE
    + RANGE_SEP
    + TEXTUAL_DMY_NO_YEAR
    + r"(?!\s+"
    + YEAR_SD
    + r")"
    + r"|"
    + TEXTUAL_DMY_NO_YEAR
    + RANGE_SEP
    + TEXTUAL_DMY_DATE
    + r")"
)
TEXTUAL_SHARED_MONTH_RANGE = (
    DAY_SD
    + RANGE_SEP
    + DAY_SD
    + r"\s+"
    + MONTH_WORD_WITH_SUFFIX
    + r"(?:\s+"
    + YEAR_SD
    + r")?"
)
TEXTUAL_MONTH_YEAR = r"\b" + MONTH_WORD_WITH_SUFFIX + r"\s+" + r"[0-9]{4}\b"
TEXTUAL_MONTH_YEAR_RANGE = (
    r"\b"
    + MONTH_WORD_WITH_SUFFIX
    + r"\s+[0-9]{4}\s*[-–—]\s*"
    + MONTH_WORD_WITH_SUFFIX
    + r"\s+[0-9]{4}\b"
)
SEASON_YEAR = r"\b" + SEASON_WORD + r"\s+" + r"[0-9]{4}\b"
NUMERIC_MONTH_YEAR_RANGE = (
    r"(?<!\d)"
    + MONTH_SD
    + r"([/-])"
    + r"[0-9]{4}"
    + RANGE_SEP
    + MONTH_SD
    + r"([/-])"
    + r"[0-9]{4}(?!\d)"
)
NUMERIC_DMY_RANGE = (
    # Ranges are expanded only when at least one side depends on the other for
    # date context. Two fully specified dates should remain separate spans.
    r"(?:"
    + DAY_SD
    + r"([/.])"
    + MONTH_SD
    + r"\1"
    + YEAR_SD
    + RANGE_SEP
    + DAY_SD
    + r"\1"
    + MONTH_SD
    + r"(?!\1"
    + YEAR_SD
    + r")"
    + r"|"
    + DAY_SD
    + r"([/.])"
    + MONTH_SD
    + RANGE_SEP
    + DAY_SD
    + r"\2"
    + MONTH_SD
    + r"\2"
    + YEAR_SD
    + r"|"
    + DAY_SD
    + r"-"
    + MONTH_SD
    + r"-"
    + YEAR_SD
    + EXPLICIT_HYPHEN_DATE_RANGE_SEP
    + DAY_SD
    + r"-"
    + MONTH_SD
    + r"(?!-"
    + YEAR_SD
    + r")"
    + r"|"
    + DAY_SD
    + r"-"
    + MONTH_SD
    + EXPLICIT_HYPHEN_DATE_RANGE_SEP
    + DAY_SD
    + r"-"
    + MONTH_SD
    + r"-"
    + YEAR_SD
    + r"|"
    + r")"
)

DATE_REGEXES = [
    re.compile(
        r"\b" + WEEKDAY + r"\s*,?\s*" + SLASH_DASH_DATE,
        re.IGNORECASE,
    ),
    re.compile(
        r"\b" + WEEKDAY + r"\s*,?\s*" + TEXTUAL_DMY_DATE,
        re.IGNORECASE,
    ),
    re.compile(r"(?<!\d)" + NUMERIC_DMY_RANGE + r"(?!\d)"),
    re.compile(
        r"(?<!\d)" + TEXTUAL_SHARED_MONTH_RANGE + r"(?!\d)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\d)" + TEXTUAL_DMY_RANGE + r"(?!\d)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?<!\d)" + TEXTUAL_DMY_HYPHEN_DATE + r"(?!\d)",
        re.IGNORECASE,
    ),
    re.compile(NUMERIC_MONTH_YEAR_RANGE),
    re.compile(TEXTUAL_MONTH_YEAR_RANGE, re.IGNORECASE),
    re.compile(r"(?<!\d)" + TEXTUAL_DMY_DATE + r"(?!\d)", re.IGNORECASE),
    re.compile(r"(?<!\d)" + TEXTUAL_DAY_MONTH + r"(?!\d)", re.IGNORECASE),
    re.compile(r"\b" + MONTH_PHASE_WITH_MONTH + r"\b", re.IGNORECASE),
    re.compile(r"(?<!\d)" + SLASH_DASH_DATE + r"(?!\d)"),
    re.compile(MONTH_YEAR_SD),
    re.compile(TEXTUAL_MONTH_YEAR, re.IGNORECASE),
    re.compile(SEASON_YEAR, re.IGNORECASE),
    re.compile(r"(?<!\d)" + DOT_DATE + r"(?!\d)"),
]


URL_REGEXES = [
    re.compile(
        r"""(?ix)
        \b
        (
            (?:https?://|ftp://|www\.)
            [^\s<>"']*
            [^\s<>"'().,!?;:]                  # allow dots/slashes internally, no trailing punctuation
        )
        """
    ),
]

AGE_UNITS = [
    "jaar",
    "jaren",
    "jarige",
    "jarig",
    "jr",
    "maand",
    "maanden",
    "mnd",
    "week",
    "weken",
    "dag",
    "dagen",
]
units_alt = "|".join(map(re.escape, AGE_UNITS))

AGE_REGEX = re.compile(
    rf"\b(\d{{1,3}}(?:[.,]\d)?)\s*(?:{units_alt})\b",
    re.IGNORECASE,
)

suffixes = ["ANTWERPEN"]
suffix_alt = "|".join(map(re.escape, suffixes))

max_digits = 1
POST_OFFICE_REGEX = re.compile(
    rf"\b(?:{suffix_alt})\b\s+" rf"(\d{{1,{max_digits}}})(?!\d)",
    re.IGNORECASE,
)

suffixes = ["Zaal", "Operatiezaal", "OK", "operatiekwartier"]
suffix_alt = "|".join(map(re.escape, suffixes))

max_digits = 2
ORG_REGEX = re.compile(
    rf"\b(?:{suffix_alt})\b\s+" rf"(\d{{1,{max_digits}}})(?!\d)",
    re.IGNORECASE,
)


REGEX_BY_LABEL = {
    "Date": DATE_REGEXES,
    "Contactdetails": URL_REGEXES,
    "Age_Birthdate": [AGE_REGEX, *DATE_REGEXES],
    "Address_Location:Patient": [POST_OFFICE_REGEX],
    "Address_Location:Other": [POST_OFFICE_REGEX],
    "Address_Location:Caregiver": [POST_OFFICE_REGEX],
    "Organization:Healthcare": [ORG_REGEX, POST_OFFICE_REGEX],
}
"""
Maps entity labels to regexes that can safely expand partial annotations.

The regex pass is conservative: a span is only extended when it is fully
contained in a regex match, and the smallest containing match is preferred.
"""

REGEX_RULE_NAMES_BY_LABEL = {
    "Date": [
        "weekday_numeric_date",
        "weekday_textual_dmy_date",
        "numeric_dmy_range",
        "textual_shared_month_range",
        "textual_dmy_range",
        "textual_dmy_hyphen_date",
        "numeric_month_year_range",
        "textual_month_year_range",
        "textual_dmy_date",
        "textual_day_month",
        "month_phase_with_month",
        "numeric_slash_dash_date",
        "numeric_month_year",
        "textual_month_year",
        "season_year",
        "numeric_dot_date",
    ],
    "Contactdetails": ["url"],
    "Age_Birthdate": [
        "age_expression",
        "weekday_numeric_date",
        "weekday_textual_dmy_date",
        "numeric_dmy_range",
        "textual_shared_month_range",
        "textual_dmy_range",
        "textual_dmy_hyphen_date",
        "numeric_month_year_range",
        "textual_month_year_range",
        "textual_dmy_date",
        "textual_day_month",
        "month_phase_with_month",
        "numeric_slash_dash_date",
        "numeric_month_year",
        "textual_month_year",
        "season_year",
        "numeric_dot_date",
    ],
    "Address_Location:Patient": ["post_office_suffix"],
    "Address_Location:Other": ["post_office_suffix"],
    "Address_Location:Caregiver": ["post_office_suffix"],
    "Organization:Healthcare": ["room_reference", "post_office_suffix"],
}
"""Human-readable names aligned with ``REGEX_BY_LABEL`` for audit reports."""

SPAN_CORE_FIELDS = ("label", "begin", "end", "text")


def span_repr(span: dict) -> str:
    """Compact human-readable representation of a span."""
    if not span:
        return "<None>"
    label = span.get("label")
    b = span.get("begin")
    e = span.get("end")
    text = span.get("text")
    return f"[{label}]({b},{e}): {repr(text)}"


def snapshot_span(span: dict | None) -> dict | None:
    """Copy only the fields used by logging and equality checks."""
    if span is None:
        return None
    return {field: span.get(field) for field in SPAN_CORE_FIELDS}


def span_sort_key(span: dict) -> tuple:
    return (
        span.get("begin", -1),
        span.get("end", -1),
        span.get("label", ""),
        span.get("text", ""),
    )


def deduplicate_spans(spans: list[dict]) -> list[dict]:
    """
    Remove exact duplicate spans while preferring the richer dict when duplicates
    carry different metadata keys.
    """
    deduped: dict[tuple, dict] = {}
    order: list[tuple] = []

    for span in sorted(spans, key=span_sort_key):
        key = tuple(span.get(field) for field in SPAN_CORE_FIELDS)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = span
            order.append(key)
            continue

        if len(span) > len(existing):
            deduped[key] = span
        logger.debug("deduplicate_spans: dropping %s", span_repr(span))

    return [deduped[key] for key in order]


def log_span_change(
    step: str, before: dict, after: dict, extra: str = "", level=logging.DEBUG
):
    """
    Log a transformation step if anything relevant changed.
    Includes current doc_id (wrapped on multiple lines) if set.
    """
    # Build doc_id suffix once
    if CURRENT_DOC_ID:
        doc_suffix = f"\n[doc_id]\n{CURRENT_DOC_ID}"
    else:
        doc_suffix = ""

    if before is None or after is None:
        logger.log(
            level,
            "%s%s before=%s after=%s %s",
            step,
            doc_suffix,
            span_repr(before),
            span_repr(after),
            extra,
        )
        return

    changed = (
        before.get("label") != after.get("label")
        or before.get("begin") != after.get("begin")
        or before.get("end") != after.get("end")
        or before.get("text") != after.get("text")
    )

    if changed:
        logger.log(
            level,
            "%s%s\n%s -> %s %s\n",
            step,
            doc_suffix,
            span_repr(before),
            span_repr(after),
            extra,
        )


def is_effectively_alnum(ch: str) -> bool:
    """Return True for genuine alphanumerics, but exclude misleading symbols like º."""
    if ch in {"º", "°", "•", "·"}:  # extend if you find others
        return False
    return ch.isalnum()


def has_effective_alnum(text: str) -> bool:
    return any(is_effectively_alnum(ch) for ch in text)


def drop_non_alnum_spans(spans: list[dict]) -> list[dict]:
    """
    Drop spans whose text contains no effective alphanumeric characters.
    """
    filtered = []
    for s in spans:
        txt = s.get("text") or ""
        if has_effective_alnum(txt):
            filtered.append(s)
        else:
            logger.debug("drop_non_alnum_spans: dropping %s", span_repr(s))
    return filtered


def trim(begin: int, end: int, doc_text: str):
    """Trim non-alphanumeric characters according to the specified side."""
    if not (0 <= begin < end <= len(doc_text)):
        return begin, end

    while begin < end and not is_effectively_alnum(doc_text[begin]):
        begin += 1

    while begin < end and not is_effectively_alnum(doc_text[end - 1]):
        end -= 1

    return begin, end


def attach_dr_period_if_present(span: dict, doc_text: str) -> dict:
    """
    Extends 'Name:Caregiver' spans ending with Dr/dr/DR by one char
    if immediately followed by a '.' in doc_text.
    Raises SpanCorrectionError on inconsistent or unexpected state.
    """
    if span.get("label") != "Name:Caregiver":
        return span

    b, e = span.get("begin"), span.get("end")
    if (
        not isinstance(b, int)
        or not isinstance(e, int)
        or not (0 <= b < e <= len(doc_text))
    ):
        raise ValueError(
            f"Invalid span indices: {b=} {e=} for text length {len(doc_text)}"
        )

    text = doc_text[b:e]
    if text != span.get("text"):
        raise ValueError(
            "Span text does not match document text slice — possible offset mismatch."
        )

    ends_with_dr = text.endswith(("dr", "Dr", "DR"))
    if ends_with_dr:
        if e >= len(doc_text):
            return span
        next_char = doc_text[e]
        if next_char == ".":
            s2 = dict(span)
            s2["end"] = e + 1
            s2["text"] = doc_text[b : e + 1]
            return s2

    # If it doesn't end with 'Dr' or next '.', just return
    return span


def attach_plus_if_present_for_contactdetails(span: dict, doc_text: str) -> dict:
    """
    For 'Contactdetails' spans, after trimming non-alphanumerics,
    re-attach a leading '+' if it is immediately to the left in doc_text.
    """
    if span.get("label") != "Contactdetails":
        return span

    b, e = span.get("begin"), span.get("end")
    if (
        not isinstance(b, int)
        or not isinstance(e, int)
        or not (0 <= b < e <= len(doc_text))
    ):
        return span

    # If already starts with '+', nothing to do
    if doc_text[b] == "+":
        return span

    # If there is a '+' immediately before the span, extend one char left
    if b > 0 and doc_text[b - 1] == "+":
        s2 = dict(span)
        s2["begin"] = b - 1
        s2["text"] = doc_text[b - 1 : e]
        return s2

    return span


def attach_initial_period_if_present(span: dict, doc_text: str) -> dict:
    """
    If a Name:Patient or Name:Caregiver span ends in a single capital initial
    (preceded by a non-alphanumeric char) and is immediately followed by '.',
    extend the span to include the '.'.
    """
    if span.get("label") not in {"Name:Patient", "Name:Caregiver"}:
        return span

    b, e = span.get("begin"), span.get("end")
    if (
        not isinstance(b, int)
        or not isinstance(e, int)
        or not (0 <= b < e <= len(doc_text))
    ):
        return span

    text = doc_text[b:e]
    if text != span.get("text"):
        # Safety: offsets out of sync, don't touch
        return span

    # Last character in the span
    letter_pos = e - 1
    last_char = doc_text[letter_pos]

    # Must end with a single capital letter
    if not ("A" <= last_char <= "Z"):
        return span

    # Char before that capital (if any) must be non-alphanumeric
    if letter_pos - 1 >= 0 and is_effectively_alnum(doc_text[letter_pos - 1]):
        return span

    # Immediately followed by a '.'
    if e >= len(doc_text) or doc_text[e] != ".":
        return span

    # All conditions satisfied: extend to include the '.'
    s2 = dict(span)
    s2["end"] = e + 1
    s2["text"] = doc_text[b : e + 1]
    return s2


BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
OPENERS = set(BRACKET_PAIRS.keys())
CLOSERS = set(BRACKET_PAIRS.values())


def balance_enclosing_brackets(span: dict, doc_text: str) -> dict:
    """
    After trimming, complete a single missing bracket on the left or right
    if the immediate neighbor in doc_text would fix an imbalance.
    Handles (), [], {}.
    """
    b, e = span.get("begin"), span.get("end")
    if (
        not isinstance(b, int)
        or not isinstance(e, int)
        or not (0 <= b < e <= len(doc_text))
    ):
        return span

    text = doc_text[b:e]
    if text != span.get("text"):
        # Keep conservative if upstream text doesn't match
        return span

    # Count brackets in current span
    counts = {op: 0 for op in OPENERS}
    for ch in text:
        if ch in OPENERS:
            counts[ch] += 1
        elif ch in CLOSERS:
            # find its opener
            for op, cl in BRACKET_PAIRS.items():
                if ch == cl:
                    counts[op] -= 1
                    break

    s2 = dict(span)

    # Try to fix missing LEFT opener if there are more closers than openers for some type
    if b > 0:
        left_char = doc_text[b - 1]
        if left_char in OPENERS:
            op = left_char
            if counts.get(op, 0) < 0:  # deficit of this opener
                s2["begin"] = b - 1
                s2["text"] = doc_text[b - 1 : e]
                b = s2["begin"]
                text = s2["text"]
                # Update counts for accuracy if we also try right side
                counts[op] += 1

    # Try to fix missing RIGHT closer if there are more openers than closers for some type
    if e < len(doc_text):
        right_char = doc_text[e]
        if right_char in CLOSERS:
            # if any opener still in surplus AND this closer matches the last unmatched opener, accept it
            # Heuristic: accept if right_char closes *any* surplus opener type.
            for op, surplus in counts.items():
                if surplus > 0 and BRACKET_PAIRS[op] == right_char:
                    s2["end"] = e + 1
                    s2["text"] = doc_text[s2["begin"] : e + 1]
                    break

    return s2


def normalize_span(span: dict, doc_text: str, idx: int) -> dict:
    """
    Run the per-span cleanup steps that only depend on the span and document text.

    The order is deliberate:
    - trim stray outer punctuation first
    - re-attach punctuation that is semantically part of the entity
    - balance enclosing brackets after trimming and punctuation fixes
    """
    b, e = span["begin"], span["end"]
    nb, ne = trim(b, e, doc_text)
    if nb >= ne:
        nb, ne = b, e

    current = dict(span)
    current["begin"], current["end"] = nb, ne
    current["text"] = doc_text[nb:ne]
    log_span_change(f"span {idx} - trim", snapshot_span(span), snapshot_span(current))

    transforms = (
        ("attach_dr_period_if_present", attach_dr_period_if_present),
        ("attach_initial_period_if_present", attach_initial_period_if_present),
        (
            "attach_plus_if_present_for_contactdetails",
            attach_plus_if_present_for_contactdetails,
        ),
        ("balance_enclosing_brackets", balance_enclosing_brackets),
    )

    for step_name, transform in transforms:
        before = snapshot_span(current)
        current = transform(current, doc_text)
        log_span_change(
            f"span {idx} - {step_name}",
            before,
            snapshot_span(current),
        )

    return current


def merge_adjacent_name_patient(spans: list[dict], doc_text: str) -> list[dict]:
    """
    Merge consecutive Name:Patient spans if only whitespace (space/newline/etc.)
    lies between them.
    Assumes spans are sorted by begin.
    """
    if not spans:
        return spans

    spans = sorted(spans, key=lambda s: s["begin"])
    merged: list[dict] = []
    i = 0
    n = len(spans)

    while i < n:
        cur = dict(spans[i])
        j = i + 1

        while j < n:
            nxt = spans[j]
            if cur.get("label") == nxt.get("label") == "Name:Patient":
                gap = doc_text[cur["end"] : nxt["begin"]]
                # Only whitespace between -> merge
                if gap and gap.strip() == "":
                    before = snapshot_span(cur)
                    cur["end"] = nxt["end"]
                    cur["text"] = doc_text[cur["begin"] : cur["end"]]
                    log_span_change(
                        "merge_adjacent_name_patient",
                        before,
                        snapshot_span(cur),
                        extra=f" merged_with={span_repr(nxt)}",
                        level=logging.INFO,
                    )
                    j += 1
                    continue
            break

        merged.append(cur)
        i = j

    return merged


def relabel_date_if_degree_prefix(spans: list[dict], doc_text: str) -> list[dict]:
    """
    If a Date span is immediately preceded by '°'/'º' in the document text,
    relabel it to Age_Birthdate.
    """
    new_spans = []
    for s in spans:
        if s.get("label") == "Date":
            b = s.get("begin")
            if isinstance(b, int) and b > 0 and doc_text[b - 1] in {"°", "º"}:
                before = snapshot_span(s)
                s2 = dict(s)
                s2["label"] = "Age_Birthdate"
                log_span_change(
                    "relabel_date_if_degree_prefix",
                    before,
                    snapshot_span(s2),
                    level=logging.INFO,
                )
                new_spans.append(s2)
                continue
        new_spans.append(s)
    return new_spans


RELATIVE_DATE_FALSE_POSITIVE_RE = re.compile(r"^\s*volgend\s+jaar\s*$", re.IGNORECASE)
MONTH_PHASE_ONLY_RE = re.compile(
    r"^\s*(?:begin|midden|half|eind)(?:/(?:begin|midden|half|eind))*\s*$",
    re.IGNORECASE,
)
TWO_NUMBER_HYPHEN_RE = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
DATE_COMPONENT_CONTEXT_RE = re.compile(
    r"(?:"
    r"\b(?:birth|birty|sample|geboorte)[\w\s:._/-]{0,24}?"
    r"(?:year|month|day|jaar|maand|dag)\b"
    r"|"
    r"\b(?:jaar|maand|dag|year|month|day)\b"
    r")",
    re.IGNORECASE,
)


def filter_invalid_date_spans(spans: list[dict], doc_text: str) -> list[dict]:
    """Drop common false-positive Date spans that cannot be valid dates alone."""
    filtered: list[dict] = []
    for span in spans:
        if span.get("label") != "Date":
            filtered.append(span)
            continue

        text = span.get("text") or ""
        if should_drop_date_span_text(text, span, doc_text):
            logger.debug("filter_invalid_date_spans: dropping %s", span_repr(span))
            continue
        filtered.append(span)
    return filtered


def should_drop_date_span_text(text: str, span: dict, doc_text: str) -> bool:
    if RELATIVE_DATE_FALSE_POSITIVE_RE.fullmatch(text):
        return True
    if MONTH_PHASE_ONLY_RE.fullmatch(text):
        return True

    two_number_hyphen = TWO_NUMBER_HYPHEN_RE.fullmatch(text)
    if not two_number_hyphen:
        return False

    first = int(two_number_hyphen.group(1))
    second = int(two_number_hyphen.group(2))
    return not (1 <= first <= 31 and 1 <= second <= 12)


def has_date_component_context(span: dict, doc_text: str) -> bool:
    begin = span.get("begin")
    end = span.get("end")
    if not isinstance(begin, int) or not isinstance(end, int):
        return False
    context = doc_text[max(0, begin - 50) : min(len(doc_text), end + 50)]
    return DATE_COMPONENT_CONTEXT_RE.search(context) is not None


WEEKDAY_RE = re.compile(
    r"(?i)\b(maandag|dinsdag|woensdag|donderdag|vrijdag|zaterdag|zondag|ma|di|wo|do|vr|za|zo)\b(?:\s*,)?\s*$"
)


def extend_date_with_weekday(
    span: dict, doc_text: str, all_spans_for_doc: list
) -> dict:
    """
    If span is a Date and immediately preceded by a Dutch weekday (full or abbr),
    extend the span leftwards to include the weekday and optional comma/space.
    Prevents overlaps with other spans in the same document.
    """
    if span.get("label") != "Date":
        return span

    b, e = span.get("begin"), span.get("end")
    if not (isinstance(b, int) and isinstance(e, int) and 0 <= b < e <= len(doc_text)):
        return span

    # Look back up to 20 chars for "weekday[,]? <spaces>" directly before the date
    lookback = 20
    left_start = max(0, b - lookback)
    left_ctx = doc_text[left_start:b]

    m = WEEKDAY_RE.search(left_ctx)
    if not m or m.end() != len(left_ctx):
        return span  # no weekday ending right before the date

    new_begin = left_start + m.start()

    # Avoid overlap with other spans (excluding itself)
    for other in all_spans_for_doc:
        if other is span:
            continue
        ob, oe = other.get("begin"), other.get("end")
        if not (isinstance(ob, int) and isinstance(oe, int)):
            continue
        # if [new_begin, e) intersects [ob, oe)
        if not (e <= ob or new_begin >= oe):
            # only allow overlap if the other is the same Date span (already excluded) -> skip extension
            return span

    s2 = dict(span)
    s2["begin"] = new_begin
    s2["text"] = doc_text[new_begin:e]
    return s2


def extend_dates_with_weekdays(spans: list[dict], doc_text: str) -> list[dict]:
    """
    Apply weekday extension against the full document span set so results do not
    depend on input ordering.
    """
    new_spans: list[dict] = []
    for idx, span in enumerate(spans):
        before = snapshot_span(span)
        updated = extend_date_with_weekday(span, doc_text, spans)
        log_span_change(
            f"span {idx} - extend_date_with_weekday",
            before,
            snapshot_span(updated),
        )
        new_spans.append(updated)
    return new_spans


def regex_match_priority(text: str) -> int:
    """Prefer period/range matches over their smaller single-date submatches."""
    if re.search(r"\d[/-]\d{4}\s*[-–—]\s*\d+[/-]\d{4}", text):
        return 0
    if re.search(
        MONTH_WORD_WITH_SUFFIX + r"\s+\d{4}\s*[-–—]\s*" + MONTH_WORD_WITH_SUFFIX,
        text,
        re.IGNORECASE,
    ):
        return 0
    if re.search(r"\d[/.]\d+(?:[/.]\d+)?\s*[-–—]\s*\d+[/.]\d", text):
        return 0
    if re.search(
        r"\d+\s+"
        + MONTH_WORD_WITH_SUFFIX
        + r"(?:\s+"
        + YEAR_SD
        + r")?"
        + RANGE_SEP
        + r"\d+\s+"
        + MONTH_WORD_WITH_SUFFIX,
        text,
        re.IGNORECASE,
    ):
        return 0
    if re.search(
        r"\d+-\d+(?:-\d+)?" + EXPLICIT_HYPHEN_DATE_RANGE_SEP + r"\d+-\d+",
        text,
    ):
        return 0
    if re.search(
        r"\d\s*[-–—]\s*\d+\s+" + MONTH_WORD_WITH_SUFFIX,
        text,
        re.IGNORECASE,
    ):
        return 0
    if re.search(r"\b" + MONTH_PHASE_WITH_MONTH + r"\b", text, re.IGNORECASE):
        return 0
    return 1


def build_regex_match_index(
    doc_text: str,
    regex_by_label: dict[str, list[re.Pattern]],
) -> dict[str, list[tuple[int, int]]]:
    """
    Precompute regex matches once per label so span extension does not rescan the
    whole document for every single span.
    """
    match_index: dict[str, list[tuple[int, int]]] = {}

    for label, regexes in regex_by_label.items():
        seen: set[tuple[int, int]] = set()
        matches: list[tuple[int, int]] = []

        for rx in regexes:
            for match in rx.finditer(doc_text):
                bounds = (match.start(), match.end())
                if bounds in seen:
                    continue
                seen.add(bounds)
                matches.append(bounds)

        matches.sort(
            key=lambda bounds: (
                regex_match_priority(doc_text[bounds[0] : bounds[1]]),
                bounds[1] - bounds[0],
                bounds[0],
                bounds[1],
            )
        )
        match_index[label] = matches

    return match_index


def regex_rule_name(label: str, regex_index: int) -> str:
    """Return a stable display name for a label-specific regex rule."""
    names = REGEX_RULE_NAMES_BY_LABEL.get(label, [])
    if 0 <= regex_index < len(names):
        return names[regex_index]
    return f"regex_{regex_index + 1}"


def build_regex_rule_match_index(
    doc_text: str,
    regex_by_label: dict[str, list[re.Pattern]],
) -> dict[str, list[tuple[int, int, int, re.Pattern]]]:
    """
    Precompute regex matches with their rule identity for audit/report callers.
    """
    match_index: dict[str, list[tuple[int, int, int, re.Pattern]]] = {}

    for label, regexes in regex_by_label.items():
        seen: set[tuple[int, int, int]] = set()
        matches: list[tuple[int, int, int, re.Pattern]] = []

        for regex_index, rx in enumerate(regexes):
            for match in rx.finditer(doc_text):
                key = (match.start(), match.end(), regex_index)
                if key in seen:
                    continue
                seen.add(key)
                matches.append((match.start(), match.end(), regex_index, rx))

        matches.sort(
            key=lambda item: (
                regex_match_priority(doc_text[item[0] : item[1]]),
                item[1] - item[0],
                item[0],
                item[1],
                item[2],
            )
        )
        match_index[label] = matches

    return match_index


def extend_spans_to_regex(
    spans: list[dict],
    doc_text: str,
    regex_by_label: dict[str, list[re.Pattern]],
    regex_observer=None,
) -> list[dict]:
    """
    For labels with regexes, extend a span to the smallest regex match that fully contains it.
    Example for Date: if '2021' is annotated but regex matches '1/1/2021',
    span is extended to the full match.

    This phase does not resolve semantic conflicts between two different
    expanded spans; it only grows each span independently and leaves duplicate
    cleanup to the caller.
    """
    if not regex_by_label:
        return spans

    match_index = (
        build_regex_rule_match_index(doc_text, regex_by_label)
        if regex_observer
        else build_regex_match_index(doc_text, regex_by_label)
    )
    new_spans: list[dict] = []

    for s in spans:
        label = s.get("label")
        matches = match_index.get(label)
        if not matches:
            new_spans.append(s)
            continue

        b, e = s.get("begin"), s.get("end")
        if not (isinstance(b, int) and isinstance(e, int)):
            new_spans.append(s)
            continue

        best_match = next(
            (match for match in matches if match[0] <= b <= e <= match[1]),
            None,
        )
        if best_match is None:
            new_spans.append(s)
        else:
            nb, ne = best_match[0], best_match[1]
            if (nb, ne) == (b, e):
                new_spans.append(s)
            else:
                before = snapshot_span(s)
                s2 = dict(s)
                s2["begin"], s2["end"] = nb, ne
                s2["text"] = doc_text[nb:ne]
                log_span_change(
                    "extend_spans_to_regex",
                    before,
                    snapshot_span(s2),
                    extra=f" label={label}",
                    level=logging.INFO,
                )
                if regex_observer:
                    regex_index = best_match[2]
                    regex = best_match[3]
                    regex_observer(
                        label,
                        regex_rule_name(label, regex_index),
                        regex,
                        before,
                        snapshot_span(s2),
                    )
                new_spans.append(s2)

    return new_spans


def auto_add_patient_name_spans(
    spans: list[dict],
    text: str,
    patient_name: dict | None,
) -> list[dict]:
    """
    Add patient-name spans from metadata when they can be found in the text.

    Blacklisted connector tokens such as ``van`` or ``de`` are only allowed when
    they are adjacent to an accepted patient-name token or an existing
    ``Name:Patient`` span. This prevents isolated function words from being
    added as standalone names.
    """
    if not patient_name:
        return spans

    spans = list(spans)

    fam = (patient_name.get("family_name") or "").strip()
    given = (patient_name.get("given_name") or "").strip()

    if not fam and not given:
        return spans

    tokens = []
    if given:
        tokens.extend(given.split())
    if fam:
        tokens.append(fam)

    parts = [rf"\b{re.escape(t)}\b" for t in tokens]
    if not parts:
        return spans

    rx = re.compile("|".join(parts), re.IGNORECASE)

    # Existing spans
    existing_positions = {(s["begin"], s["end"]) for s in spans}
    existing_patient_spans = [s for s in spans if s.get("label") == "Name:Patient"]

    def containing_spans(b: int, e: int) -> list[dict]:
        """
        Return existing spans that strictly contain [b, e)
        (any label). Equal boundaries are *not* treated as substrings.
        """
        containers = []
        for s in spans:
            sb, se = s["begin"], s["end"]
            if sb <= b and e <= se and (sb, se) != (b, e):
                containers.append(s)
        return containers

    def contained_patient_spans(b: int, e: int) -> list[dict]:
        """Return existing patient spans that lie strictly inside [b, e)."""
        contained = []
        for s in spans:
            if s.get("label") != "Name:Patient":
                continue
            sb, se = s["begin"], s["end"]
            if b <= sb and se <= e and (sb, se) != (b, e):
                contained.append(s)
        return contained

    def overlaps_non_patient_span(b: int, e: int) -> bool:
        """Return True if [b, e) overlaps any existing non-patient annotation."""
        for s in spans:
            if s.get("label") == "Name:Patient":
                continue
            if b < s["end"] and s["begin"] < e:
                return True
        return False

    def rebuild_existing_caches() -> None:
        nonlocal existing_positions, existing_patient_spans
        existing_positions = {(s["begin"], s["end"]) for s in spans}
        existing_patient_spans = [s for s in spans if s.get("label") == "Name:Patient"]

    # --- Collect all matches first ---
    matches = []
    for m in rx.finditer(text):
        b, e = m.start(), m.end()
        tok = m.group(0)
        matches.append(
            {
                "m": m,
                "begin": b,
                "end": e,
                "token": tok,
                "token_l": tok.strip().lower(),
                "is_blacklisted": tok.strip().lower() in BLACKLIST_NAME_TOKENS,
            }
        )

    if not matches:
        return spans

    # --- Adjacency helpers (same logic as your merge function: whitespace-only gap) ---

    def whitespace_gap(a_begin: int, a_end: int, b_begin: int, b_end: int) -> bool:
        # returns True if [a, b] have only whitespace between them (in either order)
        if a_end <= b_begin:
            gap = text[a_end:b_begin]
        elif b_end <= a_begin:
            gap = text[b_end:a_begin]
        else:
            # overlapping or reversed; not "adjacent" in our sense
            return False
        return gap and gap.strip() == ""

    def adj_to_existing_patient(mi) -> bool:
        b, e = mi["begin"], mi["end"]
        for s in existing_patient_spans:
            if whitespace_gap(b, e, s["begin"], s["end"]):
                return True
        return False

    def adj_between_matches(i: int, j: int) -> bool:
        mi, mj = matches[i], matches[j]
        return whitespace_gap(mi["begin"], mi["end"], mj["begin"], mj["end"])

    # --- Decide which matches are allowed to become Name:Patient spans ---

    n = len(matches)
    allowed = [False] * n

    # 1) Non-blacklisted matches are always allowed
    for i, mi in enumerate(matches):
        if not mi["is_blacklisted"]:
            allowed[i] = True

    # 2) Any match (blacklisted or not) adjacent to an existing Name:Patient span is allowed
    for i, mi in enumerate(matches):
        if adj_to_existing_patient(mi):
            allowed[i] = True

    # 3) For blacklisted tokens, allow them if they are adjacent (whitespace) to any
    #    *allowed* match or existing patient span. Propagate until stable so chains
    #    like "de la Croix" work.
    changed = True
    while changed:
        changed = False
        for i, mi in enumerate(matches):
            if allowed[i]:
                continue  # already allowed
            if not mi["is_blacklisted"]:
                continue  # non-blacklisted already handled

            # Adjacent to existing Name:Patient?
            if adj_to_existing_patient(mi):
                allowed[i] = True
                changed = True
                continue

            # Adjacent to an allowed match?
            for j in range(n):
                if i == j or not allowed[j]:
                    continue
                if adj_between_matches(i, j):
                    allowed[i] = True
                    changed = True
                    break

    # --- Now actually add spans for allowed matches, respecting existing_positions/subspans ---

    # Keep original order in text
    for i, mi in enumerate(matches):
        if mi["is_blacklisted"] and not allowed[i]:
            # blacklisted and not connected to any Name:Patient -> skip
            continue

        b, e = mi["begin"], mi["end"]

        # Skip if identical [begin, end) already exists (any label)
        if (b, e) in existing_positions:
            continue

        containers = containing_spans(b, e)
        if containers:
            # Existing patient spans already covering the metadata match are at
            # least as recall-friendly as the candidate. Other labels should
            # block only this token, not later non-overlapping name parts.
            continue

        if overlaps_non_patient_span(b, e):
            continue

        contained_patients = contained_patient_spans(b, e)
        if contained_patients:
            spans = [s for s in spans if s not in contained_patients]
            rebuild_existing_caches()

        new_span = {
            "label": "Name:Patient",
            "begin": b,
            "end": e,
            "text": text[b:e],
            "Category": "Name",
            "Subtype": "Patient",
        }
        spans.append(new_span)

        # Update caches so later matches see this span as well
        existing_positions.add((b, e))
        if new_span["label"] == "Name:Patient":
            existing_patient_spans.append(new_span)

        logger.log(
            logging.INFO,
            "auto_add_patient_name_span\n[doc_id]\n%s\n%s\n",
            CURRENT_DOC_ID,
            span_repr(new_span),
        )

    spans.sort(key=lambda s: s["begin"])
    return spans


def post_process_spans(spans, text, metadata=None, *, regex_observer=None):
    """
    Apply the full post-processing pipeline to a document.

    Parameters
    ----------
    spans:
        Raw input spans. Each span is expected to contain ``label``, ``begin``,
        ``end``, and ``text``.
    text:
        Full document text the spans refer to.
    metadata:
        Optional document metadata. Currently ``metadata["patient_name"]`` is
        used to recover missing patient-name spans.
    regex_observer:
        Optional callback used by reporting wrappers. When provided, it is
        called for each regex-backed span correction with the label, rule name,
        regex object, before snapshot, and after snapshot.

    Returns
    -------
    list[dict]
        A new list of processed spans sorted deterministically by position.

    Notes
    -----
    The function is designed to be robust for imperfect upstream spans:
    it trims, extends, relabels, and deduplicates rather than assuming perfect
    offsets from the recognizer.
    """
    ordered_spans = sorted(spans, key=span_sort_key)
    new_spans = [
        normalize_span(span, text, idx) for idx, span in enumerate(ordered_spans)
    ]

    # 1) drop spans that are only non-alphanumeric
    new_spans = drop_non_alnum_spans(new_spans)

    # 2) automatically add patient name spans
    patient_name = metadata.get("patient_name") if metadata else None
    if patient_name:
        new_spans = auto_add_patient_name_spans(new_spans, text, patient_name)

    # 3) merge adjacent Name:Patient spans
    new_spans = merge_adjacent_name_patient(new_spans, text)

    # 4) regex-based extension (e.g. partial dates, URLs)
    new_spans = extend_spans_to_regex(
        new_spans,
        text,
        REGEX_BY_LABEL,
        regex_observer=regex_observer,
    )
    new_spans = deduplicate_spans(new_spans)
    new_spans = filter_invalid_date_spans(new_spans, text)

    # 5) weekday extension must consider the full span set to avoid order-sensitive overlaps
    new_spans = extend_dates_with_weekdays(new_spans, text)

    # 6) Date → Age_Birthdate when preceded by °/º
    new_spans = relabel_date_if_degree_prefix(new_spans, text)

    return deduplicate_spans(new_spans)
