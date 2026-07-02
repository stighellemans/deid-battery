"""deduce / belgian-deduce runner, with optional per-document METADATA
(patient + caregiver names, addresses, document date) injected as known
identifiers. Requires the corresponding package (``deduce`` or ``belgian_deduce``).

config block (battery.yaml)::

    - id: belgian-deduce
      runner: deduce
      params: {engine: belgian-deduce}   # or engine: deduce

Metadata is taken from doc["_meta"] (resolved by the orchestrator from the
configured metadata.source).
"""
from __future__ import annotations

import logging

from ..checkpoint import ordered, resume_split, track
from ..schema import make_span

# belgian_deduce calls logging.basicConfig(level=DEBUG) on import; keep it quiet.
logging.disable(logging.INFO)

# Plain deduce (Dutch tags, after its PersonAnnotationConverter). Generic detected
# persons -> Name:Caregiver (the common case in clinical notes).
DEDUCE_MAP = {
    "patient": "Name:Patient", "persoon": "Name:Caregiver", "datum": "Date",
    "leeftijd": "Age_Birthdate", "telefoonnummer": "Contactdetails",
    "emailadres": "Contactdetails", "url": "Contactdetails", "id": "ID:Patient",
    "bsn": "ID:Patient", "locatie": "Address_Location:Other", "straat": "Address_Location:Other",
    "ziekenhuis": "Organization:Healthcare", "zorginstelling": "Organization:Healthcare",
    "instelling": "Organization:Healthcare",
}
# belgian-deduce (English tags).
BELGIAN_MAP = {
    "patient": "Name:Patient", "person": "Name:Caregiver", "age": "Age_Birthdate",
    "date": "Date", "email": "Contactdetails", "phone_number": "Contactdetails",
    "url": "Contactdetails", "id": "ID:Patient", "national_register_number": "ID:Patient",
    "hospital": "Organization:Healthcare", "healthcare_institution": "Organization:Healthcare",
    "location": "Address_Location:Other", "street": "Address_Location:Other",
}


def _date_variants(value) -> list[str]:
    """Common written forms of a date, so a metadata document-date matches the
    text regardless of the source format. Returns [original] if it can't parse."""
    from datetime import datetime
    s = str(value).strip()
    d = None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            d = datetime.strptime(s, fmt)
            break
        except ValueError:
            continue
    if d is None:
        return [s]
    months = ["januari", "februari", "maart", "april", "mei", "juni", "juli",
              "augustus", "september", "oktober", "november", "december"]
    out = [s]
    for f in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d"):
        out.append(d.strftime(f))
    out += [f"{d.day}/{d.month}/{d.year}", f"{d.day}-{d.month}-{d.year}",
            f"{d.day}.{d.month}.{d.year}", f"{d.day} {months[d.month - 1]} {d.year}"]
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _doc_to_spans(document, text, label_map):
    out = []
    for ann in document.annotations:
        label = label_map.get(str(ann.tag))
        if not label:
            continue
        b, e = int(ann.start_char), int(ann.end_char)
        if e > b:
            out.append(make_span(b, e, label, text[b:e]))
    out.sort(key=lambda s: (s["begin"], s["end"]))
    return out


def _deduce_metadata(meta, engine, classes):
    """Map canonical metadata -> the engine's deidentify(metadata=...) dict."""
    if not meta:
        return {}
    Person = classes["Person"]
    kw = {}
    p = meta.get("patient")
    if p:
        if engine == "deduce":  # plain deduce: patient name only
            kw["patient"] = Person(
                first_names=p.get("first_names") or None,
                initials=p.get("initials") or None,
                surname=p.get("surname") or None,
            )
        else:
            person_kw = dict(first_names=p.get("first_names") or None,
                             initials=p.get("initials") or None,
                             surname=p.get("surname") or None)
            if p.get("aliases"):
                person_kw["aliases"] = p["aliases"]
            if p.get("birth_date"):
                person_kw["birth_date"] = p["birth_date"]
            if p.get("addresses") and classes.get("Address"):
                person_kw["addresses"] = [classes["Address"](**a) for a in p["addresses"]]
            kw["patient"] = Person(**person_kw)
    if engine != "deduce" and meta.get("caregivers"):
        kw["persons"] = [Person(first_names=c.get("first_names") or None,
                                surname=c.get("surname") or None,
                                aliases=c.get("aliases") or None)
                         for c in meta["caregivers"]]
    if engine != "deduce" and meta.get("document_date") and classes.get("MetadataEntity"):
        v = _date_variants(meta["document_date"])
        kw["entities"] = [classes["MetadataEntity"](text=v[0], tag="date", variants=v[1:] or None)]
    return kw


def run(docs, params) -> dict[str, list[dict]]:
    ckpt, todo, by_doc = resume_split(docs, params)
    if not todo:  # everything already checkpointed -> don't construct the engine
        return ordered(docs, by_doc)
    engine = (params or {}).get("engine", "belgian-deduce")
    if engine == "deduce":
        from deduce import Deduce
        from deduce.person import Person
        classes, label_map = {"Person": Person}, DEDUCE_MAP
    elif engine == "belgian-deduce":
        from belgian_deduce import Address, Deduce, MetadataEntity, Person
        classes = {"Person": Person, "Address": Address, "MetadataEntity": MetadataEntity}
        label_map = BELGIAN_MAP
    else:
        raise ValueError(f"deduce runner: unknown engine {engine!r}")

    model = Deduce()
    for d in track(todo, params):
        text = d["text"]
        md = _deduce_metadata(d.get("_meta") or {}, engine, classes)
        document = model.deidentify(text, metadata=md) if md else model.deidentify(text)
        spans = _doc_to_spans(document, text, label_map)
        by_doc[d["doc_id"]] = spans
        ckpt.record(d["doc_id"], spans)
    return ordered(docs, by_doc)
