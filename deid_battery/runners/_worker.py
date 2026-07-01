"""Subprocess entrypoint to run one runner under a different interpreter/venv
(e.g. the transformers>=5.7 environment for the privacy filters).

    python -m deid_battery.runners._worker <runner> <params.json> <docs.jsonl> <out.jsonl>
"""
from __future__ import annotations

import json
import sys

from deid_battery.runners import get_runner
from deid_battery.schema import read_jsonl, write_by_doc


def main():
    runner, params_path, docs_path, out_path = sys.argv[1:5]
    params = json.loads(open(params_path, encoding="utf-8").read())
    docs = read_jsonl(docs_path)  # each: {doc_id, text, _meta}
    write_by_doc(out_path, get_runner(runner)(docs, params))


if __name__ == "__main__":
    main()
