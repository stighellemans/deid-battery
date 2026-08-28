# Paper reproducibility

`deid-battery` is the research harness used for the MedDeID manuscript. It is
published separately from the production MedDeID suite because it coordinates
paper-specific comparator systems, evaluation, and figures rather than serving
as production clinical software.

## Qwen3-8B comparator

The complete prompt materials are committed under `prompts/`:

- `dict_prompt.txt`: the Dutch prompt template, split into system and user
  messages at `Originele tekst:`;
- `labels.csv`: the allowed labels and their Dutch definitions;
- `dict_example.txt`: the two fixed few-shot examples.

`deid_battery/runners/llm.py` loads these files, constructs the request, removes
Qwen reasoning blocks before parsing, recovers complete items from truncated
JSON where possible, and maps returned substrings back to character offsets.
`configs/battery.yaml` records the evaluated generation settings: temperature
0.6, top-p 0.95, thinking enabled, an 8,000-token output limit, and two workers.

The paper run used Qwen3-8B with Q4_K_M GGUF quantisation through a local Ollama
endpoint backed by llama.cpp. The Ollama context length was 16,384 tokens. The
model tag alone does not establish the loaded quantisation, so verify Q4_K_M in
the local Ollama installation before rerunning the benchmark.

When `thinking: true`, the runner asks for JSON in the prompt but does not send a
strict response schema because that schema would prevent Qwen's reasoning block.
When `thinking: false`, it appends `/no_think` and sends the JSON schema supported
by OpenAI-compatible endpoints.

## Repository boundaries

The standard checkout has the following code dependencies:

- `deid-schema` is required at runtime for canonical span and taxonomy helpers.
  The paper source lock pins commit
  `cfa99eb04e7884a13e05df27f942fa03855f4209`.
- The post-processing implementation is vendored under
  `deid_battery/_vendor/post_process`; no separate `post-process` checkout is
  needed for the paper run.
- The quantitative evaluation and plotting modules used by the paper are
  vendored under `deid_battery/_vendor`; their source commits and file hashes are
  recorded in `deid_battery/_vendor/VENDORED.json`. The source repositories are
  not runtime dependencies.
- `span-annotations` is needed only when optional INCEpTION-token normalization
  is enabled. That option is disabled in the paper configuration and is not a
  dependency of the Qwen runner.
- `belgian-deduce` is needed only for the Belgian DEDUCE comparator. The source
  lock records the comparator commit used by the study.
- Other comparator libraries and model revisions are declared in
  `requirements/`, `configs/battery.yaml`, and
  `deployment/hospital-source-lock.json`.

The repository deliberately excludes clinical text, gold annotations,
per-document predictions, model weights, logs, and local timing files. These
paths are covered by `.gitignore`. Public synthetic data can be substituted for
the restricted hospital inputs when checking the execution path.
