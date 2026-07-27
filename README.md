# CaseFile

CaseFile is a Public Forum debate evidence and coaching agent. It retrieves card-atomic
evidence with the citation physically attached, creates drills from durable coaching
records, enforces student/coach permissions in code, and stages every DOCX ingest behind a
human confirmation gate.

The app runs offline by default. Anthropic, Chroma, MCP, and Google Calendar are optional
depth layers; the deterministic ingestion, JSON retrieval fallback, role checks, mock
calendar, FastAPI app, and regression evals work without credentials.

## What is implemented

- Run-level OOXML ingestion with no text normalization and no `python-docx` dependency.
- Compact paragraph-index boundary pass with 18-paragraph windows and 4-paragraph overlap
  when an Anthropic key is configured; a measured deterministic fallback otherwise.
- Per-document and per-card marking detection. White highlight is discarded as paste noise;
  read and yellow-emphasis spans are stored as character offsets.
- Deterministic validation for coverage, duplicate assignment, invalid indices, missing
  citations, cite-count divergence, truncation, raw entities, fragmentation, duplicates,
  and incomplete cards.
- Source-hashed preview/confirm workflow. A changed source or failed validation cannot be
  confirmed. Corrupt, incomplete, and ungrounded paraphrase cards remain auditable but do
  not enter search.
- Resolution/side filtering before similarity search, with a relevance threshold and an
  explicit no-result response. Chroma uses stable local feature-hash embeddings, so the
  demo never downloads an embedding model; JSON cosine search is the fallback.
- Named LangGraph nodes (`receive_request`, `classify_intent`, `ask_clarification`,
  `route_on_intent`, `execute_tools`, `call_model`, `should_continue`) and an equivalent
  sequential runtime when LangGraph is not installed.
- Student and coach tools with both role-filtered availability and in-function checks.
- Atomic progress writes, argument/role/chunk-id audit logs, mock or Google Calendar events,
  FastAPI plus a small chat UI, and portable search/drill MCP tools.
- A 20-case golden regression set across citation faithfulness, routing/authorization, and
  evidence integrity, plus a 50-query candidate generator in the planned 20/15/10/5 mix.

## Important source finding

The supplied `background/Copy of Pro Cards - Crypto.docx` is not the document described by
the legacy ground truth:

| Artifact | Non-empty paragraphs | Cards | Source filename |
|---|---:|---:|---|
| Supplied DOCX | 26 | 8 | `Copy of Pro Cards - Crypto.docx` |
| Legacy benchmark | 124 (per plan) | 36 | `Pro_Cards_-_Crypto.docx` |

The supplied document was rendered and inspected across all four pages, then labeled in
`casefile/ingest/ground_truth_sample.json`. The offline boundary pass scores 8/8 exact
boundaries on that file. The legacy 36-card labels remain in
`casefile/ingest/ground_truth_crypto.json`, but their matching DOCX is absent, so the claimed
83% baseline and any LLM improvement over it cannot honestly be rerun here. The scorer now
rejects mismatched source filenames instead of emitting a misleading score.

## Quick start

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

Install all extension dependencies when you want the live LLM, LangGraph, Chroma, MCP, and
Google Calendar paths:

```bash
python -m pip install -e '.[all,dev]'
cp .env.example .env
```

Environment variables are read from the shell; `.env` is a template, not automatically
loaded. Never commit API keys, OAuth credentials, or tokens.

## Ingest the sample

Preview with the deterministic fallback:

```bash
python -m casefile.ingest.pipeline \
  'background/Copy of Pro Cards - Crypto.docx' \
  --resolution 2026-09-CRYPTO \
  --side pro \
  --no-model
```

Review the counts and flags, then use the printed token:

```bash
python -m casefile.ingest.pipeline --confirm YOUR_32_CHARACTER_TOKEN
```

With `ANTHROPIC_API_KEY` set, omit `--no-model`. The boundary model sees only compact
paragraph metadata and 60-character previews and returns indices. The field model may read
one card but can return labels only; original text and spans always come from the OOXML.

Useful diagnostics:

```bash
python -m casefile.ingest.serialize_index 'background/Copy of Pro Cards - Crypto.docx'
python -m casefile.ingest.boundary_pass --no-model 'background/Copy of Pro Cards - Crypto.docx' > pred.json
python -m casefile.ingest.score_boundaries pred.json casefile/ingest/ground_truth_sample.json
python -m casefile.ingest.build_index
```

## Run the agent

CLI:

```bash
casefile --role student --user-id student-1 --resolution 2026-09-CRYPTO
```

API and browser UI:

```bash
uvicorn casefile.api.main:app --reload
```

Open `http://127.0.0.1:8000`. The JSON endpoint is `POST /chat`; ingestion uses
`POST /ingest/preview` followed by `POST /ingest/confirm`.

Example requests:

```text
What Pro evidence do we have about regulation?
Give me a summary drill for the Pro side.
Show progress for student-2.
Write my final focus speech for me.
```

The third request is denied for a student other than `student-2`. The fourth hits the
evidence-integrity boundary: CaseFile retrieves evidence and creates drills but does not
write competition speeches, fabricate citations, or repair evidence text.

A coach can log an assessment and book the follow-up in one turn:

```text
Log assessment for student-1: needs cleaner collapse. weaknesses: collapse; and schedule a session at 2026-08-04T16:00.
```

## Rules corpus

No authoritative NSDA rule documents were supplied, so CaseFile deliberately does not ship
invented or silently stale rule text. Until current, authorized Markdown sources are added
under `casefile/rules/`, rule questions fail closed rather than answer from model memory.

Preserve section numbers as Markdown headings, record the document edition/effective date
and canonical source URL, then run `python -m casefile.ingest.build_index`. Rule responses
return the document, section number, title, and exact indexed text.

## Personas and authorization

| Tool | Student | Coach |
|---|:---:|:---:|
| Search cards/rules | Yes | Yes |
| Generate own/any drill | Own | Any |
| Read progress | Own | Any |
| Log assessment | No | Yes |
| Ingest cards | No | Yes |
| Schedule session | Own | Any |

Checks live in `casefile/agent/tools.py`; prompt instructions are not treated as access
control. Denials are structured strings such as:

```text
[DENIED] role 'student' cannot read progress records for another student.
```

## MCP

Only retrieval and drill generation are exposed because they are the capabilities another
client would plausibly discover. Progress, ingestion, and calendar writes stay internal.

```bash
python -m casefile.mcp.server
python -m casefile.mcp.client_demo
```

Both MCP tools require explicit role, user, and resolution context and re-run the same
in-function authorization checks as the API.

## Calendar

`MOCK_CALENDAR=true` is the default and writes local demo events to
`casefile/data/calendar_events.json`. For Google Calendar:

1. Create an installed-app OAuth client in Google Cloud and save it outside version control.
2. Install the `calendar` or `all` extra.
3. Set `GOOGLE_CALENDAR_CREDENTIALS`, optionally set `GOOGLE_CALENDAR_TOKEN`, and set
   `MOCK_CALENDAR=false`.
4. Run a scheduling request and complete the local OAuth flow.

The integration requests only `https://www.googleapis.com/auth/calendar.events`. Verify the
current Google consent-screen and token-lifetime requirements before relying on a stage
demo; the committed mock remains the network-failure path.

## Evaluation

```bash
python -m casefile.evals.generate_testset
python -m casefile.evals.run_eval
python -m casefile.evals.run_eval --llm-judge
```

The committed `casefile/evals/eval_results.json` is a real deterministic regression run over
synthetic fixtures: 20/20 cases, with 5.0/5.0 in each dimension. It is a code-path regression
result, not a claim about the missing 36-card corpus or real-world answer quality. With an
Anthropic key, `--llm-judge` replaces the deterministic scores for citation faithfulness and
evidence integrity while keeping routing/authorization deterministic.

## Repository map

```text
casefile/
  ingest/       OOXML serializer, boundary/field passes, spans, validators, benchmarks
  rules/        authorized Markdown rule sources (empty/fail-closed by default)
  chroma_db/    optional persistent Chroma collections
  agent/        state, nodes, graph, prompts, roles, and tools
  api/          FastAPI backend and demo chat UI
  mcp/          stdio server and client demo
  evals/        candidate generation, golden set, rubric, judge, cached results
  data/         card ledger and progress records
background/     original plan, context, sample, and reference prototypes
tests/          ingestion, retrieval, authorization, agent, audit, and eval regressions
```

## Guardrails and known limits

- A resolution argument is strongly preferred. Filename inference is marked low confidence.
- Unknown-side cards do not leak into a Pro/Con query; supply `--side` when the file does not
  encode its side.
- Corrupt, incomplete, or paraphrased-without-source cards are retained for review but are
  excluded from search.
- The offline boundary fallback is strong on the supplied conventional 8-card file; it does
  not replace the planned LLM judgment test on the missing 36-card benchmark.
- Multi-user authentication and tournament/local-circuit rule variations remain out of
  scope. Role is caller-provided session context, exactly as scoped in the plan.
- CaseFile does not cut new evidence, repair source text, fabricate cards/citations, or write
  competition speeches.
