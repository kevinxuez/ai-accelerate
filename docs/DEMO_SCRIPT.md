# Four-agent demo script

This walkthrough demonstrates the production architecture, typed artifacts, ordered handoffs,
and one visible failure. Allow about ten minutes.

## Preparation

Install the application, provision the pinned embedding model, and export a valid model key as
described in the root README. Opt into the synthetic providers for the local demonstration:

```bash
export CASEFILE_CALENDAR_PROVIDER=fixture
export CASEFILE_NSDA_PROVIDER=fixture
uvicorn casefile.api.main:app --reload
```

Verify `GET /health/ready` reports `ready`, `langgraph`, `in_memory`, the pinned embedding model,
and the two explicitly selected fixture providers. Open <http://127.0.0.1:8000>.

The right panel should show the active agent, response status, request/session IDs, ordered agent
events, tool calls grouped under the initiating agent, and model-call metadata.

## 1. Research confirmed evidence

Choose **Research confirmed evidence** and run the preset.

Point out:

- The exact submitted prompt remains visible.
- The Supervisor hands the request to the Evidence Librarian.
- The result is rendered from an `EvidencePacket`, not parsed from prose.
- Every card includes its complete citation, body, card ID, source file, retrieval score, and
  preserved read/emphasis spans.
- Provenance names the in-memory backend, pinned embedding model, ledger version, and
  confirmed-only policy.

## 2. Generate and revise an argument

Choose **Build or revise an argument** and run it. After the first result, submit:

> Revise the warrant and impact to make the internal link clearer. Preserve every citation.

Point out:

- Research and argument generation are separate Librarian and Strategist handoffs.
- All six `ArgumentDraft` sections display support badges.
- Unsupported facts are prominent and cannot cite cards.
- Every cited card ID resolves to a full confirmed source card below the draft.
- The second turn reuses versioned session state and returns a new typed draft.

## 3. Ingest a DOCX through the Librarian

Choose **Ingest a DOCX**, select a `.docx` evidence file, and run the preset.

Point out:

- The browser uploads an opaque attachment handle; the filesystem path is never inserted into
  the user's prompt.
- `needs_confirmation` is distinct from `needs_input`.
- The `IngestionPreview` shows every proposed card's full text, citation, side, evidence type,
  marked spans, model provenance, flags, explanations, and exclusion status.
- Nothing has entered the evidence ledger yet.

Select **Confirm evidence import**. Show the real `IngestionCommitResult`: written/searchable
counts, ledger version, retrieval-ready status, source filename, and job ID.

## 4. Practice with the Skills Coach

Choose **Practice with Skills Coach** and run it. Reply to the coach's question in the same
session.

Point out:

- Every `CoachTurn` is labeled **Simulated coach — not a human coach**.
- The Skills Coach may request evidence through the Supervisor and Librarian.
- Replies continue through versioned session state.
- No progress record is written automatically.

## 5. View progress and schedule human coaching

Choose **Progress and human coaching** and run it.

Point out:

- The Skills Coach returns a typed `ProgressSummary` subject to ownership rules.
- Scheduling returns to the Supervisor, its only tool owner.
- With the fixture provider selected, the resulting `CalendarEvent` is visibly synthetic.
- With the Google provider selected, a real write is staged and requires an explicit confirmation
  token before the provider is called.

## 6. Inspect a visible failure

Choose **Inspect a visible failure** and run it as `student-1`.

The request intentionally asks for another student's progress. Point out:

- The HTTP response is non-200 and `status` is `failed`.
- The UI displays `AUTHORIZATION_DENIED`, the stage, agent, tool, retryability, request ID, and
  session ID.
- Agent/tool/model traces collected before the failure remain visible.
- The UI does not replace the error with an empty state or infer failure from response wording.

## Closing summary

The demo has exercised all four agents, typed artifacts, specialist handoffs, in-memory semantic
retrieval, lossless evidence handling, versioned sessions, confirmation-gated writes, explicit
providers, grouped traces, and fail-closed error behavior through one LangGraph runtime.
