# Orchestration and Workflow Justification

CaseFile operates in a trust-sensitive domain: it handles evidence, citations, student records, document ingestion, and external writes. Its architecture therefore prioritizes provenance, controlled delegation, and visible failure over maximum agent autonomy.

## Core orchestration decisions

| Decision | Justification | Tradeoff |
| --- | --- | --- |
| One Supervisor owns routing | Gives the conversation one authority for sequencing, clarification, completion, and scheduling. Specialists cannot independently create uncontrolled agent loops. | The Supervisor is a bottleneck and adds a model call after most specialist actions. |
| Three domain specialists | Evidence management, argument construction, and coaching have meaningfully different permissions and validation rules. Separating them reduces prompt complexity and limits each agent's authority. | More contracts and handoff code than a single-agent implementation. |
| One compiled LangGraph runtime | API, CLI, browser, and MCP behavior cannot drift into separate orchestration implementations. The production graph is explicit and inspectable. | LangGraph becomes a required dependency rather than an optional enhancement. |
| Specialists always return to the Supervisor | Allows multi-stage workflows—such as Coach → evidence request → Librarian → Coach—without permitting peer-to-peer delegation or ambiguous ownership. | Extra routing latency, especially for simple requests. |
| Typed handoffs and artifacts | Converts model output into validated application state. Invalid agent decisions fail before tools or downstream agents act. | Schemas must evolve whenever capabilities change. |
| Agent-scoped tool registry | Enforces least privilege in code: only the Librarian retrieves or ingests, only the Coach accesses progress, and only the Supervisor schedules. | Some otherwise harmless direct workflows require another handoff. |
| Explicit human confirmation for writes | Ingestion, assessments, and real calendar events have consequences that should not follow from probabilistic model intent alone. | Workflows become multi-turn and require persisted pending state. |
| Fail closed, with no synthetic fallback | Fabricated evidence or a plausible-looking substitute argument would be worse than an explicit failure. Stable error codes also make failures operationally actionable. | The system can feel less helpful during outages or malformed model output. |
| Versioned, bounded session state | Supports revision, coaching continuity, clarification, and confirmation while preventing unbounded histories and incompatible state loading. | Requires schema migration planning and careful state-size management. |
| Deterministic checks around model judgment | Models handle semantic tasks—routing, document boundaries, labels, and drafting—while code enforces identities, citations, sides, hashes, permissions, and limits. | More implementation work, but materially better integrity. |

These choices are enforced in the graph rather than merely described in prompts. The main implementation points are [`casefile/agents/graph.py`](casefile/agents/graph.py), [`casefile/agents/contracts.py`](casefile/agents/contracts.py), and [`casefile/tools/registry.py`](casefile/tools/registry.py).

## Why the four-agent split makes sense

The division follows data ownership and risk boundaries:

- The **Supervisor** owns user intent, sequencing, clarification, and the only outward-facing scheduling capability.
- The **Evidence Librarian** owns information provenance: confirmed evidence, rules, topics, and ingestion.
- The **Argument Strategist** is deliberately downstream of an `EvidencePacket`. It cannot retrieve independently, and its citations are checked against the exact supplied card IDs in [`casefile/agents/argument_strategist.py`](casefile/agents/argument_strategist.py).
- The **Skills Coach** owns practice and progress workflows, but requests evidence through the Supervisor instead of bypassing the Librarian.

This split is stronger than dividing agents only by user-facing features. It aligns authority with the artifacts each agent is permitted to produce.

## Workflow justification

### Research and argument construction

```text
User → Supervisor → Librarian → Supervisor → Strategist → Supervisor → User
```

Retrieval and writing are intentionally separate. The Librarian produces a provenance-bearing `EvidencePacket`; the Strategist transforms only that packet into an argument. This makes it possible to prove that every argument citation came from confirmed retrieval and to represent evidence gaps explicitly instead of inventing support.

### Evidence-aware coaching

```text
Supervisor → Coach
                  └─ needs evidence → Supervisor → Librarian
                                                   ↓
Supervisor ← EvidencePacket ←──────────────────────┘
     ↓
   Coach → DrillPlan or CoachTurn
```

The Coach identifies its learning need, but the Librarian remains the evidence authority. This prevents coaching logic from silently broadening a search, accessing unconfirmed documents, or fabricating factual support. The orchestration is implemented in [`casefile/agents/graph.py`](casefile/agents/graph.py).

### Document ingestion

```text
Upload → safe extraction → document screening
       → model boundary detection and labeling
       → deterministic validation
       → complete preview
       → explicit confirmation
       → hash verification → atomic ledger commit
```

A model is useful for interpreting irregular debate-card formatting, but it should not directly mutate the evidence ledger. The confirmation token, source-hash verification, atomic write, and token retirement turn an uncertain extraction process into a controlled transaction. See [`casefile/ingest/pipeline.py`](casefile/ingest/pipeline.py) and [`casefile/ingest/commit.py`](casefile/ingest/commit.py).

### Clarification and confirmation

`needs_input` and `needs_confirmation` are separate states because they represent different conditions:

- `needs_input`: the workflow lacks required information.
- `needs_confirmation`: the system has enough information but requires authorization before a consequential action.

That distinction is encoded as a state invariant in [`casefile/agents/state.py`](casefile/agents/state.py), making it reliable for API and UI consumers instead of forcing them to infer status from prose.

### Errors and observability

Model, tool, provider, validation, authorization, and storage failures become typed failed states and non-200 API responses. Agent, tool, and model traces retain the path leading to failure. This is important for debugging multi-agent systems because conversational text alone cannot reliably explain which layer failed.

## Important limitations

The architecture is defensible, but several decisions deserve refinement:

- The Supervisor adds latency and cost because all specialist results route back through it. A future optimization could deterministically finish simple, single-artifact requests without weakening multi-stage workflows.
- `step_count` persists with session state. As currently implemented, a sufficiently long valid session can eventually reach the 64-step limit across separate requests. A per-turn counter plus a separate session-level bound would be clearer.
- Confirmation through chat still depends on the Supervisor correctly interpreting the user's confirmation message. Explicit confirmation endpoints are stronger and should remain available for consequential integrations.
- Filesystem session and ledger stores are appropriate for a local demonstration, but a multi-process deployment would require transactional shared storage and stronger concurrency control.
- The project does not currently contain formal architecture decision records comparing alternatives such as a single tool-calling agent, direct specialist-to-specialist routing, or deterministic routing. The code demonstrates the chosen approach, but documenting rejected alternatives would make the justification easier to defend.

## Conclusion

The orchestration is intentionally conservative: probabilistic components decide meaning, while deterministic code controls authority, state transitions, evidence integrity, and writes. That is the right bias for this application.

The current test suite contains 96 passing tests covering delegation, confirmation, session persistence, prompt-injection handling, tool authorization, and failure-without-fallback behavior.
