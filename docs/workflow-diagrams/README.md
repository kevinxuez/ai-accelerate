# Four-agent workflow diagrams

These Mermaid sources document the production runtime:

1. [`01-four-agent-architecture.mmd`](src/01-four-agent-architecture.mmd) — ownership,
   delegation, tools, and storage.
2. [`02-request-and-error-flow.mmd`](src/02-request-and-error-flow.mmd) — the only request path,
   including clarification, confirmation, completion, and typed failure.
3. [`03-ingestion-confirmation.mmd`](src/03-ingestion-confirmation.mmd) — lossless DOCX preview,
   atomic ledger commit and direct retrieval.
4. [`04-session-and-traces.mmd`](src/04-session-and-traces.mmd) — versioned continuation and
   observable traces.
5. [`05-extended-langgraph-runtime.mmd`](src/05-extended-langgraph-runtime.mmd) — complete
   LangGraph routing, specialist operations, confirmation loops, failures, and persistence.
6. [`06-langgraph-slide.svg`](06-langgraph-slide.svg) — slide-ready 16:9 overview of the
   supervisor loop, typed specialist handoffs, explicit outcomes, and shared state. A
   1920×1080 [`PNG`](06-langgraph-slide.png) is included for direct use in slides.
7. [`07-evals-slide.svg`](07-evals-slide.svg) — slide-ready evaluation scorecard showing the
   curated challenge set, retrieval quality signals, the abstention gap, and system-wide test
   coverage. A 1920×1080 [`PNG`](07-evals-slide.png) is included for direct use in slides.
8. [`08-agent-tools-slide.svg`](08-agent-tools-slide.svg) — slide-ready map of each agent's main
   responsibility, scoped tools, primary outputs, and the shared registry guardrails. A
   1920×1080 [`PNG`](08-agent-tools-slide.png) is included for direct use in slides.

Render a source with Mermaid CLI if desired:

```bash
npx --yes @mermaid-js/mermaid-cli \
  -c docs/workflow-diagrams/mermaid-config.json \
  -i docs/workflow-diagrams/src/01-four-agent-architecture.mmd \
  -o four-agent-architecture.svg
```

Generated images are intentionally not committed; the Mermaid source is authoritative.
