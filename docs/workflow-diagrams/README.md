# Four-agent workflow diagrams

These Mermaid sources document the production runtime:

1. [`01-four-agent-architecture.mmd`](src/01-four-agent-architecture.mmd) — ownership,
   delegation, tools, and storage.
2. [`02-request-and-error-flow.mmd`](src/02-request-and-error-flow.mmd) — the only request path,
   including clarification, confirmation, completion, and typed failure.
3. [`03-ingestion-confirmation.mmd`](src/03-ingestion-confirmation.mmd) — lossless DOCX preview,
   atomic commit, and Chroma rebuild.
4. [`04-session-and-traces.mmd`](src/04-session-and-traces.mmd) — versioned continuation and
   observable traces.

Render a source with Mermaid CLI if desired:

```bash
npx --yes @mermaid-js/mermaid-cli \
  -c docs/workflow-diagrams/mermaid-config.json \
  -i docs/workflow-diagrams/src/01-four-agent-architecture.mmd \
  -o four-agent-architecture.svg
```

Generated images are intentionally not committed; the Mermaid source is authoritative.
