# CaseFile agent workflows

These diagrams describe the code that runs today. They distinguish the agent's
LangGraph orchestration from direct API ingestion endpoints and direct MCP tool calls.
When LangGraph is not installed, `CaseFileAgent` executes the same nodes in the same order
with its sequential fallback.

## 1. Shared request lifecycle

Every `/chat` request goes through this graph before a domain tool is called.

```mermaid
flowchart TD
    A[POST /chat] --> B{Pending session?}
    B -- yes --> C{New intent or stop coaching?}
    C -- stop --> Z[Clear session and end coaching]
    C -- changed task --> D[Clear old pending task]
    C -- reply to same task --> E[Append reply to original request]
    B -- no --> F[receive_request]
    D --> F
    E --> F
    F --> G[screen_request]
    G --> H{Security action}
    H -- block --> I[block_prompt_injection]
    H -- allow --> J[classify_intent]
    J --> K{Missing or ambiguous input?}
    K -- yes --> L[ask_clarification]
    L --> M[Save session-bound pending task]
    K -- no --> N[route_on_intent]
    N --> O[Build strict task DAG]
    O --> P[execute_tools]
    P --> Q[observe_results]
    Q -- one bounded evidence retry --> P
    Q -- finish --> R[call_model / grounded renderer]
    R --> S[should_continue]
    S --> T[Return response, traces, observations, and structured data]
    I --> T
    M --> T
```

The security decision is deterministic. The optional model classifier is used only when
the deterministic classifier still has a general, read-only ambiguity; it cannot create a
write intent or override a block. Clarifications are limited to six turns and are bound to
the original user and resolution.

## 2. Find evidence and format an argument

Intent: `retrieve_evidence`  
Task action: `search_cards`

```mermaid
flowchart TD
    A[Evidence request] --> B{Pro or Con present?}
    B -- no --> C[Ask for side and save pending task]
    B -- yes --> D[Filter index by exact resolution and side]
    D --> E[Rank intact cards by similarity]
    E --> F{Any cards above threshold?}
    F -- no --> G[Remove request boilerplate and retry once]
    G --> H[Discard retry results without explicit topic overlap]
    H --> I{Grounded results?}
    F -- yes --> J[Build structured grounding cards]
    I -- no --> K[Return explicit no-card answer]
    I -- yes --> J
    J --> L[Full citation, tag, body, read spans, emphasis spans]
    L --> M{Anthropic configured and request safe?}
    M -- yes --> N[Send bounded marked excerpts to argument formatter]
    N --> O{Strict schema and cited headers valid?}
    O -- yes --> P[Claim + warrant + impact]
    O -- no --> Q[Deterministic grounded outline]
    M -- no --> Q
    P --> R[Render argument and every retrieved card]
    Q --> R
    R --> S[UI marks read and emphasized source ranges]
```

The model receives only bounded source excerpts and card metadata. It may organize those
sources into an argument, but it cannot repair evidence, invent a citation, or add outside
facts. Citation headers in its structured output must be a subset of the retrieved cards.
The browser renders the full original card bodies separately from generated analysis.

## 3. Explain a rule

Intent: `explain_rule`  
Task action: `search_rules`

```mermaid
flowchart LR
    A[Rule question] --> B[Screen untrusted question]
    B --> C[Search only indexed, non-quarantined rule chunks]
    C --> D{Authoritative chunks found?}
    D -- yes --> E[Return document, section number, heading, and exact text]
    D -- no --> F[Fail closed; do not answer from model memory]
```

There is currently no bundled authoritative rules corpus, so the normal result is the
fail-closed response until approved rule documents are indexed.

## 4. Generate a drill

Intent: `generate_drill`  
Task action: `generate_drill`

```mermaid
flowchart TD
    A[Drill request] --> B{Side supplied?}
    B -- no --> C[Ask for Pro or Con]
    B -- yes --> D[Use named speech position or general]
    D --> E[Read the student's latest progress record]
    E --> F{Weakness tags present?}
    F -- yes --> G[Use weakness tags as retrieval query]
    F -- no --> H[Use speech-position evidence query]
    G --> I[Retrieve up to three cards for side and resolution]
    H --> I
    I --> J[Build deterministic practice instructions]
    J --> K[Return drill plus cited card references]
```

Drill creation does not generate a competition speech. It creates a practice task that
requires the student to supply the debating.

## 5. Simulated coach

Intent: `coach_simulation`  
Task action: `coach_simulation`

```mermaid
flowchart TD
    A[Start or continue coaching] --> B{Side and speech focus supplied?}
    B -- no --> C[Ask a session-bound clarification]
    B -- yes --> D[Read student's latest progress]
    D --> E[Choose weakness tag or requested speech focus]
    E --> F[Retrieve two cards for the active side and resolution]
    F --> G[Extract bounded read and emphasis excerpts from both cards]
    G --> H{Anthropic configured and safe?}
    H -- yes --> I[Generate one schema-validated feedback turn and question]
    H -- no --> J[Use deterministic coaching turn]
    I --> K[Return labeled simulated-coach turn]
    J --> K
    K --> L[Render both full cards with citation and marked source text]
    L --> M[Save coaching session for next reply]
    M --> N{Next message}
    N -- student reply --> D
    N -- end coaching --> O[Clear session]
    N -- twelve-turn limit --> O
```

The coach is an agent behavior inside the student's session, not a second frontend persona.
It receives bounded marked excerpts from both retrieved cards, asks one question at a time,
and never writes the student's speech.

## 6. Check progress

Intent: `progress`  
Task action: `progress`

```mermaid
flowchart LR
    A[Progress request] --> B[Resolve requested student ID]
    B --> C[Enforce ownership in tool code]
    C -- own record --> D[Read progress.json]
    C -- another student's record --> E[Return structured denial]
    D --> F[Return matching records or no-record result]
```

The current browser uses one student identity. A legacy trusted `coach` caller can write an
assessment through the internal tool/API path, but that is not a separate browser persona
and is not exposed over MCP.

## 7. Attach and ingest evidence

The browser attachment workflow begins with the agent for request screening and intent
routing, then uses the ingestion service directly for the file lifecycle.

```mermaid
flowchart TD
    A[Attach DOCX + import prompt] --> B[POST /chat/with-attachment]
    B --> C[Agent security screen and ingest-intent check]
    C --> D{Allowed import request?}
    D -- no --> E[Reject without parsing or indexing]
    D -- yes --> F[Validate filename, size, ZIP structure, and document.xml]
    F --> G[Stage upload under randomized private path]
    G --> H[Serialize original OOXML paragraphs and markings]
    H --> I[Screen document and cards for indirect injection]
    I --> J[Boundary detection: optional model or deterministic fallback]
    J --> K[Field labeling: optional model or deterministic fallback]
    K --> L[Reconstruct citation, tag, body, read spans, emphasis spans]
    L --> M[Run deterministic validation and source hash]
    M --> N[Return preview and single-use confirmation token]
    N --> O{User confirms?}
    O -- no --> P[Nothing enters retrieval]
    O -- yes --> Q[POST /ingest/confirm]
    Q --> R[Recheck token, caller context, validation, and source hash]
    R --> S[Index valid non-quarantined cards atomically]
    R --> T[Keep risky cards quarantined and out of search]
```

The original text is the source of truth. Model passes return indices or labels only; they
do not rewrite the document. The preview/confirm gate is why attaching a file does not
immediately modify the evidence index.

## 8. Schedule a session

Intent: `schedule_session`  
Task action: `schedule_session`

```mermaid
flowchart TD
    A[Scheduling request] --> B{Start time supplied?}
    B -- no --> C[Ask for ISO start time]
    B -- yes --> D[Validate owner, time, duration, timezone, and attendee]
    D --> E{MOCK_CALENDAR}
    E -- true --> F[Write local calendar_events.json immediately]
    E -- false --> G[Stage Google event and return confirmation token]
    G --> H[Explicit confirmation request]
    H --> I[Verify single-use token belongs to caller]
    I --> J[Create Google Calendar event]
    F --> K[Audit and return event]
    J --> K
```

The mock is a local integration substitute for demos; it does not contact a calendar
provider. Real Google Calendar writes require the extra confirmation turn.

## 9. Compound requests and replanning

```mermaid
flowchart TD
    A[Request names multiple supported actions] --> B[Build validated task DAG]
    B --> C{Ready task wave}
    C --> D[Run independent read tasks in parallel]
    C --> E[Run writes and policy tasks serially]
    D --> F[Record status, attempts, dependencies, and safe trace args]
    E --> F
    F --> G{Dependency failed?}
    G -- yes --> H[Skip dependent task]
    G -- no --> I[Render each result section]
    H --> I
```

Read tasks may retry transient failures within their task limit. Write and policy tasks get
one attempt. When assessment logging and scheduling appear together, scheduling depends on
the assessment result.

## 10. External MCP workflow

MCP is a direct tool surface, not a remote entry point into the complete chat graph.

```mermaid
flowchart LR
    A[External MCP client] --> B[FastMCP stdio server]
    B --> C[Per-user rate limit]
    C --> D{Discovered tool}
    D --> E[search_cards]
    D --> F[search_rules]
    D --> G[generate_drill]
    D --> H[get_progress]
    E --> I[Strict validation, security screening, and authorization]
    F --> I
    G --> I
    H --> I
    I --> J[Direct structured tool result]
```

Because MCP calls tools directly, they do not use chat intent classification, clarification
sessions, the empty-search replan, simulated coaching, or the AI argument formatter.
Ingestion, quarantine approval, assessment writes, and calendar writes remain internal.

## 11. Synthetic NSDA provider

The NSDA integration is a read-only provider boundary for development and demos. The
bundled data is fictional and does not enter the authoritative rules index.

```mermaid
flowchart TD
    A[CaseFile consumer or external test client] --> B{Entry surface}
    B -- Mock REST API --> C[GET /mock/nsda/v1 resource]
    B -- Provider adapter --> D{NSDA_BASE_URL configured?}
    D -- no --> E[Read validated bundled mock dataset]
    D -- yes --> F[Call compatible HTTPS provider with timeout and optional bearer token]
    C --> G[Validate query parameters]
    G --> E
    F --> H[Validate response envelope]
    H --> I{Requested resource}
    E --> I
    I --> J[Metadata]
    I --> K[Current topic by event and date]
    I --> L[Search synthetic rule fixtures]
    I --> M[Filter tournaments]
    I --> N[Look up synthetic member eligibility]
    J --> O[Return versioned response envelope]
    K --> O
    L --> P[Keep separate from authoritative search_rules index]
    P --> O
    M --> O
    N --> O
    O --> Q[Response includes mock, synthetic, dataset version, and disclaimer]
```

The HTTP adapter requires HTTPS except for loopback development and rejects base URLs with
embedded credentials, query strings, or fragments. Mock rules are never promoted into the
authoritative `search_rules` corpus, which continues to fail closed until approved source
documents are indexed.
