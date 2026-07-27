# Rules corpus

CaseFile intentionally ships without invented or stale rule text. Add the authoritative
NSDA documents as Markdown files in this directory, preserving their published section
numbers as headings, then run:

```bash
python -m casefile.ingest.build_index
```

Expected heading form:

```markdown
## 7.2.B Evidence Availability
Exact text from the authorized source document...
```

Keep the source filename, edition, effective date, and canonical URL at the top of each
file. The agent returns `document`, `section_number`, and `section_title` with every rule
answer. If this directory has no rule corpus, rule questions fail closed with an explicit
"no authoritative rule on file" response.
