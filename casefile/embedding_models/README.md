# Required embedding model

Provision `sentence-transformers/all-MiniLM-L6-v2` into the
`all-MiniLM-L6-v2/` directory beside this file, or set
`CASEFILE_EMBEDDING_MODEL_PATH` to an existing local copy of that exact model.

CaseFile loads the model with local-files-only behavior. Missing or invalid assets fail
startup; no model is downloaded or substituted at runtime.
