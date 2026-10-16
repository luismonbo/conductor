# Scripts

Local tools for looking inside the harness, run by you or by Claude. None ships in an image
(`.dockerignore` excludes `scripts/`); production jobs live in `src/harness/cli/`.

Run them from the repo root: settings, `.env` and the default paths (`./harness.sqlite`,
`./data/...`) are relative to it. Every read command takes `--json`, and `--help` on any command
lists its options. `_common.py` holds what both scripts share; it isn't run on its own.

## `db.py`: conversations and run history

Reads the API's database read-only. With `HARNESS_CHECKPOINTER=sqlite` (the default) that's the
file `HARNESS_CHECKPOINTER_URL` names, or `--db PATH`; with `HARNESS_CHECKPOINTER=postgres` it's the
Postgres `HARNESS_CHECKPOINTER_URL` names, taken from the environment or `.env`, never the command
line. A query against Neon wakes it for a few minutes. Neon's production connection string names
the image's Debian CA file (`sslrootcert=/etc/ssl/certs/ca-certificates.crt`); on a Mac, point
`sslrootcert` at certifi's bundle instead (`uv run python -m certifi` prints its path).

| Command | Shows |
|---|---|
| `uv run scripts/db.py threads` | Conversations, newest first: agent, turns, tokens, first message (`--agent`, `--limit`) |
| `uv run scripts/db.py thread 10fa` | One conversation (any unique id prefix): messages, tool calls and results, then its runs and guard decisions (`--full`, `--no-tools`) |
| `uv run scripts/db.py thread 10fa --markdown > thread.md` | The conversation as Markdown, to share it or turn it into an eval case |
| `uv run scripts/db.py usage` | Tokens per UTC day against `HARNESS_DAILY_TOKEN_BUDGET` (`--days`) |
| `uv run scripts/db.py guard --blocked` | Input-guard decisions, newest first (`--limit`) |

Exports and `--json` output can contain visitors' messages: don't commit them.

## `rag.py`: the vector index

Milvus or pgvector: `--store`, default `HARNESS_RAG_VECTOR_STORE_BACKEND`. `--dsn` points pgvector
at another database, such as a remote server with `sslmode=require`.

| Command | Shows |
|---|---|
| `uv run scripts/rag.py docs` | Every collection and its documents, with the embedding model ingest recorded (`--collection`) |
| `uv run scripts/rag.py show papers/5045` | A document's chunks in order (any unique id prefix); `papers/5045:12` shows one chunk in full |
| `uv run scripts/rag.py search "how many encoder layers?"` | What the agent's `search_documents` gets: the same embedder, k and per-document quota (`--raw` turns the quota off; `--k`, `--collection`) |
| `uv run scripts/rag.py delete --collection papers` | Deletes a collection's chunks, or only `--document-id` ones, after asking |

- Milvus Lite allows one process: stop `make api` and any eval run first.
- Take a `search` query from a `search_documents` call in `db.py thread`: the model rewrites the
  visitor's question before searching. Each search makes one embedding call.
- Without a terminal, `delete` needs `--yes`. Claude never passes it without the owner's OK.

## `eval/`: eval-dataset upkeep

- `label_assist.py`: grade a retrieval pool for one case and append the case to a dataset.
- `refresh_corpus_fingerprint.py`: fold the live index's document counts into a dataset after a
  re-ingest.

## `infra/`: used by make and compose

- `validate_litellm_config.py`: checks `litellm_config.yaml`; `make up` runs it first.
- `create-databases.sql`: creates the `litellm` database in a new Postgres volume.
