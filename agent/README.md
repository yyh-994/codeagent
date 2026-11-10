# Claude Code Agent

**LangGraph-powered coding agent** — supports Anthropic and OpenAI-compatible backends.

A production-ready coding agent that uses LangGraph's `StateGraph` to orchestrate the agent loop, with full support for:
tools, permissions, context compression, memory recall, skills, sub-agents, MCP, plan mode, auto mode, goal pursuit, and loop scheduling.

## Architecture

```
START → preprocess → agent → router → tools → agent
                              → END
```

- **preprocess**: Memory recall, context compaction, compression pipeline
- **agent**: LLM call via ChatAnthropic / ChatOpenAI
- **tools**: Permission checking + tool execution (read, write, edit, grep, shell, etc.)
- **router**: Conditional edge — tools if tool_calls detected, END otherwise

Each node is a pure function that reads state and returns a partial update. LangGraph handles the looping, state management, and streaming.

## Quick Start

```bash
cd claude-code-agent
pip install -e .

# Set API key
export ANTHROPIC_API_KEY=your-key-here

# Start interactive REPL
claude-code-agent

# Or one-shot
claude-code-agent "fix the bug in src/app.py"
```

### OpenAI-compatible backend

```bash
export OPENAI_API_KEY=sk-xxx
export OPENAI_BASE_URL=https://api.openai.com/v1
claude-code-agent --api-base $OPENAI_BASE_URL --model gpt-4o "hello"
```

## Features

| Feature | Description |
|---------|-------------|
| **Agent Loop** | LangGraph `StateGraph` with preprocess → agent → tools nodes |
| **Dual Backend** | Anthropic (ChatAnthropic) + OpenAI-compatible (ChatOpenAI) |
| **10+ Tools** | read_file, write_file, edit_file, list_files, grep_search, run_shell, skill, agent, tool_search |
| **3 Permission Modes** | default, plan, bypassPermissions |
| **Context Compression** | 3-tier: budget, snip stale, microcompact |
| **Memory System** | Project-isolated Milvus entities, HNSW/COSINE search, LLM fact extraction |
| **Skills** | .claude/skills/\*/SKILL.md — inline + fork modes |
| **Sub-agents** | general + custom via .claude/agents/\*.md |
| **MCP** | JSON-RPC over stdio, config from .claude/settings.json |
| **Plan Mode** | Read-only planning with approval workflow |
| **Auto Mode** | LLM classifier judges each action |
| **/goal** | Session-scoped stop hook with evaluator |
| **/loop** | Recurring (interval) or self-paced (dynamic) |
| **Session** | Auto-save to ~/.claude-code-agent/sessions/ |

## Long-term memory

Long-term memories live entirely in **Milvus**, in one entity with four fields:

| Field | Milvus type | Purpose |
| --- | --- | --- |
| `id` | VARCHAR, primary key | Stable ID for search, update and deletion |
| `type` | VARCHAR | user / feedback / project / reference |
| `content` | VARCHAR | Complete fact or experience, up to 4000 characters |
| `vector` | FLOAT_VECTOR | Embedding of content |

There is no SQLite memory table, vector BLOB table or MEMORY.md index. SQLite
checkpoints and session JSON keep their existing short-term conversation roles.
Old memory.db files and embedding sidecars are left untouched and are not read
or migrated automatically. Only a local memory.lock file is used to coordinate
agents sharing a project; it does not contain memories.

Each startup directory gets a stable Milvus alias derived from its normalized
absolute path. A nested directory uses its own collection, without searching
for a parent Git repository. All four memory types are isolated by project.
Moving the project changes its alias. Use the same Milvus endpoint and database
to reopen it; switching endpoints/databases does not migrate data between servers.

### Start the local Milvus server

Install the updated dependencies, then start Docker Desktop with Linux containers:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
docker compose -f docker-compose.milvus.yml up -d
docker compose -f docker-compose.milvus.yml ps
```

The included single-container Milvus 2.6 configuration uses embedded etcd and
local storage, with a persistent Docker volume. It binds the SDK and health
ports to localhost. Wait for the health check to pass. To stop it while retaining
data, run `docker compose -f docker-compose.milvus.yml stop`.
For an existing Milvus server or Zilliz Cloud, configure its URI and credentials
instead of starting this local container.

Configure `.env`:

```dotenv
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_DB_NAME=default
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_LOCAL_PATH=D:/models/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu
```

`MILVUS_TOKEN` is optional for the included local server. Use the token required
by your remote server. The SDK targets Milvus 2.6; Lite is not used.

### Local embedding model

Install the local runtime with `python -m pip install -e ".[local]"` and point
`EMBEDDING_LOCAL_PATH` at the existing model snapshot directory containing the
weights, tokenizer and Sentence Transformers configuration. For example:

```dotenv
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_LOCAL_PATH=D:/models/bge-small-zh-v1.5
EMBEDDING_DEVICE=cpu
```

The embedding client uses Sentence Transformers locally, on CPU by default. It caches
the model in the process and runs encoding in a worker thread. Loading uses
`local_files_only=True`, `trust_remote_code=False` and `token=False`: a missing
model raises an error instead of downloading or falling back to paid inference.
No Hugging Face token is needed. Text stays on the local machine until the memory
entity is sent to the configured Milvus server. Inputs exceeding the model's
token limit are truncated by its tokenizer, so keep each memory concise.

The local model path forms part of the embedding identity; after
changing model paths, restart and run `/memory reindex` for existing
records. Use a new path when replacing model weights rather than overwriting them
in place. Short-term conversation storage is independent of this configuration.

The CLI loads `.env` from the startup directory first, then fills missing variables
from the agent source directory. When using `CodingAgent` as a Python library,
call `dotenv.load_dotenv` before constructing it.

### Save, search and rebuild

The LLM uses the `remember` skill to extract independent facts or complete
experiences. It processes each fact separately: call `memory_search`, compare
the result, then skip it or call `memory_save` before processing the next fact.
`memory_save` accepts one object, for example
`{"type": "project", "content": "Use uv for Python dependencies."}`.
Omit `id` to create a memory; supply an existing ID to revise it. Only the fact
being saved is embedded and written, and `saved` contains one memory object.
Identical existing facts are skipped by the LLM; exact duplicates reaching the
store are reused. `memory_delete` removes the entire entity, including its
vector. Default mode asks for write confirmation, plan mode permits searching
but blocks writes, and yolo mode permits writes.

Milvus builds a **HNSW / COSINE** index and returns the nearest records with
their text. Python no longer reads all vectors or computes their similarities.
Search defaults to top 5 and a similarity threshold of 0.35; `memory_search`
accepts `limit` and `threshold` overrides. HNSW is approximate retrieval;
calibrate its recall and the threshold with your embedding model and examples.
Strong consistency makes acknowledged updates visible to subsequent reads.

Automatic recall finishes before the current turn's model call. Its text is
limited to about 8000 characters. The base SystemMessage stays unchanged. For
each model request, the current HumanMessage is copied and prefixed with
`【相关历史记忆】`, followed by `【当前用户问题】` and the original user content.
The graph message ID identifies the current HumanMessage even when tool results
follow it. The request copy is never written back to conversation history,
session JSON or checkpoints, and repeated tool iterations rebuild it from the
original user message. No results means no prefix. Recall failures produce a notice
and allow the main conversation to continue. Explicit tool failures are returned
to the agent. A network timeout can have an uncertain write outcome; search
before retrying. Milvus stores each fact's content and vector together. Each
save stands independently: a later failure does not roll back earlier saves.

Use `/memory` to list IDs, types and contents. The embedding model identity is
recorded in the collection schema description, and the dimension in the vector
field. After changing the embedding model, restart and run `/memory reindex`.
Reindex embeds existing memories one at a time, writes and verifies a replacement
collection, switches the project alias,
then removes the obsolete collection. A failed build leaves the old collection
active; unfinished collections may need manual cleanup. Local agents sharing
this directory coordinate writes using a file lock. Run reindex without writers
on other machines sharing the same alias.

The implementation follows the official
[Milvus collection API](https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Collections/create_collection.md),
[search API](https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Vector/search.md),
and [alias API](https://milvus.io/api-reference/pymilvus/v2.6.x/MilvusClient/Collections/alter_alias.md).
The local Docker configuration adapts the
[embedded standalone setup](https://github.com/milvus-io/milvus/blob/v2.6.0/scripts/standalone_embed.sh).

### Verification

Offline SDK and graph integration tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

With the local Milvus server healthy, run the live persistence/search test.
It uses synthetic embeddings, a temporary project identity and only removes
collections created for that temporary project:

```powershell
$env:RUN_MILVUS_TESTS = "1"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_milvus_live.py -v
```

To verify the configured local BGE model without network access during encoding,
and optionally test its real vectors against Milvus:

```powershell
$env:RUN_LOCAL_EMBEDDING_TESTS = "1"
$env:RUN_MILVUS_TESTS = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## CLI Options

```
--yolo, -y          Skip all confirmation prompts
--plan              Plan mode: read-only
--model, -m         Model name
--api-base URL      OpenAI-compatible API endpoint
--resume            Resume last session
--max-cost USD      Stop at cost limit
--max-turns N       Stop after N turns
```

## REPL Commands

```
/clear              Clear conversation
/plan               Toggle plan mode
/cost               Show token usage
/compact            Manually compact
/goal <condition>   Set a goal stop hook
/loop [int] <prompt> Recurring prompt
/memory             List memories
/skills             List skills
/<skill-name>       Invoke a skill
```

## Project Structure

```
src/
├── __init__.py
├── __main__.py      # CLI entry point
├── graph.py         # LangGraph agent graph (core)
├── state.py         # AgentState type definition
├── tools.py         # Tool definitions + execution
├── prompt.py        # System prompt construction
├── memory.py        # Semantic memory system
├── skills.py        # Skill discovery + execution
├── mcp_client.py    # MCP protocol client
├── autonomy.py      # Goal/loop/auto mode
├── session.py       # Session persistence
├── ui.py            # Terminal UI
└── frontmatter.py   # YAML frontmatter parser
```

## License

MIT
