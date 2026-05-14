# ACE — Attention-Weighted Context Eviction

Content-aware context compression for multi-turn agentic LLM inference.

When an agentic loop runs for many turns, tool results accumulate in the context
window until the model hits its limit. The standard fix is to truncate the oldest
messages — but head/tail truncation discards structure indiscriminately, often
deleting the error lines or file paths the model needs most while keeping verbose
status boilerplate.

ACE replaces truncation with **line-level importance scoring**. Each line in an
old tool result is scored by content type, the top-scoring lines are kept, and the
rest are replaced with a single `[...N lines omitted by ACE...]` marker. The result
fits in the same context budget but preserves more of the signal the model acts on.

---

## How it works

### Line classifier

Each line gets a score in [0, 1]:

| Score | Content type |
|-------|-------------|
| 1.00 | Tool-call JSON (`"name"` + `"arguments"`) |
| 0.95 | Error / exception lines |
| 0.90 | File/workspace paths (`/tmp/`, `/results/`, `/workspace/`) |
| 0.85 | Task-framing keywords (`task:`, `step`, `verify`, `inspect`) |
| 0.70 | Numeric data (percentages, large numbers, units) |
| 0.60 | Shell commands (`$`, `curl`, `python3`, `docker`…) |
| 0.50 | Default (unknown content) |
| 0.30 | Success boilerplate (`done`, `ok`, `installed`, `saved`) |
| 0.20 | Very long lines (>200 chars — usually verbose dumps) |
| 0.10 | Meta-commentary ("here are the results", "i executed") |
| 0.00 | Blank lines |

### Compression

Given a `targetRatio` (default 0.4):

1. Score every line.
2. Keep the top `ceil(n_lines × targetRatio)` lines by score.
3. Always keep the first and last line.
4. Replace consecutive dropped runs with `[...N lines omitted by ACE...]`.

### Eviction loop

When the total context exceeds a char budget:

1. Collect all `[Tool result]: …` messages from oldest to newest.
2. Skip the `keepRecent` most-recent ones (they're the active working memory).
3. ACE-compress each eligible message until the budget is met.

---

## Usage

### Node.js

```js
const { aceCompress, aceEvict } = require('./ace');

// Compress one tool result
const compressed = aceCompress(toolResultText, 0.4);

// Evict old results from a messages array in-place
const charsSaved = aceEvict(messages, {
  budgetChars: 40000,
  keepRecent:  2,
  targetRatio: 0.4,
});
```

### Python

```python
from ace import ace_compress, ace_evict

# Compress one tool result
compressed = ace_compress(tool_result_text, target_ratio=0.4)

# Evict old results from a messages list in-place
chars_saved = ace_evict(messages, budget_chars=40_000, keep_recent=2)
```

### Drop-in for agent-bridge.js

ACE is already integrated into [agent-bridge.js](agent-bridge.js). Enable it with:

```bash
ACE_EVICT=1 KV_EVICT=1 node agent-bridge.js
```

`KV_EVICT=1` enables the eviction loop; `ACE_EVICT=1` switches it from head/tail
truncation to ACE compression.

---

## Benchmark: SWE-bench Lite (multi-turn)

**Setup:**
- Model: Qwen3-Next-80B-A3B-4bit (MLX, Apple Silicon)
- Dataset: 5 SWE-bench Lite problems (astropy)
- Agent: multi-turn (up to 6 turns), file-read tool, 2+ file reads required before patch
- KV budget: 8,000 chars — triggers compression on most multi-turn conversations
- Eval: structural patch similarity (file + line overlap, threshold >0.5)

### Results

| Condition | Accuracy | Avg turns | Chars evicted | Tokens saved |
|-----------|----------|-----------|---------------|--------------|
| No eviction | **0/5 (0%)** | 5.8 | 0 | 0 |
| KV eviction (head/tail) | **1/5 (20%)** | 6.0 | 23,407 | ~5,852 |
| ACE eviction | **1/5 (20%)** | 5.4 | 10,132 | ~2,533 |

**ACE matches KV accuracy while evicting 57% fewer characters**, preserving 2.3× more
context within the same budget.

**Both eviction modes beat no-eviction.** Problem 5 (astropy-6938) is solved by both
KV and ACE but not by no-eviction — unconstrained context growth (16k+ chars of
accumulated file reads) caused the model to produce a wrong patch. Eviction removed
low-value early exploration and surfaced the most recent/relevant content.

**ACE uses fewer turns** (5.4 avg vs 6.0 for KV), suggesting that structurally
preserved context helps the model reach a conclusion faster.

### Efficiency comparison

Simple KV eviction and ACE both keep the context under budget, but they achieve it differently:

- **KV (head/tail):** drops the first 600 chars and last 300 chars of old messages and
  marks everything in between as evicted. Fast, but loses structure.
- **ACE:** keeps the 40% highest-scoring lines, scattered across the full message.
  Error lines, paths, and task instructions survive; blank lines and boilerplate do not.

In the 5-problem test, ACE had to evict only 10k chars to stay under budget (vs 23k for
KV), because the retained content was already denser — the model needed fewer follow-up
turns to find what it was looking for.

---

## Integration: agent-bridge.js

`agent-bridge.js` is an Anthropic Messages API → OpenAI Chat Completions proxy that
lets Claude Code (and any Anthropic-API client) drive a locally-served MLX or llama.cpp
model. ACE is one of several context-management features it provides:

| Feature | Env var | What it does |
|---------|---------|-------------|
| Thought stripping | `STRIP_THOUGHTS=1` | Removes `<think>…</think>` blocks from old turns |
| KV eviction | `KV_EVICT=1` | Enables the eviction loop |
| ACE mode | `ACE_EVICT=1` | Switches eviction from head/tail to content-aware |
| Budget | `MAX_CONTEXT_CHARS=N` | Eviction threshold in chars (default 40000) |

```bash
# Start bridge with ACE
BACKEND_URL=http://localhost:8000/v1 \
BACKEND_MODEL=your-model-id \
KV_EVICT=1 ACE_EVICT=1 \
MAX_CONTEXT_CHARS=40000 \
node agent-bridge.js

# Health check
curl -s http://localhost:3210/health | jq '{kv_evict,ace_evict,kv_evicted_chars}'
```

---

## Files

| File | Description |
|------|-------------|
| `ace.js` | Standalone Node.js module (no dependencies) |
| `ace.py` | Python port |
| `agent-bridge.js` | Full bridge with ACE, thought-stripping, tool compaction |
| `benchmark/run_swebench_agent.py` | Multi-turn SWE-bench runner used for the benchmark above |

---

## Limitations

- The line classifier uses heuristics, not learned weights. It works well for
  tool-use / bash output but hasn't been tested on structured data (JSON, CSV, XML)
  where a different scoring function would be appropriate.
- The 5-problem benchmark is small. Accuracy numbers are directional, not statistically
  robust.
- ACE scores lines independently; it doesn't consider cross-line context (e.g. a number
  on line N is only meaningful if line N-1 says what it refers to).

---

## License

MIT
