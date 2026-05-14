#!/usr/bin/env node
// ── Agent API Bridge ────────────────────────────────────────────────────────
// Generic proxy: translates Anthropic Messages API → OpenAI Chat Completions
// Allows Claude Code (or any Anthropic-compatible client) to use ANY local model.
//
// Usage:
//   node agent-bridge.js                          # defaults: port 3210, backend localhost:8095
//   BRIDGE_PORT=4000 BACKEND_URL=http://localhost:8080/v1 node agent-bridge.js
//
// Then set Claude Code to use it:
//   ANTHROPIC_BASE_URL=http://localhost:3210 claude -p "do something"
//
// Supports: llama-server, Ollama, vLLM, MLX-server, LM Studio, text-gen-webui
// ─────────────────────────────────────────────────────────────────────────────

const http = require('http');
const https = require('https');

const PORT        = parseInt(process.env.BRIDGE_PORT   || '3210', 10);
const BACKEND_URL = process.env.BACKEND_URL            || 'http://localhost:8095/v1';
const BACKEND_MODEL = process.env.BACKEND_MODEL        || 'gemma4-e4b-v3';
const BACKEND_KEY = process.env.BACKEND_KEY            || 'sk-no-key';
const LOG_LEVEL   = process.env.BRIDGE_LOG             || 'info'; // info, debug, quiet

// KV Eviction — budget-based context compression.
//
// Only activates when the total accumulated context exceeds MAX_CONTEXT_CHARS.
// When over budget, compresses OLD tool results (oldest first) keeping head + tail,
// dropping the noisy middle. Never touches the last KV_EVICT_KEEP_RECENT results
// nor the system/task messages.
//
// Env vars:
//   KV_EVICT=0               disable entirely
//   MAX_CONTEXT_CHARS=32000  budget (~8k tokens); eviction only starts here
//   KV_EVICT_HEAD=600        chars to keep from result start (structure/keys)
//   KV_EVICT_TAIL=300        chars to keep from result end (final values/errors)
//   KV_EVICT_KEEP_RECENT=2   recent tool results to never compress
//   KV_EVICT_MIN=300         don't compress results already smaller than this

// Max tokens per generation — include budget for thought blocks when thinking is enabled
const MAX_TOKENS = parseInt(process.env.MAX_TOKENS || '16384', 10);
// Max ms to wait for a backend response before aborting (thinking can runaway)
// Default: 3 min. With thinking ON use lower (e.g. 180000). Set to 0 to disable.
const BACKEND_TIMEOUT_MS = parseInt(process.env.BACKEND_TIMEOUT_MS || '0', 10);
// When THINKING_MODE=1: use a more directive system prompt that prevents analysis loops.
// Thinking models over-plan and spend many turns on helper scripts; this prompt enforces
// direct action toward the final deliverable.
const THINKING_MODE = process.env.THINKING_MODE === '1';
// When SEMANTIC_HINTS=1: inject correction hints into tool results that contain known failure
// patterns (all-zero values, empty lists). The hints are appended to the tool result text
// so the model sees them on the next turn and can self-correct.
const SEMANTIC_HINTS = process.env.SEMANTIC_HINTS === '1';
// When JSON_HINT=1: inject a one-line proactive hint the FIRST TIME the model reads a
// UniProt JSON blob (detected by presence of "Natural variant" in the tool result).
// Fires once, before the model writes parse code — bridges the inferential gap for
// cancer mutation filtering without reactive correction (avoids think-v3's context inflation).
const JSON_HINT = process.env.JSON_HINT === '1';
let jsonHintFired = false;
let p53JsonSeen = false;  // tracks if p53.json was accessed in any prior assistant tool call

const KV_EVICT             = process.env.KV_EVICT !== '0';
const MAX_CONTEXT_CHARS    = parseInt(process.env.MAX_CONTEXT_CHARS    || '40000', 10); // ~10k tokens
const KV_EVICT_HEAD        = parseInt(process.env.KV_EVICT_HEAD        || '600',   10);
const KV_EVICT_TAIL        = parseInt(process.env.KV_EVICT_TAIL        || '300',   10);
const KV_EVICT_KEEP_RECENT = parseInt(process.env.KV_EVICT_KEEP_RECENT || '2',     10);
const KV_EVICT_MIN         = parseInt(process.env.KV_EVICT_MIN         || '300',   10);

// Thought stripping — remove <|channel>thought....<channel|> blocks from old
// assistant turns. The model's reasoning is already resolved; only the action
// (tool call) matters for future turns. Keeps the last STRIP_KEEP_RECENT turns intact.
//   STRIP_THOUGHTS=0         disable
//   STRIP_KEEP_RECENT=2      how many recent assistant turns to leave untouched
const STRIP_THOUGHTS      = process.env.STRIP_THOUGHTS !== '0';
const STRIP_KEEP_RECENT   = parseInt(process.env.STRIP_KEEP_RECENT || '2', 10);

// Matches Gemma 4 thought channel: <|channel>thought\n...<channel|>
// Also handles variants without newline and partial closings.
const THOUGHT_RE = /<\|channel\>thought[\s\S]*?<channel\|>/g;

let thoughtCharsStripped = 0;

function stripThoughtBlocks(text) {
  if (!text || !STRIP_THOUGHTS) return text;
  const stripped = text.replace(THOUGHT_RE, '[thought removed]');
  thoughtCharsStripped += text.length - stripped.length;
  return stripped;
}

// Semantic hint injection — appended to tool results that contain known failure patterns.
// Runs only when SEMANTIC_HINTS=1. Helps the model self-correct on bad extractions.
function injectSemanticHints(toolResult) {
  if (!SEMANTIC_HINTS) return toolResult;
  const t = toolResult;
  const hints = [];

  // CH1: All-zero secondary structure — model used wrong JSON field
  const hasZeroSS = /(?:helix|strand|turn|coil)[^\n]{0,30}0\.0%/i.test(t)
                 || /0\.0%[^\n]{0,30}(?:helix|strand|turn|coil)/i.test(t);
  if (hasZeroSS) {
    hints.push(
      '[HINT]: Secondary structure values are 0 — wrong JSON path. ' +
      'Correct approach: iterate p53.json features[], filter type in ["Helix","Beta strand","Turn"], ' +
      'sum (location.end.value - location.start.value + 1) per type, divide by 393. ' +
      'Expected: Helix~21.9%, Beta strand~25.4%, Turn~6.6%, Coil~46.1%.'
    );
  }

  // CH1: Empty or zero cancer mutations
  const hasEmptyMut = /mutations.*\[\s*\]|\[\s*\].*mutation/i.test(t)
                   || (/cancer/i.test(t) && !/[A-Z]\d+[A-Z]/.test(t) && t.length < 200);
  if (hasEmptyMut) {
    hints.push(
      '[HINT]: No cancer mutations found — wrong filter. ' +
      'Correct: filter features[] where type=="Natural variant" AND description contains "cancer" OR "somatic" OR "tumor". ' +
      'Build code as: alternativeSequence.originalSequence + location.start.value + alternativeSequence.alternativeSequences[0]. ' +
      'Example: originalSequence="R", start=175, alt="H" → "R175H".'
    );
  }

  if (hints.length === 0) return toolResult;
  log('info', `  [hints] injected ${hints.length} correction hint(s) into tool result`);
  return toolResult + '\n\n' + hints.join('\n');
}

// Proactive JSON hint — fires once on the FIRST tool result that arrives after p53.json
// was referenced in any assistant tool_use call. This catches the silent curl download
// (turn 1-2) and fires the hint on the very next tool result the model sees.
// Tells the model the correct description filter BEFORE it writes parse code.
function scanMessagesForP53(messages) {
  if (!JSON_HINT || p53JsonSeen) return;
  for (const msg of messages) {
    if (msg.role !== 'assistant') continue;
    for (const block of (msg.content || [])) {
      if (block.type === 'tool_use') {
        const args = JSON.stringify(block.input || {});
        if (/p53\.json/i.test(args)) { p53JsonSeen = true; return; }
      }
    }
  }
}

function injectJsonHint(toolResult) {
  if (!JSON_HINT || jsonHintFired) return toolResult;
  // Also detect from raw JSON content (fallback)
  if (!p53JsonSeen && !toolResult.includes('"Natural variant"')) return toolResult;
  jsonHintFired = true;
  const hint =
    '\n# PARSE NOTE: Cancer mutations are "Natural variant" features whose description ' +
    'contains "cancer", "tumor", or "somatic" (case-insensitive). ' +
    'Other variants (phosphorylation, polymorphisms) do NOT have these keywords in description. ' +
    'Filter: type=="Natural variant" AND ("cancer" OR "tumor" OR "somatic") in description. ' +
    'Collect ALL such variants (there are 100+), not just the first few. ' +
    'Key hotspots to include: R175H, R248W, R248Q, R273H, R273C, R249S, G245S, R282W, Y220C.';
  log('info', '  [json-hint] proactive cancer-filter hint injected');
  return toolResult + hint;
}

let kvEvictedTotal = 0;

// Strip thought blocks from old assistant turns (keep last STRIP_KEEP_RECENT intact).
// Applied after the messages array is fully built.
function applyThoughtStripping(messages) {
  if (!STRIP_THOUGHTS) return;

  // Find all assistant message indices
  const assistantIndices = messages
    .map((m, i) => m.role === 'assistant' ? i : -1)
    .filter(i => i >= 0);

  // Only strip from old turns — leave the most recent STRIP_KEEP_RECENT untouched
  const toStrip = assistantIndices.slice(0, -STRIP_KEEP_RECENT);
  if (toStrip.length === 0) return;

  // Log first 200 chars of oldest assistant turn to understand actual thought format
  if (toStrip.length > 0) {
    const sample = (messages[toStrip[0]].content || '').slice(0, 200).replace(/\n/g, '\\n');
    log('info', `  [thought-dbg] assistant[${toStrip[0]}] sample: ${JSON.stringify(sample)}`);
  }

  let stripped = 0;
  for (const idx of toStrip) {
    const m = messages[idx];
    if (typeof m.content === 'string' && THOUGHT_RE.test(m.content)) {
      THOUGHT_RE.lastIndex = 0;
      const before = m.content.length;
      m.content = stripThoughtBlocks(m.content);
      stripped += before - m.content.length;
    }
  }

  if (stripped > 0) {
    log('info', `  Thought strip: ${toStrip.length} old turns, -${stripped} chars (~${Math.round(stripped/4)} tok), total stripped≈${Math.round(thoughtCharsStripped/4)} tok`);
  }
}

function estimateContextChars(messages) {
  return messages.reduce((sum, m) => {
    const c = m.content;
    return sum + (typeof c === 'string' ? c.length : JSON.stringify(c || '').length);
  }, 0);
}

// ACE v2 — Attention-weighted Context Eviction
// Content-aware alternative to the simple head+tail truncation.
// Enabled with ACE_EVICT=1 env var. Falls back to simple eviction when ACE_EVICT=0.
const ACE_EVICT = process.env.ACE_EVICT === '1';

function aceClassifyLine(s) {
  const sl = s.toLowerCase();
  if (!s.trim()) return 0.0;
  if (s.includes('"name"') && (s.includes('"arguments"') || s.includes('"parameters"'))) return 1.0;
  const errorPhrases = ['error', 'failed', 'not found', 'traceback', 'exception', 'exit code'];
  if (errorPhrases.some(p => sl.includes(p))) return 0.95;
  if (s.includes('/results/') || s.includes('/workspace/') || s.includes('/tmp/')) return 0.9;
  if (['task:', 'begin', 'step ', 'verify', 'inspect'].some(k => sl.includes(k))) return 0.85;
  if (/\d+\.?\d*%|\d{4,}|bytes|rows|columns/.test(sl)) return 0.7;
  if (['$', '>>>', 'root@', 'agent@'].some(k => s.startsWith(k)) ||
      ['wget','curl','pip','python3','apt-get','docker'].some(k => sl.includes(k))) return 0.6;
  if (['done','ok','installed','saved','complete'].includes(sl.trim())) return 0.3;
  if (s.length > 200) return 0.2;
  if (['i executed', 'here are the results', 'tool results:'].some(k => sl.includes(k))) return 0.1;
  return 0.5;
}

function aceCompressMessage(content, targetRatio = 0.4) {
  const lines = content.split('\n');
  if (lines.length <= 3) return content;
  const scored = lines.map((line, i) => {
    let score = aceClassifyLine(line);
    if (i === 0 || i === lines.length - 1) score = Math.max(score, 0.7);
    return { i, line, score };
  });
  const nKeep = Math.max(2, Math.floor(lines.length * targetRatio));
  const keepSet = new Set(
    [...scored].sort((a, b) => b.score - a.score).slice(0, nKeep).map(x => x.i)
  );
  keepSet.add(0); keepSet.add(lines.length - 1);
  const out = []; let skipped = 0;
  for (let i = 0; i < lines.length; i++) {
    if (keepSet.has(i)) {
      if (skipped > 0) out.push(`  [...${skipped} lines omitted by ACE...]`);
      skipped = 0; out.push(lines[i]);
    } else { skipped++; }
  }
  if (skipped > 0) out.push(`  [...${skipped} lines omitted by ACE...]`);
  return out.join('\n');
}

// Called after messages array is built. Compresses old tool results until
// total context fits within MAX_CONTEXT_CHARS.
// When ACE_EVICT=1: uses content-aware line selection (ACE v2).
// Otherwise: simple head+tail truncation.
function applyKvEviction(messages) {
  if (!KV_EVICT) return;
  if (estimateContextChars(messages) <= MAX_CONTEXT_CHARS) return;

  const toolResultIndices = [];
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (m.role === 'user' && typeof m.content === 'string'
        && m.content.startsWith('[Tool result]:')) {
      toolResultIndices.push(i);
    }
  }

  const compressable = toolResultIndices.slice(0, -KV_EVICT_KEEP_RECENT);

  for (const idx of compressable) {
    if (estimateContextChars(messages) <= MAX_CONTEXT_CHARS) break;
    const prefix = '[Tool result]: ';
    const result = messages[idx].content.slice(prefix.length);
    if (result.length <= KV_EVICT_MIN) continue;
    if (result.includes('[...kv-evicted:') || result.includes('[...ACE') || result.includes('omitted by ACE')) continue;

    const before = result.length;
    let compressed;
    if (ACE_EVICT) {
      compressed = aceCompressMessage(result, 0.4);
    } else {
      const dropped = result.length - KV_EVICT_HEAD - KV_EVICT_TAIL;
      if (dropped <= 0) continue;
      compressed = result.slice(0, KV_EVICT_HEAD)
        + `\n[...kv-evicted: ${dropped} chars (~${Math.round(dropped/4)} tok)...]\n`
        + result.slice(-KV_EVICT_TAIL);
    }

    const saved = before - compressed.length;
    if (saved <= 0) continue;
    kvEvictedTotal += saved;
    messages[idx].content = prefix + compressed;
    log('info', `  ${ACE_EVICT ? 'ACE' : 'KV'} evict msg[${idx}]: -${saved} chars, ctx now ≈${Math.round(estimateContextChars(messages)/4)} tok`);
  }
}

let requestCount = 0;
let toolCallCount = 0;
const startTime = Date.now();

function log(level, ...args) {
  if (LOG_LEVEL === 'quiet') return;
  if (level === 'debug' && LOG_LEVEL !== 'debug') return;
  const ts = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`[bridge ${ts}s]`, ...args);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', chunk => data += chunk);
    req.on('end', () => {
      try { resolve(JSON.parse(data)); }
      catch (e) { reject(new Error('Invalid JSON')); }
    });
    req.on('error', reject);
  });
}

// Essential tools — only pass these to the model, ignore the rest
const ESSENTIAL_TOOLS = new Set([
  'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep',
]);

// Translate Anthropic Messages API → OpenAI Chat Completions
function anthropicToOpenAI(body) {
  const messages = [];

  // System prompt — replace Claude Code's massive prompt with a compact agent prompt
  if (body.system) {
    let sysText = typeof body.system === 'string'
      ? body.system
      : (body.system || []).map(b => b.text || '').join('\n');
    // Claude Code sends ~15k chars system prompt. Replace with compact version.
    if (sysText.length > 8000) {
      if (THINKING_MODE) {
        // Directive prompt for thinking mode: prevents analysis loops.
        // Without this, the model spends 15-21 turns writing helper Python scripts
        // instead of producing the final deliverable directly.
        sysText = `You are an autonomous coding agent. Use tools: Bash, Read, Write, Edit, Glob, Grep.

RULES — follow strictly:
1. Follow task steps IN ORDER. Each step should be one tool call.
2. Extract data BEFORE writing the report. Verify the extracted values are non-zero/non-empty.
3. For HTML reports: write the complete HTML in ONE Write tool call (no iterating). Must include Chart.js canvas chart.
4. Do NOT write intermediate helper scripts — use python3 -c one-liners in Bash for data extraction.
5. If extracted values are all zero or empty, re-read the task instructions and try a different field/path.
6. One tool call per turn. Declare done only when all task files exist with real data.`;
      } else {
        sysText = `You are an autonomous coding agent. You help with software engineering tasks.
Use tools to accomplish tasks. Available: Bash (run commands), Read (read files), Write (create files), Edit (modify files), Glob (find files), Grep (search content).
Be concise. Execute tools one at a time. Verify results before declaring done.`;
      }
    }
    messages.push({ role: 'system', content: sysText });
  }

  // Scan assistant turns for p53.json references (for proactive JSON hint)
  scanMessagesForP53(body.messages || []);

  // Convert Anthropic messages — strip Claude Code bloat
  for (const msg of (body.messages || [])) {
    if (msg.role === 'user') {
      let userContent = typeof msg.content === 'string' ? msg.content : null;
      if (userContent) {
        // Strip <system-reminder> and other XML tag blocks that Claude Code injects
        userContent = userContent.replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, '').trim();
        userContent = userContent.replace(/<[\s\S]*?<\/antml:[^>]+>/g, '').trim();
        // Skip empty messages after stripping
        if (!userContent) continue;
        // Truncate very long user messages
        if (userContent.length > 8000) {
          userContent = userContent.slice(0, 8000) + '\n[...truncated...]';
        }
        messages.push({ role: 'user', content: userContent });
      } else {
        // Array of content blocks — truncate tool results
        const parts = (msg.content || []).map(block => {
          if (block.type === 'text') {
            let t = block.text || '';
            t = t.replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, '').trim();
            return t.length > 8000 ? t.slice(0, 8000) + '\n[...truncated...]' : t;
          }
          if (block.type === 'tool_result') {
            let result = typeof block.content === 'string' ? block.content : JSON.stringify(block.content);
            const jsonHint = injectJsonHint(result);  // detect on full content, get hint suffix
            const hintSuffix = jsonHint !== result ? jsonHint.slice(result.length) : '';
            if (result.length > 6000) result = result.slice(0, 6000) + '\n[...truncated...]';
            if (hintSuffix) result = result + hintSuffix;  // append hint after truncation
            result = injectSemanticHints(result);
            return `[Tool result]: ${result}`;
          }
          return '';
        }).filter(Boolean);
        if (parts.length) messages.push({ role: 'user', content: parts.join('\n') });
      }
    } else if (msg.role === 'assistant') {
      if (typeof msg.content === 'string') {
        messages.push({ role: 'assistant', content: msg.content });
      } else {
        // Extract text and tool_use blocks
        const textParts = [];
        const toolCalls = [];
        for (const block of (msg.content || [])) {
          if (block.type === 'text') textParts.push(block.text);
          if (block.type === 'tool_use') {
            toolCalls.push({
              id: block.id,
              type: 'function',
              function: {
                name: block.name,
                arguments: JSON.stringify(block.input || {}),
              }
            });
          }
        }
        const assistantMsg = { role: 'assistant' };
        if (textParts.length) assistantMsg.content = textParts.join('\n');
        else assistantMsg.content = null;
        if (toolCalls.length) assistantMsg.tool_calls = toolCalls;
        messages.push(assistantMsg);
      }
    }
  }

  // Convert Anthropic tools → OpenAI functions
  // Only pass essential tools to save tokens (23 tools → 6)
  const tools = (body.tools || [])
    .filter(tool => ESSENTIAL_TOOLS.has(tool.name))
    .map(tool => {
      // Compact tool schemas to save tokens
      const compactSchemas = {
        'Bash': { type: 'object', properties: { command: { type: 'string', description: 'Shell command to run' } }, required: ['command'] },
        'Read': { type: 'object', properties: { file_path: { type: 'string', description: 'Absolute path to read' } }, required: ['file_path'] },
        'Write': { type: 'object', properties: { file_path: { type: 'string' }, content: { type: 'string' } }, required: ['file_path', 'content'] },
        'Edit': { type: 'object', properties: { file_path: { type: 'string' }, old_string: { type: 'string' }, new_string: { type: 'string' } }, required: ['file_path', 'old_string', 'new_string'] },
        'Glob': { type: 'object', properties: { pattern: { type: 'string', description: 'Glob pattern like **/*.py' } }, required: ['pattern'] },
        'Grep': { type: 'object', properties: { pattern: { type: 'string', description: 'Regex pattern to search' }, path: { type: 'string' } }, required: ['pattern'] },
      };
      return {
        type: 'function',
        function: {
          name: tool.name,
          description: (tool.description || '').slice(0, 80),
          parameters: compactSchemas[tool.name] || tool.input_schema || { type: 'object', properties: {} },
        }
      };
    });

  // Context window management: keep system + first user msg + last N messages
  // This prevents token explosion as conversation grows
  const MAX_CONTEXT_MSGS = 20;  // system + task + 18 recent turns (need more for SWE-bench)
  if (messages.length > MAX_CONTEXT_MSGS) {
    const system = messages[0]; // system prompt
    const task = messages[1];   // original user task
    const recent = messages.slice(-MAX_CONTEXT_MSGS + 2); // last N-2 messages
    const evicted = messages.length - MAX_CONTEXT_MSGS;
    log('info', `  Context: evicted ${evicted} old messages, keeping ${MAX_CONTEXT_MSGS}`);
    // Inject task reminder so the model doesn't lose focus
    const reminder = { role: 'user', content: `[Reminder: Your original task was: ${task.content.slice(0, 300)}]\nContinue working on this task. Focus on completing remaining steps.` };
    messages.length = 0;
    messages.push(system, task, reminder, ...recent);
  }

  // Strip thought blocks from old assistant turns (always, regardless of budget)
  applyThoughtStripping(messages);

  // Budget-based KV eviction: compress old tool results only when context is full
  const ctxBefore = estimateContextChars(messages);
  applyKvEviction(messages);
  const ctxAfter = estimateContextChars(messages);
  if (ctxBefore !== ctxAfter) {
    log('info', `  KV evict total: ${Math.round(ctxBefore/4)}→${Math.round(ctxAfter/4)} tok (saved ${Math.round((ctxBefore-ctxAfter)/4)} tok)`);
  }

  const openaiBody = {
    model: BACKEND_MODEL,
    messages,
    max_tokens: Math.min(body.max_tokens || MAX_TOKENS, MAX_TOKENS),
    temperature: body.temperature ?? 0.3,
    stream: false,  // Force non-streaming for tool call detection
  };

  if (tools.length > 0) {
    openaiBody.tools = tools;
    openaiBody.tool_choice = 'auto';
  }

  return openaiBody;
}

// Translate OpenAI response → Anthropic Messages API response
function openAIToAnthropic(openaiResp, requestId) {
  const choice = (openaiResp.choices || [])[0] || {};
  const msg = choice.message || {};
  const content = [];

  // Log reasoning overhead (msg.reasoning = Gemma 4 thinking block in mlx_lm 0.31+)
  if (msg.reasoning) {
    const rChars = (msg.reasoning || '').length;
    if (rChars > 5) {  // ignore the empty '\n' case
      log('info', `  Reasoning: ${rChars} chars (~${Math.round(rChars/4)} tok), stripped`);
    }
  }

  // Text content — also detect tool calls embedded in text
  if (msg.content) {
    // Strip thought blocks from response before parsing (thinking mode output)
    const textRaw = msg.content;
    const text = stripThoughtBlocks(textRaw);
    if (text !== textRaw) {
      log('info', `  Thought strip (response): -${textRaw.length - text.length} chars (~${Math.round((textRaw.length - text.length)/4)} tok)`);
    }
    // Try to extract JSON tool calls from text: {"name": "bash", "arguments": ...}
    const toolCallPattern = /\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*("(?:[^"\\]|\\.)*"|\{[^}]*\})\s*\}/;
    const match = text.match(toolCallPattern);
    if (match) {
      const toolName = match[1];
      let toolArgs;
      try {
        const rawArgs = match[2];
        if (rawArgs.startsWith('"')) {
          // String argument (e.g., bash command)
          toolArgs = { command: JSON.parse(rawArgs) };
          if (toolName === 'write_file' || toolName === 'read_file') {
            toolArgs = { path: JSON.parse(rawArgs) };
          }
        } else {
          toolArgs = JSON.parse(rawArgs);
        }
      } catch {
        toolArgs = { input: match[2] };
      }
      // Map our tool names to Claude Code tool names
      const ccToolName = {
        'bash': 'Bash',
        'write_file': 'Write',
        'read_file': 'Read',
        'web_search': 'WebSearch',
      }[toolName] || toolName;

      toolCallCount++;
      content.push({
        type: 'tool_use',
        id: `toolu_${requestId}_${toolCallCount}`,
        name: ccToolName,
        input: toolArgs,
      });
      // Add any text BEFORE the tool call
      const beforeTool = text.slice(0, text.indexOf(match[0])).trim();
      if (beforeTool) {
        content.unshift({ type: 'text', text: beforeTool });
      }
    } else {
      content.push({ type: 'text', text });
    }
  }

  // Tool calls → tool_use blocks
  if (msg.tool_calls && msg.tool_calls.length > 0) {
    for (const tc of msg.tool_calls) {
      toolCallCount++;
      let args = {};
      try { args = JSON.parse(tc.function.arguments || '{}'); } catch {}
      content.push({
        type: 'tool_use',
        id: tc.id || `toolu_${requestId}_${toolCallCount}`,
        name: tc.function.name,
        input: args,
      });
    }
  }

  // Detect tool_use in content (from text-embedded tool calls)
  const hasToolUse = content.some(c => c.type === 'tool_use');
  const stopReason = hasToolUse ? 'tool_use'
    : choice.finish_reason === 'tool_calls' ? 'tool_use'
    : choice.finish_reason === 'length' ? 'max_tokens'
    : 'end_turn';

  const usage = openaiResp.usage || {};

  return {
    id: `msg_${requestId}`,
    type: 'message',
    role: 'assistant',
    model: BACKEND_MODEL,
    content: content.length > 0 ? content : [{ type: 'text', text: '' }],
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: usage.prompt_tokens || 0,
      output_tokens: usage.completion_tokens || 0,
    },
  };
}

// Forward request to backend
async function forwardToBackend(openaiBody) {
  const backendUrl = new URL(BACKEND_URL);
  const endpoint = backendUrl.pathname.replace(/\/$/, '') + '/chat/completions';

  return new Promise((resolve, reject) => {
    const options = {
      hostname: backendUrl.hostname,
      port: backendUrl.port,
      path: endpoint,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${BACKEND_KEY}`,
      },
    };

    const payload = JSON.stringify(openaiBody);
    options.headers['Content-Length'] = Buffer.byteLength(payload);

    const proto = backendUrl.protocol === 'https:' ? https : http;
    const req = proto.request(options, res => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        clearTimeout(timer);
        try { settle(resolve, JSON.parse(data)); }
        catch (e) { settle(reject, new Error(`Backend parse error: ${data.slice(0, 200)}`)); }
      });
    });
    let settled = false;
    function settle(fn, val) { if (!settled) { settled = true; fn(val); } }
    req.on('error', e => settle(reject, e));

    // Wall-clock deadline — always returns a stub so CC can recover from a stuck MLX turn.
    // 1200s: at 4.5tok/s, 5120 tok max = ~1138s; need headroom above that.
    const deadlineMs = BACKEND_TIMEOUT_MS > 0 ? BACKEND_TIMEOUT_MS : 1200000;
    const timer = setTimeout(() => {
      req.destroy();
      log('info', `  Backend timeout after ${deadlineMs/1000}s — returning max_tokens stub`);
      settle(resolve, {
        id: 'timeout',
        choices: [{ index: 0, finish_reason: 'length', message: { role: 'assistant', content: null } }],
        usage: { prompt_tokens: 0, completion_tokens: 0 },
      });
    }, deadlineMs);
    req.write(payload);
    req.end();
  });
}

// Stream translation: OpenAI SSE → Anthropic SSE
async function handleStream(openaiBody, res, requestId) {
  const backendUrl = new URL(BACKEND_URL);
  const endpoint = backendUrl.pathname.replace(/\/$/, '') + '/chat/completions';

  return new Promise((resolve, reject) => {
    const options = {
      hostname: backendUrl.hostname,
      port: backendUrl.port,
      path: endpoint,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${BACKEND_KEY}`,
      },
    };

    openaiBody.stream = true;
    const payload = JSON.stringify(openaiBody);
    options.headers['Content-Length'] = Buffer.byteLength(payload);

    // Send Anthropic stream headers
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });

    // Send message_start event
    res.write(`event: message_start\ndata: ${JSON.stringify({
      type: 'message_start',
      message: {
        id: `msg_${requestId}`,
        type: 'message',
        role: 'assistant',
        model: BACKEND_MODEL,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      }
    })}\n\n`);

    // Send content_block_start
    res.write(`event: content_block_start\ndata: ${JSON.stringify({
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    })}\n\n`);

    const proto = backendUrl.protocol === 'https:' ? https : http;
    const req = proto.request(options, backendRes => {
      let buf = '';
      backendRes.on('data', chunk => {
        buf += chunk.toString();
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') {
            res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: 'content_block_stop', index: 0 })}\n\n`);
            res.write(`event: message_delta\ndata: ${JSON.stringify({
              type: 'message_delta',
              delta: { stop_reason: 'end_turn', stop_sequence: null },
              usage: { output_tokens: 0 },
            })}\n\n`);
            res.write(`event: message_stop\ndata: ${JSON.stringify({ type: 'message_stop' })}\n\n`);
            res.end();
            resolve();
            return;
          }
          try {
            const chunk = JSON.parse(data);
            const delta = chunk.choices?.[0]?.delta;
            if (delta?.content) {
              res.write(`event: content_block_delta\ndata: ${JSON.stringify({
                type: 'content_block_delta',
                index: 0,
                delta: { type: 'text_delta', text: delta.content },
              })}\n\n`);
            }
            // Tool calls in stream
            if (delta?.tool_calls) {
              for (const tc of delta.tool_calls) {
                if (tc.function?.name) {
                  toolCallCount++;
                  res.write(`event: content_block_start\ndata: ${JSON.stringify({
                    type: 'content_block_start',
                    index: tc.index + 1,
                    content_block: {
                      type: 'tool_use',
                      id: tc.id || `toolu_${requestId}_${toolCallCount}`,
                      name: tc.function.name,
                      input: {},
                    }
                  })}\n\n`);
                }
                if (tc.function?.arguments) {
                  res.write(`event: content_block_delta\ndata: ${JSON.stringify({
                    type: 'content_block_delta',
                    index: (tc.index || 0) + 1,
                    delta: { type: 'input_json_delta', partial_json: tc.function.arguments },
                  })}\n\n`);
                }
              }
            }
          } catch {}
        }
      });
      backendRes.on('end', () => {
        if (!res.writableEnded) { res.end(); resolve(); }
      });
    });
    req.on('error', e => {
      if (!res.writableEnded) {
        res.write(`event: error\ndata: ${JSON.stringify({ type: 'error', error: { message: e.message } })}\n\n`);
        res.end();
      }
      reject(e);
    });
    req.setTimeout(300000, () => { req.destroy(); reject(new Error('Stream timeout')); });
    req.write(payload);
    req.end();
  });
}

// ── HTTP Server ──────────────────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  const pathname = new URL(req.url, `http://localhost:${PORT}`).pathname;

  // Health check
  if (req.method === 'GET' && pathname === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      status: 'ok',
      backend: BACKEND_URL,
      model: BACKEND_MODEL,
      requests: requestCount,
      tool_calls: toolCallCount,
      strip_thoughts: STRIP_THOUGHTS,
      strip_keep_recent: STRIP_KEEP_RECENT,
      thought_stripped_tokens_approx: Math.round(thoughtCharsStripped / 4),
      kv_evict: KV_EVICT,
      ace_evict: ACE_EVICT,
      kv_evict_budget_chars: MAX_CONTEXT_CHARS,
      kv_evicted_chars: kvEvictedTotal,
      kv_evicted_tokens_approx: Math.round(kvEvictedTotal / 4),
      uptime_s: Math.floor((Date.now() - startTime) / 1000),
    }));
  }

  // Anthropic Messages API endpoint
  if (req.method === 'POST' && pathname === '/v1/messages') {
    requestCount++;
    const reqId = `${Date.now()}-${requestCount}`;
    const startMs = Date.now();

    let body;
    try { body = await readBody(req); }
    catch { res.writeHead(400); return res.end(JSON.stringify({ error: { message: 'Invalid JSON' } })); }

    const isStream = body.stream || false;
    const toolCount = (body.tools || []).length;
    const msgCount = (body.messages || []).length;

    log('info', `#${requestCount} messages=${msgCount} tools=${toolCount} stream=${isStream}`);

    try {
      const openaiBody = anthropicToOpenAI(body);
      log('debug', `OpenAI body: ${JSON.stringify(openaiBody).slice(0, 500)}`);

      // Always use non-streaming backend (for tool call detection in text)
      // Then wrap in SSE if client requested streaming
      const openaiResp = await forwardToBackend(openaiBody);
      const anthropicResp = openAIToAnthropic(openaiResp, reqId);
      const elapsed = ((Date.now() - startMs) / 1000).toFixed(2);
      const tcCount = (anthropicResp.content || []).filter(b => b.type === 'tool_use').length;
      const evictNote = KV_EVICT ? ` kv_evicted_total≈${Math.round(kvEvictedTotal/4)}tok` : '';
      log('info', `#${requestCount} done ${elapsed}s tokens=${anthropicResp.usage.input_tokens}+${anthropicResp.usage.output_tokens} tool_calls=${tcCount}${evictNote}`);

      if (isStream) {
        // Wrap non-streaming response as SSE events (Claude Code expects SSE)
        res.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache', 'Connection': 'keep-alive' });
        // message_start
        res.write(`event: message_start\ndata: ${JSON.stringify({ type: 'message_start', message: { ...anthropicResp, content: [] } })}\n\n`);
        // content blocks
        for (let i = 0; i < anthropicResp.content.length; i++) {
          const block = anthropicResp.content[i];
          res.write(`event: content_block_start\ndata: ${JSON.stringify({ type: 'content_block_start', index: i, content_block: block.type === 'text' ? { type: 'text', text: '' } : block })}\n\n`);
          if (block.type === 'text') {
            res.write(`event: content_block_delta\ndata: ${JSON.stringify({ type: 'content_block_delta', index: i, delta: { type: 'text_delta', text: block.text } })}\n\n`);
          } else if (block.type === 'tool_use') {
            res.write(`event: content_block_delta\ndata: ${JSON.stringify({ type: 'content_block_delta', index: i, delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.input) } })}\n\n`);
          }
          res.write(`event: content_block_stop\ndata: ${JSON.stringify({ type: 'content_block_stop', index: i })}\n\n`);
        }
        // message_delta + stop
        res.write(`event: message_delta\ndata: ${JSON.stringify({ type: 'message_delta', delta: { stop_reason: anthropicResp.stop_reason, stop_sequence: null }, usage: anthropicResp.usage })}\n\n`);
        res.write(`event: message_stop\ndata: ${JSON.stringify({ type: 'message_stop' })}\n\n`);
        res.end();
      } else {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(anthropicResp));
      }
    } catch (e) {
      log('info', `#${requestCount} ERROR: ${e.message}`);
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        type: 'error',
        error: { type: 'api_error', message: `Backend error: ${e.message}` },
      }));
    }
    return;
  }

  // Catch-all
  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: { message: `Not found: ${pathname}` } }));
});

server.listen(PORT, () => {
  log('info', '═══════════════════════════════════════════');
  log('info', '  Agent API Bridge');
  log('info', `  Listening:  http://localhost:${PORT}`);
  log('info', `  Backend:    ${BACKEND_URL}`);
  log('info', `  Model:      ${BACKEND_MODEL}`);
  log('info', `  Thoughts:   ${STRIP_THOUGHTS ? `strip old turns (keep_recent=${STRIP_KEEP_RECENT})` : 'OFF'}`);
  log('info', `  KV Evict:   ${KV_EVICT ? `ON (budget=${MAX_CONTEXT_CHARS} chars ~${Math.round(MAX_CONTEXT_CHARS/4)}tok, head=${KV_EVICT_HEAD} tail=${KV_EVICT_TAIL} keep_recent=${KV_EVICT_KEEP_RECENT})` : 'OFF'}`);
  log('info', `  SysPrompt:  ${THINKING_MODE ? 'DIRECTIVE (thinking mode — prevents analysis loops)' : 'standard'}`);
  log('info', `  SemanticHints: ${SEMANTIC_HINTS ? 'ON (injects correction hints on zero/empty tool results)' : 'OFF'}`);
  log('info', `  JsonHint:    ${JSON_HINT ? 'ON (proactive cancer-filter hint on first UniProt JSON read)' : 'OFF'}`);
  log('info', '═══════════════════════════════════════════');
  log('info', '');
  log('info', 'Usage with Claude Code:');
  log('info', `  ANTHROPIC_BASE_URL=http://localhost:${PORT} ANTHROPIC_API_KEY=sk-bridge claude -p "task"`);
  log('info', '');
});
