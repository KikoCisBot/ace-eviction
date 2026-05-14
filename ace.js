/**
 * ACE — Attention-Weighted Context Eviction
 *
 * Content-aware context compression for multi-turn agentic LLM inference.
 * Replaces head/tail truncation with line-level importance scoring so that
 * tool results are compressed rather than blindly sliced.
 *
 * Usage (Node.js):
 *   const { aceCompress, aceEvict } = require('./ace');
 *
 *   // Compress a single tool-result string to ~40% of its original size,
 *   // keeping the highest-importance lines.
 *   const compressed = aceCompress(toolResultText, 0.4);
 *
 *   // Evict old tool results from a messages array in-place until the
 *   // total estimated size is under budgetChars.
 *   aceEvict(messages, { budgetChars: 40000, keepRecent: 2, targetRatio: 0.4 });
 */

'use strict';

/**
 * Score a single line for importance.
 * Returns a float in [0, 1] — higher = keep.
 *
 * Heuristic priority order:
 *  1.0  tool call JSON (name + arguments)
 *  0.95 error / exception lines
 *  0.9  file / workspace paths
 *  0.85 task-framing keywords (task:, step, verify…)
 *  0.7  numeric data (%, large numbers, units)
 *  0.6  shell commands
 *  0.5  default (unknown content)
 *  0.3  success boilerplate (done, ok, installed…)
 *  0.2  very long lines (>200 chars — usually verbose dumps)
 *  0.1  meta-commentary ("here are the results", "i executed"…)
 *  0.0  blank lines
 */
function aceClassifyLine(s) {
  const sl = s.toLowerCase();
  if (!s.trim()) return 0.0;
  // Tool call JSON
  if (s.includes('"name"') && (s.includes('"arguments"') || s.includes('"parameters"'))) return 1.0;
  // Errors
  const errorPhrases = ['error', 'failed', 'not found', 'traceback', 'exception', 'exit code'];
  if (errorPhrases.some(p => sl.includes(p))) return 0.95;
  // File paths
  if (s.includes('/results/') || s.includes('/workspace/') || s.includes('/tmp/')) return 0.9;
  // Task keywords
  if (['task:', 'begin', 'step ', 'verify', 'inspect'].some(k => sl.includes(k))) return 0.85;
  // Numeric data
  if (/\d+\.?\d*%|\d{4,}|bytes|rows|columns/.test(sl)) return 0.7;
  // Shell commands
  if (['$', '>>>', 'root@', 'agent@'].some(k => s.startsWith(k)) ||
      ['wget', 'curl', 'pip', 'python3', 'apt-get', 'docker'].some(k => sl.includes(k))) return 0.6;
  // Success boilerplate
  if (['done', 'ok', 'installed', 'saved', 'complete'].includes(sl.trim())) return 0.3;
  // Long verbose dump
  if (s.length > 200) return 0.2;
  // Meta-commentary
  if (['i executed', 'here are the results', 'tool results:'].some(k => sl.includes(k))) return 0.1;
  return 0.5;
}

/**
 * Compress a multi-line string to approximately `targetRatio` of its original
 * line count, keeping the highest-scoring lines.
 *
 * Always keeps the first and last line.
 * Omitted runs are replaced with a single "[...N lines omitted by ACE...]" marker.
 *
 * @param {string}  content      - The text to compress.
 * @param {number}  targetRatio  - Fraction of lines to keep (default 0.4).
 * @returns {string}
 */
function aceCompress(content, targetRatio = 0.4) {
  const lines = content.split('\n');
  if (lines.length <= 3) return content;

  const scored = lines.map((line, i) => {
    let score = aceClassifyLine(line);
    // Always protect first and last line
    if (i === 0 || i === lines.length - 1) score = Math.max(score, 0.7);
    return { i, line, score };
  });

  const nKeep = Math.max(2, Math.floor(lines.length * targetRatio));
  const keepSet = new Set(
    [...scored].sort((a, b) => b.score - a.score).slice(0, nKeep).map(x => x.i)
  );
  keepSet.add(0);
  keepSet.add(lines.length - 1);

  const out = [];
  let skipped = 0;
  for (let i = 0; i < lines.length; i++) {
    if (keepSet.has(i)) {
      if (skipped > 0) out.push(`  [...${skipped} lines omitted by ACE...]`);
      skipped = 0;
      out.push(lines[i]);
    } else {
      skipped++;
    }
  }
  if (skipped > 0) out.push(`  [...${skipped} lines omitted by ACE...]`);
  return out.join('\n');
}

/**
 * Evict (compress) old tool results from an Anthropic-style messages array
 * until the estimated total size is under `budgetChars`.
 *
 * Operates on messages that start with "[Tool result]: " (the format used by
 * agent-bridge.js). Skips the `keepRecent` most-recent tool results.
 * Messages are mutated in-place.
 *
 * @param {Array}   messages           - Anthropic messages array.
 * @param {Object}  opts
 * @param {number}  opts.budgetChars   - Target max total chars (default 40000).
 * @param {number}  opts.keepRecent    - Tool-result messages to leave untouched (default 2).
 * @param {number}  opts.targetRatio   - Compression ratio for aceCompress (default 0.4).
 * @param {number}  opts.minSize       - Don't compress messages shorter than this (default 200).
 * @returns {number} Total chars removed.
 */
function aceEvict(messages, opts = {}) {
  const {
    budgetChars  = 40000,
    keepRecent   = 2,
    targetRatio  = 0.4,
    minSize      = 200,
  } = opts;

  const estimateSize = (msgs) =>
    msgs.reduce((s, m) => s + (typeof m.content === 'string' ? m.content.length : 0), 0);

  if (estimateSize(messages) <= budgetChars) return 0;

  const toolIndices = messages
    .map((m, i) => (m.role === 'user' && typeof m.content === 'string'
      && m.content.startsWith('[Tool result]: ') ? i : -1))
    .filter(i => i >= 0);

  const compressable = toolIndices.slice(0, toolIndices.length - keepRecent);
  let totalSaved = 0;

  for (const idx of compressable) {
    if (estimateSize(messages) <= budgetChars) break;
    const prefix = '[Tool result]: ';
    const result = messages[idx].content.slice(prefix.length);
    if (result.length <= minSize) continue;
    if (result.includes('[...ACE') || result.includes('omitted by ACE')) continue;

    const compressed = aceCompress(result, targetRatio);
    const saved = result.length - compressed.length;
    if (saved <= 0) continue;

    totalSaved += saved;
    messages[idx].content = prefix + compressed;
  }

  return totalSaved;
}

module.exports = { aceClassifyLine, aceCompress, aceEvict };
