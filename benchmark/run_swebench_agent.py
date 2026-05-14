#!/usr/bin/env python3
"""
SWE-bench multi-turn agent via agent-bridge.
Clones repo at correct commit, lets model use Read/Bash tools to explore,
generates a patch, evaluates it.

Usage:
  python3 run_swebench_agent.py --api-url http://0.0.0.0:3210/v1 --limit 5 --label baseline
  ACE_EVICT=1 → restart bridge with ACE_EVICT=1 env var first
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PROBLEMS_PATH = "/tmp/swebench_problems_20.json"


def call_api(api_url, messages, model="bridge", max_tokens=4096, system=""):
    """Call the agent-bridge (Anthropic Messages API format)."""
    # Separate system message from user/assistant messages
    anthropic_messages = [m for m in messages if m["role"] != "system"]
    payload = {
        "model":      model,
        "messages":   anthropic_messages,
        "max_tokens": max_tokens,
        "system":     system,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        api_url.rstrip("/") + "/messages", data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer bridge-key",
                 "anthropic-version": "2023-06-01"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        body = json.loads(resp.read())
        # Normalize to OpenAI-style for caller
        text = ""
        for block in body.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}
    except Exception as e:
        return {"error": str(e)}


def extract_patch(text):
    for p in [r'```diff\n(.*?)```', r'```patch\n(.*?)```']:
        m = re.findall(p, text, re.DOTALL)
        if m:
            return m[0].strip()
    lines = text.split('\n')
    diff = [l for l in lines if re.match(r'^(---|\\+\\+\\+|@@)', l)]
    return '\n'.join(diff).strip() if len(diff) > 2 else ""


def evaluate_patch(predicted, ground_truth):
    if not predicted:
        return {"match": False, "score": 0, "reason": "no_patch"}
    pred_files = set(re.findall(r'[ab]/(\S+)', predicted))
    gt_files   = set(re.findall(r'[ab]/(\S+)', ground_truth))
    file_overlap = len(pred_files & gt_files) / max(len(gt_files), 1)
    pred_adds = set(l.strip() for l in predicted.split('\n') if l.startswith('+') and not l.startswith('+++'))
    gt_adds   = set(l.strip() for l in ground_truth.split('\n') if l.startswith('+') and not l.startswith('+++'))
    pred_dels = set(l.strip() for l in predicted.split('\n') if l.startswith('-') and not l.startswith('---'))
    gt_dels   = set(l.strip() for l in ground_truth.split('\n') if l.startswith('-') and not l.startswith('---'))
    add_ov = len(pred_adds & gt_adds) / max(len(gt_adds), 1)
    del_ov = len(pred_dels & gt_dels) / max(len(gt_dels), 1)
    score = file_overlap * 0.3 + add_ov * 0.4 + del_ov * 0.3
    return {"match": score > 0.5, "score": round(score, 3), "file_overlap": round(file_overlap, 3)}


def clone_repo(repo, commit, workdir):
    """Clone repo at specific commit into workdir."""
    repo_url = f"https://github.com/{repo}.git"
    clone_dir = os.path.join(workdir, repo.replace("/", "__"))
    if os.path.exists(clone_dir):
        return clone_dir
    try:
        subprocess.run(["git", "clone", "--depth=50", repo_url, clone_dir],
                       capture_output=True, timeout=60)
        subprocess.run(["git", "checkout", commit], cwd=clone_dir,
                       capture_output=True, timeout=30)
    except Exception:
        pass
    return clone_dir if os.path.exists(clone_dir) else None


def run_agent(api_url, problem, repo_dir, max_turns=6, min_explore_turns=2):
    """Run multi-turn agent on a SWE-bench problem.
    min_explore_turns: require this many file-read turns before accepting a patch.
    """
    repo = problem["repo"]
    statement = problem["problem_statement"]
    hints = problem.get("hints_text", "")

    system = (
        "You are a software engineer fixing bugs in a GitHub repository. "
        "You have access to the repository files. "
        "You MUST first explore at least 2-3 relevant source files before generating a patch. "
        "Say 'read <filename>' to read a file. "
        "After exploring the code, output the fix as a unified diff in a ```diff block."
    )

    user_first = (
        f"Repository: {repo}\n"
        f"Repository is available at: {repo_dir}\n\n"
        f"Issue:\n{statement[:2000]}\n\n"
        + (f"Hints: {hints[:500]}\n\n" if hints else "")
        + "Step 1: Read the most relevant source files to understand the bug.\n"
        + "Step 2: After reading files, generate the fix as a unified diff."
    )

    messages = [{"role": "user", "content": user_first}]
    total_chars_per_turn = []
    turns_used = 0
    explore_turns = 0  # number of turns where we read a file

    for turn in range(max_turns):
        total_chars_per_turn.append(sum(len(m.get("content", "")) for m in messages))
        resp = call_api(api_url, messages)
        if "error" in resp:
            break

        choice = resp.get("choices", [{}])[0]
        assistant_msg = choice.get("message", {})
        content = assistant_msg.get("content", "") or ""
        turns_used += 1

        messages.append({"role": "assistant", "content": content})

        # Only accept a patch after min_explore_turns of file reading
        if extract_patch(content) and explore_turns >= min_explore_turns:
            break

        # Check if model is asking to read a file (simple heuristic)
        file_read_match = re.search(
            r'(?:read|open|look at|examine|check)\s+[`"]?([/\w\-_.]+\.\w+)[`"]?',
            content, re.IGNORECASE
        )
        if file_read_match and repo_dir:
            fname = file_read_match.group(1)
            fpath = os.path.join(repo_dir, fname.lstrip("/"))
            if os.path.exists(fpath):
                try:
                    file_content = open(fpath).read()[:3000]
                    tool_result = f"Content of {fname}:\n```\n{file_content}\n```"
                    explore_turns += 1
                except Exception:
                    tool_result = f"Could not read {fname}"
            else:
                # Try to find the file
                found = list(Path(repo_dir).rglob(os.path.basename(fname)))
                if found:
                    try:
                        file_content = open(found[0]).read()[:3000]
                        tool_result = f"Content of {found[0].relative_to(repo_dir)}:\n```\n{file_content}\n```"
                        explore_turns += 1
                    except Exception:
                        tool_result = f"Found {found[0]} but could not read it"
                else:
                    tool_result = f"File {fname} not found in repository"
            suffix = "" if explore_turns >= min_explore_turns else \
                " Read more files if needed before generating the patch."
            messages.append({"role": "user", "content": f"[Tool result]: {tool_result}{suffix}"})
        else:
            # No file read detected
            if explore_turns < min_explore_turns:
                messages.append({"role": "user", "content":
                    f"You need to read source files first ({explore_turns}/{min_explore_turns} done). "
                    "Say 'read <filename>' to read a file from the repository."})
            else:
                messages.append({"role": "user", "content":
                    "Continue. Generate the patch now in ```diff format."})

    # Extract final patch from conversation
    for m in reversed(messages):
        if m["role"] == "assistant":
            patch = extract_patch(m.get("content", ""))
            if patch:
                return patch, turns_used, total_chars_per_turn

    return "", turns_used, total_chars_per_turn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://0.0.0.0:3210/v1")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--label", default="bridge")
    parser.add_argument("--no-clone", action="store_true", help="Skip repo cloning")
    args = parser.parse_args()

    problems = json.load(open(PROBLEMS_PATH))[:args.limit]

    # Check bridge health
    try:
        r = urllib.request.urlopen(f"{args.api_url.rstrip('/v1')}/health", timeout=5)
        health = json.loads(r.read())
        ace = health.get("ace_evict", False)
        kv  = health.get("kv_evict", False)
        print(f"Bridge: kv_evict={kv}  ace_evict={ace}  model={health.get('model','?')[:40]}")
    except Exception as e:
        print(f"Bridge unreachable: {e}")
        sys.exit(1)

    output_dir = "/tmp/swebench-results"
    os.makedirs(output_dir, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="swe_repos_")

    results = []
    correct = 0; total = 0
    start = time.time()
    all_chars_growth = []

    for i, p in enumerate(problems):
        instance_id = p["instance_id"]
        repo        = p["repo"]
        commit      = p.get("base_commit", "HEAD")
        gt          = p["patch"]

        print(f"\n[{i+1}/{args.limit}] {instance_id}", flush=True)

        repo_dir = None
        if not args.no_clone:
            print("  cloning...", flush=True)
            repo_dir = clone_repo(repo, commit, work_dir)
            if not repo_dir:
                print("  clone failed, proceeding without repo")

        t_start = time.time()
        predicted, turns, chars_per_turn = run_agent(args.api_url, p, repo_dir)
        elapsed = time.time() - t_start

        ev = evaluate_patch(predicted, gt)
        total += 1
        if ev["match"]:
            correct += 1
            print(f"  ✓ MATCH  score={ev['score']}  turns={turns}  time={elapsed:.0f}s")
        else:
            print(f"  ✗ miss   score={ev['score']}  turns={turns}  time={elapsed:.0f}s")

        if len(chars_per_turn) > 1:
            growth = chars_per_turn[-1] - chars_per_turn[0]
            print(f"    ctx: {chars_per_turn[0]}→{chars_per_turn[-1]} chars (+{growth})")
            all_chars_growth.append(growth)

        results.append({"instance_id": instance_id, "repo": repo,
                         "status": "match" if ev["match"] else "miss",
                         "eval": ev, "turns": turns,
                         "context_chars_per_turn": chars_per_turn})

    elapsed_total = time.time() - start
    accuracy = correct / total * 100 if total else 0
    avg_growth = sum(all_chars_growth) / len(all_chars_growth) if all_chars_growth else 0

    summary = {"label": args.label, "total": total, "correct": correct,
               "accuracy": round(accuracy, 2),
               "avg_context_growth_chars": int(avg_growth),
               "elapsed_seconds": int(elapsed_total)}

    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', args.label)
    out = os.path.join(output_dir, f"swebench_agent_{safe}.json")
    with open(out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n{'='*56}")
    print(f"  SWE-bench Agent — {args.label}")
    print(f"  Accuracy:     {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Ctx growth:   {int(avg_growth)} chars/problem avg")
    print(f"  Time:         {int(elapsed_total)}s")
    print(f"  Results:      {out}")
    print(f"{'='*56}")


if __name__ == "__main__":
    main()
