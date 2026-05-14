"""Tests for ace_eviction.classifier."""

import pytest
from ace_eviction.classifier import classify_line


# ---------------------------------------------------------------------------
# Blank / whitespace
# ---------------------------------------------------------------------------

class TestBlankLines:
    def test_empty_string(self):
        assert classify_line("") == 0.0

    def test_whitespace_only(self):
        assert classify_line("   \t  ") == 0.0


# ---------------------------------------------------------------------------
# Tool call JSON (1.00)
# ---------------------------------------------------------------------------

class TestToolCallJSON:
    def test_full_json_tool_call(self):
        line = '{"name": "read_file", "arguments": {"path": "/tmp/foo.py"}}'
        assert classify_line(line) == 1.0

    def test_full_json_with_input_key(self):
        line = '{"name": "bash", "input": {"command": "ls -la"}}'
        assert classify_line(line) == 1.0

    def test_inline_tool_call_pattern(self):
        line = '  {"name": "search", "arguments": {}}'
        assert classify_line(line) == 1.0


# ---------------------------------------------------------------------------
# Errors / exceptions (0.95)
# ---------------------------------------------------------------------------

class TestErrorLines:
    def test_traceback(self):
        assert classify_line("Traceback (most recent call last):") == 0.95

    def test_error_keyword(self):
        assert classify_line("FileNotFoundError: /tmp/missing.txt") == 0.95

    def test_exception_keyword(self):
        assert classify_line("Exception raised during evaluation") == 0.95

    def test_fatal(self):
        assert classify_line("FATAL: connection refused") == 0.95


# ---------------------------------------------------------------------------
# Paths (0.90)
# ---------------------------------------------------------------------------

class TestPathLines:
    def test_tmp_path(self):
        assert classify_line("Output written to /tmp/results.json") == 0.90

    def test_workspace_path(self):
        assert classify_line("/workspace/src/model.py loaded") == 0.90

    def test_py_extension(self):
        assert classify_line("Processing file main.py") == 0.90

    def test_yaml_extension(self):
        assert classify_line("Config loaded from config.yaml") == 0.90


# ---------------------------------------------------------------------------
# Task-framing keywords (0.85)
# ---------------------------------------------------------------------------

class TestTaskKeywords:
    def test_task_colon(self):
        assert classify_line("task: run the evaluation suite") == 0.85

    def test_step_number(self):
        assert classify_line("Step 3: verify the output") == 0.85

    def test_verify_keyword(self):
        assert classify_line("Verify that the model outputs are correct") == 0.85

    def test_inspect_keyword(self):
        assert classify_line("Inspect the intermediate activations") == 0.85


# ---------------------------------------------------------------------------
# Numeric data (0.70)
# ---------------------------------------------------------------------------

class TestNumericLines:
    def test_percentage(self):
        assert classify_line("Accuracy: 87.3%") == 0.70

    def test_large_number(self):
        assert classify_line("Total tokens used: 12345") == 0.70

    def test_megabytes(self):
        assert classify_line("Memory footprint: 4096 MB") == 0.70

    def test_milliseconds(self):
        assert classify_line("Latency p50: 234ms") == 0.70


# ---------------------------------------------------------------------------
# Shell commands (0.60)
# ---------------------------------------------------------------------------

class TestShellLines:
    def test_dollar_prompt(self):
        assert classify_line("$ ls -la /workspace") == 0.60

    def test_curl(self):
        assert classify_line("curl https://api.example.com/v1/health") == 0.60

    def test_python3(self):
        # "evaluate.py" contains a .py extension, so the path rule (0.90)
        # fires before the shell rule.  Use a plain command without a .py arg.
        assert classify_line("python3 --version") == 0.60

    def test_docker(self):
        assert classify_line("docker run --rm myimage:latest") == 0.60

    def test_pip(self):
        assert classify_line("pip install -r requirements.txt") == 0.60


# ---------------------------------------------------------------------------
# Success boilerplate (0.30)
# ---------------------------------------------------------------------------

class TestSuccessBoilerplate:
    def test_done(self):
        assert classify_line("done") == 0.30

    def test_ok(self):
        assert classify_line("ok") == 0.30

    def test_installed(self):
        assert classify_line("installed") == 0.30

    def test_saved(self):
        assert classify_line("saved.") == 0.30

    def test_complete(self):
        assert classify_line("complete") == 0.30


# ---------------------------------------------------------------------------
# Very long lines (0.20)
# ---------------------------------------------------------------------------

class TestLongLines:
    def test_long_line(self):
        line = "x" * 201
        assert classify_line(line) == 0.20

    def test_exactly_200_chars_is_default(self):
        # 200 chars exactly does NOT trigger the long-line rule
        line = "a" * 200
        assert classify_line(line) == 0.50


# ---------------------------------------------------------------------------
# Meta-commentary (0.10)
# ---------------------------------------------------------------------------

class TestMetaCommentary:
    def test_here_are(self):
        assert classify_line("Here are the results of the evaluation:") == 0.10

    def test_here_is(self):
        assert classify_line("Here is the output you requested.") == 0.10

    def test_as_you_can_see(self):
        assert classify_line("As you can see, the model converged.") == 0.10

    def test_please_find(self):
        assert classify_line("Please find the attached report below.") == 0.10


# ---------------------------------------------------------------------------
# Default (0.50)
# ---------------------------------------------------------------------------

class TestDefault:
    def test_plain_prose(self):
        assert classify_line("The model was trained for three epochs.") == 0.50

    def test_short_statement(self):
        assert classify_line("Evaluation started.") == 0.50
