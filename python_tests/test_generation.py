"""
Tests for llama-cli output handling shared by the harness and the golden generator.
"""

from autoluce.bench.harness import extract_generated_text


def test_extract_generated_text_drops_timings_block():
    stdout = (
        "def factorial(n):\n"
        "    return 1 if n <= 1 else n * factorial(n - 1)\n"
        "\n"
        "llama_print_timings:        load time = ...\n"
        "llama_print_timings:      sample time = ...\n"
    )
    assert extract_generated_text(stdout) == (
        "def factorial(n):\n"
        "    return 1 if n <= 1 else n * factorial(n - 1)"
    )
