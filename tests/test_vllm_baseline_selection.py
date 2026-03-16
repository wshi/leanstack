from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from leanstack import compare_runner
from leanstack.compare_runner import CommandResult


def _payload(tokens_per_second: float) -> dict[str, object]:
    return {
        "system": "vllm",
        "variant": "openai",
        "base_url": "http://127.0.0.1:8000",
        "generated_tokens_per_second": tokens_per_second,
        "end_to_end_tokens_per_second": tokens_per_second - 1.0,
        "ttft_seconds": 0.01,
        "stream_seconds": 1.0,
        "end_to_end_seconds": 1.0,
        "generated_tokens": 64,
        "prompt_tokens": 64,
        "generated_text": "ok",
    }


def _result(tokens_per_second: float) -> CommandResult:
    return CommandResult(
        command=("remote_openai_backend_benchmark.sh",),
        stdout=json.dumps(_payload(tokens_per_second)),
        stderr="",
        returncode=0,
    )


class VllmBaselineSelectionTests(unittest.TestCase):
    def test_best_mode_selects_highest_tokens_per_second(self) -> None:
        with patch.object(
            compare_runner,
            "_run_shell_script",
            side_effect=[_result(40.0), _result(48.5), _result(44.2)],
        ) as run_mock:
            selected = compare_runner.run_vllm_benchmark(
                prompt="hello",
                baseline_mode="best",
                baseline_runs=3,
            )

        self.assertEqual(run_mock.call_count, 3)
        self.assertEqual(selected["baseline_mode"], "best")
        self.assertEqual(selected["baseline_runs"], 3)
        self.assertEqual(selected["generated_tokens_per_second"], 48.5)
        self.assertEqual(selected["baseline_candidates_tokens_per_second"], [40.0, 48.5, 44.2])

    def test_plain_mode_runs_once(self) -> None:
        with patch.object(compare_runner, "_run_shell_script", return_value=_result(42.0)) as run_mock:
            selected = compare_runner.run_vllm_benchmark(
                prompt="hello",
                baseline_mode="plain",
                baseline_runs=5,
            )

        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(run_mock.call_args.kwargs["env"]["TEMPERATURE"], "0.0")
        self.assertEqual(run_mock.call_args.kwargs["env"]["IGNORE_EOS"], "1")
        self.assertEqual(selected["baseline_mode"], "plain")
        self.assertEqual(selected["baseline_runs"], 1)
        self.assertEqual(selected["generated_tokens_per_second"], 42.0)


if __name__ == "__main__":
    unittest.main()
