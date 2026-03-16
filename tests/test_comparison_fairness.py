from __future__ import annotations

import unittest

from leanstack.compare_runner import validate_comparison_fairness


class ComparisonFairnessTests(unittest.TestCase):
    def test_fairness_gate_accepts_matching_contract(self) -> None:
        vllm = {
            "prompt_tokens": 64,
            "generated_tokens": 256,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        leanstack = {
            "prompt_tokens": 64,
            "emitted_tokens": 256,
            "max_new_tokens": 256,
            "decode_policy": "greedy",
            "ignore_eos": True,
        }

        payload = validate_comparison_fairness(
            vllm=vllm,
            leanstack=leanstack,
            expected_max_new_tokens=256,
        )
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["generated_tokens"], 256)

    def test_fairness_gate_rejects_temperature_mismatch(self) -> None:
        vllm = {
            "prompt_tokens": 64,
            "generated_tokens": 256,
            "temperature": 0.7,
            "ignore_eos": True,
        }
        leanstack = {
            "prompt_tokens": 64,
            "emitted_tokens": 256,
            "max_new_tokens": 256,
            "decode_policy": "greedy",
            "ignore_eos": True,
        }
        with self.assertRaises(RuntimeError) as ctx:
            validate_comparison_fairness(vllm=vllm, leanstack=leanstack, expected_max_new_tokens=256)
        self.assertIn("temperature must be 0.0", str(ctx.exception))

    def test_fairness_gate_rejects_generated_token_mismatch(self) -> None:
        vllm = {
            "prompt_tokens": 64,
            "generated_tokens": 248,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        leanstack = {
            "prompt_tokens": 64,
            "emitted_tokens": 256,
            "max_new_tokens": 256,
            "decode_policy": "greedy",
            "ignore_eos": True,
        }
        with self.assertRaises(RuntimeError) as ctx:
            validate_comparison_fairness(vllm=vllm, leanstack=leanstack, expected_max_new_tokens=256)
        self.assertIn("generated token mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
