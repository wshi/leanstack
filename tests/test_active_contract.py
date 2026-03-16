from __future__ import annotations

import unittest

from leanstack import compare_runner
from leanstack.benchmark import get_benchmark_profile
from leanstack.model_registry import get_model_spec


class ActiveContractTests(unittest.TestCase):
    def test_qwen_model_registry_points_to_4b_contract(self) -> None:
        spec = get_model_spec("qwen")
        self.assertEqual(spec.semantic_model_id, "Qwen/Qwen3-4B-Base")
        self.assertEqual(spec.artifact_model_id, "Qwen/Qwen3-4B-Base")
        self.assertEqual(spec.remote_model_key, "Qwen__Qwen3-4B-Base")
        self.assertEqual(spec.num_hidden_layers, 36)
        self.assertEqual(spec.hidden_size, 2560)
        self.assertEqual(spec.intermediate_size, 9728)
        self.assertEqual(spec.num_attention_heads, 32)
        self.assertEqual(spec.num_key_value_heads, 8)
        self.assertEqual(spec.head_dim, 128)

    def test_compare_runner_defaults_match_4b_contract(self) -> None:
        self.assertEqual(compare_runner.DEFAULT_MODEL_ID, "Qwen/Qwen3-4B-Base")
        self.assertEqual(compare_runner.DEFAULT_MODEL_NAME, "qwen3-4b-base")
        self.assertEqual(compare_runner.DEFAULT_MODEL_PATH_FILE, "/home/pto/lean/models/Qwen__Qwen3-4B-Base.path")
        self.assertEqual(compare_runner.DEFAULT_PACK_DIR, "/home/pto/lean/packed/Qwen__Qwen3-4B-Base")
        self.assertEqual(compare_runner.DEFAULT_VLLM_BASELINE_MODE, "best")
        self.assertEqual(compare_runner.DEFAULT_VLLM_BASELINE_RUNS, 3)
        self.assertEqual(compare_runner.DEFAULT_DECODE_POLICY, "greedy")
        self.assertEqual(compare_runner.DEFAULT_SAMPLING_TEMPERATURE, 0.0)
        self.assertTrue(compare_runner.DEFAULT_IGNORE_EOS)

    def test_primary_profile_prompt_tracks_4b_contract(self) -> None:
        profile = get_benchmark_profile("decode_64_256")
        self.assertIn("Qwen3-4B-Base", profile.goal)
        self.assertIn("Qwen3-4B-Base", profile.prompt)


if __name__ == "__main__":
    unittest.main()
