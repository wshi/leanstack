from __future__ import annotations

import unittest

from leanstack.hot_kernels import get_hot_kernel_case, list_hot_kernel_cases


class HotKernelContractTests(unittest.TestCase):
    def test_all_cases_point_to_qwen_4b(self) -> None:
        for case in list_hot_kernel_cases():
            self.assertEqual(case.model_id, "Qwen/Qwen3-4B-Base")
            self.assertEqual(case.dtype, "bfloat16")

    def test_projection_shapes_match_4b_contract(self) -> None:
        q_proj = get_hot_kernel_case("q_proj_prefill64")
        kv_proj = get_hot_kernel_case("kv_proj_prefill64")
        o_proj = get_hot_kernel_case("o_proj_prefill64")
        gate_up = get_hot_kernel_case("gate_up_proj_prefill64")
        down_proj = get_hot_kernel_case("down_proj_prefill64")
        rmsnorm = get_hot_kernel_case("rmsnorm_prefill64")

        self.assertEqual((q_proj.k, q_proj.n), (2560, 4096))
        self.assertEqual((kv_proj.k, kv_proj.n), (2560, 1024))
        self.assertEqual((o_proj.k, o_proj.n), (4096, 2560))
        self.assertEqual((gate_up.k, gate_up.n), (2560, 9728))
        self.assertEqual((down_proj.k, down_proj.n), (9728, 2560))
        self.assertEqual(rmsnorm.hidden_size, 2560)


if __name__ == "__main__":
    unittest.main()

