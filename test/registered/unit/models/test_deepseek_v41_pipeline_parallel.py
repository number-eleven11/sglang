import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.deepseek_v4_hook import (  # noqa: E402
    _validate_deepseek_v41_contiguous_pp,
)
from sglang.srt.model_executor.cuda_graph_config import Backend  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors  # noqa: E402
from sglang.srt.models.deepseek_v4 import (  # noqa: E402
    DeepseekV4Model,
    deepseek_v4_idle_engram_hash_ids,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDeepseekV41PipelineParallel(CustomTestCase):
    @staticmethod
    def _pp_config(**overrides):
        disabled = SimpleNamespace(backend=Backend.DISABLED)
        values = dict(
            pp_size=2,
            cuda_graph_config=SimpleNamespace(decode=disabled, prefill=disabled),
            speculative_algorithm=None,
            attn_cp_size=1,
            dcp_size=1,
            enable_prefill_cp=False,
            disaggregation_mode="null",
            language_model_only=True,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _model(*, first_rank, last_rank):
        model = DeepseekV4Model.__new__(DeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(
            is_first_rank=first_rank,
            is_last_rank=last_rank,
        )
        model.hc_mult = 4
        model.hidden_size = 3
        model.embed_tokens = torch.nn.Embedding(8, model.hidden_size)
        model.hc_pre_from_prev_sublayer = True
        model.dspark_layers_to_capture = None
        model._can_run_tbo = mock.Mock(return_value=False)
        return model

    def test_accepts_supported_20_20_eager_layout(self):
        # The released checkpoint carries vision weights, so PP uses the
        # standard standalone text-only mode.
        hf_config = SimpleNamespace(num_hidden_layers=40, vision_n_layers=32)
        with mock.patch.dict(os.environ, {"SGLANG_PP_LAYER_PARTITION": "20,20"}):
            _validate_deepseek_v41_contiguous_pp(self._pp_config(), hf_config)

    def test_requires_language_model_only_for_multimodal_checkpoint(self):
        hf_config = SimpleNamespace(num_hidden_layers=40, vision_n_layers=32)
        cfg = self._pp_config(language_model_only=False)
        with mock.patch.dict(os.environ, {"SGLANG_PP_LAYER_PARTITION": "20,20"}):
            with self.assertRaisesRegex(ValueError, "multimodal serving"):
                _validate_deepseek_v41_contiguous_pp(cfg, hf_config)

    def test_rejects_partition_that_splits_shared_compressed_state(self):
        hf_config = SimpleNamespace(num_hidden_layers=40, vision_n_layers=0)
        with mock.patch.dict(os.environ, {"SGLANG_PP_LAYER_PARTITION": "19,21"}):
            with self.assertRaisesRegex(ValueError, "20/20"):
                _validate_deepseek_v41_contiguous_pp(self._pp_config(), hf_config)

    def test_rejects_cuda_graphs(self):
        hf_config = SimpleNamespace(num_hidden_layers=40, vision_n_layers=0)
        enabled = SimpleNamespace(backend=Backend.FULL)
        cfg = self._pp_config(
            cuda_graph_config=SimpleNamespace(decode=enabled, prefill=enabled)
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
            with self.assertRaisesRegex(ValueError, "CUDA graphs"):
                _validate_deepseek_v41_contiguous_pp(cfg, hf_config)

    @mock.patch(
        "sglang.srt.models.deepseek_v4.get_parallel",
        return_value=SimpleNamespace(attn_dp_size=1),
    )
    def test_first_stage_sends_hc_pre_state(self, _get_parallel):
        model = self._model(first_rank=True, last_rank=False)
        expected_pre = torch.randn(2, 4, 4)
        model._forward_layers_hc_pre_from_prev = mock.Mock(
            side_effect=lambda *args, **kwargs: (args[1], expected_pre, None)
        )

        output = model.forward(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            forward_batch=SimpleNamespace(),
            input_embeds=None,
        )

        self.assertIsInstance(output, PPProxyTensors)
        self.assertIs(output["hc_prev_pre"], expected_pre)
        self.assertEqual(tuple(output["hidden_states"].shape), (2, 12))

    @mock.patch(
        "sglang.srt.models.deepseek_v4.get_parallel",
        return_value=SimpleNamespace(attn_dp_size=1),
    )
    def test_next_stage_receives_hc_pre_state(self, _get_parallel):
        model = self._model(first_rank=False, last_rank=False)
        expected_pre = torch.randn(2, 4, 4)

        def forward_layers(*args, **kwargs):
            self.assertIs(kwargs["prev_pre"], expected_pre)
            return args[1], expected_pre, None

        model._forward_layers_hc_pre_from_prev = mock.Mock(side_effect=forward_layers)
        proxy = PPProxyTensors(
            {
                "hidden_states": torch.randn(2, 12),
                "hc_prev_pre": expected_pre,
            }
        )

        output = model.forward(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            forward_batch=SimpleNamespace(),
            input_embeds=None,
            pp_proxy_tensors=proxy,
        )

        self.assertIs(output["hc_prev_pre"], expected_pre)

    def test_next_stage_requires_hc_pre_state(self):
        model = self._model(first_rank=False, last_rank=False)
        proxy = PPProxyTensors({"hidden_states": torch.randn(2, 12)})

        with self.assertRaisesRegex(ValueError, "hc_prev_pre"):
            model.forward(
                input_ids=torch.tensor([1, 2]),
                positions=torch.tensor([0, 1]),
                forward_batch=SimpleNamespace(),
                input_embeds=None,
                pp_proxy_tensors=proxy,
            )

    def test_idle_dp_rank_uses_padding_engram_hashes(self):
        hasher = SimpleNamespace(
            primes=torch.empty(2, 3, 4),
            offsets=torch.empty(2, 3),
        )
        hashes = deepseek_v4_idle_engram_hash_ids(
            hasher, torch.tensor([17, 23], dtype=torch.int32)
        )
        self.assertEqual(tuple(hashes.shape), (2, 2, 3))
        self.assertEqual(hashes.dtype, torch.int64)
        self.assertEqual(hashes.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
