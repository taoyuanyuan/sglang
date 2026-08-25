from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.server_args import MOE_A2A_BACKEND_CHOICES
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def _config():
    return MoeRunnerConfig(
        num_experts=4,
        num_local_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=128,
        top_k=2,
        num_fused_shared_experts=0,
        params_dtype=torch.bfloat16,
    )


def test_hpc_ops_is_a_first_class_a2a_backend():
    assert MoeA2ABackend("hpc_ops").is_hpc_ops()
    assert "hpc_ops" in MOE_A2A_BACKEND_CHOICES


@patch(
    "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
    return_value=SimpleNamespace(is_pplx=lambda: False),
)
@patch("sglang.srt.layers.dp_attention.get_attention_dp_size", return_value=8)
def test_peer_input_keeps_uneven_prefill_rows(_dp_size, _backend):
    assert (
        DpPaddingMode.get_dp_padding_mode(True, [3, 2, 0, 0, 0, 0, 0, 0])
        == DpPaddingMode.SUM_LEN
    )


@pytest.mark.parametrize(
    ("is_extend", "expected_pull_input"),
    [(False, False), (True, True)],
)
@patch(
    "sglang.srt.layers.moe.token_dispatcher.hpc_ops.get_dp_global_num_tokens",
    return_value=[3, 2],
)
@patch("sglang.srt.layers.moe.token_dispatcher.hpc_ops.get_is_extend_in_batch")
@patch("sglang.srt.layers.moe.token_dispatcher.hpc_ops._get_peer_workspace")
@patch("sglang.srt.layers.moe.token_dispatcher.hpc_ops.get_server_args")
@patch("sglang.srt.layers.moe.token_dispatcher.hpc_ops.get_parallel")
def test_dispatch_publishes_peer_indexed_input_without_equal_padding(
    get_parallel,
    get_server_args,
    get_workspace,
    get_is_extend,
    _global_tokens,
    is_extend,
    expected_pull_input,
):
    from sglang.srt.layers.moe.token_dispatcher.hpc_ops import HpcOpsDispatcher

    workspace = Mock()
    get_workspace.return_value = workspace
    get_is_extend.return_value = is_extend
    get_parallel.return_value = SimpleNamespace(
        moe_ep_size=2,
        moe_ep_rank=0,
        moe_ep_group=SimpleNamespace(cpu_group=object()),
    )
    get_server_args.return_value = SimpleNamespace(
        enable_dp_attention=True,
        chunked_prefill_size=8,
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=4)),
        max_running_requests=4,
    )
    dispatcher = HpcOpsDispatcher(_config())
    hidden_states = torch.randn((3, 128), dtype=torch.bfloat16)
    topk_output = StandardTopKOutput(
        topk_weights=torch.rand((3, 2), dtype=torch.float32),
        topk_ids=torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64),
        router_logits=None,
    )

    output = dispatcher.dispatch(hidden_states, topk_output)

    workspace.publish_input.assert_called_once_with(
        hidden_states,
        topk_output.topk_ids,
        topk_output.topk_weights,
    )
    assert output.workspace is workspace
    assert output.tokens_per_owner == (3, 2)
    assert output.local_output_rows == 3
    assert output.pull_input is expected_pull_input


def test_hpc_ops_runner_uses_phase_selected_peer_input():
    from sglang.srt.layers.moe.moe_runner.hpc_ops import (
        HpcOpsMoeQuantInfo,
        fused_experts_hpc_ops_to_hpc_ops,
    )
    from sglang.srt.layers.moe.token_dispatcher.hpc_ops import HpcOpsDispatchOutput

    workspace = Mock()
    workspace.consume.return_value = torch.full((3, 128), 2, dtype=torch.bfloat16)
    topk_output = StandardTopKOutput(
        topk_weights=torch.rand((3, 2), dtype=torch.float32),
        topk_ids=torch.zeros((3, 2), dtype=torch.int64),
        router_logits=None,
    )
    dispatch_output = HpcOpsDispatchOutput(
        hidden_states=torch.empty((3, 128), dtype=torch.bfloat16),
        hidden_states_scale=None,
        workspace=workspace,
        tokens_per_owner=(3, 2),
        local_output_rows=3,
        pull_input=True,
        topk_output=topk_output,
    )
    quant_info = HpcOpsMoeQuantInfo(
        w13_weight=torch.empty((2, 256, 128), dtype=torch.float8_e4m3fn),
        w2_weight=torch.empty((2, 128, 128), dtype=torch.float8_e4m3fn),
        block_quant=True,
        global_num_experts=4,
        moe_ep_rank=0,
        w13_weight_scale_inv=torch.ones((2, 2, 4)),
        w2_weight_scale_inv=torch.ones((2, 1, 4)),
        block_shape=[128, 128],
    )
    config = _config()
    config.routed_scaling_factor = 0.5
    config.swiglu_limit = 10.0

    result = fused_experts_hpc_ops_to_hpc_ops(dispatch_output, quant_info, config)

    workspace.consume.assert_called_once_with(
        w13=quant_info.w13_weight,
        w13_scale=quant_info.w13_weight_scale_inv,
        w2=quant_info.w2_weight,
        w2_scale=quant_info.w2_weight_scale_inv,
        num_experts_total=4,
        tokens_per_owner=(3, 2),
        local_output_rows=3,
        pull_input=True,
        swiglu_limit=10.0,
    )
    torch.testing.assert_close(
        result.hidden_states,
        torch.full((3, 128), 1, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    ("unsupported_flag", "message"),
    [
        ("enable_two_batch_overlap", "two-batch overlap"),
        ("enable_torch_compile", "torch.compile"),
    ],
)
@patch("sglang.srt.layers.moe.token_dispatcher.hpc_ops._get_peer_workspace")
@patch("sglang.srt.layers.moe.token_dispatcher.hpc_ops.get_server_args")
@patch("sglang.srt.layers.moe.token_dispatcher.hpc_ops.get_parallel")
def test_dispatcher_rejects_unsupported_execution_modes(
    get_parallel,
    get_server_args,
    _get_workspace,
    unsupported_flag,
    message,
):
    from sglang.srt.layers.moe.token_dispatcher.hpc_ops import HpcOpsDispatcher

    get_parallel.return_value = SimpleNamespace(
        moe_ep_size=2,
        moe_ep_group=SimpleNamespace(cpu_group=object()),
    )
    flags = {
        "enable_dp_attention": True,
        "enable_two_batch_overlap": False,
        "enable_torch_compile": False,
        "chunked_prefill_size": 8,
        "cuda_graph_config": SimpleNamespace(decode=SimpleNamespace(max_bs=4)),
        "max_running_requests": 4,
    }
    flags[unsupported_flag] = True
    get_server_args.return_value = SimpleNamespace(**flags)

    with pytest.raises(ValueError, match=message):
        HpcOpsDispatcher(_config())


def test_rank_major_source_rows_reuses_fixed_workspace():
    from sglang.srt.layers.moe.token_dispatcher.hpc_ops import _PeerIndexedWorkspace

    workspace = _PeerIndexedWorkspace.__new__(_PeerIndexedWorkspace)
    workspace.world_size = 2
    workspace.capacity = 4
    workspace._source_row_table = torch.arange(8, dtype=torch.int32).view(2, 4)
    workspace.all_source_rows = torch.empty(8, dtype=torch.int32)

    first = workspace._rank_major_source_rows((3, 1))
    first_storage = first.untyped_storage().data_ptr()
    torch.testing.assert_close(first, torch.tensor([0, 1, 2, 4], dtype=torch.int32))

    second = workspace._rank_major_source_rows((1, 3))
    assert second.untyped_storage().data_ptr() == first_storage
    torch.testing.assert_close(second, torch.tensor([0, 4, 5, 6], dtype=torch.int32))
