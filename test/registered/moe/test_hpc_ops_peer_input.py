import multiprocessing
import time
import uuid

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.hpc_ops import has_hpc_ops
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="extra-a", runner_config="2-gpu-large")


def _sm90_with_two_gpus() -> bool:
    return (
        torch.cuda.device_count() >= 2 and torch.cuda.get_device_capability(0)[0] == 9
    )


def _make_weights(device):
    torch.manual_seed(17)
    w13 = (torch.randn((4, 256, 128), device=device, dtype=torch.float32) * 0.04).to(
        torch.float8_e4m3fn
    )
    w2 = (torch.randn((4, 128, 128), device=device, dtype=torch.float32) * 0.04).to(
        torch.float8_e4m3fn
    )
    w13_scale = torch.rand((4, 2, 4), device=device, dtype=torch.float32) * 0.1
    w2_scale = torch.rand((4, 1, 4), device=device, dtype=torch.float32) * 0.1
    return w13, w13_scale, w2, w2_scale


def _peer_worker(rank, socket_name, pull_input, tokens_per_owner, result_queue):
    import hpc

    from sglang.srt.layers.moe.token_dispatcher.hpc_ops import (
        _PeerIndexedWorkspace,
    )

    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        communicator = hpc.MulticastCommunicator(rank, 2, rank, socket_name)
        workspace = _PeerIndexedWorkspace(
            communicator,
            capacity=4,
            hidden_size=128,
            top_k=2,
            device=device,
        )

        all_w13, all_w13_scale, all_w2, all_w2_scale = _make_weights(device)
        local_experts = slice(rank * 2, (rank + 1) * 2)
        token_count = tokens_per_owner[rank]
        for generation in range(2):
            torch.manual_seed(1000 * generation + 100 + rank)
            source = torch.randn(
                (token_count, 128), device=device, dtype=torch.bfloat16
            )
            route_ids = torch.tensor(
                [[[0, 2], [1, 3], [0, 3]], [[2, 0], [3, 1]]][rank],
                device=device,
                dtype=torch.int32,
            )[:token_count]
            route_weights = torch.tensor(
                [[0.6, 0.4], [0.7, 0.3], [0.55, 0.45]],
                device=device,
                dtype=torch.float32,
            )[:token_count]
            workspace.publish_input(source, route_ids, route_weights)
            actual = workspace.consume(
                w13=all_w13[local_experts].contiguous(),
                w13_scale=all_w13_scale[local_experts].contiguous(),
                w2=all_w2[local_experts].contiguous(),
                w2_scale=all_w2_scale[local_experts].contiguous(),
                num_experts_total=4,
                tokens_per_owner=tokens_per_owner,
                local_output_rows=token_count,
                pull_input=pull_input,
                swiglu_limit=0.0,
            )

            x = torch.cat(
                tuple(
                    workspace.activation_handle.get_buffer(
                        owner, 4, 128, dtype=torch.float8_e4m3fn
                    )[:count]
                    for owner, count in enumerate(tokens_per_owner)
                )
            )
            x_scale = torch.cat(
                tuple(
                    workspace.activation_scale_handle.get_buffer(
                        owner, 4, 1, dtype=torch.float32
                    )[:count]
                    for owner, count in enumerate(tokens_per_owner)
                )
            )
            all_ids = torch.cat(
                tuple(
                    workspace.route_ids_handle.get_buffer(
                        owner, 4, 2, dtype=torch.int32
                    )[:count]
                    for owner, count in enumerate(tokens_per_owner)
                )
            )
            all_weights = torch.cat(
                tuple(
                    workspace.route_weights_handle.get_buffer(
                        owner, 4, 2, dtype=torch.float32
                    )[:count]
                    for owner, count in enumerate(tokens_per_owner)
                )
            )
            expected = hpc.fuse_moe_blockwise(
                x,
                x_scale,
                all_w13,
                all_w13_scale,
                all_w2,
                all_w2_scale,
                all_ids,
                all_weights,
                0,
                4,
            )
            begin = sum(tokens_per_owner[:rank])
            torch.testing.assert_close(
                actual.float(),
                expected[begin : begin + token_count].float(),
                rtol=0.02,
                atol=0.02,
            )

            # A fast rank may publish the next layer's input, but it must not
            # reach the next output scatter until every rank has consumed this
            # generation and published that next input.
            if generation == 0 and rank == 1:
                torch.cuda.synchronize()
                time.sleep(0.5)
        result_queue.put(None)
    except Exception as exc:
        result_queue.put(repr(exc))


def _epoch_skip_worker(rank, socket_name, result_queue):
    import hpc

    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        communicator = hpc.MulticastCommunicator(rank, 2, rank, socket_name)
        _, handle = hpc.empty_multimem(
            communicator,
            1,
            dtype=torch.uint32,
            device=device,
        )
        signal_words = handle.signal_size // torch.uint32.itemsize
        local_signal = handle.get_signal(rank, signal_words)
        generation = torch.zeros((), dtype=torch.int32, device=device)

        generation.add_(1)
        hpc.fuse_moe_ep_publish(
            handle.signal_buffer_ptrs_dev,
            rank,
            2,
            generation,
            signal_words,
        )
        torch.cuda.synchronize()
        if rank == 1:
            time.sleep(0.5)
        hpc.fuse_moe_ep_wait(local_signal, 2, generation)

        generation.add_(1)
        hpc.fuse_moe_ep_publish(
            handle.signal_buffer_ptrs_dev,
            rank,
            2,
            generation,
            signal_words,
        )
        hpc.fuse_moe_ep_wait(local_signal, 2, generation)
        torch.cuda.synchronize()
        result_queue.put(None)
    except Exception as exc:
        result_queue.put(repr(exc))


@pytest.mark.parametrize("pull_input", [False, True])
@pytest.mark.parametrize("tokens_per_owner", [(3, 2), (3, 0), (0, 0)])
@pytest.mark.skipif(
    not has_hpc_ops() or not _sm90_with_two_gpus(),
    reason="requires HPC-Ops and two Hopper GPUs",
)
def test_peer_indexed_fused_moe_reuses_workspace_across_generations(
    pull_input, tokens_per_owner
):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    socket_name = f"/tmp/hpc-peer-input-{uuid.uuid4().hex}.sock"
    processes = [
        context.Process(
            target=_peer_worker,
            args=(rank, socket_name, pull_input, tokens_per_owner, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
    for rank, process in enumerate(processes):
        if process.is_alive():
            process.kill()
            pytest.fail(f"rank {rank} timed out")
        assert process.exitcode == 0, f"rank {rank} subprocess failed"
    errors = [result_queue.get(timeout=5) for _ in processes]
    assert errors == [None, None]


@pytest.mark.skipif(
    not has_hpc_ops() or not _sm90_with_two_gpus(),
    reason="requires HPC-Ops and two Hopper GPUs",
)
def test_ep_wait_accepts_a_later_published_generation():
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    socket_name = f"/tmp/hpc-epoch-skip-{uuid.uuid4().hex}.sock"
    processes = [
        context.Process(
            target=_epoch_skip_worker,
            args=(rank, socket_name, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 20
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    timed_out_ranks = [
        rank for rank, process in enumerate(processes) if process.is_alive()
    ]
    for rank in timed_out_ranks:
        processes[rank].kill()
        processes[rank].join(5)
    assert (
        not timed_out_ranks
    ), f"ranks {timed_out_ranks} timed out after missing an exact epoch"
    for rank, process in enumerate(processes):
        assert process.exitcode == 0, f"rank {rank} subprocess failed"
    errors = [result_queue.get(timeout=5) for _ in processes]
    assert errors == [None, None]
