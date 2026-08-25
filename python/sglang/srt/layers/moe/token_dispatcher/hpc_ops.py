"""Peer-indexed EP input for the HPC-Ops Fused MoE backend."""

from __future__ import annotations

import uuid
from typing import NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_is_extend_in_batch,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputChecker,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.runtime_context import get_parallel, get_resources, get_server_args


class HpcOpsDispatchOutput(NamedTuple):
    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_output: TopKOutput
    workspace: object
    tokens_per_owner: Tuple[int, ...]
    local_output_rows: int
    pull_input: bool
    hidden_states_pre_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


def _capacity_from_server_args(server_args) -> int:
    capacity = max(
        int(server_args.chunked_prefill_size or 0),
        int(server_args.cuda_graph_config.decode.max_bs or 0),
        int(getattr(server_args, "max_running_requests", 0) or 0),
    )
    if capacity <= 0:
        raise ValueError(
            "HPC-Ops peer input requires a positive prefill or decode capacity"
        )
    return capacity


class _PeerIndexedWorkspace:
    """Process-lifetime symmetric storage used by peer-indexed Fused MoE."""

    def __init__(
        self,
        communicator,
        *,
        capacity: int,
        hidden_size: int,
        top_k: int,
        device: torch.device,
    ):
        import hpc

        self.rank = communicator.GetRank()
        self.world_size = communicator.GetWorldSize()
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.top_k = top_k

        self.activation, self.activation_handle = hpc.empty_multimem(
            communicator,
            capacity,
            hidden_size,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self.activation_scale, self.activation_scale_handle = hpc.empty_multimem(
            communicator,
            capacity,
            hidden_size // 128,
            dtype=torch.float32,
            device=device,
        )
        self.route_ids, self.route_ids_handle = hpc.empty_multimem(
            communicator,
            capacity,
            top_k,
            dtype=torch.int32,
            device=device,
        )
        self.route_weights, self.route_weights_handle = hpc.empty_multimem(
            communicator,
            capacity,
            top_k,
            dtype=torch.float32,
            device=device,
        )
        self.output_slots, self.output_handle = hpc.empty_multimem(
            communicator,
            capacity,
            self.world_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )

        self.input_generation = torch.zeros((), dtype=torch.int32, device=device)
        self.output_generation = torch.zeros((), dtype=torch.int32, device=device)
        self.all_route_ids = torch.empty(
            (self.world_size * capacity, top_k), dtype=torch.int32, device=device
        )
        self.all_route_weights = torch.empty(
            (self.world_size * capacity, top_k), dtype=torch.float32, device=device
        )
        self._source_row_table = torch.arange(
            self.world_size * capacity, dtype=torch.int32, device=device
        ).view(self.world_size, capacity)
        self.all_source_rows = torch.empty(
            self.world_size * capacity, dtype=torch.int32, device=device
        )
        self._route_id_views = tuple(
            self.route_ids_handle.get_buffer(owner, capacity, top_k, dtype=torch.int32)
            for owner in range(self.world_size)
        )
        self._route_weight_views = tuple(
            self.route_weights_handle.get_buffer(
                owner, capacity, top_k, dtype=torch.float32
            )
            for owner in range(self.world_size)
        )
        self._input_signal_words = (
            self.activation_handle.signal_size // torch.uint32.itemsize
        )
        self._output_signal_words = (
            self.output_handle.signal_size // torch.uint32.itemsize
        )
        self._input_signal = self.activation_handle.get_signal(
            self.rank, self._input_signal_words
        )
        self._output_signal = self.output_handle.get_signal(
            self.rank, self._output_signal_words
        )

    def _publish(self, handle, generation: torch.Tensor, signal_words: int) -> None:
        import hpc

        hpc.fuse_moe_ep_publish(
            handle.signal_buffer_ptrs_dev,
            self.rank,
            self.world_size,
            generation,
            signal_words,
        )

    def _wait(self, signal: torch.Tensor, generation: torch.Tensor) -> None:
        import hpc

        hpc.fuse_moe_ep_wait(signal, self.world_size, generation)

    def publish_input(
        self,
        source: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        import hpc

        if source.shape[0] > self.capacity:
            raise ValueError("local token count exceeds HPC-Ops peer capacity")
        hpc.prepare_indexed_input(
            source,
            topk_ids,
            topk_weights.to(torch.float32),
            self.activation,
            self.activation_scale,
            self.route_ids,
            self.route_weights,
        )
        self.input_generation.add_(1)
        self._publish(
            self.activation_handle,
            self.input_generation,
            self._input_signal_words,
        )

    def _rank_major_source_rows(
        self, tokens_per_owner: Tuple[int, ...]
    ) -> torch.Tensor:
        if len(tokens_per_owner) != self.world_size:
            raise ValueError("HPC-Ops peer input requires one token count per EP rank")
        for count in tokens_per_owner:
            if not 0 <= count <= self.capacity:
                raise ValueError("owner token count exceeds HPC-Ops peer capacity")
        active_rows = sum(tokens_per_owner)
        torch.cat(
            tuple(
                self._source_row_table[owner, :count]
                for owner, count in enumerate(tokens_per_owner)
            ),
            out=self.all_source_rows[:active_rows],
        )
        return self.all_source_rows[:active_rows]

    def consume(
        self,
        *,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        num_experts_total: int,
        tokens_per_owner: Tuple[int, ...],
        local_output_rows: int,
        pull_input: bool,
        swiglu_limit: float,
    ) -> torch.Tensor:
        import hpc

        if len(tokens_per_owner) != self.world_size:
            raise ValueError("HPC-Ops peer input requires one token count per EP rank")
        if not 0 <= local_output_rows <= self.capacity:
            raise ValueError("local output rows exceed HPC-Ops peer capacity")

        source_rows = self._rank_major_source_rows(tokens_per_owner)
        active_rows = sum(tokens_per_owner)
        self._wait(self._input_signal, self.input_generation)
        torch.cat(
            tuple(
                view[:count]
                for view, count in zip(self._route_id_views, tokens_per_owner)
            ),
            out=self.all_route_ids[:active_rows],
        )
        torch.cat(
            tuple(
                view[:count]
                for view, count in zip(self._route_weight_views, tokens_per_owner)
            ),
            out=self.all_route_weights[:active_rows],
        )

        fused_moe = (
            hpc.fuse_moe_blockwise_indexed_pull
            if pull_input
            else hpc.fuse_moe_blockwise_indexed
        )
        partial = fused_moe(
            self.activation,
            self.activation_scale,
            self.activation_handle.data_buffer_ptrs_dev,
            self.activation_scale_handle.data_buffer_ptrs_dev,
            source_rows,
            w13,
            w13_scale,
            w2,
            w2_scale,
            self.all_route_ids[:active_rows],
            self.all_route_weights[:active_rows],
            self.rank,
            num_experts_total,
            swiglu_limit,
        )
        hpc.scatter_indexed_output(
            partial,
            self.output_handle.data_buffer_ptrs_dev,
            source_rows,
            self.capacity,
            self.rank,
            self.world_size,
        )
        self.output_generation.add_(1)
        self._publish(
            self.output_handle,
            self.output_generation,
            self._output_signal_words,
        )
        self._wait(self._output_signal, self.output_generation)
        return self.output_slots[:local_output_rows].sum(dim=1)


def _get_peer_workspace(config: MoeRunnerConfig, *, capacity: int, cpu_group):
    """Collectively allocate one workspace per EP group and input shape."""

    import hpc

    parallel = get_parallel()
    device = torch.device("cuda", torch.cuda.current_device())
    key = (
        "hpc_ops_peer_input",
        device.index,
        parallel.moe_ep_size,
        capacity,
        config.hidden_size,
        config.top_k,
    )
    workspace = get_resources().buffers.get(key)
    if workspace is not None:
        return workspace

    rank = dist.get_rank(group=cpu_group)
    world_size = dist.get_world_size(group=cpu_group)
    socket_name = [None]
    if rank == 0:
        socket_name[0] = f"/tmp/hpc-moe-{uuid.uuid4().hex}.sock"
    root = dist.get_global_rank(cpu_group, 0)
    dist.broadcast_object_list(socket_name, src=root, group=cpu_group)
    communicator = hpc.MulticastCommunicator(
        rank, world_size, device.index, socket_name[0]
    )
    workspace = _PeerIndexedWorkspace(
        communicator,
        capacity=capacity,
        hidden_size=config.hidden_size,
        top_k=config.top_k,
        device=device,
    )
    get_resources().buffers[key] = workspace
    return workspace


class HpcOpsDispatcher(BaseDispatcher):
    """Publish EP inputs for direct or local-persistent HPC-Ops W13."""

    def __init__(self, config: MoeRunnerConfig):
        super().__init__()
        parallel = get_parallel()
        server_args = get_server_args()
        if not server_args.enable_dp_attention:
            raise ValueError("HPC-Ops peer input requires DP Attention")
        if getattr(server_args, "enable_two_batch_overlap", False):
            raise ValueError("HPC-Ops peer input does not support two-batch overlap")
        if getattr(server_args, "enable_torch_compile", False):
            raise ValueError("HPC-Ops peer input does not support torch.compile")
        if parallel.moe_ep_size <= 1:
            raise ValueError("HPC-Ops peer input requires EP size greater than one")
        self.world_size = parallel.moe_ep_size
        self.capacity = _capacity_from_server_args(server_args)
        self.workspace = _get_peer_workspace(
            config,
            capacity=self.capacity,
            cpu_group=parallel.moe_ep_group.cpu_group,
        )

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> HpcOpsDispatchOutput:
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise TypeError("HPC-Ops peer input requires standard Top-K output")
        global_num_tokens = get_dp_global_num_tokens()
        if global_num_tokens is None or len(global_num_tokens) != self.world_size:
            raise RuntimeError(
                "HPC-Ops peer input requires the complete DP token-count vector"
            )
        tokens_per_owner = tuple(int(count) for count in global_num_tokens)
        if max((*tokens_per_owner, hidden_states.shape[0])) > self.capacity:
            raise ValueError("HPC-Ops peer input capacity exceeded")
        self.workspace.publish_input(
            hidden_states, topk_output.topk_ids, topk_output.topk_weights
        )
        return HpcOpsDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=None,
            topk_output=topk_output,
            workspace=self.workspace,
            tokens_per_owner=tokens_per_owner,
            local_output_rows=hidden_states.shape[0],
            pull_input=get_is_extend_in_batch(),
        )

    def combine(self, combine_input: CombineInput) -> torch.Tensor:
        if not CombineInputChecker.format_is_standard(combine_input):
            raise TypeError("HPC-Ops peer input requires standard combine input")
        return combine_input.hidden_states
