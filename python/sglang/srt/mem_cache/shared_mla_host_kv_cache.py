"""
SharedMLA: Shared L2 KV cache for MLA models in NVLink domain.

For MLA models (DeepSeek V2/V3, Kimi, GLM-4, etc.), all TP ranks produce
identical compressed KV cache. This module implements a shared host (L2) pool:

- rank 0 allocates a POSIX shared memory slab + cudaHostRegisterPortable
- rank 1-7 attach to the shared slab (no memory allocation)
- Only rank 0 writes L2 (backup) and reads L3 (prefetch)
- All ranks read from the shared slab (load_back via ld.global)

Host_value synchronization uses NCCL broadcast (already available via TP group).
This eliminates (TP-1)/TP of L2 DRAM usage and L1->L2 DMA transfers.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import threading
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Optional

import numpy as np
import psutil
import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


def _mbind_interleave(addr: int, length: int) -> bool:
    """Set MPOL_INTERLEAVE on a virtual address range via the mbind(2) syscall.

    The shared slab is read fan-out by all GPUs. Placing all physical pages on
    a single NUMA node makes the cross-socket GPUs saturate one UPI direction
    (measured ~82 GB/s for 4 GPUs vs ~154 GB/s local). Interleaving pages across
    both nodes spreads the traffic over both full-duplex UPI directions, raising
    8-GPU aggregate read bandwidth from ~212 to ~281 GB/s with no extra DRAM.

    Must be called BEFORE the pages are first touched (memset), since mbind only
    governs future page faults, not already-resident pages.

    Returns True on success; on any failure the caller falls back to default
    (first-touch) placement, which is still correct, just less balanced.
    """
    # Build a node mask covering all online NUMA nodes.
    try:
        with open("/sys/devices/system/node/online") as f:
            online = f.read().strip()
    except OSError:
        return False

    node_ids = []
    for part in online.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            node_ids.extend(range(int(lo), int(hi) + 1))
        else:
            node_ids.append(int(part))
    if len(node_ids) < 2:
        return False  # single NUMA node: interleave is a no-op

    max_node = max(node_ids)
    nodemask = 0
    for n in node_ids:
        nodemask |= 1 << n

    MPOL_INTERLEAVE = 3
    # maxnode counts bits; +1 for the inclusive upper bound the kernel expects.
    maxnode = max_node + 2

    page_size = os.sysconf("SC_PAGESIZE")
    aligned_addr = addr & ~(page_size - 1)
    aligned_len = length + (addr - aligned_addr)

    # glibc does not export mbind as a symbol on all builds, so invoke it via
    # the raw syscall (x86_64: __NR_mbind = 237, aarch64: 235).
    machine = platform.machine()
    if machine == "x86_64":
        NR_mbind = 237
    elif machine in ("aarch64", "arm64"):
        NR_mbind = 235
    else:
        return False

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    mask_arr = ctypes.c_ulong(nodemask)
    libc.syscall.restype = ctypes.c_long
    # mbind(addr, len, mode, nodemask*, maxnode, flags)
    rc = libc.syscall(
        ctypes.c_long(NR_mbind),
        ctypes.c_void_p(aligned_addr),
        ctypes.c_ulong(aligned_len),
        ctypes.c_int(MPOL_INTERLEAVE),
        ctypes.byref(mask_arr),
        ctypes.c_ulong(maxnode),
        ctypes.c_uint(0),
    )
    if rc != 0:
        errno = ctypes.get_errno()
        logger.warning(
            f"SharedMLA: mbind(MPOL_INTERLEAVE) failed (errno={errno}), "
            f"falling back to default NUMA placement"
        )
        return False
    return True


class SharedMLATokenToKVPoolHost:
    """MLA shared L2 pool: rank 0 owns, rank 1-7 read-only.

    Uses POSIX shared_memory + cudaHostRegisterPortable so all GPUs
    in the NVLink domain can ld.global the shared slab.

    This is NOT a subclass of HostKVCache — it replaces the entire
    HostKVCache for MLA models in shared mode. The interface is
    compatible with HiCacheController's expectations.
    """

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        tp_rank: int,
        tp_size: int,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.is_owner = tp_rank == 0
        self.page_size = page_size
        self.layout = layout
        self.device = device
        self.device_pool = device_pool
        self.dtype = device_pool.store_dtype
        self.layer_num = device_pool.layer_num
        self.start_layer = device_pool.start_layer or 0
        self.end_layer = device_pool.end_layer or self.layer_num - 1

        # Compute kv_cache_dim
        self.kv_lora_rank = device_pool.kv_lora_rank
        self.qk_rope_head_dim = device_pool.qk_rope_head_dim
        self.kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim

        # Size computation
        self.size_per_token = self.kv_cache_dim * self.dtype.itemsize * self.layer_num
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size

        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        total_bytes = self.size * self.layer_num * self.token_stride_size

        if self.is_owner:
            host_mem = psutil.virtual_memory()
            available_bytes = host_mem.available - 10 * (1024**3)
            if total_bytes > available_bytes:
                raise ValueError(
                    f"Not enough host memory for SharedMLA. "
                    f"Need {total_bytes / 1e9:.2f} GB, "
                    f"have {available_bytes / 1e9:.2f} GB."
                )

        # Shared memory name — include rank 0's PID to avoid collision across
        # instances. Broadcast it from rank 0 so all ranks agree on the name.
        owner_pid = os.getpid() if self.is_owner else 0
        pid_tensor = torch.tensor(
            [owner_pid], dtype=torch.int64, device=device_pool.device
        )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(pid_tensor, src=0)
        self._shm_name = f"shared_mla_tp{tp_size}_p{pid_tensor.item()}"

        if self.is_owner:
            logger.info(
                f"SharedMLA rank 0: creating shared slab "
                f"{total_bytes / 1e9:.2f} GB (name={self._shm_name})"
            )
            # A previous run that crashed after create but before shutdown()
            # leaves the POSIX slab on /dev/shm; recreating with the same name
            # would raise FileExistsError, and the stale pages defeat the NUMA
            # interleave applied below. Unlink any leftover before creating.
            try:
                stale = shared_memory.SharedMemory(name=self._shm_name)
                stale.close()
                stale.unlink()
                logger.warning(f"SharedMLA rank 0: removed stale slab {self._shm_name}")
            except FileNotFoundError:
                pass

            self._shm = shared_memory.SharedMemory(
                name=self._shm_name, create=True, size=total_bytes
            )

            # Interleave the slab's physical pages across all NUMA nodes BEFORE
            # first-touch, so cross-socket GPUs spread their reads over both
            # full-duplex UPI directions instead of saturating one. mbind only
            # affects future faults, so we set the policy then touch the pages.
            owner_np = np.ndarray((total_bytes,), dtype=np.uint8, buffer=self._shm.buf)
            owner_tensor = torch.from_numpy(owner_np)
            interleaved = _mbind_interleave(owner_tensor.data_ptr(), total_bytes)
            owner_np[:] = 0  # first-touch: realize pages under the chosen policy
            logger.info(
                f"SharedMLA rank 0: slab pages "
                f"{'interleaved across NUMA nodes' if interleaved else 'default (first-touch) placement'}"
            )

        # All ranks barrier so rank 1-7 wait for rank 0 to finish creating shm
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if not self.is_owner:
            self._shm = shared_memory.SharedMemory(name=self._shm_name)
            logger.info(f"SharedMLA rank {self.tp_rank}: attached to shared slab")

        # Map as torch tensor
        np_array = np.ndarray((total_bytes,), dtype=np.uint8, buffer=self._shm.buf)
        shm_tensor = torch.from_numpy(np_array)

        # Every process must cudaHostRegister the shared memory in its own
        # CUDA context. "Portable" means all GPUs within ONE process can access
        # it, but each process still needs its own registration.
        if _is_cuda:
            cudart = torch.cuda.cudart()
            rc = cudart.cudaHostRegister(
                shm_tensor.data_ptr(), total_bytes, 1  # cudaHostRegisterPortable
            )
            if int(rc) != 0:
                raise RuntimeError(
                    f"cudaHostRegister failed on rank {self.tp_rank} (rc={int(rc)})"
                )

        # Barrier after all ranks register
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # View as KV buffer: [layer_num, size, 1, kv_cache_dim] (layer first)
        self.kv_buffer = shm_tensor.view(self.dtype).view(
            self.layer_num, self.size, 1, self.kv_cache_dim
        )

        # Per-layer data_refs and data_ptrs (for transfer kernels)
        # data_refs[i] is layer i's buffer: [size, 1, kv_cache_dim]
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=device_pool.device,
        )

        # Slot management (rank 0 only)
        self.lock = threading.RLock()
        if self.is_owner:
            self.free_slots = torch.arange(self.size, dtype=torch.int64)
        else:
            self.free_slots = None

        self.mem_usage = total_bytes / (1024**3)
        self.layer_transfer_counter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

        logger.info(
            f"SharedMLA rank {self.tp_rank}: ready, "
            f"size={self.size} tokens, {self.mem_usage:.2f} GB"
        )

    # ---- Slot management ----

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Only rank 0 allocates. Returns host slot indices."""
        if not self.is_owner:
            return None
        assert need_size % self.page_size == 0
        if need_size > len(self.free_slots):
            return None
        select = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return select

    def free(self, indices: torch.Tensor):
        """Only rank 0 frees."""
        if not self.is_owner:
            return
        if indices.numel() == 0:
            return
        self.free_slots = torch.cat([self.free_slots, indices.cpu()])

    def available_size(self):
        if self.is_owner:
            return len(self.free_slots)
        return 0

    def clear(self):
        if self.is_owner:
            self.free_slots = torch.arange(self.size, dtype=torch.int64)

    # ---- Transfer operations ----

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        """Only rank 0: GPU DMA → shared slab."""
        if not self.is_owner:
            return
        if io_backend != "kernel" or not _is_cuda:
            return

        from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla

        element_size = self.kv_cache_dim * self.dtype.itemsize
        try:
            from sglang.jit_kernel.hicache import (
                can_use_hicache_jit_kernel,
            )
            from sglang.jit_kernel.hicache import (
                transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
            )

            use_jit = can_use_hicache_jit_kernel(element_size=element_size)
        except ImportError:
            use_jit = False

        if use_jit:
            jit_transfer_hicache_all_layer_mla(
                ptr_dst=self.data_ptrs,
                indices_dst=host_indices,
                ptr_src=device_pool.data_ptrs,
                indices_src=device_indices,
                cache_dst_stride_bytes=self.token_stride_size,
                cache_src_stride_bytes=self.token_stride_size,
                element_size=element_size,
            )
        else:
            transfer_kv_all_layer_mla(
                src_layers=device_pool.data_ptrs,
                dst_layers=self.data_ptrs,
                src_indices=device_indices,
                dst_indices=host_indices,
                item_size=self.token_stride_size,
                num_layers=self.layer_num,
            )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        """All ranks: GPU ld.global [shared slab] → local L1."""
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        if io_backend != "kernel" or not _is_cuda:
            return

        from sgl_kernel.kvcacheio import transfer_kv_per_layer_mla

        try:
            from sglang.jit_kernel.hicache import (
                can_use_hicache_jit_kernel,
            )
            from sglang.jit_kernel.hicache import (
                transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
            )

            use_jit = can_use_hicache_jit_kernel(
                element_size=self.kv_cache_dim * self.dtype.itemsize
            )
        except ImportError:
            use_jit = False

        if use_jit:
            jit_transfer_hicache_one_layer_mla(
                cache_dst=device_pool.kv_buffer[layer_id],
                cache_src=self.data_refs[layer_id],
                indices_dst=device_indices,
                indices_src=host_indices,
                element_dim=self.kv_cache_dim,
            )
        else:
            transfer_kv_per_layer_mla(
                src=self.data_refs[layer_id],
                dst=device_pool.kv_buffer[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                item_size=self.token_stride_size,
            )

    def get_contiguous_buf_infos(self):
        data_ptrs = [int(self.data_ptrs[i].item()) for i in range(self.layer_num)]
        data_lens = [self.data_refs[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.token_stride_size * self.page_size] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def get_page_buffer_meta(self, indices):
        """Return (ptr_list, size_list) for RDMA/Mooncake zero-copy.
        Layout is layer_first: [layer_num, size, 1, kv_cache_dim].
        Returns per-page per-layer pointers, matching MLATokenToKVPoolHost.
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        base_ptr = self.kv_buffer.data_ptr()
        indices_list = indices.tolist()
        for i in range(0, len(indices_list), self.page_size):
            for layer_id in range(self.layer_num):
                ptr = (
                    base_ptr
                    + indices_list[i] * self.kv_cache_dim * self.dtype.itemsize
                    + layer_id * self.size * self.kv_cache_dim * self.dtype.itemsize
                )
                ptr_list.append(ptr)
        elem_size = self.dtype.itemsize * self.page_size * self.kv_cache_dim
        return ptr_list, [elem_size] * len(ptr_list)

    def get_size_per_token(self):
        return self.kv_cache_dim * self.dtype.itemsize * self.layer_num

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def get_kv_size_bytes(self):
        return self.size * self.layer_num * self.token_stride_size

    def register_layer_transfer_counter(
        self, layer_transfer_counter: "LayerDoneCounter"
    ):
        self.layer_transfer_counter = layer_transfer_counter

    def maybe_get_custom_mem_pool(self):
        return None

    def shutdown(self):
        if self._shm is not None:
            self._shm.close()
            if self.is_owner:
                try:
                    self._shm.unlink()
                except Exception:
                    pass
