"""Worker-side hooks for TriAttention — intentionally empty.

Why no worker patch is needed in sglang
---------------------------------------
In vLLM V1, compression must happen in the worker because the scheduler
and worker run in separate processes.  The scheduler sends a compression
*signal* via the batch, and the worker executes the actual GPU operations
(gather, score, compact, reclaim) inside ``execute_model``.

sglang's architecture is fundamentally different: scheduler and worker
share the same process and address space (``event_loop_normal``).  The
scheduler has direct access to GPU tensors (KV pool, req_to_token_pool)
without cross-process IPC.  Therefore, the entire compression pipeline
runs inside ``Scheduler.get_next_batch_to_run`` (see
``scheduler_hooks.py``), *before* the batch is sent to the worker.

By the time ``TpModelWorker.forward_batch_generation`` is called:
  1. KV cache has already been compacted in-place.
  2. Freed slots have been reclaimed to the allocator.
  3. KV metadata (``kv_committed_len``, ``kv_allocated_len``) on each
     ``Req`` object has been synced.
  4. The effective-length tracker is up to date.

The worker simply runs the model forward pass on the batch as-is.

CUDA graph compatibility
~~~~~~~~~~~~~~~~~~~~~~~~
CUDA graphs capture kernel launch patterns, not data contents.  Our
compression modifies KV buffer *contents* and req_to_token *mappings*
(both are data in GPU memory), not the execution graph.  The
``can_run_cuda_graph`` flag from the forward pass remains valid because
the batch shape (number of requests, sequence lengths) is determined
after compression, not before.

Phase G (input_patches)
~~~~~~~~~~~~~~~~~~~~~~~
The remaining GPU-side concern -- making ``seq_lens`` reflect effective
lengths while ``positions`` remain absolute -- is handled by patching
``ForwardBatch.init_new`` (Phase G), which is orthogonal to this module.
That patch reads effective-length information from the ``Req`` objects
and does not require any worker-level hook.
"""
