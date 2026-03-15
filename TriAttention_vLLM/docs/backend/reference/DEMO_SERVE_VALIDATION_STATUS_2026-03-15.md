# Demo Serve Validation Status

Updated: 2026-03-15
Status: Not ready to merge to main
Owner: Codex
Branch: `codex/vllm-async-commit-fix-20260315`

## 1. Why this document exists

This document records the latest validation status after the async boundary fix
was proven on the long probe case, followed by a more realistic demo-serve
validation on a fresh machine.

The main purpose is to keep these two statements separate:

1. the async boundary fix itself is real and validated on the probe path
2. the broader demo serve/gateway path is not yet fully cleared for merge

## 2. What was already known before this round

The isolated branch fix had already established:

1. old async path could drift after compression and end in obvious repetition
2. the queue-boundary fix repaired that behavior on the long probe case
3. repaired async matched the sync control on that probe case
4. repaired async did not collapse into a full-time sync solution

Reference docs:

1. `TriAttention_vLLM/docs/backend/reference/VLLM_ASYNC_BOUNDARY_FIX_2026-03-15.md`
2. `TriAttention_vLLM/docs/backend/reference/VLLM_ASYNC_ROOT_CAUSE_TRACE_2026-03-15.md`

## 3. What was tested in this round

This round intentionally moved beyond the probe runner and exercised the demo
serve stack more directly.

Tested layers:

1. `demo/vllm` gateway
2. baseline vLLM server
3. TriAttention vLLM server
4. direct requests through the gateway

The machine used in this round had no active GPU conflicts and was warmed up
before the formal comparisons.

## 4. Confirmed good result on the probe path

On the new machine, the old/new/sync probe comparison was rerun and remained
consistent with earlier findings.

Confirmed results:

1. repaired async output matched sync output exactly
2. old async still degraded
3. repaired async no longer showed the old repeated-tail failure

Reference outputs:

1. old async:
   - `/tmp/tri_phase2_probe/v2async_old_newmachine_compare/shard00/run000.jsonl`
2. repaired async:
   - `/tmp/tri_phase2_probe/v2async_new_newmachine_compare/shard00/run000.jsonl`
3. sync control:
   - `/tmp/tri_phase2_probe/v2sync_new_newmachine_compare/shard00/run000.jsonl`

Key reading:

1. `new async == sync`
2. `old async != new async`
3. old async still falls into visible repetition near the tail

## 5. Demo-style serve validation that was attempted

Because the request for this round was to validate something closer to the real
demo, the following path was exercised:

1. start baseline serve
2. start TriAttention serve
3. start `demo.vllm.server` gateway
4. send long requests through the gateway instead of the offline runner

Two request families were tested:

1. OpenClaw-style `/v1/completions`
2. browser-demo-style `/v1/chat/completions`

## 6. What was confirmed on the demo serve path

### 6.1 Compression really does trigger

This is not a "compression failed to activate" problem.

Multiple runs showed clear compression activity in the TriAttention serve logs:

1. `compression applied before=... after=7000`
2. scheduler reclaim and worker reclaim both appeared
3. KV usage dropped after compression events

So the serve-path issue is not simply "compression never happened".

### 6.2 Baseline remained usable

On the same machine and same gateway stack, baseline responses stayed in a
normal readable band.

This matters because it rules out:

1. gateway totally broken
2. model installation totally broken
3. request payload being universally unusable

## 7. What failed on the demo serve path

### 7.1 OpenClaw-style completions route still degrades

Direct gateway requests to `/v1/completions` with the long weekly-report prompt
did not clear the quality bar on the TriAttention backend.

Observed behavior:

1. compression triggered
2. output did not become pure garbage
3. but the tail still drifted into obvious bad repetition

Representative artifact:

1. `/tmp/demo_gateway_triattention_completion.json`
2. `/tmp/demo_gateway_triattention_completion_v2map.json`

The second run added the formal-path compatibility debug rewrite used in probe
validation, but the result still did not become merge-ready.

### 7.2 Browser-demo-style chat route is better, but still not clean enough

The browser UI actually uses `/v1/chat/completions`, so that path was tested
separately.

Results:

1. baseline chat path remained readable:
   - `/tmp/demo_gateway_baseline_chat_fixture_prompt.json`
2. TriAttention chat path did not collapse into obvious repeated-token loops on
   every run, but it still showed degraded "meta thinking" / repeated planning
   behavior and, in async mode, visible repeated `**` tail patterns on one run:
   - `/tmp/demo_gateway_triattention_sync_chat_fixture_prompt.json`
   - `/tmp/demo_gateway_triattention_async_chat_fixture_prompt.json`

This is an important distinction:

1. chat route looks better than completions route
2. but it is still not clean enough to call "demo fully fixed"

## 8. Important diagnostic conclusions from this round

### 8.1 The demo serve issue is not just the old async-boundary bug

This was checked explicitly.

Why:

1. the probe path already proved the queue-boundary fix
2. on the serve path, forcing sync scheduling did not cleanly eliminate the
   quality issue

So the current serve-path issue is not explained by:

1. "async boundary bug simply reappeared"

### 8.2 The demo serve issue is not just "compression failed to trigger"

Again, compression clearly triggered many times on the serve path.

So the remaining suspect is narrower:

1. something about the serve/gateway runtime path
2. or request-shaping differences between probe runner and serve path

### 8.3 Probe success must not be over-read as demo success

This is the most important process takeaway.

The branch now has a validated fix for:

1. long probe path async corruption

But it does not yet have a validated fix for:

1. the broader demo serve/gateway path

## 9. Most likely remaining difference classes

The following are the strongest remaining suspects after this round:

1. serve path request shaping differs materially from the probe runner
2. serve path and probe path do not expose exactly the same runtime scheduling
   and lifecycle behavior
3. gateway/API usage may be exercising a different compression-adjacent code
   path than the validated probe run

What now looks less likely:

1. "the async boundary fix never worked"
2. "compression is simply not activating"
3. "baseline/model installation is broken on this machine"

## 10. Merge decision

Current decision:

1. do not merge this branch to `main` or `master` yet

Reason:

1. the branch is strong enough for the probe milestone
2. but the demo serve/gateway path has not passed a clean end-to-end quality bar

## 11. Practical next step

Recommended next debugging step:

1. compare the validated probe request path against the serve/gateway request
   path module by module
2. focus on differences that are specific to the serve stack
3. do not reopen the already-closed "async boundary" question unless new
   evidence directly points back to it

## 12. Bottom line

The branch contains a real and validated repair for the probe-path async
compression corruption.

However, the demo serve/gateway path still shows remaining quality degradation.

So the correct status is:

1. probe fix: validated
2. demo serve fix: not yet validated
3. merge to main: not yet safe
