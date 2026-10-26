# Unified paged cache

Each Engine owns one pool of equal-sized physical slabs. A runner declares an
ordered `CacheLayout`; every `CacheGroupLayout` explains how one model state is
packed into those slabs:

- `family=HISTORY` stores token-addressed attention or rolling history.
- `family=STATE` stores recurrent checkpoints.
- `retention=HISTORY` keeps the full logical history. State groups retain only
  their current and preceding checkpoints.
- `retention=WINDOW` keeps a bounded chronological suffix.
- `window_tokens` is that suffix's physical retention budget;
  `read_window_tokens` is the logical window including the current token.
  It defaults to `window_tokens`. To write a chunk of `Q` tokens while its
  earliest query reads a window of `W` tokens, reserve at least `W + Q - 1`
  retained tokens and declare `read_window_tokens=W`.
- `blocks_per_slab` says how many blocks of that type fit in one slab.

Slabs are fungible. They are not divided into fixed History, Window, or State
regions. Allocation binds a free slab to whichever group currently needs it,
and retirement returns it to the common pool. Admission estimates the slab
demand for every request and group, including packing, before admitting work.

The runner allocates the matching device arena once and receives its pool ID and
frozen layout through `bind_resources`. A `BatchPlan` does not repeat that
identity. It contains only the ordered `CacheGroupPlan`s whose structured
physical addresses refer to the already-bound arena.

## Batch plans

`BatchPlan.cache_groups` has one plan per declared block type—such as full
attention, sliding-window attention, or recurrent state—not one per sequence.
The cache is a pool of equally sized physical slabs. Each slab in use is
assigned to one cache group and fits that group's `blocks_per_slab` blocks.
A physical block is identified by `(slab_id, slotnum)`. A cached token location is
`(block, block_offset)`, equivalently `(slab_id, slotnum, block_offset)`. Each
group plan covers every batch sequence and has one CSR block table:

- `block_indptr` slices the parallel `block_slab_ids` and
  `block_slotnums` arrays by sequence.
- `block_table_start_indices[i]` is the absolute chunk index represented by
  sequence `i`'s first table entry. For a group with 64-token blocks, block 37
  covers token positions 2368 through 2431. Full-history rows begin at zero;
  sliding-window rows advance when earlier chunks are evicted.
- `append_slab_ids`, `append_slotnums`, and `append_block_offsets` give one
  exact cached-token write location per scheduled token.

The Engine allocates blocks, performs copy-on-write decisions, and holds the
reader/writer lease through synchronous execution and native commit. Model code maps
each `(slab_id, slotnum)` pair into its model-specific tensor row. Slab zero is
reserved for inactive/null rows; live slab IDs are one-based. Keeping the
address structured prevents runners from assuming a different packing than the
allocator.

## Prefix caching

`Engine(..., prefix_cache=True)` enables an exact-token block trie with LRU
entries. Prefixes are published only at an alignment shared by every cache
group. A hit attaches immutable references to the cached pages and schedules
only the unmatched suffix.

A model can declare `CacheLayout(replay_suffix_tokens=N)` when a restored
history needs its final `N` inputs recomputed. Prefix lookup caps the hit at
`prefill_length - N` before applying cache alignment. The scheduler does not
interpret those inputs or choose model layers. `BatchPlan.prefill_lengths`
provides each request's final prefill endpoint across chunks, including
generated inputs when reconstructing a retracted request. Scored prompts still
take the full prefill path.

Live requests have priority over cached entries. When allocation needs more
slabs, the least-recently-used prefix is evicted. Advancing a shared Window or
State ring uses copy-on-write so one request cannot overwrite another request's
prefix.

State groups use two checkpoint pages per live sequence. At a restored boundary,
the cached input page remains read-only and the next checkpoint page is private
and writable. Within a checkpoint interval, decode updates the private page in
place. This keeps recurrent state prefix-compatible without a second slot
allocator.

## Physical sizing

DeepSeek chooses slab capacity from free VRAM after model load and before
cache allocation and CUDA graph capture, subject to its memory-utilization
target. Other runners may supply a fixed capacity. The slab byte size is the
largest group allocation. Smaller groups set `blocks_per_slab` from their
exact byte ratio, so the allocator can compare all demand in slab units.

DeepSeek-V4.1 packs its smaller compressed-history and rolling state groups into
the same slab size. GPT-OSS uses full-history and sliding attention groups with
equal-sized pages.

Slab zero is reserved as the device null target. Positive slab IDs are stable for
the lifetime of the Engine-issued lease.

Window read history and all scheduled writes share one chronological table.
The Engine caps prefill chunks by the smallest `window_tokens -
read_window_tokens + 1` across Window groups and rejects decode lookahead that
exceeds it. The same bound is checked before native cache growth or copy-on-write.
This prevents a later write in a chunk from overwriting the earlier queries'
history, including when a prefix hit shares those pages with another request.

DeepSeek-V4.1 defaults to 4096-token prefill chunks and reserves this guard band
for sliding attention and compressor tails. The runner's `prefill_chunk_size`
parameter controls both execution size and retained capacity. This preserves
vectorized prefill but uses more cache memory per live request than a decode-only
tail; the scheduler accounts for that physical demand. A future separate
transient prefill workspace could reduce the retained footprint.

With DSpark enabled, draft attention reads the preceding 128 main-hidden rows
before target verification consumes the current anchor. Its shared attention
group therefore declares a 129-token read envelope. This preserves that oldest
row even for short, unaligned verification blocks; default 4096-token chunks
use the same number of physical blocks after rounding.

## MLX arena

`resin.backend.mlx.SlabArena` backs every group with one contiguous byte array
shaped `[slab, byte]`. The total byte count is not an array dimension, so
multi-GiB arenas do not hit MLX's signed 32-bit dimension limit.
The model supplies a `[layer, ...]` block shape and dtype for each group.
The largest packed byte allocation determines the slab stride, aligned for
every group dtype. BF16 history and FP32 recurrent state therefore share the
Engine's fungible pool without allocating a second capacity-sized arena.
Physical `(slab_id, slotnum)` addresses use 64-bit element/byte arithmetic;
they do not wrap at the 8 GiB BF16 boundary. No request-to-cache map exists.

MLX slice assignment can copy the entire allocation while an old view is live.
The arena queues writes, evaluates their values and model outputs together,
then commits one two-axis scatter in the widest commonly aligned integer word
type, without flattening the arena. Typed views are ephemeral. Fresh blocks and
null state are cleared before execution, including reused padding. `commit()`
synchronizes before the Engine releases its leases.

The public backend API is:

| Name | Contract |
| --- | --- |
| `SlabArena(layout, group_shapes, group_dtypes=...)` | Allocate one mixed-precision byte arena; default group dtype is BF16. Permanently bind with `bind_resources(ResourceBinding)`. |
| `SlabArena.storage` | Contiguous uint8 array shaped `[parent_capacity + 1, slab_bytes]`; row zero is the null slab. |
| `SlabArena.read_layer(...)` | Read complete layer pages with a compact-address Metal kernel, without per-element indices. |
| `SlabArena.write_layer(...)` | Queue whole-state or token-offset history writes. |
| `SlabArena.commit(*outputs)` | Evaluate reads and optional model outputs together, then synchronously persist one scatter. |
| `SlabArena.initialize_fresh(plan)` | Clear newly entered blocks and null state before execution. |
| `paged_attention(...)` | One-token decode over paged history plus separate uncommitted K/V; FP32 softmax accumulation. |
| `DevicePageGroup` | Device slab IDs, slots, first logical blocks, and valid column counts. |
| `stage_page_groups(PagedCachePlan)` | Lower native CSR tables into zero-padded MLX arrays. |
| `MlxSampler` | Greedy or stateless seeded temperature/top-k/top-p sampling, stable under batch reordering and chunking. |
| `prompt_scores(...)` | Score requested prompt predecessors and target IDs in bounded vocabulary projections. |
| `PromptScores` | Concatenate selected, top, and target prompt score columns across prefill rows. |
| `format_output(...)` | Materialize native `ForwardOutput` columns, including raw-model output and prompt logprobs. |

Both runners retain 128-token KV pages. Their `prefill_chunk_size` also defines
the STATE checkpoint interval: **2048 tokens for LFM2, 4096 for Qwen3.5** by
default. This follows the CUDA Qwen runner's geometry rather than changing
native retention. Equal-length prefill rows execute together; unequal lengths
retain independent row execution without padding.

Only a forward's final recurrent state is stored. Exact prefix hits therefore
have 2048- or 4096-token granularity, not the former 128-token granularity.
Changing `prefill_chunk_size` changes both scheduling and checkpoint geometry;
it must be a positive multiple of `block_tokens`. Smaller scheduler chunks are
safe, but a chunk must not exceed the runner's checkpoint interval: the native
two-page ring must still contain the state from which that chunk starts.

Decode attention accepts `[batch, 1, heads, dim]` Q/K/V, with head dimensions
divisible by 32 and full-history slab/slot tables starting at logical block zero.
Cache pages are `[layer, token, 2, kv_heads, dim]`. History and queries have the
same dtype; current K/V are separate because cache writes land only after the
whole forward. Independent 32-token partitions visit each live token once,
mask before loading padding, and combine FP32 softmax partials. Prefill still
uses compact page reads followed by MLX SDPA.

Qwen runners must declare FP32 Gated DeltaNet state separately from BF16 KV,
size packing from **bytes**, and preserve checkpoint alignment. MoE weights and
routing are outside this arena; neither changes Engine page ownership. Keep
arena views within a forward so writes can reuse the allocation.
