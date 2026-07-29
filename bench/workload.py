"""Deterministic request-stream generator for the paged attention engine.

Gate 1, Step 0a. This module is the single source of the request stream that
every engine in this project consumes. It emits token IDs rather than text, so
the paged engine and the HF baseline see a byte-identical stream: the same
integers, in the same order, with the same decode budgets.

Two streams are produced:

  mixed    Lengths drawn from the clamped lognormal bound in STEP 0A CHOICES
           below. This is the workload of record.
  control  Rectangular 512-prompt / 512-decode. Every request is the same
           shape, so any difference between two engines on this stream is the
           engine and not the workload.

Both are sized by the concurrency level C, which is the sweep variable: a run
sweeps C from 1 to collapse, and at each level the stream holds
N = max(4 * C, 64) requests. C is a parameter of every builder, never a derived
constant. DEFAULT_C and DEFAULT_CONTROL_C are only defaults -- the level at
which each stream's block demand meets cache capacity.

Stream contract. Both engines must honour all three or the comparison is void:

  * Requests are offline. The whole stream exists at t=0 and is admitted in
    emission order; there is no arrival process. `admission = strict_fcfs`
    consumes this list front to back.
  * `max_new_tokens` is exact. Decode to exactly that many tokens and ignore
    EOS. Otherwise the two engines perform different amounts of work and no
    comparison between them means anything.
  * `stream_digest` over the canonical serialization is the proof that two runs
    saw the same stream. Compare digests, not descriptions.

Reproducibility. Every draw is built on `random.Random.getrandbits` -- the most
stable primitive the stdlib generator exposes -- plus an explicit inverse-CDF
transform. Nothing here depends on CPython's internal choice of algorithm for
`randrange`, `random`, or `lognormvariate`, so the stream survives interpreter
upgrades. Seeds are namespaced strings, which `random.Random` hashes with
SHA-512, so the mixed and control streams never share a token sequence.

Because token IDs are drawn sequentially, the stream at a given C is a strict
prefix of the stream at any larger C. Raising C extends a run; it never
reshuffles the requests already in it.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

# --- LOCKED (Gate 1 spec) --------------------------------------------------
# Restated here only because engine/ does not exist yet at Step 0a, and the
# sweep points below are derived from KV cache capacity. When engine/cache.py
# lands, a test in tests/ should assert these equal the engine's own constants
# so the two cannot drift.
BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 3584

# --- Draw machinery knobs (not Step 0a choices) ----------------------------
# Fixed and deliberately independent of the caller's seed, so every derived
# quantity below is a function of the config alone.
_MEAN_ESTIMATE_SAMPLES = 100_000
_MEAN_ESTIMATE_SEED = "pae-gate1/mean-estimate/v1"

_STD_NORMAL = statistics.NormalDist()

# inv_cdf is undefined at 0.0; getrandbits(53) can produce it with probability
# 2**-53. Clamp rather than resample so the draw consumes a fixed bit budget.
_UNIT_EPSILON = 1e-12


@dataclass(frozen=True)
class LengthDist:
    """A lognormal length draw, clamped (not truncated) to [lo, hi].

    Clamping keeps the bit budget per draw fixed, which keeps the stream stable
    under changes to lo/hi in a way rejection sampling would not.
    """

    mu: float
    sigma: float
    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.lo < 1:
            raise ValueError(f"lo must be at least 1, got {self.lo}")
        if self.hi < self.lo:
            raise ValueError(f"hi ({self.hi}) must be >= lo ({self.lo})")


@dataclass(frozen=True)
class Request:
    """One request. Immutable so a stream cannot be edited after digesting."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.max_new_tokens


@dataclass(frozen=True)
class WorkloadConfig:
    """Every Step 0a choice, with no defaults, so DEFAULT_CONFIG shows them all."""

    prompt: LengthDist
    decode: LengthDist
    vocab_limit: int
    block_size: int
    num_gpu_blocks: int
    min_requests: int
    requests_per_concurrency: int

    @property
    def total_token_slots(self) -> int:
        return self.num_gpu_blocks * self.block_size


# --- RNG primitives --------------------------------------------------------


def _rng(seed: int | str, stream_name: str) -> random.Random:
    """Namespaced generator, so two streams at one seed never share draws."""
    return random.Random(f"pae-gate1/{stream_name}/{seed}")


def _unit(rng: random.Random) -> float:
    """Uniform in [0, 1) from raw bits, matching what `random()` does."""
    return rng.getrandbits(53) / (1 << 53)


def _below(rng: random.Random, n: int) -> int:
    """Uniform integer in [0, n) by rejection sampling on raw bits.

    Built on getrandbits so the token stream does not depend on CPython's
    internal implementation of randrange.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    bits = (n - 1).bit_length()
    if bits == 0:
        return 0
    while True:
        candidate = rng.getrandbits(bits)
        if candidate < n:
            return candidate


def _draw_length(rng: random.Random, dist: LengthDist) -> int:
    """One clamped lognormal length, via explicit inverse-CDF transform."""
    u = min(max(_unit(rng), _UNIT_EPSILON), 1.0 - _UNIT_EPSILON)
    value = math.exp(dist.mu + dist.sigma * _STD_NORMAL.inv_cdf(u))
    return max(dist.lo, min(dist.hi, round(value)))


# --- Moments of the draw process -------------------------------------------


@lru_cache(maxsize=None)
def _length_moments(
    prompt: LengthDist, decode: LengthDist, block_size: int
) -> tuple[float, float]:
    """(mean total length, mean blocks per request) by fixed-seed Monte Carlo.

    Estimated over the exact same draw function the real stream uses, rather
    than solved in closed form, because the closed form would ignore the
    integer rounding and the ceil into blocks.

    Deliberately not measured on a realized stream. A single 304-request draw
    puts the mean blocks per request anywhere in roughly 45.4 to 49.6 depending
    on seed, which would swing any derived sweep point by several levels. Taken
    over the draw process instead, the sweep points are a pure function of the
    config and do not move when the seed changes.
    """
    rng = random.Random(_MEAN_ESTIMATE_SEED)
    total_len = 0
    total_blocks = 0
    for _ in range(_MEAN_ESTIMATE_SAMPLES):
        length = _draw_length(rng, prompt) + _draw_length(rng, decode)
        total_len += length
        total_blocks += math.ceil(length / block_size)
    return total_len / _MEAN_ESTIMATE_SAMPLES, total_blocks / _MEAN_ESTIMATE_SAMPLES


def mean_total_len(prompt: LengthDist, decode: LengthDist, block_size: int = BLOCK_SIZE) -> float:
    """Expected prompt + decode length under the clamped lognormal draws."""
    return _length_moments(prompt, decode, block_size)[0]


def mean_blocks_per_request(
    prompt: LengthDist, decode: LengthDist, block_size: int = BLOCK_SIZE
) -> float:
    """Expected ceil(total_len / block_size), the per-request block footprint."""
    return _length_moments(prompt, decode, block_size)[1]


def concurrency_for_block_demand(blocks_per_request: float, target_blocks: int) -> int:
    """Smallest C whose resident block demand reaches target_blocks."""
    if blocks_per_request <= 0.0:
        raise ValueError(f"blocks_per_request must be positive, got {blocks_per_request}")
    return max(1, math.ceil(target_blocks / blocks_per_request))


def capacity_concurrency(cfg: WorkloadConfig) -> int:
    """C at which mean-sized sequences exactly fill the KV cache."""
    return max(1, int(cfg.total_token_slots // mean_total_len(cfg.prompt, cfg.decode, cfg.block_size)))


# --- STEP 0A CHOICES -------------------------------------------------------
# Two symbols in the Gate 1 spec were unbound: `<DISTRIBUTION>` was a literal
# unfilled placeholder, and `C` in "at least 4C requests" was never defined.
# Both are bound here, in one place, so re-binding either is a one-line change.
# Nothing downstream reads them except through WorkloadConfig and the sweep
# points below. Every derived value here is computed, never typed by hand.
#
#   <DISTRIBUTION>  Clamped lognormal, prompt and decode drawn independently.
#                   The heavy right tail is what produces mixed block-table
#                   occupancy and preemption pressure; a uniform draw would not.
#
#   C               The sweep variable, not a constant. A run sweeps C from 1
#                   to collapse and the 4C rule applies per level:
#                       N = max(requests_per_concurrency * C, min_requests)
#                   C is a required parameter of every builder. The named
#                   levels below are the ones worth pinning a run to.

VOCAB_LIMIT = 151643  # verified: lowest special/added id for both Qwen2.5 models

DEFAULT_CONFIG = WorkloadConfig(
    prompt=LengthDist(mu=6.0, sigma=0.8, lo=16, hi=2048),
    decode=LengthDist(mu=5.0, sigma=0.9, lo=8, hi=1024),
    # Ids drawn from [0, vocab_limit) are ordinary text tokens for both
    # Qwen/Qwen2.5-0.5B-Instruct and Qwen/Qwen2.5-1.5B-Instruct, whose base
    # vocabularies are identical. `verify_vocab_limit` re-checks this live.
    vocab_limit=VOCAB_LIMIT,
    block_size=BLOCK_SIZE,
    num_gpu_blocks=NUM_GPU_BLOCKS,
    min_requests=64,
    requests_per_concurrency=4,
)

CONTROL_PROMPT_LEN = 512
CONTROL_DECODE_LEN = 512
CONTROL_BLOCKS_PER_REQUEST = math.ceil((CONTROL_PROMPT_LEN + CONTROL_DECODE_LEN) / BLOCK_SIZE)

# The oversubscription point the sweep is pinned against: demand exceeding the
# 3584-block cache by enough to force eviction rather than merely fill it.
BLOCK_DEMAND_TARGET = 4096

MEAN_BLOCKS_PER_REQUEST = mean_blocks_per_request(DEFAULT_CONFIG.prompt, DEFAULT_CONFIG.decode)

# Defaults: the level at which each stream's demand meets cache capacity.
# These reproduce the Step 0a seed-0 digests and are only defaults.
DEFAULT_C = capacity_concurrency(DEFAULT_CONFIG)
DEFAULT_CONTROL_C = max(1, NUM_GPU_BLOCKS // CONTROL_BLOCKS_PER_REQUEST)

# Named sweep points, both at BLOCK_DEMAND_TARGET blocks of demand.
PRESSURE_C = concurrency_for_block_demand(CONTROL_BLOCKS_PER_REQUEST, BLOCK_DEMAND_TARGET)
FRAG_C = concurrency_for_block_demand(MEAN_BLOCKS_PER_REQUEST, BLOCK_DEMAND_TARGET)


# --- Stream sizing ---------------------------------------------------------


def request_count(c: int, cfg: WorkloadConfig = DEFAULT_CONFIG) -> int:
    """N = max(4C, 64): the spec's 4C rule, applied per concurrency level."""
    if c < 1:
        raise ValueError(f"concurrency must be at least 1, got {c}")
    return max(cfg.min_requests, cfg.requests_per_concurrency * c)


def block_demand(c: int, blocks_per_request: float) -> float:
    """Resident blocks demanded by C concurrent requests of that footprint."""
    return c * blocks_per_request


# --- Stream construction ---------------------------------------------------


def _make_request(
    rng: random.Random, request_id: str, prompt_len: int, max_new_tokens: int, vocab_limit: int
) -> Request:
    prompt = tuple(_below(rng, vocab_limit) for _ in range(prompt_len))
    return Request(request_id=request_id, prompt_token_ids=prompt, max_new_tokens=max_new_tokens)


def build_mixed_stream(
    seed: int | str, c: int = DEFAULT_C, cfg: WorkloadConfig = DEFAULT_CONFIG
) -> tuple[Request, ...]:
    """The workload of record: clamped lognormal prompt and decode lengths."""
    rng = _rng(seed, "mixed")
    requests = []
    for index in range(request_count(c, cfg)):
        # Lengths first, then the prompt, so the draw order is fixed per request.
        prompt_len = _draw_length(rng, cfg.prompt)
        max_new_tokens = _draw_length(rng, cfg.decode)
        requests.append(
            _make_request(rng, f"mixed-{index:06d}", prompt_len, max_new_tokens, cfg.vocab_limit)
        )
    return tuple(requests)


def build_control_stream(
    seed: int | str, c: int = DEFAULT_CONTROL_C, cfg: WorkloadConfig = DEFAULT_CONFIG
) -> tuple[Request, ...]:
    """Rectangular 512-prompt / 512-decode. Shapes fixed; only tokens vary."""
    rng = _rng(seed, "control")
    return tuple(
        _make_request(
            rng, f"ctrl-{index:06d}", CONTROL_PROMPT_LEN, CONTROL_DECODE_LEN, cfg.vocab_limit
        )
        for index in range(request_count(c, cfg))
    )


STREAM_BUILDERS = {"mixed": build_mixed_stream, "control": build_control_stream}
DEFAULT_STREAM_C = {"mixed": DEFAULT_C, "control": DEFAULT_CONTROL_C}


def build_stream(
    name: str,
    seed: int | str,
    c: int | None = None,
    cfg: WorkloadConfig = DEFAULT_CONFIG,
) -> tuple[Request, ...]:
    """Build a named stream. `c=None` uses that stream's capacity default."""
    if name not in STREAM_BUILDERS:
        raise ValueError(f"unknown stream {name!r}, expected one of {sorted(STREAM_BUILDERS)}")
    return STREAM_BUILDERS[name](seed, DEFAULT_STREAM_C[name] if c is None else c, cfg)


# --- Serialization ---------------------------------------------------------


def canonical_bytes(requests: Sequence[Request]) -> bytes:
    """The one on-disk form. Sorted keys, no optional whitespace, LF endings."""
    lines = [
        json.dumps(
            {
                "request_id": request.request_id,
                "prompt_token_ids": list(request.prompt_token_ids),
                "max_new_tokens": request.max_new_tokens,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for request in requests
    ]
    return "".join(f"{line}\n" for line in lines).encode("utf-8")


def stream_digest(requests: Sequence[Request]) -> str:
    """SHA-256 over `canonical_bytes`. Two engines compare this, not prose."""
    return hashlib.sha256(canonical_bytes(requests)).hexdigest()


def write_jsonl(requests: Sequence[Request], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_bytes(requests))
    return destination


def read_jsonl(path: str | Path) -> tuple[Request, ...]:
    """Reload a stream. `stream_digest` of the result must match the writer's."""
    requests = []
    for line_number, line in enumerate(Path(path).read_bytes().decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = {"request_id", "prompt_token_ids", "max_new_tokens"} - record.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} missing field(s) {sorted(missing)}")
        requests.append(
            Request(
                request_id=record["request_id"],
                prompt_token_ids=tuple(record["prompt_token_ids"]),
                max_new_tokens=record["max_new_tokens"],
            )
        )
    return tuple(requests)


# --- Reporting -------------------------------------------------------------


def summarize(requests: Sequence[Request], c: int | None = None) -> dict[str, object]:
    """Shape of the stream. Counts and lengths only; this file times nothing."""
    prompts = [request.prompt_len for request in requests]
    decodes = [request.max_new_tokens for request in requests]
    blocks = [math.ceil(request.total_len / BLOCK_SIZE) for request in requests]
    summary: dict[str, object] = {
        "concurrency_c": c,
        "requests": len(requests),
        "prompt_tokens": sum(prompts),
        "decode_tokens": sum(decodes),
        "prompt_len_min_median_max": (min(prompts), int(statistics.median(prompts)), max(prompts)),
        "decode_len_min_median_max": (min(decodes), int(statistics.median(decodes)), max(decodes)),
        "mean_blocks_per_request": round(sum(blocks) / len(blocks), 4),
    }
    if c is not None:
        summary["resident_blocks_at_c"] = round(c * sum(blocks) / len(blocks), 1)
        summary["cache_blocks"] = NUM_GPU_BLOCKS
    summary["digest"] = stream_digest(requests)
    return summary


def verify_vocab_limit(model_id: str, cfg: WorkloadConfig = DEFAULT_CONFIG) -> bool:
    """True if every id in [0, vocab_limit) is an ordinary text token.

    Imported lazily: generating a stream must not require transformers.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    reserved = set(tokenizer.get_added_vocab().values()) | set(tokenizer.all_special_ids)
    lowest_reserved = min(reserved, default=len(tokenizer))
    return cfg.vocab_limit <= lowest_reserved


# --- CLI -------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a deterministic request stream as JSONL of token IDs."
    )
    parser.add_argument("--stream", choices=sorted(STREAM_BUILDERS), default="mixed")
    parser.add_argument("--seed", default="0", help="int or string; namespaced per stream")
    parser.add_argument(
        "-C",
        "--concurrency",
        type=int,
        default=None,
        help=f"sweep level C; N = max(4C, 64). "
        f"default: {DEFAULT_C} (mixed) / {DEFAULT_CONTROL_C} (control). "
        f"named levels: PRESSURE_C={PRESSURE_C}, FRAG_C={FRAG_C}",
    )
    parser.add_argument("--out", type=Path, default=None, help="default: stdout")
    parser.add_argument("--digest-only", action="store_true", help="print the digest and exit")
    args = parser.parse_args(argv)

    concurrency = DEFAULT_STREAM_C[args.stream] if args.concurrency is None else args.concurrency
    requests = build_stream(args.stream, args.seed, concurrency)

    if args.digest_only:
        print(stream_digest(requests))
        return 0

    if args.out is None:
        sys.stdout.buffer.write(canonical_bytes(requests))
    else:
        write_jsonl(requests, args.out)

    # stderr, so stdout stays pure JSONL for piping.
    print(f"stream={args.stream} seed={args.seed}", file=sys.stderr)
    for key, value in summarize(requests, concurrency).items():
        print(f"  {key} = {value}", file=sys.stderr)
    if args.out is not None:
        print(f"  written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
