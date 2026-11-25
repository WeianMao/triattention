import torch

from triattention.methods.pruning_utils import (
    score_keys_for_round,
    score_keys_for_round_reference,
)

# D = {1, 2, 4, ..., 2^16}, the geometric offset schedule the score averages over.
OFFSETS = [float(2**i) for i in range(17)]
BANDS = 64
THETA = 10000.0


def _inputs(
    keys: int = 256,
    round_start: int = 5_000,
    aggregation: str = "mean",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple:
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=generator, dtype=torch.float64).to(dtype)

    omega = (THETA ** (-torch.arange(BANDS, dtype=torch.float64) / BANDS)).to(dtype)
    return (
        torch.arange(keys),
        round_start,
        rand(keys, BANDS),
        (rand(keys, BANDS) * 2 - 1) * torch.pi,
        omega,
        rand(keys, BANDS),
        torch.tensor(OFFSETS, dtype=dtype),
        aggregation,
        rand(BANDS),
    )


def test_folded_score_matches_reference_for_mean_aggregation() -> None:
    # The offset average is separable, so folding it into a per-band weight is an
    # identity. In float64 the two forms should agree to round-off. Round-off grows
    # with the phase magnitude, hence with round_start, so the bound is loose enough
    # to stay valid at long context.
    for round_start in (500, 5_000, 32_000, 131_072, 1_000_000):
        args = _inputs(round_start=round_start, dtype=torch.float64)
        reference = score_keys_for_round_reference(*args)
        folded = score_keys_for_round(*args)
        relative = (folded - reference).abs().max() / reference.abs().max()
        assert relative < 1e-11, f"round_start={round_start}, relative={relative:g}"


def test_non_folding_paths_are_unchanged() -> None:
    # The fold applies to the mean aggregation only: max is nonlinear, and the
    # disable_trig score carries no offset dependence to begin with. Both cases
    # defer to the original implementation and must be bit-identical.
    for aggregation in ("mean", "max"):
        for disable_trig in (False, True):
            if aggregation == "mean" and not disable_trig:
                continue
            args = _inputs(aggregation=aggregation)
            reference = score_keys_for_round_reference(*args, disable_trig)
            folded = score_keys_for_round(*args, disable_trig)
            assert torch.equal(folded, reference), (
                f"aggregation={aggregation}, disable_trig={disable_trig}"
            )


def test_empty_key_set_returns_empty_scores() -> None:
    args = _inputs(keys=0)
    assert score_keys_for_round(*args).numel() == 0
