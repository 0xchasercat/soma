# Conditional flow: current measured result

Measured 2026-08-07 against the saved session-grouped validation cohort. These
numbers describe the current schema-v2 pair `flow/flow.pt` and
`model/flow.onnx`; they do not carry forward claims from older 255-dimensional
experiments.

## Corpus and model

- Raw trusted movements: 40,591.
- Canonicalized: 40,055 (98.68%).
- Behavioral distance/duration/curvature filters: zero.
- Unrepresentable: 536 zero start-to-end displacement records. They remain
  valid discriminator evidence but have no target axis for this flow.
- Split: 36,044 train and 4,011 validation movements across disjoint capture
  sessions (3,294/349 session groups).
- Model: 5,360,103 parameters, six rational-quadratic coupling layers, 159
  modeled dimensions, three context dimensions.
- Best validation NLL: -606.645 total, -3.8154 per dimension.
- Sample smoke test: 512/512 finite; generated duration p5/p50/p95 was
  137/628/6,581 ms.

The vector drops constrained middle delta 40 rather than the final delta, so
touchdown velocity remains modeled. The conditioner retains temporal phase
instead of globally pooling it. Context is
`log(distance), log(target_size), terminal_stop`.

## Runtime fidelity audit

`python3 flow/audit_eval.py --n 3000 --seed 0` compares held-out human and
generated movements under identical contexts after the same 60 Hz dispatch
resampling. The shipped path uses the measured 89.92% stationary-touchdown
mixture and no second AR(2) tremor layer.

| Diagnostic | Separability |
|---|---:|
| touchdown final-speed ratio | 0.4% |
| duration | 7.7% |
| curvature | 7.7% |
| overshoot occurrence | 8.9% |
| peak speed | 18.1% |
| peak timing | 22.4% |
| submovement count | 28.9% |
| maximum lateral deviation | 29.3% |
| Sigma-Lognormal fit | 30.4% |
| overshoot fraction | 31.5% |
| high-frequency lateral energy | 40.8% |

Mean rank separability across the 11 diagnostics is 20.6% (0% means no rank
information; 100% means complete separation). This is an audit, not a claim of
indistinguishability. High-frequency lateral energy, overshoot magnitude, and
Sigma-Lognormal structure remain the clearest gaps.

The legacy AR(2) postprocessor is disabled on the flow path: applying it to the
earlier raw flow increased mean separability by 3.2 points and made terminal
behavior worse. Full captured microstructure is now left to the learned model.

## Artifact verification

The original nflows ONNX export used data-dependent boolean indexing. Legacy
tracing froze the dummy input's spline-tail mask, causing valid runtime samples
to crash with a reshape error. The current export uses a mathematically
equivalent branch-free spline with `torch.where` tails.

`python3 flow/verify_onnx.py --n 128` reports:

```text
finite=True, mean_abs_error=2.381e-07, max_abs_error=2.074e-05
```

The verifier is a required promotion check; successful PyTorch training alone
does not establish a usable runtime artifact.

## Remaining scope boundary

The positive corpus represents one operator. Uniform-time canonicalization also
compresses long idle gaps into a fixed 81-point sequence. Future work should
evaluate a multi-operator cohort and a representation that models active motion
and within-gesture pauses separately instead of interpreting one as the other.
