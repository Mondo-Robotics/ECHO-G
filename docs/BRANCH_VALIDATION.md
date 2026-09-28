# V2 release-branch validation

This report concerns the new `src/echo_g` package on `release/v2-20260928`. It is separate from
the earlier standalone implementation's full common3242 parity and historical benchmark.

## Real checkpoint parity — passed

The packaged implementation was compared with the frozen original V2 implementation on five
real frozen-condition clips, using the actual best15k EMA checkpoint:

```text
checkpoint SHA256: c1e7e863e28b76b710bc18f9e9029836771fc86c1bb9fca42aa9d7655f74a616
sampling: seed 0, 8 Euler steps, CFG=1
runtime: NVIDIA A800-SXM4-80GB, PyTorch 2.7.0+cu128
```

| Canary | Frames | Text tokens |
|---|---:|---:|
| 1 | 60 | 8 |
| 2 | 312 | 40 |
| 3 | 591 | 73 |
| 4 | 600 | 52 |
| 5 | 588 | 247 |

All five comparisons passed:

- Strict loading and model state agreed with the original implementation.
- Collated model inputs, including signed time distances, were exactly equal.
- Nonzero flow-velocity outputs were exactly equal.
- Eight-step Euler sampling and physical-unit denormalization were exactly equal.
- Maximum absolute difference was **0** for each comparison.

A signed-time check also replaced the distance with its absolute value. This changed the
flow-velocity output for all five clips (maximum absolute differences 0.000232–0.001388),
confirming that the sign affects this trained model. It does not establish a benchmark gain
from timing or learned lag.

The canaries exercise the valid frame limit, long text beyond the historical 64-token limit,
and real trained parameters. This is stronger than comparing freshly initialized zero-output
models. [Machine-readable report](../validation/real_checkpoint_parity.json).

## Package tests

The final CPU unit/integration suite completed with **42 passed, 1 skipped, and no failures**
(43 tests). The one skip was the MuJoCo render test in the test environment without MuJoCo.
A separate environment with MuJoCo 3.10.0 successfully rendered a three-frame synthetic motion.
Repository-wide Ruff checks also passed.

Additional comparisons recorded for this branch:

| Check | Result and scope |
|---|---|
| Packaged model versus independent V2 reference | Initial state and nonzero forwards exactly equal on synthetic cases, including 256 tokens; [report](../validation/model_port_reference_parity.json) |
| Training math versus frozen original trainer | Loss, gradients, RNG, optimizer and EMA exactly equal for dropout 0, 0.1 and 1; [report](../validation/training_math_reference_parity.json) |
| Latest evaluator computational definitions | 33 core definitions unchanged by AST comparison, excluding imports/docstrings and CLI/main; [report](../validation/benchmark_visualization_parity.json) |
| Evaluation and visualization entry points | Help/import checks passed; dependency and motion-export computations checked in the same report |

These checks cover specific implementation contracts. The synthetic render checks the renderer
with a three-frame test motion; it does not establish final external G1 asset validation or a
real-data video. None of these checks substitutes for the real-checkpoint comparisons above or
a complete dataset benchmark.

## Scope and remaining measurements

These checks establish parity on the five specified real frozen-condition inputs in the stated
runtime. They do not establish a new full common3242 benchmark, MM20 run, end-to-end raw-encoder
parity, or cross-hardware bitwise equality for this branch. The pretrained checkpoint, complete
data, encoders, evaluator assets, and robot XML/meshes remain externally provisioned assets.

The historical scores and previous 3,242-clip standalone-port comparison remain documented in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md#historical-results), with their original model and evaluator
identities. They have not been relabeled as measurements of this new package.

## Package installation

The `0.2.0.dev0` wheel was built and installed in an isolated target directory. All eight console commands passed `--help` from outside the checkout. See [installation report](../validation/package_installation.json) and [test summary](../validation/test_summary.json).
