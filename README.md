# CRD: Robust Counter-Cascade Defense

C++17 implementation accompanying **Robust Counter-Cascade Defense against
Opinion-Conflict Attacks in Online Social Networks**.

This compact source distribution contains CRD, its linearized variant CRD+,
the baselines and ablations, and one preprocessed example network (Convote).

## Build

Requirements: a C++17 compiler, CMake 3.16 or newer, and Make.
Eigen 3.4.1 and AMGCL headers are bundled; no dependency download is needed.

Run from the repository root:

```bash
make
```

This configures a Release build and compiles with four build jobs. Equivalently:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

The executable is `build/crd`. Release builds with GCC use `-O3`.

## Quick Example

```bash
./build/crd \
  --graph dataset/convote/convote.txt \
  --s-file dataset/convote/innate_uniform.txt \
  --algorithm crd \
  --conflict-backend dense \
  --crd-mode engineering \
  --k-A-relative 0.5% \
  --k-B-relative 0.5% \
  --T 3 \
  --theta-A 1000 \
  --theta-B 1000 \
  --theta-emp 1000 \
  --epsilon-relative 10% \
  --mc-simulations 100 \
  --seed 1 \
  --output-dir results/convote_demo \
  --output-prefix crd
```

The example uses IC and polarization, with reduced sampling budgets for a quick
run; it is not the paper's full experimental configuration. Results are written
as TXT, CSV, and a complete seed-set TXT file in the specified output directory.

## Other Settings

- Always specify `--graph` and `--s-file` when using the bundled dataset.
- For CRD+, use `--algorithm crd-linear --conflict-backend amgcl`.
- FJ-Iter (`--algorithm fjc-iter`) requires `--conflict-backend dense`.
- For HIST and HIST-Iter, use `--epsilon 0.1` instead of `--epsilon-relative`.
- For LT, use `--propagation-model lt --p-file dataset/convote/lt_weights.txt`.
- For disagreement, use `--metric disagreement`.

See `./build/crd --help` for all algorithms and parameters, and
[the dataset README](dataset/convote/README.md) for data provenance.
Third-party licenses are preserved in the Eigen and AMGCL directories.
