# HamiltonZero

HamiltonZero is Simulacra Research's research release for compiled neural
wavefunctions of quantum spin Hamiltonians. It exposes three workflows:

- learned-router multisystem training;
- compiled single-system fine-tuning;
- compiled single-system evaluation, with optional router contest or large-N
  execution.

## Installation

HamiltonZero requires Python 3.12 and JAX-compatible accelerator drivers.

```bash
python -m pip install .
```

The package pins the Python `jax` package to a
[`TakeOver/jax` commit](https://github.com/TakeOver/jax/commit/79f82535b15a444516d4a5e2beb71d283665b2ff),
also published as `hamiltonzero-jax-v0.11.0-spin.1`, and pins
`jaxlib==0.11.0`. The fork contains the symbolic-zero JVP support used by the
tuned Pallas attention kernel; stock Python JAX 0.11.0 is not sufficient for
that pathway. Install the accelerator plugin appropriate for the host using
the standard JAX instructions.

Learned-router training uses eight visible accelerators and requires an MCMC
batch size divisible by eight. Fine-tuning uses all visible accelerators and
requires its MCMC batch size to be divisible by their count. Evaluation chooses
a visible-device subset compatible with its walker batch.

## Foundation checkpoint

The HamiltonZero v1 foundation checkpoint is hosted in the
[Hugging Face repository](https://huggingface.co/simulacra-research/HamiltonZero).
Download it into the path used by the example configurations:

```bash
hf download simulacra-research/HamiltonZero \
  weights/hamiltonzero_v1.eqx \
  --local-dir .
```

The checkpoint contains the complete foundation wavefunction and its learned
router. `router` is the checkpoint kind, not a router-only parameter file.

To load the model directly, construct an architecture template and deserialize
its array leaves:

```python
import jax

from hamiltonzero.checkpoint import load_model
from hamiltonzero.config import ModelConfig
from hamiltonzero.model import build_model

template = build_model(
    ModelConfig(),
    jax.random.PRNGKey(0),
    n_max=64,
)
model = load_model("weights/hamiltonzero_v1.eqx", template)
```

The template key initializes placeholder values only; deserialization replaces
all serialized array leaves. Set `n_max` to the padded width of the system when
constructing a template for direct model use. The command-line evaluation path
does this from the input system automatically.

## Hamiltonians and NetworkX

The public API follows the textbook convention

\[
H = \sum_{i<j} S_i^T J_{ij} S_j + \sum_i h_i^T S_i,
\qquad S=\sigma/2.
\]

Construct and save a system from a simple undirected NetworkX graph:

```python
from pathlib import Path

import networkx as nx

from hamiltonzero import SpinHamiltonian
from hamiltonzero.data import save_system

graph = nx.path_graph(8)
nx.set_edge_attributes(graph, 1.0, "J")
nx.set_node_attributes(graph, 0.0, "h")

system = SpinHamiltonian.from_networkx(graph)
save_system(Path("outputs/systems/chain_8.json"), system)
```

The same example is runnable as `python examples/networkx_system.py`. An edge
`J` may be an isotropic scalar, a length-three diagonal, or a 3-by-3 exchange
matrix. A node `h` may be a scalar z-field or a length-three field vector.
`SpinHamiltonian.from_arrays` accepts dense arrays instead.

HamiltonZero converts public inputs to the model's internal `-J/2` and `-h`
representation. The `SpinHamiltonian.J` and `SpinHamiltonian.h` properties
return the public textbook values. If `mu` is omitted, a conservative value is
computed from the Hamiltonian.

## Standalone compiled inference

The compact inference API loads the foundation checkpoint, runs the
beam-8 router, permutes the Hamiltonian, and compiles the selected physical
wavefunction in one call:

```python
import jax
import networkx as nx

from hamiltonzero import SpinHamiltonian, burn_in, energy, prepare, spin, step

graph = nx.path_graph(8)
nx.set_edge_attributes(graph, 1.0, "J")
system = SpinHamiltonian.from_networkx(graph)

route_key, mcmc_key = jax.random.split(jax.random.PRNGKey(0))
compiled, order = prepare(
    system,
    "weights/hamiltonzero_v1.eqx",
    route_key,
)
state, q = burn_in(
    compiled,
    mcmc_key,
    batch_size=256,
    replicas=8,
    burn_in=1024,
    walker_chunk_size=16,
)
local_energy = energy(compiled, q)
local_spin = spin(compiled, q)
state, q = step(compiled, state, steps=24, walker_chunk_size=16)
```

`state` is the complete replica-exchange MCMC state and `q` is its cold-chain
population. `energy` returns named `total`, `exchange`, `casimir`, and `field`
local-energy samples. `spin` returns the complex local spin estimator in the
routed `(site, x/y/z)` order; contracting it with the routed public field
reproduces the energy field channel. The selected padded-site permutation is
returned as `order`. `order.leaf_to_input[leaf]` is the public input-site index
assigned to a compiled tree leaf; `order.input_to_leaf[site]` is its inverse.
The first mapping is also available as `compiled.route`. The public NetworkX
path starts in exactly the supplied `system.nodes` order, applies this route
once to the context and walkers, and then compiles an identity-routed tree.
There is no additional bit reversal: applying one would corrupt the mapping.
Both arrays include padded virtual leaves when the model width exceeds the
physical site count. A complete runnable version that prints sample means and
standard deviations is in
[`examples/compiled_inference.py`](examples/compiled_inference.py).
[`examples/j1j2_4x4_route.ipynb`](examples/j1j2_4x4_route.ipynb) constructs a
periodic 4-by-4 J1-J2 model from NetworkX and visualizes the returned order as
the successive cells of the compiled binary merge tree. Install its plotting
dependencies with `python -m pip install '.[notebooks]'`.

For a normalized pure state, the full-state `Tr(|psi><psi|)` is exactly one.
The nontrivial purity observable is the subsystem second Renyi value
`Tr(rho_A^2)`. It uses two independent computational-basis chains and a
two-replica SWAP estimator:

```python
import jax

from hamiltonzero import burn_in_basis, measure_renyi2

x_key, y_key = jax.random.split(jax.random.PRNGKey(1))
x_state, x = burn_in_basis(compiled, x_key, batch_size=256, burn_in=1024)
y_state, y = burn_in_basis(compiled, y_key, batch_size=256, burn_in=1024)
x_state, y_state, result = measure_renyi2(
    compiled,
    x_state,
    y_state,
    subsystem=range(4),
    blocks=16,
    steps_between=24,
)
print(result.purity, result.standard_error, result.renyi2_nats)
print(result.resolved, result.failure_reasons)
```

`subsystem` accepts public site indices or a boolean mask. The result also
retains each SWAP sample in stable log-polar form. Entropy is reported only
when the block-count, effective-sample-size, autocorrelation,
heavy-tail, imaginary-null, and physical-bound checks resolve the estimate;
otherwise `renyi2_nats` is `None` and `failure_reasons` says why.
`burn_in_basis` samples the computational basis required by this estimator.
The SU(2)-quaternion walkers returned by `burn_in` cannot be substituted for
those samples.

## Datasets

The repository includes the exact 5,000-system foundation training panel and
the evaluation systems with at least 256 physical spins. Every file uses the
public textbook units above.

- `datasets/train/foundation_5000.jsonl` contains systems from 2 through 64
  spins, fixed WL1/FWL2 dispatch, and available exact-diagonalization energies.
- `datasets/eval/` contains the PPP-Ohno, RUDY, square-lattice J1-J2, and
  triangular-Heisenberg evaluation systems from 256 through 8,100 physical
  spins.

Large-N files store physical sites only; the loader reconstructs power-of-two
padding in memory. See [`datasets/README.md`](datasets/README.md) for the full
inventory and sparse exchange encoding.

## Train

The training command starts a new learned-router multisystem run and writes one
final full foundation-model checkpoint, including its learned router, plus a
metadata sidecar:

```bash
hamiltonzero train examples/train.json
```

The example uses `datasets/train/foundation_5000.jsonl`, writes
`outputs/foundation.eqx`, and exposes model, MCMC, KFAC, router, and energy
parameters through JSON. The command writes the trained model at the end of
the run.

To skip burn-in using compatible post-burn-in sampler states:

```bash
hamiltonzero train examples/train.json --reuse-mcmc path/to/mcmc-states
```

For multisystem training, the path is a directory containing
`<system-index>.eqx` files. For a one-system training panel it may be a single
file.

## Fine-tune

Fine-tuning selects and freezes a route from a router checkpoint, compiles the
single-system wavefunction, and optimizes that compiled model:

```bash
hamiltonzero finetune examples/finetune.json
```

The example fine-tunes on the 256-spin PPP-Ohno system and writes
`outputs/ppp_ohno_n256.eqx`. A neighboring `.eqx.json` sidecar records the
compiled-fine-tune kind, frozen model width, and configured ranks. A compatible
single post-burn-in state can also be supplied:

```bash
hamiltonzero finetune examples/finetune.json --reuse-mcmc path/to/state.eqx
```

## Evaluate

Compiled evaluation uses the route selected by a router checkpoint, or the
embedded frozen route in a compiled fine-tune checkpoint:

```bash
hamiltonzero eval examples/eval.json
```

Use router contest to compare candidate routes before evaluating the winner:

```bash
hamiltonzero eval examples/eval.json --contest
```

Use the sequence-sharded large-N implementation for the large systems:

```bash
hamiltonzero eval examples/eval_large_n.json --large-n
```

Each evaluation writes `eval.json` and `eval.metrics.jsonl` inside its
configured output directory.

Training and fine-tuning metrics are written beside the final checkpoint as
`<checkpoint>.metrics.jsonl`. Evaluation writes the same per-measurement fields
to `eval.metrics.jsonl`. These JSONL rows contain step, energy, energy standard
deviation, step wall time, and total wall time. Final `eval.json` additionally
reports exchange/field channels and lag-one autocorrelation when available.

## Configuration

Every command accepts one JSON configuration. The files in `examples/` are
minimal runnable configurations; omitted parameters use the defaults in
`hamiltonzero.config`.

The KFAC-JAX fork is vendored under `src/kfac_jax`.

## License

HamiltonZero first-party source, datasets, and released model weights are
licensed under Apache-2.0, copyright Simulacra Research Inc. The vendored
KFAC-JAX fork and JAX-derived large-N attention kernel remain under
Apache-2.0. The Microsoft-Folx-derived attention forward and reverse-mode
kernels remain under MIT. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
