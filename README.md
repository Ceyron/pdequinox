<h4 align="center">PDE Emulator Architectures for <a href="https://github.com/patrick-kidger/equinox" target="_blank">Equinox</a>.</h4>

<p align="center">
<a href="https://pypi.org/project/pdequinox/">
  <img src="https://img.shields.io/pypi/v/pdequinox.svg" alt="PyPI">
</a>
<a href="https://github.com/ceyron/pdequinox/actions/workflows/test.yml">
  <img src="https://github.com/ceyron/pdequinox/actions/workflows/test.yml/badge.svg" alt="Tests">
</a>
<a href="https://fkoehler.site/pdequinox/">
  <img src="https://img.shields.io/badge/docs-latest-green" alt="docs-latest">
</a>
<a href="https://github.com/ceyron/pdequinox/releases">
  <img src="https://img.shields.io/github/v/release/ceyron/pdequinox?include_prereleases&label=changelog" alt="Changelog">
</a>
<a href="https://github.com/ceyron/pdequinox/blob/main/LICENSE.txt">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#background">Background</a> •
  <a href="#features">Features</a> •
  <a href="#boundary-conditions">Boundary Conditions</a> •
  <!-- <a href="#constructors">Constructors</a> • -->
  <a href="#acknowledgements">Acknowledgements</a>
</p>

<p align="center">
    <img width=600 src="https://github.com/user-attachments/assets/8948f0e8-b879-468e-aaa2-158788b4d3f2">
</p>

A collection of neural architectures for emulating Partial Differential Equations (PDEs) in JAX agnostic to the spatial dimension (1D, 2D, 3D) and boundary conditions (Dirichlet, Neumann, Periodic). This package is built on top of [Equinox](https://github.com/patrick-kidger/equinox).

## Installation

```bash
pip install pdequinox
```

Requires Python 3.10+ and JAX 0.4.13+. 👉 [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).

## Documentation

The documentation is available at [fkoehler.site/pdequinox](https://fkoehler.site/pdequinox/).


## Quickstart

Train a UNet to become an emulator for the 1D Poisson equation.

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import optax  # `pip install optax`
import pdequinox as pdeqx
from tqdm import tqdm  # `pip install tqdm`

force_fields, displacement_fields = pdeqx.sample_data.poisson_1d_dirichlet(
    key=jax.random.PRNGKey(0)
)

force_fields_train = force_fields[:800]
force_fields_test = force_fields[800:]
displacement_fields_train = displacement_fields[:800]
displacement_fields_test = displacement_fields[800:]

unet = pdeqx.arch.ClassicUNet(1, 1, 1, key=jax.random.PRNGKey(1))

def loss_fn(model, x, y):
    y_pref = jax.vmap(model)(x)
    return jnp.mean((y_pref - y) ** 2)

opt = optax.adam(3e-4)
opt_state = opt.init(eqx.filter(unet, eqx.is_array))

@eqx.filter_jit
def update_fn(model, state, x, y):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, new_state = opt.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss

loss_history = []
shuffle_key = jax.random.PRNGKey(151)
for epoch in tqdm(range(100)):
    shuffle_key, subkey = jax.random.split(shuffle_key)

    for batch in pdeqx.dataloader(
        (force_fields_train, displacement_fields_train),
        batch_size=32,
        key=subkey
    ):
        unet, opt_state, loss = update_fn(
            unet,
            opt_state,
            *batch,
        )
        loss_history.append(loss)
```
## Background

Neural Emulators are networks learned to efficienty predict physical phenomena,
often associated with PDEs. In the simplest case this can be a linear advection
equation, all the way to more complicated Navier-Stokes cases. If we work on
Uniform Cartesian grids* (which this package assumes), one can borrow plenty of
architectures from image-to-image tasks in computer vision (e.g., for
segmentation). This includes:

* Standard Feedforward ConvNets
* Convolutional ResNets ([He et al.](https://arxiv.org/abs/1512.03385))
* U-Nets ([Ronneberger et al.](https://arxiv.org/abs/1505.04597))
* Dilated ResNets ([Yu et al.](https://arxiv.org/abs/1511.07122), [Stachenfeld et al.](https://arxiv.org/abs/2112.15275))
* Fourier Neural Operators ([Li et al.](https://arxiv.org/abs/2010.08895))

It is interesting to note that most of these architectures resemble classical
numerical methods or at least share similarities with them. For example,
ConvNets (or convolutions in general) are related to finite differences, while
U-Nets resemble multigrid methods. Fourier Neural Operators are related to
spectral methods. The difference is that the emulators' free parameters are
found based on a (data-driven) numerical optimization not a symbolic
manipulation of the differential equations.

(*) This means that we essentially have a pixel or voxel grid on which space is
discretized. Hence, the space can only be the scaled unit cube $\Omega = (0,
L)^D$

## Features

* Based on [JAX](https://github.com/google/jax):
  * One of the best Automatic Differentiation engines (forward & reverse)
  * Automatic vectorization
  * Backend-agnostic code (run on CPU, GPU, and TPU)
* Based on [Equinox](https://github.com/patrick-kidger/equinox):
  * Single-Batch by design
  * Integration into the Equinox SciML ecosystem
* Agnostic to the spatial dimension (works for 1D, 2D, and 3D)
* Agnostic to the boundary condition (works for Dirichlet, Neumann, and periodic
  BCs)
* Composability
* Tools to count parameters and assess receptive fields

## Boundary Conditions

This package assumes that the boundary condition is baked into the neural
emulator. Hence, most components allow setting `boundary_mode` which can be
`"dirichlet"`, `"neumann"`, or `"periodic"`. This affects what is considered a
degree of freedom in the grid.

![three_boundary_conditions](https://github.com/user-attachments/assets/a46c276c-4c4b-4890-aca2-49c8b04d1948)

Dirichlet boundaries fully eliminate degrees of freedom on the boundary.
Periodic boundaries only keep one end of the domain as a degree of freedom (This
package follows the convention that the left boundary is the degree of freedom). Neumann boundaries keep both ends as degrees of freedom.

## Acknowledgements

### Related Work

Similar packages that provide a collection of emulator architectures are
[PDEBench](https://github.com/pdebench/PDEBench) and
[PDEArena](https://github.com/pdearena/pdearena). With focus on Phyiscs-informed
Neural Networks and Neural Operators, there are also
[DeepXDE](https://github.com/lululxvi/deepxde) and [NVIDIA
Modulus](https://developer.nvidia.com/modulus).

### Citation

This package was developed as part of the [APEBench paper
(arxiv.org/abs/2411.00180)](https://arxiv.org/abs/2411.00180) (accepted at
Neurips 2024). If you find it useful for your research, please consider citing
it:

```bibtex
@article{koehler2024apebench,
  title={{APEBench}: A Benchmark for Autoregressive Neural Emulators of {PDE}s},
  author={Felix Koehler and Simon Niedermayr and R{\"}udiger Westermann and Nils Thuerey},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={38},
  year={2024}
}
```

(Feel free to also give the project a star on GitHub if you like it.)

[Here](https://github.com/tum-pbs/apebench) you can find the APEBench benchmark
suite.

### Funding

The main author (Felix Koehler) is a PhD student in the group of [Prof. Thuerey at TUM](https://ge.in.tum.de/) and his research is funded by the [Munich Center for Machine Learning](https://mcml.ai/).

### License

MIT, see [here](LICENSE.txt)

---

> [fkoehler.site](https://fkoehler.site/) &nbsp;&middot;&nbsp;
> GitHub [@ceyron](https://github.com/ceyron) &nbsp;&middot;&nbsp;
> X [@felix_m_koehler](https://twitter.com/felix_m_koehler) &nbsp;&middot;&nbsp;
> LinkedIn [Felix Köhler](www.linkedin.com/in/felix-koehler)
