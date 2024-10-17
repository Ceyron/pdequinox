# Getting Started

## Installation

Clone the repository, navigate to the folder and install the package with pip:
```bash
pip install pdequinox
```

Requires Python 3.10+ and JAX 0.4.13+. 👉 [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).


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

## Features

* Based on [JAX](https://github.com/google/jax):
    * One of the best Automatic Differentiation engines (forward & reverse)
    * Automatic vectorization
    * Backend-agnostic code (run on CPU, GPU, and TPU)
* Built on top of [Equinox](https://github.com/patrick-kidger/equinox):
    * Single-Batch by design
    * Integration into the Equinox SciML ecosystem
* Agnostic to the spatial dimension (works for 1D, 2D, and 3D)
* Agnostic to the boundary condition (works for Dirichlet, Neumann, and periodic
  BCs)
* Composability
* Tools to count parameters and assess receptive fields


## License

MIT, see [here](https://github.com/Ceyron/pdequinox/blob/main/LICENSE.txt)

---

> [fkoehler.site](https://fkoehler.site/) &nbsp;&middot;&nbsp;
> GitHub [@ceyron](https://github.com/ceyron) &nbsp;&middot;&nbsp;
> X [@felix_m_koehler](https://twitter.com/felix_m_koehler)

