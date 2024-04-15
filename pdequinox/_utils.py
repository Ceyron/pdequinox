from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree


class ConstantEmbeddingMetadataNetwork(eqx.Module):
    """
    Wrap an equinox Module based on a convolutional format (channels,
    *spatial_dims) to take an additional scalar argument which will be
    concatenated to the other's input channel as a constant field.

    **Arguments:**

    - `network`: eqx.Module. The network to be wrapped (which has a
        convolutional input structure)
    - `normalization_factor`: float. The scalar used to normalize the input to
        ensure its order of magnitude is similar to the other inputs.
    """

    network: eqx.Module
    normalization_factor: float

    def __call__(
        self,
        x,
        meta_data,
    ):
        """
        Take state of shape (n_channels, n_dof_1, n_dof_2, ..., n_dof_d) and
        metadata of the shape (n_metadata,) concatenate them into a single
        tensor of shape (n_channels + n_metadata, n_dof_1, n_dof_2, ..., n_dof_d)
        and pass it through the network.
        """

        normalization_factor_detached = jax.lax.stop_gradient(self.normalization_factor)

        meta_data = meta_data / normalization_factor_detached

        meta_data_embedded = meta_data.reshape(
            (
                -1,
                *(
                    [
                        1,
                    ]
                    * (x.ndim - 1)
                ),
            )
        ) * jnp.ones((1, *x.shape[1:]))
        x_with_embedding = jnp.concatenate((x, meta_data_embedded), axis=0)
        return self.network(x_with_embedding)


def count_parameters(model: eqx.Module):
    """
    Count parameters in an equinox Module (this includes the
    architectures of PDEquinox).
    """
    return sum(p.size for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))


def dataloader(
    data: Union[PyTree, Array],
    *,
    batch_size: int,
    key: PRNGKeyArray,
):
    """
    A generator for looping over the data in batches.

    The data is shuffled before looping. The length is based on how many
    minibatches are needed to loop over the data once (n_samples // batch_size +
    1). In deep learning terminology, one looping over the dataloader represents
    one epoch.

    For a supervised learning problem use

    ```python

    dataloader(
        (inputs, targets), batch_size=batch_size, key=key,
    )

    ```

    **Arguments:**

    - `data`: Union[PyTree, Array]. The data to be looped over. This must be
        JAX-compatible PyTree; in the easiest case an array. Each leaf array in
        the PyTree must be array-like with a leading a batch axis. These leading
        batch axes must be identical for all leafs.
    - `batch_size`: int. The size of the minibatches. (keyword-based argument)
    - `key`: JAX PRNGKey. The key to be used for shuffling the data; required
        for reproducible randomness. (keyword-based argument)
    """

    n_samples_list = [a.shape[0] for a in jtu.tree_leaves(data)]

    if not all(n == n_samples_list[0] for n in n_samples_list):
        raise ValueError(
            "All arrays / PyTree leaves must have the same number of samples. (Leading array axis)"
        )

    n_samples = n_samples_list[0]

    n_batches = int(jnp.ceil(n_samples / batch_size))

    permutation = jax.random.permutation(key, n_samples)

    for batch_id in range(n_batches):
        start = batch_id * batch_size
        end = min((batch_id + 1) * batch_size, n_samples)

        batch_indices = permutation[start:end]

        sub_data = jtu.tree_map(lambda a: a[batch_indices], data)

        yield sub_data


def cycling_dataloader(
    data: Union[PyTree, Array],
    *,
    batch_size: int,
    num_steps: int,
    key: PRNGKeyArray,
    return_info: bool = False,
):
    """
    A generator for looping over the data in batches for a fixed number of
    steps.

    It performs as many epochs (one full iteration over the data) as needed to
    produce `num_steps` batches. Note that one batch will never contain data
    from two epochs. Internally, this generator uses the `dataloader` generator.
    Hence, if `batch_size` is chosen larger than the length of batch axis in the
    leaf arrays of `data`, the batch will be of the size of the data.

    For a supervised learning problem use

    ```python

    cycling_dataloader(
        (inputs, targets), batch_size=batch_size, num_steps=num_steps, key=key,
    )

    ```

    **Arguments:**

    - `data`: Union[PyTree, Array]. The data to be looped over. This must be
        JAX-compatible PyTree; in the easiest case an array. Each leaf array in
        the PyTree must be array-like with a leading a batch axis. These leading
        batch axes must be identical for all leafs.
    - `batch_size`: int. The size of the minibatches. (keyword-based argument)
    - `num_steps`: int. The number of steps to loop over the data.
      (keyword-based argument)
    - `key`: JAX PRNGKey. The key to be used for shuffling the data; required
        for reproducible randomness. (keyword-based argument)
    - `return_info`: bool. Whether to return the epoch and batch indices in
        addition to the data. (keyword-based argument)
    """
    epoch_id = 0
    total_step_id = 0

    while True:
        key, subkey = jax.random.split(key)

        for batch_id, sub_data in enumerate(
            dataloader(data, batch_size=batch_size, key=subkey)
        ):
            if total_step_id == num_steps:
                return

            if return_info:
                yield sub_data, epoch_id, batch_id
            else:
                yield sub_data

            total_step_id += 1

        epoch_id += 1


def extract_from_ensemble(ensemble: eqx.Module, i: int):
    """
    Given an ensemble of equinox Modules, extract its i-th element.

    If you create an ensemble, e.g., with

    ```python

    import equinox as eqx

    ensemble = eqx.filter_vmap(
        lambda k: eqx.nn.Conv1d(1, 1, 3)
    )(jax.random.split(jax.random.PRNGKey(0), 5) ```

    its weight arrays have an additional batch/ensemble axis. It cannot be used
    natively on its corresponding data. This function extracts the i-th element
    of the ensemble.

    **Arguments:**

    - `ensemble`: eqx.Module. The ensemble of networks.
    - `i`: int. The index of the network to be extracted. This can also be a
        slice!
    """
    params, static = eqx.partition(ensemble, eqx.is_array)
    params_extracted = jtu.tree_map(lambda x: x[i], params)
    network_extracted = eqx.combine(params_extracted, static)
    return network_extracted


def combine_to_ensemble(networks: list[eqx.Module]) -> eqx.Module:
    """
    Given a list of multiple equinox Modules of the same PyTree structure
    combine them into an essemble (to have the weight arrays with an additional
    batch/ensemble axis).

    **Arguments:**

    - `networks`: list[eqx.Module]. The networks to be combined.

    **Returns:**

    - `ensemble`: eqx.Module. The ensemble of networks.
    """
    _, static = eqx.partition(networks[0], eqx.is_array)
    params = [eqx.filter(network, eqx.is_array) for network in networks]
    params_combined = jtu.tree_map(lambda *x: jnp.stack(x), *params)
    ensemble = eqx.combine(params_combined, static)
    return ensemble


def sum_receptive_fields(
    receptive_fields: tuple[tuple[tuple[float, float], ...], ...]
) -> tuple[tuple[float, float], ...]:
    """
    Given a list of receptive fields (each a tuple of tuples of floats) sum them
    up to get the total receptive field.

    The receptive field is a structure with three nested tuples. The outer-most
    tuple refers to the collection of receptive fields to be added up. The
    mid-level tuple is over the number of spatial dimensions; this can be of
    length one, two, or three. The inner-most tuple always contains two floats
    representing the receptive field in downward and upward direction.

    **Arguments:**

    - `receptive_fields`: tuple[tuple[tuple[float, float], ...], ...]. The
        receptive fields to be summed up.

    **Returns:**

    - `total_receptive_field`: tuple[tuple[float, float], ...]. The total
        receptive field.
    """
    num_spatial_dims = len(receptive_fields[0])
    return tuple(
        tuple(sum(r[i][direction] for r in receptive_fields) for direction in range(2))
        for i in range(num_spatial_dims)
    )
