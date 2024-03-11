from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree


class ConstantEmbeddingMetadataNetwork(eqx.Module):
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
    return sum(p.size for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))


def dataloader(
    data: Union[PyTree, Array],
    *,
    batch_size: int,
    key: PRNGKeyArray,
):
    """
    Loop generator over the data. The data can be a PyTree or an Array. For
    supervised learning problems you can also hand over a tuple of Arrays
    (=PyTree).
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
    n_steps: int,
    key: PRNGKeyArray,
    return_info: bool = False,
):
    epoch_id = 0
    total_step_id = 0

    while True:
        key, subkey = jax.random.split(key)

        for batch_id, sub_data in enumerate(
            dataloader(data, batch_size=batch_size, key=subkey)
        ):
            if total_step_id == n_steps:
                return

            if return_info:
                yield sub_data, epoch_id, batch_id
            else:
                yield sub_data

            total_step_id += 1

        epoch_id += 1


def extract_from_ensemble(ensemble, i):
    params, static = eqx.partition(ensemble, eqx.is_array)
    params_extracted = jtu.tree_map(lambda x: x[i], params)
    network_extracted = eqx.combine(params_extracted, static)
    return network_extracted
