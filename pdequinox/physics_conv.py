from jaxtyping import PRNGKeyArray

from periodic_conv import PeriodicConv

def PhysicsConv(
    num_spacial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dilation: int = 1,
    *,
    boundary_mode: str,
    key: PRNGKeyArray,
    use_bias: bool = True,
    zero_bias_init: bool = False,
    **boundary_kwargs,
):
    """
    Convenience constructor
    """
    if boundary_mode.lower() == "periodic":
        return PeriodicConv(
            num_spacial_dims,
            in_channels,
            out_channels,
            kernel_size,
            dilation,
            key=key,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            **boundary_kwargs,
        )
    else:
        raise ValueError(f"boundary_mode={boundary_mode} not implemented")
