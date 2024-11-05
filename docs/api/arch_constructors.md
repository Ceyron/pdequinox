# Architectural Constructors

There are two primary architectural constructors for Sequential and Hierarchical
Networks that allow for composability with the `PDEquinox` blocks.

## Sequential Constructor

![sequential_net](https://github.com/user-attachments/assets/866f9cb9-5d6f-462e-8621-26b74526ae68)

The [`pdequinox.Sequential`][] network constructor is defined by:

* a lifting block $\mathcal{L}$
* $N$ blocks $\left \{ \mathcal{B}_i \right\}_{i=1}^N$
* a projection block $\mathcal{P}$
* the hidden channels within the sequential processing
* the number of blocks $N$ (one can also supply a list of hidden channels if they shall be different between blocks)

## Hierarchical Constructor

![hierarchical_net](https://github.com/user-attachments/assets/b574c834-b8c8-476d-aabb-c121ba41d5c3)

The [`pdequinox.Hierarchical`][] network constructor is defined by:

* a lifting block $\mathcal{L}$
* The number of levels $D$ (i.e., the number of additional hierarchies). Setting $D = 0$ recovers the sequential processing.
* a list of $D$ blocks $\left \{ \mathcal{D}_i \right\}_{i=1}^D$ for
  downsampling, i.e. mapping downwards to the lower hierarchy (oftentimes this
  is that they halve the spatial axes while keeping the number of channels)
* a list of $D$ blocks $\left \{ \mathcal{B}_i^l \right\}_{i=1}^D$ for
  processing in the left arc (oftentimes this changes the number of channels,
  e.g. doubles it such that the combination of downsampling and left processing
  halves the spatial resolution and doubles the feature count)
* a list of $D$ blocks $\left \{ \mathcal{U}_i \right\}_{i=1}^D$ for upsamping,
  i.e., mapping upwards to the higher hierarchy (oftentimes this doubles the
  spatial resolution; at the same time it halves the feature count such that we
  can concatenate a skip connection)
* a list of $D$ blocks $\left \{ \mathcal{B}_i^r \right\}_{i=1}^D$ for
  processing in the right arc (oftentimes this changes the number of channels,
  e.g. halves it such that the combination of upsampling and right processing
  doubles the spatial resolution and halves the feature count)
* a projection block $\mathcal{P}$
* the hidden channels within the hierarchical processing (if just an integer is
  provided; this is assumed to be the number of hidden channels in the highest
  hierarchy.)

## Beyond Architectural Constructors

For completion, `pdequinox.arch` also provides a [`pdequinox.arch.ConvNet`][] which is a simple
feed-forward convolutional network. It also provides [`pdequinox.arch.MLP`][] which is a dense
networks which also requires pre-defining the number of resolution points. -->

## API

::: pdequinox.Sequential
    options:
        members:
            - __init__
            - __call__

---

::: pdequinox.Hierarchical
    options:
        members:
            - __init__
            - __call__
