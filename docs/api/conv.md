# Low-Level Convolution Routines

They wrap the `equinox.conv` module to provide an interface based on `"SAME"`
convolutions with variable boundary modes or implement spectral convolutions.

::: pdequinox.conv.PhysicsConv
    options:
        members:
            - __init__
            - __call__

---

::: pdequinox.conv.PhysicsConvTranspose
    options:
        members:
            - __init__
            - __call__

---

::: pdequinox.conv.SpectralConv
    options:
        members:
            - __init__
            - __call__

---

::: pdequinox.conv.PointwiseLinearConv
    options:
        members:
            - __init__
            - __call__