"""Neural spatial super-resolution for a 2D shallow-water solver.

The numpy import below is load-order insurance, not decoration. Measured on this system:

    python -c "import torch, h5py"          -> ImportError, GLIBCXX_3.4.30 not found
    python -c "import numpy, torch, h5py"   -> fine
    python -c "import h5py, torch"          -> fine

`import torch` on its own pulls in the system `/lib64/libstdc++.so.6`, which lacks the
`GLIBCXX_3.4.30` that conda's `libicuuc.so.78` (an h5py dependency) needs, so a later
`import h5py` fails. Importing numpy or h5py first pins conda's newer library and the two
then coexist.

Every entry point here already imports numpy before torch, which is why nothing was broken;
this line makes that guarantee explicit rather than accidental, since reordering an import
block is an innocuous-looking change that would reintroduce the failure.

Scope of the guard, stated honestly: it only helps when `swe_sr` is imported before torch,
which holds for `python -m swe_sr.*` because Python executes this file first. External code
that imports torch before `swe_sr` is beyond its reach. The underlying problem is
environmental, and the durable fix is to make the dynamic loader prefer conda's libstdc++.
"""

import numpy as _numpy  # noqa: F401  # see the GLIBCXX note above; must precede any torch import
