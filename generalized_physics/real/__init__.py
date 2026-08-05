"""Real-data, real-Wan execution of the generalized_physics phases.

The CPU package (``generalized_physics.training.*``) trains against
``wan/mock_wan.py`` on procedurally generated episodes. This subpackage swaps
in (a) latents encoded from the ``f1_10h`` dataset by a real Wan VAE and
(b) a real frozen Wan DiT, **without editing a single phase module**.

Two properties of the existing code make that possible:

  * every phase consumes exactly one object -- the cache dict built by
    ``data/counterfactual_dataset.py::build_cache`` -- so producing that same
    dict from real data is sufficient (see ``cache_io.load_cache``);
  * every phase reaches the mock only through ``training/common.py`` symbols,
    and does so as attribute lookups at call time (``from . import common``
    ... ``common.fm_loss(...)``), so rebinding those module attributes is
    sufficient (see ``backend.install_real_backend``).
"""
