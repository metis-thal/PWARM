"""Material experiments — the scientist's apparatus catalogue.

Each module defines one experiment type: what it measures (claim), the
candidate designs (grid), the expected measurement resolution, and how to
derive the claim from an observation record. ``REGISTRY`` maps experiment
kinds to their modules; the designer scores designs by expected information
gain, the laboratory executes them, and the modules derive the physics.
"""

from . import buoyancy_test, drop_test, immersion_test, slide_test

REGISTRY = {
    drop_test.KIND: drop_test,
    slide_test.KIND: slide_test,
    buoyancy_test.KIND: buoyancy_test,
    immersion_test.KIND: immersion_test,
}

__all__ = ["REGISTRY", "buoyancy_test", "drop_test", "immersion_test",
           "slide_test"]
