"""Contract tests for the pwarm.physics package surface.

Pins the lazy interface re-export introduced when the physics<->interface
import cycle was removed (TEST_POLICY.md §7 defect 3): interface names
resolve through the module ``__getattr__`` regardless of import order,
unknown attributes fail loudly, and a fresh interpreter whose FIRST pwarm
import is ``pwarm.interface`` succeeds — the exact failure mode that broke
multiprocessing workers before the fix.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_lazy_interface_reexport_resolves() -> None:
    """Interface names exported through __getattr__ are the real objects."""
    import pwarm.physics
    from pwarm.interface.gui import GUI

    assert pwarm.physics.GUI is GUI


def test_unknown_attribute_raises_loudly() -> None:
    """The __getattr__ miss branch raises AttributeError (not silent None)."""
    import pwarm.physics

    with pytest.raises(AttributeError, match="has no attribute"):
        pwarm.physics.definitely_not_an_attribute  # noqa: B018 -- intentional miss


def test_interface_first_import_order_in_fresh_process() -> None:
    """A fresh process importing pwarm.interface FIRST must not hit the cycle."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", "import pwarm.interface, pwarm.physics"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
