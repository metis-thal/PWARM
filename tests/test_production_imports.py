"""Production import contract: every shipped pwarm module must import.

Each import runs in a FRESH ``python -c`` subprocess (``PYTHONPATH=src`` only,
cwd = repo root): ``tests/conftest.py`` — including its legacy ``pwarm.kernel``
stub — never loads, the pytest process's ``sys.modules`` is never inherited,
and the child additionally asserts the stub and the tests tree never appear in
its ``sys.modules``.  This is the regression gate for DEFERRED_WORK.md,
"Production Import Integrity": the coverage gate measures logic executed under
stubs; this gate measures importability.

Environment contract: the gate assumes the dev/CI dependency profile (extras
``viz`` + ``dev``).  Outcome categories — deliberately distinct, do not
collapse them:

* importable          — must succeed;
* optional dependency — a ModuleNotFoundError rooted at a dist declared in
  ``OPTIONAL_DEPS`` is a documented skip; any other error fails;
* legacy exemption    — must fail with the expected legacy error (a missing
  ``pwarm.kernel*`` module, or ``ray`` for ``ray_parallel``).  If an exempt
  module ever imports, the test FAILS with "stale exemption", so
  ``LEGACY_EXEMPTIONS`` can only shrink;
* anything else       — fails, with the child's stderr attached.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_DIR = _REPO_ROOT / "src" / "pwarm"

#: Modules that still carry module-level imports of the deleted
#: ``pwarm.kernel``.  Every entry MUST keep its registered debt item in
#: DEFERRED_WORK.md, "Production Import Integrity", including reason and
#: disposition plan.  Entries may only be REMOVED (once the module is fixed
#: or retired) — never added to absorb a new failure.
LEGACY_EXEMPTIONS: dict[str, str] = {
    "pwarm.viz.viewer":
        "renders the deleted kernel World; retire-or-migrate onto SceneSnapshot (P2)",
    "pwarm.viz.viewer3d":
        "renders the deleted kernel World3D; retire-or-migrate onto SceneSnapshot (P2)",
    "pwarm.ai.experiment":
        "builds deleted kernel worlds; migrate onto WorldEngine or retire (P2)",
    "pwarm.parallel.ray_parallel":
        "module-level ray import + deleted kernel World3D; engine migration (P2)",
}

#: Modules whose import legitimately depends on an optional distribution:
#: module -> (acceptable missing dist roots, reason).  A ModuleNotFoundError
#: rooted at one of these dists is a documented skip; anything else fails.
OPTIONAL_DEPS: dict[str, tuple[tuple[str, ...], str]] = {
    "pwarm.ai.pinn": (("torch",), "optional extra 'ai' (torch)"),
    "pwarm.viz.text_overlay": (
        ("PIL",),
        (
            "Pillow is required by the GPU text overlay but undeclared in any "
            "extra (packaging gap — see DEFERRED_WORK.md)"
        ),
    ),
    "pwarm.viz.gl_renderer": (
        ("glfw", "moderngl"),
        "optional extra 'viz' (glfw/moderngl)",
    ),
}

# Child interpreter protocol: exit 0 = imported (after asserting the legacy
# stub and the tests tree never entered sys.modules); exit 3 =
# ModuleNotFoundError (missing module printed as "MISSING:<name>"); anything
# else = unexpected crash, stderr carries the traceback.
_CHILD = (
    "import sys\n"
    "try:\n"
    "    __import__(sys.argv[1])\n"
    "except ModuleNotFoundError as exc:\n"
    "    print('MISSING:' + str(exc.name))\n"
    "    raise SystemExit(3)\n"
    "assert 'pwarm.kernel' not in sys.modules, 'legacy kernel in sys.modules'\n"
    "assert not any(m == 'tests' or m.startswith('tests.') for m in sys.modules), "
    "'tests package in sys.modules'\n"
    "print('OK')\n"
)


def _iter_module_names() -> list[str]:
    """Enumerate shipped modules from the filesystem (no imports performed)."""
    names: list[str] = []
    for dirpath, dirnames, filenames in os.walk(_PKG_DIR):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            parts = list(
                (Path(dirpath) / fname)
                .relative_to(_PKG_DIR)
                .with_suffix("")
                .parts
            )
            if parts[-1] == "__init__":
                parts = parts[:-1]
            if parts and parts[-1] == "__main__":
                continue  # never import a __main__ file
            names.append(".".join(["pwarm", *parts]) if parts else "pwarm")
    return sorted(names)


def _probe(module: str) -> tuple[int, str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    # Keep the child a plain interpreter, not a pytest-cov subprocess hook.
    for key in [k for k in env if k.startswith(("COV_CORE", "COVERAGE_"))]:
        env.pop(key, None)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, module],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        timeout=600,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr


def _is_legacy_failure(module: str, missing: str) -> bool:
    if missing.startswith("pwarm.kernel"):
        return True
    return module == "pwarm.parallel.ray_parallel" and missing == "ray"


@pytest.mark.parametrize("module", _iter_module_names())
def test_module_imports_in_clean_interpreter(module: str) -> None:
    """The module must import in a fresh stub-free interpreter."""
    code, out, err = _probe(module)

    if module in LEGACY_EXEMPTIONS:
        assert code == 3, (
            f"stale exemption: {module} now imports — remove it from "
            f"LEGACY_EXEMPTIONS (registered reason was: {LEGACY_EXEMPTIONS[module]})"
        )
        missing = out.removeprefix("MISSING:") or "<unnamed>"
        assert _is_legacy_failure(module, missing), (
            f"exempted {module} fails on an unexpected module '{missing}' — "
            "this is a new import failure, not the registered legacy debt\n"
            f"{err[-2000:]}"
        )
        pytest.skip(f"legacy exemption (missing {missing}): {LEGACY_EXEMPTIONS[module]}")

    if code == 0:
        return
    if code == 3:
        missing = out.removeprefix("MISSING:") or "<unnamed>"
        option = OPTIONAL_DEPS.get(module)
        if option and missing in option[0]:
            pytest.skip(f"optional dependency '{missing}' absent — {option[1]}")
        pytest.fail(
            f"{module} failed to import (missing: {missing}). "
            "If this is registered debt, add it to LEGACY_EXEMPTIONS with a "
            "DEFERRED_WORK.md entry; otherwise fix the import.\n" + err[-2000:]
        )
    pytest.fail(f"{module} import crashed unexpectedly (exit {code})\n" + err[-2000:])


def test_legacy_exemptions_reference_existing_modules() -> None:
    """Exemption keys must be real shipped modules (catches renames/typos)."""
    enumerated = set(_iter_module_names())
    stale = set(LEGACY_EXEMPTIONS) - enumerated
    assert not stale, f"exemption keys no longer exist — remove them: {sorted(stale)}"


def test_optional_dep_map_references_existing_modules() -> None:
    """Optional-dep keys must be real shipped modules (catches renames/typos)."""
    enumerated = set(_iter_module_names())
    stale = set(OPTIONAL_DEPS) - enumerated
    assert not stale, f"optional-dep keys no longer exist — remove them: {sorted(stale)}"
