"""Tests for the GPU text overlay (viz.text_overlay).

Requires a standalone moderngl context (available on Windows developer
machines); tests skip gracefully when no context can be created. Scope: the
font loader fallback chain, the panel state machine (dirty tracking,
rasterise triggers, release), and the actual uploaded pixel content.
"""

from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")
PILImage = pytest.importorskip("PIL.Image")
ImageFont = pytest.importorskip("PIL.ImageFont")

from pwarm.viz.text_overlay import (
    _ACCENT_BAR,
    _BG,
    _DEFAULT_COLOR,
    TextPanel,
    _load_font,
)


@pytest.fixture(scope="module")
def gl_context():
    """A standalone moderngl context, skipped when the driver refuses."""
    try:
        return moderngl.create_context(standalone=True)
    except Exception as error:  # noqa: BLE001 - any driver failure skips
        pytest.skip(f"no standalone GL context: {error}")


def test_load_font_returns_usable_font() -> None:
    """_load_font resolves a truetype face on normal systems."""
    font = _load_font(15)
    assert font is not None


def test_load_font_falls_back_when_truetype_missing(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """When every truetype name fails, load_default is used."""
    real_truetype = ImageFont.truetype

    def no_system_fonts(name, size, **kwargs):
        if isinstance(name, str):  # fail the consola/segoe/arial lookups...
            raise OSError(f"no font {name}")
        return real_truetype(name, size, **kwargs)  # ...but let load_default work

    monkeypatch.setattr(ImageFont, "truetype", no_system_fonts)
    font = _load_font(15)
    assert font is not None


def test_panel_set_lines_and_clear(gl_context) -> None:
    """set_lines marks the panel dirty; clear resets to an empty list."""
    panel = TextPanel(gl_context, width_px=200)
    assert panel._dirty is True
    panel._dirty = False
    panel.set_lines(["hello", ("world", "#ff0000")])
    assert panel._lines == ["hello", ("world", "#ff0000")]
    assert panel._dirty is True
    panel.clear()
    assert panel._lines == []
    assert panel._dirty is True


def test_panel_draw_before_any_content_is_noop(gl_context) -> None:
    """draw() without content and without dirt is a no-op."""
    panel = TextPanel(gl_context)
    panel._dirty = False
    panel.draw((320, 240))  # must not raise
    assert panel._texture is None


def test_panel_draw_zero_framebuffer_is_noop(gl_context) -> None:
    """A degenerate framebuffer leaves the dirty flag untouched."""
    panel = TextPanel(gl_context)
    panel.set_lines(["hello"])
    panel.draw((0, 0))
    assert panel._dirty is True
    assert panel._texture is None


def test_panel_draw_rasterises_to_texture(gl_context) -> None:
    """draw() uploads a width_px x framebuffer-height texture."""
    panel = TextPanel(gl_context, width_px=120)
    panel.set_lines(["PWARM", ("t=1.0 s", "#7dffa8")])
    panel.draw((320, 240))
    assert panel._texture is not None
    assert panel._texture.size == (120, 240)
    assert panel._dirty is False
    # pinned quirk: _img_height is a dead field (never updated), so every
    # draw() re-rasterizes; it stays 0 for the panel's whole life
    assert panel._img_height == 0


def test_panel_rerasterises_on_height_change_only(gl_context) -> None:
    """Same-size draws reuse the texture; a height change re-creates it.

    The (dead) _img_height field makes every draw re-rasterize, but the
    upload only allocates a new texture when the pixel size changed.
    """
    panel = TextPanel(gl_context, width_px=100)
    panel.set_lines(["hello"])
    panel.draw((320, 200))
    texture_first = panel._texture
    panel.draw((400, 200))  # width change only: same texture, new pixels
    assert panel._texture is texture_first
    panel.draw((400, 160))  # height change: new texture
    assert panel._texture is not texture_first
    assert panel._texture.size == (100, 160)


def test_panel_pixel_content(gl_context) -> None:
    """The uploaded texture carries the accent bar and the background colour."""
    panel = TextPanel(gl_context, width_px=120, bg_alpha=214)
    panel.set_lines(["x"])
    panel.draw((120, 64))
    data = np.frombuffer(panel._texture.read(), dtype=np.uint8)
    image = data.reshape(64, 120, 4)  # rows top-to-bottom
    # left edge is the accent bar
    assert tuple(image[0, 0]) == tuple(_ACCENT_BAR)
    # far corner is the panel background
    assert tuple(image[63, 119]) == (*_BG, 214)
    # the accent bar does not span the full width
    assert tuple(image[0, 10]) != tuple(_ACCENT_BAR)


def test_panel_release_invalidates_for_reuse(gl_context) -> None:
    """release() drops the texture; the released panel cannot draw again.

    Pinned as implemented: release() also releases the VBO/VAO/program but
    draw() only re-creates the texture, so reuse fails on the dead VBO.
    """
    panel = TextPanel(gl_context, width_px=100)
    panel.set_lines(["hello"])
    panel.draw((200, 100))
    assert panel._texture is not None
    panel.release()
    assert panel._texture is None
    panel._dirty = True
    with pytest.raises(AttributeError):
        panel.draw((200, 100))
    # a fresh panel keeps working after another panel was released
    fresh = TextPanel(gl_context, width_px=100)
    fresh.set_lines(["again"])
    fresh.draw((200, 100))
    assert fresh._texture is not None


def test_default_color_constant() -> None:
    """The default text colour is the documented hex value."""
    assert _DEFAULT_COLOR == "#d8e2f0"
