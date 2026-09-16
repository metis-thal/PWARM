"""Text overlay panel for the GL renderer — PIL-rasterized text on a texture quad.

The GL renderer is a moderngl core-profile pipeline with no text support; this
module adds a self-contained HUD panel: lines of text are rasterized with PIL
into an RGBA image, uploaded as a texture, and drawn as an alpha-blended quad
over the 3D frame (after the scene pass, before the buffer swap — see
``GLRenderer.post_draw_hook``).

The panel is a pure view component: it never reads or writes simulation state;
callers push plain strings (optionally with a hex color) via ``set_lines``.
"""

from __future__ import annotations

from collections.abc import Sequence

import moderngl
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

_VERTEX_SHADER = """
#version 330
in vec2 in_pos;   // NDC
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 out_color;
void main() {
    out_color = texture(u_tex, v_uv);
}
"""

_DEFAULT_COLOR = "#d8e2f0"
_ACCENT_BAR = (90, 200, 255, 255)
_BG = (13, 16, 28)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Prefer a real TrueType font; fall back to PIL's scalable default."""
    for name in ("consola.ttf", "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


class TextPanel:
    """A right-anchored HUD panel that draws text lines over the 3D view.

    Usage::

        panel = TextPanel(renderer.ctx, width_px=430)
        renderer.post_draw_hook = lambda r: panel.draw(r.framebuffer_size())
        panel.set_lines([("PWARM", "#7fd7ff"), "hello world"])
    """

    def __init__(self, ctx: moderngl.Context, width_px: int = 430,
                 font_size: int = 15, line_spacing: float = 1.55,
                 margin: int = 18, bg_alpha: int = 214) -> None:
        self._ctx = ctx
        self._width_px = int(width_px)
        self._font = _load_font(font_size)
        self._line_h = max(1, int(font_size * line_spacing))
        self._margin = margin
        self._bg_alpha = bg_alpha
        self._lines: list[str | tuple[str, str]] = []
        self._dirty = True

        self._prog = ctx.program(vertex_shader=_VERTEX_SHADER,
                                 fragment_shader=_FRAGMENT_SHADER)
        self._vbo = ctx.buffer(reserve=4 * 4 * 4)  # 4 verts x (2 pos + 2 uv) f32
        self._vao = ctx.vertex_array(self._prog,
                                     [(self._vbo, "2f 2f", "in_pos", "in_uv")])
        self._texture: moderngl.Texture | None = None
        self._img_height = 0

    # -- content ------------------------------------------------------------

    def set_lines(self, lines: Sequence[str | tuple[str, str]]) -> None:
        """Replace the panel content.

        Each item is a plain string (default color) or a ``(text, hex_color)``
        pair. Long content is clipped at the panel bottom.
        """
        self._lines = list(lines)
        self._dirty = True

    def clear(self) -> None:
        self.set_lines([])

    # -- rendering ----------------------------------------------------------

    def _rasterize(self, height_px: int) -> None:
        img = Image.new("RGBA", (self._width_px, height_px),
                        (*_BG, self._bg_alpha))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 3, height_px - 1], fill=_ACCENT_BAR)

        y = self._margin
        bottom = height_px - self._line_h
        for item in self._lines:
            if y > bottom:
                break
            if isinstance(item, tuple):
                text, color = item
                rgb = ImageColor.getrgb(color)
            else:
                text, rgb = item, ImageColor.getrgb(_DEFAULT_COLOR)
            draw.text((self._margin, y), text, font=self._font,
                      fill=(*rgb, 255))
            y += self._line_h

        self._upload(img)
        self._dirty = False

    def _upload(self, img: Image.Image) -> None:
        data = img.tobytes()
        if self._texture is None or self._texture.size != img.size:
            if self._texture is not None:
                self._texture.release()
            self._texture = self._ctx.texture(img.size, 4, data)
            self._texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._texture.repeat_x = False
            self._texture.repeat_y = False
        else:
            self._texture.write(data)

    def draw(self, framebuffer_size: tuple[int, int]) -> None:
        """Draw the panel over the last rendered frame.

        Call from ``GLRenderer.post_draw_hook`` — after the 3D pass, before
        the buffer swap. The panel is re-rasterized when its content or the
        framebuffer height changed.
        """
        if self._texture is None and not self._dirty:
            return
        fw, fh = int(framebuffer_size[0]), int(framebuffer_size[1])
        if fw <= 0 or fh <= 0:
            return
        if self._dirty or self._img_height != fh:
            self._rasterize(fh)
        if self._texture is None:
            return

        # Right-anchored quad, full height. PIL's first byte row is the image
        # TOP and maps to GL texel row 0 (v=0), so v=0 goes with NDC y=+1.
        x0 = 1.0 - 2.0 * self._width_px / fw
        verts = np.array([
            [x0,  1.0, 0.0, 0.0],   # top-left
            [1.0, 1.0, 1.0, 0.0],   # top-right
            [x0, -1.0, 0.0, 1.0],   # bottom-left
            [1.0, -1.0, 1.0, 1.0],  # bottom-right
        ], dtype=np.float32)
        self._vbo.write(verts.tobytes())

        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.enable(moderngl.BLEND)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._texture.use(0)
        self._prog["u_tex"].value = 0
        self._vao.render(moderngl.TRIANGLE_STRIP)

        # Restore the exact 3D pipeline state the renderer set up at init().
        self._ctx.disable(moderngl.BLEND)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.CULL_FACE)

    def release(self) -> None:
        for res in (self._texture, self._vao, self._vbo, self._prog):
            if res is not None:
                res.release()
        self._texture = None
