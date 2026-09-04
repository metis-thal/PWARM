# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['scripts\\terrain_viewer.py'],
    pathex=[],
    binaries=[],
    datas=[('data/terrain', 'data/terrain')],
    hiddenimports=['pyvista', 'pyvista.plotting', 'pyvista.utilities', 'scipy.ndimage', 'scipy.interpolate', 'scipy.spatial', 'vtkmodules.vtkRenderingOpenGL2', 'vtkmodules.vtkInteractionStyle'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'tensorflow', 'IPython', 'jupyter', 'notebook'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PWARM_TerrainViewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
