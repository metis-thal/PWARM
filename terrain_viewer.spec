# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PWARM Terrain Viewer."""

a = Analysis(
    ['scripts/terrain_viewer.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('data/terrain', 'data/terrain'),
    ],
    hiddenimports=[
        'pyvista',
        'pyvista.plotting',
        'pyvista.utilities',
        'pyvista.themes',
        'vtkmodules',
        'vtkmodules.vtkRenderingOpenGL2',
        'vtkmodules.vtkInteractionStyle',
        'scipy.ndimage',
        'scipy.interpolate',
        'scipy.spatial',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PIL',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'IPython',
        'jupyter',
        'notebook',
        'torch',
        'tensorflow',
        'setuptools._vendor.jaraco',
        'setuptools._vendor.backports',
        'setuptools._vendor.wheel',
    ],
    noarchive=False,
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
    icon=None,
)
