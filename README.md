# Terrain to STL

This project contains:

- an interactive Python converter that turns either an HEC-RAS terrain HDF or a DEM GeoTIFF into a watertight STL shell
- a native Python desktop GUI with an integrated STL viewer
- a standalone desktop STL mesh viewer so you can rotate the mesh, inspect the shell interior, look up at the terrain underside, and verify the flat top surface

The converter:

- asks for an input terrain source path: `.hdf`, `.tif`, or `.tiff`
- if the source is an HDF, finds the matching sibling `.vrt`, `.tif`, or `.tiff`
- asks for a flat top elevation
- asks for a raster sample step
- writes a binary `.stl` file in the same folder as the selected terrain source

The script entrypoint is:

```powershell
python terrain_to_stl.py
```

The converter also accepts optional command-line flags so the desktop bundle and CI can run it non-interactively:

```powershell
python terrain_to_stl.py --input example\example.hdf --top-elevation 100 --sample-step 8
```

The mesh viewer entrypoint is:

```powershell
python mesh_viewer.py
```

The native desktop GUI entrypoint is:

```powershell
python desktop_gui.py
```

The structured console backend entrypoint is:

```powershell
python desktop_console.py
```

## Static Web App

This repo also now contains a separate browser package in:

- `webapp`

The web app is designed to be deployable as a static GitHub Pages site.

It includes:

- a browser converter page for terrain `.hdf` + `.vrt` / `.tif` / `.tiff` uploads, or one standalone DEM `.tif` / `.tiff`
- a browser STL viewer page with preview rendering, clipping, profile lines, bottom-only mode, full-shell mode, exact profile measurement, wireframe, and opacity controls

The browser converter runs in a Web Worker with Pyodide and reuses the same Python conversion core that powers `terrain_to_stl.py`.

The browser viewer is a separate client-side implementation built for static hosting and does not replace the desktop `mesh_viewer.py`.

### Web App Requirements

- Node.js 20 or newer
- npm

Important:

- the web app uses `Node.js` and `npm`
- it does not use the Python `.venv`
- the Python setup in this repo is for the desktop tools such as `desktop_gui.py`, `desktop_console.py`, `terrain_to_stl.py`, and `mesh_viewer.py`

### Run the Web App in VS Code

1. Open this repo folder in Visual Studio Code.
2. Open a terminal with `Terminal > New Terminal`.
3. In the VS Code terminal, run:

```powershell
cd webapp
```

4. Install the web app packages the first time:

```powershell
npm install
```

5. Start the development server:

```powershell
npm run dev
```

6. Open the URL printed by Vite in your browser, usually:

```text
http://localhost:5173/
```

Useful routes:

- `#/convert` = browser terrain conversion
- `#/viewer` = browser STL viewer

Examples:

```text
http://localhost:5173/#/convert
http://localhost:5173/#/viewer
```

While `npm run dev` is running, Vite hot reload will refresh the page automatically when you save changes in VS Code.

To stop the dev server, press `Ctrl+C` in the VS Code terminal.

### Quick Commands

If you just want the short version, run:

```powershell
cd webapp
npm install
npm run dev
```

Vite will print a local development URL, typically:

```text
http://localhost:5173/
```

Routes:

- `#/convert` = browser terrain conversion
- `#/viewer` = browser STL viewer

### Building the Web App

To create the production build:

```powershell
cd webapp
npm run build
```

To run the webapp tests:

```powershell
cd webapp
npm test
```

The static output is written to:

- `webapp\dist`

To preview the production build locally:

```powershell
cd webapp
npm run preview
```

### GitHub Pages Deployment

The Pages workflow is:

- `.github\workflows\pages.yml`

It installs the `webapp` dependencies, runs the production build, and publishes `webapp\dist` to GitHub Pages.

### Web App Notes

- the browser converter is upload-based, so in HDF mode the matching `.vrt`, `.tif`, or `.tiff` must be uploaded in the same browser session
- the browser converter supports standalone DEM GeoTIFF uploads and terrain-style one-band VRT files with `SimpleSource` / `ComplexSource`
- if a `.vrt` references additional raster files, those files must also be uploaded in the same session
- browser VRT source paths are matched to uploaded files by basename, so absolute Windows paths inside the VRT can still resolve if the referenced TIFF basename is uploaded
- unsupported browser VRT features such as warped or derived VRT bands fail with a clear error instead of partial conversion
- browser inspection now reads raster headers first, so large HDF terrains can finish inspection without decoding the full raster into browser memory
- stitched browser sample steps now use local refinement around stitch components and expose preset stitch-aware steps `1`, `2`, `4`, `8`, `16`, and `32`
- browser conversion keeps `sample step 1` on the full-resolution decode path, but higher browser sample steps now read only the exact sampled raster rows plus any stitch refinement windows they need
- that sparse browser decode does not add any extra approximation beyond the selected raster sample step; the sampled elevations and stitch-aware refinement geometry are still exact for that chosen step
- browser conversion is blocked before raster decode if either limit is exceeded:
  - estimated peak working set above `2.5 GiB`
  - estimated STL size above `256 MiB`
- browser STL viewer uploads above `256 MiB` are blocked before the file is read into browser memory
- files near those limits can still slow down or crash the browser, even if they are not blocked
- the web app points large or near-limit browser runs to the portable Windows desktop workflow download published on GitHub Releases
- the browser viewer renders a lighter preview mesh for interaction and keeps the exact STL in a worker for on-demand profile measurement
- the browser viewer defaults to surface opacity `0.8`, just like the desktop viewer
- the desktop Python scripts remain the reference workflow for large local runs
- a separate GitHub Actions workflow builds a portable Windows desktop ZIP and publishes it as a GitHub Release asset
- the desktop bundle launchers are generated into the extracted bundle root during the build; the runnable local test target is the built bundle under `dist`, not any source asset folder

### Desktop Bundle Development

To build the portable Windows bundle locally:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_desktop_bundle.ps1 -BundleVersion local-test -OutputRoot dist\desktop-smoke
```

After the build finishes, run the desktop workflow from the extracted bundle folder:

```text
dist\desktop-smoke\terrain-to-stl-desktop-windows-x64\
```

Run these files from that extracted folder:

- `Run Desktop GUI.cmd`

The `app\` folder beside it is bundle internals and should stay together with `python.exe`.
If `Run Desktop GUI.cmd` does not show the viewer loading overlay while opening a large STL, rebuild the bundle from source because the copied desktop app under `dist` is stale.
The desktop GUI accepts a terrain file set such as `HDF + VRT/TIF` selections, shows which files are associated with the primary terrain input, includes the same vertical exaggeration control used by the browser viewer, uses fixed sample-step presets `1`, `2`, `4`, `8`, `16`, and `32`, shows per-step STL size estimates plus a calibrated time estimate for the currently selected step, and includes the integrated STL viewer in the same app.

## Requirements

- Windows
- Python 3.11 or newer
- Visual Studio Code
- VS Code Python extension

Python packages required by the script:

- `numpy`
- `h5py`
- `rasterio`
- `pyvista`
- `matplotlib`
- `PySide6`
- `pyvistaqt`

These packages are pinned in:

- `requirements.txt`

## Recommended Setup: Local Virtual Environment

Do not rely on global Python packages. Create a local virtual environment for this project.

### 1. Open the project in VS Code

Open this folder in Visual Studio Code.

### 2. Open a terminal in VS Code

Use:

```text
Terminal > New Terminal
```

### 3. Create a virtual environment

If your default `python` command points to the Python version you want to use:

```powershell
python -m venv .venv
```

If you want to use your installed Python 3.11 directly:

```powershell
C:/Users/gtmen/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv
```

### 4. Activate the virtual environment in PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, your prompt should include `(.venv)`.

### 5. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 6. Install the required packages

```powershell
python -m pip install -r requirements.txt
```

## Select the Virtual Environment in VS Code

After creating `.venv`, tell VS Code to use it.

1. Open the Command Palette.
2. Run:

```text
Python: Select Interpreter
```

3. Choose the interpreter inside this project's `.venv`.
4. Open a new terminal in VS Code so the environment auto-activates.

If VS Code selected the correct interpreter, this command should point to `.venv`:

```powershell
python -c "import sys; print(sys.executable)"
```

## Running the Script

Once the virtual environment is active, run:

```powershell
python terrain_to_stl.py
```

Optional non-interactive arguments:

```powershell
python terrain_to_stl.py --input input\Terrain_Example.hdf --top-elevation 100 --sample-step 8 --output input\Terrain_Example.stl
```

You can also use the VS Code Python extension Run button, but the integrated terminal is the simplest option because the script is interactive.

## Running the Mesh Viewer

Once the virtual environment is active, you can open an STL mesh in the viewer.

Run with an explicit STL path:

```powershell
python mesh_viewer.py input\Terrain_Example.stl
```

Or run without an argument and enter the STL path when prompted:

```powershell
python mesh_viewer.py
```

The viewer opens as a desktop window, not a browser window.

## Mesh Viewer Features

The viewer is designed for shell inspection.

It lets you:

- rotate, pan, and zoom the mesh
- open large STL files in a lighter preview mode for smoother interaction
- switch to top view
- switch to bottom view so you can inspect the terrain underside
- turn the clip plane on only when you need to cut the shell open
- enable a wireframe overlay
- color the shell by elevation with a terrain-style legend
- turn the profile line on only when you need a section
- start with a default surface opacity of `0.8` so the shell is easier to inspect
- use `Bottom Only` profiles by default to show just the terrain underside elevation
- switch to `Full Shell` to show clean top and bottom profile envelopes together
- run an exact full-resolution profile measurement on demand
- restore the viewer back to a known default state with one button

### Mesh Viewer Controls

Visible controls appear on the left side of the viewer window.

Buttons and toggles:

- `Top`, `Bottom`, `Isometric`, `Reset Camera`
- `Clip Tool`, `Reset Clip`
- `Profile Tool`, `Reset Profile`, `Measure Exact`
- `Full Shell`
- `Wireframe`, `Restore All`
- `Surface opacity` slider

Defaults:

- surface opacity starts at `0.8`
- profile mode starts as `Bottom Only`

Recommended recovery action:

- if the shell seems to disappear or you lose track of the state, click `Restore All`
- `Restore All` resets the viewer to `Bottom Only` mode and restores surface opacity to `0.8`

Keyboard shortcuts are still available as secondary controls:

- `T` = top view
- `B` = bottom view
- `I` = isometric view
- `R` = reset camera
- `X` = reset clipping and show the full STL again
- `C` = toggle clip tool
- `P` = toggle profile tool
- `F` = toggle full shell
- `M` = measure exact profile
- `A` = restore all

### Viewer Workflow

To inspect the bottom terrain surface:

1. Open the STL in `mesh_viewer.py`.
2. Press `B` to look from below.
3. Turn on `Clip Tool` and move the plane to cut away part of the shell.
4. Turn on `Wireframe` or lower the surface opacity if needed.
5. Turn on `Profile Tool` and move the line to create a preview section.
6. Leave `Full Shell` off if you want only the terrain underside along that line.
7. Turn `Full Shell` on if you want the chart and 3D profile to show both the shell bottom and shell top envelopes.
8. Click `Measure Exact` only when you want a full-resolution profile for the current line, clip, and profile mode.

The viewer colors the mesh by elevation, so the flat top should appear as one constant elevation band while the terrain bottom varies.

### Preview vs Exact

For large STL files, the viewer now opens with a lighter preview mesh so clip and profile interactions stay responsive.

This means:

- the desktop GUI shows a loading overlay while it reads the STL and builds the preview
- large binary STL files build that preview from a streamed reader instead of fully materializing the mesh first
- normal rotation, clip, and profile updates use the preview mesh
- `Measure Exact` loads the full STL only when needed, and very large files warn before the full-resolution load starts
- `Full Shell` keeps the profile clean by plotting top and bottom envelopes instead of the noisy raw shell loop
- the rendered shell stays in preview mode even after an exact profile, so interaction remains fast

## Interactive Flow

When the script starts, it asks for three values.

### 1. Input terrain source path

Example:

```text
Input terrain source path (.hdf, .tif, .tiff):
```

Enter a full path or a relative path from the project folder.

You can enter:

- an HEC-RAS terrain `.hdf`
- a DEM `.tif`
- a DEM `.tiff`

Example:

```text
input\Terrain_Example.hdf
```

### 2. Top elevation for the STL shell

The script reads the terrain and shows the maximum elevation before asking for the top elevation.

Example:

```text
Top elevation for the STL shell (terrain maximum elevation is 91.437500):
```

Rules:

- the top elevation must be greater than or equal to the terrain maximum elevation
- if it is lower, the script stops and tells you to reduce the terrain extent and try again

### 3. Raster sample step

Example:

```text
Raster sample step (1, 2, 4, 8, ...):
```

Guidance:

- `1` = full raster resolution
- larger values reduce detail and reduce STL size
- the script always includes the last raster row and column
- stitched terrains support preset stitch-aware steps `1`, `2`, `4`, `8`, `16`, and `32`
- non-stitched terrains still accept any integer sample step greater than or equal to `1`

## Example Session

```text
(.venv) PS C:\Users\gtmen\Desktop\ProjectsCodex\terrain-to-stl> python terrain_to_stl.py
Input terrain source path (.hdf, .tif, .tiff): input\Terrain_Example.hdf
Loading raster surface from C:\Users\gtmen\Desktop\ProjectsCodex\terrain-to-stl\input\Terrain_Example.vrt...
Reading terrain metadata from C:\Users\gtmen\Desktop\ProjectsCodex\terrain-to-stl\input\Terrain_Example.hdf...
Top elevation for the STL shell (terrain maximum elevation is 91.437500): 100
Raster sample step (1, 2, 4, 8, ...): 8
No populated stitch TIN was found. The raster will be converted by itself.
Writing STL to C:\Users\gtmen\Desktop\ProjectsCodex\terrain-to-stl\input\Terrain_Example.stl...
Writing bottom and top surfaces...
Finished writing C:\Users\gtmen\Desktop\ProjectsCodex\terrain-to-stl\input\Terrain_Example.stl
```

## Output

The STL is written beside the input terrain source.

Examples:

- `Terrain_Example.hdf` -> `Terrain_Example.stl`
- `Terrain_Example.tif` -> `Terrain_Example.stl`
- if that file already exists -> `Terrain_Example_1.stl`
- next run -> `Terrain_Example_2.stl`

The generated STL:

- is binary STL
- is watertight
- uses the terrain as the bottom surface
- uses your entered elevation as the flat top surface

## File Requirements

If you use an HDF input, the HDF file must have a matching sibling raster file with the same base name:

- `.vrt`, or
- `.tif`, or
- `.tiff`

You can also convert a standalone DEM directly:

- `.tif`
- `.tiff`

Examples:

- `input\Terrain_Example.hdf` works because `input\Terrain_Example.vrt` exists
- `inputs\terrain.hdf` works because `inputs\terrain.vrt` exists
- `input\Terrain_Example.tif` works by itself as a standalone DEM input

The converter does not build the full terrain surface from the `.hdf` alone.

## Notes About Large Terrain Files

Large terrain rasters can create very large STL files.

Large STL files can still take time to open, but the desktop viewer now shows a loading overlay while it works.

For large binary STL files, the viewer builds the interactive preview from a streamed reader instead of waiting for a full-resolution VTK load first.

After that preview load, the viewer uses the lighter mesh for interaction and only loads the exact mesh when you click `Measure Exact`.

Use these rules of thumb:

- use sample step `1` only when you need native detail
- use larger sample steps to reduce output size for any terrain, but stitched terrains are limited to the preset values `1`, `2`, `4`, `8`, `16`, and `32`
- if top elevation is lower than the terrain maximum, crop the terrain extent externally and try again
- use `Restore All` if the current clip/profile state becomes confusing
- use `Bottom Only` for underside inspection and `Full Shell` when you want to compare bottom and top together

## Troubleshooting

### `ModuleNotFoundError: No module named 'h5py'`

This means the current Python interpreter does not have the required packages installed.

Use the project virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then verify the package is installed in the active environment:

```powershell
python -m pip show h5py
```

The same approach also applies if VS Code reports `ModuleNotFoundError` for `rasterio` or `pyvista`.

### VS Code is using the wrong Python interpreter

Run:

```text
Python: Select Interpreter
```

Then choose the `.venv` interpreter for this workspace.

You can confirm with:

```powershell
python -c "import sys; print(sys.executable)"
```

The interpreter path should point into this project's `.venv`. If you create the environment with Python 3.11, VS Code should also be set to that same `.venv` interpreter.

### "Could not find a sibling raster"

In HDF mode, make sure the HDF has a matching `.vrt`, `.tif`, or `.tiff` beside it with the same base name.

### "The top elevation must be greater than or equal to the max terrain elevation"

Choose a larger top elevation, or reduce the terrain extent to the area you want to export.

### "Supported stitch-aware raster sample steps are 1, 2, 4, 8, 16, 32"

The stitched terrain path now supports these preset sample steps:

```text
Raster sample step = 1, 2, 4, 8, 16, or 32
```

### The viewer opens but interaction feels slow

Very large STL files still take time to open, and the first exact profile on a large STL can take several seconds because it loads the full-resolution mesh after a warning prompt.

Try one or more of these:

- create a smaller STL with a larger raster sample step
- crop the terrain extent before converting
- use the preview section first and click `Measure Exact` only when needed
- use `Restore All` instead of repeatedly dragging widgets back into place

### The viewer looks blank or the shell seems to disappear

This usually means the clip plane removed almost everything you were looking at, or you have several tools active at once.

Use one of these recovery steps:

- click `Restore All` to return to the default preview state
- click `Reset Clip` to show the full shell again
- click `Bottom` or `Isometric` if you are stuck in a flat top-down view

## Current Repository Layout

Important files in this folder:

- `terrain_to_stl.py` - interactive converter
- `mesh_viewer.py` - interactive desktop STL viewer
- `requirements.txt` - pinned Python dependencies
- `input\Terrain_Example.hdf` - example terrain HDF
- `input\Terrain_Example.tif` - example DEM GeoTIFF
- `input\Terrain_Example.stl` - example generated STL
- `input\Terrain_Example.vrt` and `input\Terrain_Example.tif` - example raster sidecars for the HDF workflow
- `inputs\terrain.hdf` - larger terrain example with populated stitch TIN data
