Terrain to STL Desktop Workflow
===============================

This portable Windows bundle runs offline after download and extraction.

What is included
- Native desktop GUI with integrated STL viewer
- Integrated terrain to STL conversion workflow
- Integrated STL viewer
- Private Python runtime and vendored dependencies

How to use it
1. Extract this ZIP to a normal folder on disk.
2. Open the extracted folder.
3. Double-click "Run Desktop GUI.cmd".

Desktop GUI workflow
- Use "Run Desktop GUI.cmd" to inspect a terrain file set, run conversion with progress, and open the STL in the integrated viewer.
- The terrain picker accepts one HDF with any associated VRT/TIF files, or one standalone DEM GeoTIFF.
- The GUI shows which selected files are associated with the primary terrain input and which extra files are not used for conversion.
- The integrated viewer includes vertical exaggeration controls alongside clip, profile, wireframe, and opacity tools.
- The Viewer tab can also open an existing STL directly from disk.
- The sample-step picker uses the fixed preset values 1, 2, 4, 8, 16, and 32.
- The GUI shows an STL size estimate for every preset and benchmarks the currently selected preset to estimate conversion time on that PC.

Notes
- Run "Run Desktop GUI.cmd" from the extracted bundle folder, not from the source repository.
- Keep the bundled files together. The launcher expects the bundled python.exe and app folder beside it.
- The app folder is part of the bundle internals and should stay next to python.exe.
- This v1 desktop package is unsigned. Windows or browser reputation systems may still warn on some systems.
- If Microsoft Edge or Defender flags the ZIP and you know it came from the official project release page, use the browser or Windows false-positive reporting flow for that download.
