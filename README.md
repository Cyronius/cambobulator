# Cambobulator

*Be in the meeting. Not in the meeting.*

Cambobulator is a virtual webcam for Windows, for people who heard "cameras on, everyone!" and took it
personally. It reads your real webcam, quietly removes you from it, and hands the result to Teams, Discord, Zoom
or Meet as a perfectly ordinary camera called **OBS Virtual Camera**. Your colleagues see your office. Your
office looks the same as ever, just without you in it.

```
real webcam ──► capture thread ──► Fade into Background ──► virtual camera  ("OBS Virtual Camera")
                                                       └──► browser preview (raw | output)
```

It was built for a coworker who would rather not be perceived. They now blend into their own background, like
a chameleon with a Microsoft 365 license.

* **Fade.** It finds you (MediaPipe's selfie segmentation model, bundled, nothing to download), softens your
  edges, and blends you toward a clean shot of your empty room: `out = lerp(frame, background, fade * mask)`.
  At 0 you are you. At 1 you are drywall.
* **Background plate.** Click *Capture background*, then leave. You have 5 seconds. No capture yet? It pieces
  the room together from every pixel you haven't been sitting in front of, and makes a blurry, educated guess
  about the rest. The plate keeps adapting slowly, so the sun going down won't blow your cover.
* **Auto-fade.** Sit still and you dissolve, along with your person-look settings. Move or talk and you snap
  back instantly, so nobody hears a question coming from an empty chair.
* **Predator shimmer.** A refraction ripple through your silhouette at partial fade, for when you want to be
  invisible but also want credit for it.
* **Person look.** Brightness, contrast, saturation and pixelate, applied to you and never the room.
  Pixelate averages only your pixels in each block, so you look like a witness-protection interview instead
  of a smudge on the wall.

---

## Install

Windows 10/11, x64. Nothing else to install: Cambobulator bundles its own virtual-camera driver and
person-segmentation model.

1. Download `cambobulator.exe` from the [latest release](https://github.com/Cyronius/cambobulator/releases/latest)
   (or build it yourself, see [Development](#development)) and double-click it. The control panel opens in
   your browser. Windows SmartScreen may warn about an
   unrecognized app the first time; click *More info → Run anyway*.
2. The first time the virtual camera starts, Windows asks for administrator permission **once**. That
   installs the driver: a copy of OBS Studio's virtual-camera module, in `C:\Program Files\Cambobulator\`.
   Video apps list it as **OBS Virtual Camera**.

That's it. You can also install the driver ahead of time with `cambobulator driver install`.

**From source** (any Python 3.10+): `py -m venv .venv`, `.venv\Scripts\activate`, `pip install .`, then run
`cambobulator`.

**Already have OBS Studio?** Its driver is the same one, so Cambobulator just uses it and skips the install.
Don't run OBS's own virtual camera at the same time as Cambobulator.

**Driver commands:**

```
cambobulator driver status      # which driver is registered, and whose
cambobulator driver install     # install or update Cambobulator's copy (one admin prompt)
cambobulator driver uninstall   # remove Cambobulator's copy (leaves an OBS install alone)
```

**macOS and Linux** are not supported: there is no driver to bundle. The camera code paths still exist, and
pyvirtualcam can use OBS 30+'s camera extension on macOS or a v4l2loopback device on Linux if you set one up
yourself, but none of that is tested.

---

## Use it

```
cambobulator                 # same as "cambobulator ui": control panel at http://127.0.0.1:8765
cambobulator run             # headless, using the saved settings
cambobulator cameras         # list cameras (index: name)
cambobulator filters         # list the effect's parameters
cambobulator driver status   # is the virtual camera driver installed?
```

**Control panel.** Pick the source camera and drag the effect's sliders (grouped as Fade, Person, Mask and
Background); the *On* box bypasses the effect. You see the raw camera and the processed output side by side.
Capture the background, and toggle the virtual camera with the button in the top bar. Every change is saved automatically to the config file
(path shown at the bottom of the panel), so `cambobulator run` uses exactly what you set up. Captured
background plates are saved next to it as PNG files.

**Headless.** `cambobulator run` starts the virtual camera with the saved settings and prints a one-line
summary. Useful options:

| option | what it does |
|---|---|
| `--source 1` / `--source synthetic` / `--source clip.mp4` | override the source (a camera index, a test pattern, or a looping video file) |
| `--width 1920 --height 1080 --fps 30` | override resolution / frame rate (default 1280×720 @ 30) |
| `--capture-background` | start with a background-capture countdown (step out of frame!) |
| `--no-vcam --record out.mp4 --max-frames 300` | don't touch the virtual camera; record to a file instead (handy for testing) |
| `--segmenter none` | skip person segmentation (fade uses the background-difference fallback) |
| `--segmenter synthetic` | exact mask for the synthetic test pattern (demos) |
| `--config path.json` | use a different settings file |

### Pick the camera in your video app

Until Cambobulator starts sending, the camera shows a "Cambobulator is not running" card. Pick it like this:

* **Teams (new):** Settings → *Devices* (or the camera drop-down before joining) → Camera → **OBS Virtual Camera**.
  In a meeting: click the arrow next to the camera button → OBS Virtual Camera.
* **Discord:** User Settings → *Voice & Video* → Camera → **OBS Virtual Camera**.
* **Zoom:** Settings → *Video* → Camera → **OBS Virtual Camera**.
* **Browser apps (Meet, Teams web):** the camera picker in the site's settings; you may need to reload the tab
  after starting Cambobulator.

Tips:

* Pick your **real** webcam as Cambobulator's source, and the **virtual** camera in Teams. On Windows,
  usually only one app can use a physical webcam at a time. If Teams grabs it first, Cambobulator will report
  that the camera is busy.
* Teams/Zoom mirror your *self-view* only; other people see the image un-mirrored.
* If **OBS Virtual Camera** doesn't appear in Teams, quit Teams from its tray icon and start it again. A Teams
  that was already running when the driver was installed never notices the new camera.
* Turn off Teams' own background effects; they fight with the fade.
* If Teams shows the "Cambobulator is not running" card, the virtual camera is off. Look at the top bar of
  the control panel.

### Getting a good fade

1. Sit where you'll sit in the meeting, then click **Capture background** and get out of the shot within 5 s.
2. Raise **Fade** slowly. If you see an outline, raise **Grow mask**.
3. If parts of you flicker in and out, lower **Mask threshold**. If the room near you smears, raise it.
4. **Background adapt rate** follows slow lighting changes. Set it to 0 to freeze the plate.
5. If the camera moves, or the lighting changes a lot, capture again (or *Forget background*).

---

## Configuration file

JSON, stored at `%APPDATA%\cambobulator\config.json`:

```json
{
  "source": "0",
  "width": 1280, "height": 720, "fps": 30,
  "virtual_camera": true,
  "vcam_backend": null,
  "vcam_device": null,
  "segmenter": "auto",
  "segmenter_model": null,
  "ui_host": "127.0.0.1", "ui_port": 8765,
  "filters": [
    {"type": "fade_into_background", "enabled": true, "id": "3f2a91c0",
     "params": {"fade": 0.85, "auto_fade": true, "shimmer": 0.2}}
  ]
}
```

`vcam_backend` / `vcam_device` are passed to [pyvirtualcam](https://github.com/letmaik/pyvirtualcam). Leave them
`null` to use the bundled driver; any other backend (for example `"unitycapture"`) is yours to install.
`segmenter_model` can point at a different ONNX model with the same input and output as the bundled one
(see `src/cambobulator/models/MODEL.txt`); `null` uses the bundled model.

---

## Change the effect

The effect is `FadeIntoBackground` in `src/cambobulator/filters/fade.py`. Frames are `numpy.uint8` arrays in
OpenCV's **BGR** order, shape `(H, W, 3)`; `process(frame, ctx)` returns a frame of the same shape and type.

* The control panel is built from `PARAMS` and `ACTIONS`. Add a `Param` (with a `group`, which becomes a
  heading) and the slider appears; an `Action("name", "Label")` button calls `self.action_name()`.
* Read parameters with `self["name"]`. Values are already converted to the right type and clamped to
  [min, max]. Unknown or removed params in an old config are ignored with a warning.
* Use `ctx.timestamp` (seconds) for anything time-based, not `time.time()`. That keeps it testable.
* An exception in `process` doesn't kill the stream: the frame passes through and the error shows in the UI.

The config keeps a `filters` list for compatibility, but only its `fade_into_background` entry is used.
Configs from before the single-effect UI may list Mirror, Adjust or Pixelate entries; those are skipped.

---

## Development

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The tests need no camera and no virtual-camera driver (they never install it). They use synthetic frames, a
fake segmenter, a synthetic source, and a stubbed `pyvirtualcam`. To see the whole pipeline without hardware:

```powershell
cambobulator run --source synthetic --segmenter synthetic --no-vcam --record demo.mp4 --max-frames 150
cambobulator ui  --source synthetic --segmenter synthetic --no-vcam --config demo.json
```

`--segmenter synthetic` finds the test pattern's "person" by its exact colour, since the model can't see a
cartoon. Point it at a real video of a person to see real segmentation: `--source me.mp4`. Use a separate
`--config` for demos so you don't overwrite your real settings.

### Build the exe

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-exe.ps1
```

This writes `dist\cambobulator.exe` (~70 MB, one file, no Python needed). PyInstaller bundles `web/`,
`models/` and `virtualcam/` from the package. The exe is unsigned, hence the SmartScreen warning.

### Vendored files

* `src/cambobulator/virtualcam/`: OBS Studio's virtual-camera DLLs, unmodified (version, hashes and source
  link in `SOURCE.txt`), plus our own `placeholder.png`. pyvirtualcam's OBS backend only checks that the OBS
  Virtual Camera CLSID is registered, then writes frames into the shared-memory queue the DLL reads, so it can't
  tell this copy from a real OBS install. `cambobulator driver install` also writes
  `%APPDATA%\obs-virtualcam.txt` (the format apps see before Cambobulator sends anything); OBS normally
  writes that file, and without it apps that open the camera early get no usable format.
* `src/cambobulator/models/selfie_segmenter_landscape.onnx`: MediaPipe's selfie segmenter, converted from
  `.tflite` by `scripts/convert_segmenter.py` (instructions inside). `tests/test_misc.py` checks it against a
  mask MediaPipe itself produced.

### Code map

| file | what |
|---|---|
| `filters/base.py`, `params.py` | `Filter` base class, `Param`/`Action` metadata |
| `filters/__init__.py` | registry: config name → filter class |
| `filters/fade.py` | Fade into Background, the one effect (incl. the person-only look) |
| `segmentation.py` | `Segmenter` protocol, the bundled ONNX model via OpenCV dnn (lazy-loaded), null/stub segmenters |
| `sources.py` | camera / video file / synthetic sources, `CaptureThread` (keeps only the newest frame) |
| `pipeline.py` | runs the effect on a processing thread; outputs + preview |
| `outputs.py` | virtual camera (pyvirtualcam), video file recorder |
| `driver.py` | install/uninstall the bundled virtual-camera driver (elevated helper, registry checks) |
| `app.py` | `Controller`: ties settings, pipeline and virtual camera together; autosave |
| `web/` | FastAPI control panel + MJPEG preview |
| `cli.py` | `cambobulator ui / run / cameras / filters / driver` |
| `models/`, `virtualcam/` | vendored model and driver (see above) |
| `scripts/` | exe build, model conversion |

### Why a local web UI instead of Tkinter?

* **There's no GUI toolkit to install or bundle.** Tk's image widgets are slow at pushing 720p video, and some
  Python builds ship without Tkinter. A browser just displays an MJPEG stream.
* **It keeps processing separate from the UI.** The camera pipeline runs in its own threads. The UI is just a
  JSON API plus two `<img>` streams, so a slow or closed browser tab never stalls the virtual camera.
* **It is easy to test.** The API is tested with FastAPI's test client, and no display is needed.
* It listens on `127.0.0.1` only by default, so nobody else on the network can see your camera.

### Latency & performance

The capture thread keeps only the newest frame, so the processing loop never works on stale frames (extra
latency stays at about one frame). Segmentation takes ~9 ms per frame (Windows 11 laptop, OpenCV 5.0). On a
4-core cloud VM at 1280×720, the fade filter took ~13 ms per frame with a captured plate (~25 ms while still
learning the plate without one, ~22 ms with shimmer/auto-fade). That gives a steady 30 fps. On the laptop, the
person look adds ~5 ms for brightness/contrast/saturation and ~7 ms for pixelate at 1280×720. Mask processing
runs at reduced resolution (≤ 480 px wide). If your machine struggles, use 960×540.

## Known limits

* The person mask is only as good as the selfie model: hair, fingers and fast motion can leave faint
  edges at full fade. **Grow mask** hides most of this.
* The bundled driver was checked on Windows 11 by reading Cambobulator's output back through DirectShow
  (OpenCV) and in Edge. Teams itself was not part of that check.
* x64 only: the driver DLLs are x64 and x86, so ARM64-native video apps can't load them.
* Only one virtual-camera output at a time (that's what the OBS driver supports).
* The motion detector behind auto-fade doesn't hear audio. "Talking" means visible mouth/head movement.
  Tune **Motion sensitivity** for your camera's noise level.

## License

Cambobulator's own code is MIT. Two dependencies are GPL-2.0: the bundled OBS virtual-camera DLLs (see
`src/cambobulator/virtualcam/SOURCE.txt`) and pyvirtualcam. The built exe contains both, so if you give it to
someone, give them a link to this repository's source as well.
