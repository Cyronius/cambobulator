# Cambobulator

A software-defined virtual webcam. Cambobulator reads your real webcam, runs each frame through a chain of
filters, and sends the result to a **virtual camera** that you can pick in Teams, Discord, Zoom, Meet or OBS.

```
real webcam ──► capture thread ──► [filter 1] ► [filter 2] ► … ──► virtual camera  ("OBS Virtual Camera")
                                                              └──► browser preview (raw | output)
```

The first filter is **Fade into Background**. It's for a coworker who would rather not be perceived: he
slowly blends into his own background.

* It segments the person (MediaPipe selfie segmentation), softens the mask edge, and blends the person's pixels
  toward a clean background plate: `out = lerp(frame, background, fade * mask)`.
* **Background plate.** Step out of frame and click *Capture background now*, or use *Capture after countdown*
  (3 s by default). With no capture yet, it learns the background from every pixel it has seen that is not you,
  and fills in never-seen areas with an inpainted, blurred guess. Either way, the plate keeps adapting slowly
  from non-person pixels, so lighting drift doesn't give you away.
* **Auto-fade.** Fade slowly ramps up while you sit still, and snaps back the moment you move or talk.
* **Predator shimmer.** A refraction ripple through your silhouette at partial fade.

There are also a few simple filters (Mirror, Brightness/Contrast/Saturation, Pixelate). They are useful
examples, and they let you test reordering.

---

## Install

You need **Python 3.10 – 3.12** (MediaPipe may not have wheels for the newest Python yet) and a
virtual-camera driver.

### Windows (Teams)

1. Install **OBS Studio 28+** from <https://obsproject.com>. Open it once, click **Start Virtual Camera**, then
   **Stop Virtual Camera**, and close OBS. This registers the "OBS Virtual Camera" driver. You do *not* need
   OBS running afterwards. Don't run the OBS virtual camera at the same time as Cambobulator.
2. Install Cambobulator:
   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install .
   ```
3. Run `cambobulator` (the control panel opens in your browser).

### macOS

1. Install **OBS Studio 30+**. Click **Start Virtual Camera** once, and approve the system extension when macOS
   asks (System Settings → Privacy & Security). Then stop it and quit OBS.
2. `python3 -m venv .venv && source .venv/bin/activate && pip install .`
3. Run `cambobulator`. The first run asks for camera permission for your terminal app.

### Linux

1. Install and load v4l2loopback:
   ```bash
   sudo apt install v4l2loopback-dkms        # Fedora: akmod-v4l2loopback, Arch: v4l2loopback-dkms
   sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="Cambobulator"
   ```
   (`exclusive_caps=1` is required for Chrome/Teams/Zoom to see the device. To load it at boot, add the
   options to `/etc/modprobe.d/v4l2loopback.conf` and `v4l2loopback` to `/etc/modules-load.d/`.)
2. `python3 -m venv .venv && source .venv/bin/activate && pip install .`
3. Run `cambobulator`.

### Segmentation model

On first use, Cambobulator downloads MediaPipe's ~250 KB `selfie_segmenter_landscape.tflite` into your cache
folder (`%LOCALAPPDATA%\cambobulator`, `~/Library/Caches/cambobulator` or `~/.cache/cambobulator`). If the
machine is offline, download it from the URL in `src/cambobulator/segmentation.py` and set
`"segmenter_model": "C:/path/to/selfie_segmenter_landscape.tflite"` in the config. Without a model,
the fade filter still works after you capture a background plate: it then finds you by comparing each frame
to the plate (this is less robust, but it works).

---

## Use it

```
cambobulator                 # same as "cambobulator ui": control panel at http://127.0.0.1:8765
cambobulator run             # headless, using the saved settings
cambobulator cameras         # list cameras (index: name)
cambobulator filters         # list filters and their parameters
```

**Control panel.** Pick the source camera, turn filters on and off, reorder them (↑/↓), and drag sliders.
You see the raw camera and the processed output side by side. Trigger a background capture, and toggle the
virtual camera with the button in the top bar. Every change is saved automatically to the config file
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
| `--segmenter none` | skip MediaPipe (fade uses the background-difference fallback) |
| `--segmenter synthetic` | exact mask for the synthetic test pattern (demos) |
| `--config path.json` | use a different settings file |

### Pick the camera in your video app

Start Cambobulator first (the virtual camera must be running), then:

* **Teams (new):** Settings → *Devices* (or the camera drop-down before joining) → Camera → **OBS Virtual Camera**.
  In a meeting: click the arrow next to the camera button → OBS Virtual Camera.
* **Discord:** User Settings → *Voice & Video* → Camera → **OBS Virtual Camera**.
* **Zoom:** Settings → *Video* → Camera → **OBS Virtual Camera**.
* **Browser apps (Meet, Teams web):** the camera picker in the site's settings; you may need to reload the tab
  after starting Cambobulator.
* On Linux the device is called whatever you set as `card_label` ("Cambobulator").

Tips:

* Pick your **real** webcam as Cambobulator's source, and the **virtual** camera in Teams. On Windows,
  usually only one app can use a physical webcam at a time. If Teams grabs it first, Cambobulator will report
  that the camera is busy.
* Teams/Zoom mirror your *self-view* only; other people see the image un-mirrored. Don't add the Mirror filter
  to "fix" this unless you want everyone to see you flipped.
* Turn off Teams' own background effects; they fight with the fade.
* If the image in Teams is black, the virtual camera isn't running. Look at the top bar of the control panel.

### Getting a good fade

1. Sit where you'll sit in the meeting, then click **Capture after countdown** and get out of the shot.
2. Raise **Fade** slowly. If you see an outline, raise **Grow mask** and/or **Edge feather**.
3. If parts of you flicker in and out, lower **Mask threshold**. If the room near you smears, raise it.
4. **Background adapt rate** follows slow lighting changes. Set it to 0 to freeze the plate.
5. If the camera moves, or the lighting changes a lot, capture again (or *Forget background*).

---

## Configuration file

JSON, stored at `%APPDATA%\cambobulator\config.json`, `~/Library/Application Support/cambobulator/config.json`
or `~/.config/cambobulator/config.json`:

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

`vcam_backend` / `vcam_device` are passed to [pyvirtualcam](https://github.com/letmaik/pyvirtualcam). Examples:
`"unitycapture"` on Windows, or `"/dev/video10"` on Linux if you have several loopback devices.

---

## Write a new filter

A filter is a class with metadata and a `process(frame, ctx)` method. Frames are `numpy.uint8` arrays in
OpenCV's **BGR** order, with shape `(H, W, 3)`. Return a frame of the same shape and type.

```python
# src/cambobulator/filters/sepia.py
import cv2
import numpy as np

from cambobulator.filters import register_filter
from cambobulator.filters.base import Filter, FrameContext
from cambobulator.params import Action, Param


@register_filter
class Sepia(Filter):
    NAME = "sepia"                   # unique id, stored in the config
    LABEL = "Sepia"                  # shown in the UI
    DESCRIPTION = "Old-timey."
    PARAMS = (                       # the UI builds sliders/checkboxes/drop-downs from these
        Param("amount", "float", 1.0, 0.0, 1.0, 0.01, "Amount"),
    )
    ACTIONS = ()                     # optional buttons: Action("name", "Label") -> calls self.action_name()
    USES_SEGMENTER = False           # True if you use self.segmenter.segment(frame)

    KERNEL = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])

    def process(self, frame: np.ndarray, ctx: FrameContext | None = None) -> np.ndarray:
        sepia = cv2.transform(frame, self.KERNEL[::-1, ::-1])  # BGR in, BGR out
        return cv2.addWeighted(sepia, self["amount"], frame, 1 - self["amount"], 0)
```

Then add it to the import line at the bottom of `src/cambobulator/filters/__init__.py`. Or, from a separate
package, expose it as an entry point and Cambobulator loads it at startup:

```toml
[project.entry-points."cambobulator.filters"]
sepia = "my_package.sepia"
```

Notes:

* Read parameters with `self["name"]`. Values are already converted to the right type and clamped to
  [min, max].
* Use `ctx.timestamp` (seconds) for anything time-based, not `time.time()`. That keeps your filter testable.
* If your filter needs the person mask, call `self.segmenter.segment(frame)` and handle `None`
  (no segmenter available).
* Optional hooks: `reset()` (the source changed), `status()` (a small dict shown under the controls), and
  `save_assets()/load_assets()` (persist images next to the config).
* An exception in `process` doesn't kill the stream: the frame passes through and the error shows in the UI.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests need no camera, no virtual-camera driver and no model download. They use synthetic frames, a
fake segmenter, a synthetic source, and a stubbed `pyvirtualcam`. To see the whole pipeline without hardware:

```bash
cambobulator run --source synthetic --segmenter synthetic --no-vcam --record demo.mp4 --max-frames 150
cambobulator ui  --source synthetic --segmenter synthetic --no-vcam --config demo.json
```

`--segmenter synthetic` finds the test pattern's "person" by its exact colour, since MediaPipe can't see a
cartoon. Point it at a real video of a person to see real segmentation: `--source me.mp4`. Use a separate
`--config` for demos so you don't overwrite your real settings.

### Code map

| file | what |
|---|---|
| `filters/base.py`, `params.py` | `Filter` base class, `Param`/`Action` metadata |
| `filters/__init__.py` | registry (`register_filter`, `create_filter`, plugin entry points) |
| `filters/fade.py` | Fade into Background |
| `filters/basic.py` | Mirror, Adjust, Pixelate |
| `segmentation.py` | `Segmenter` protocol, MediaPipe implementation (lazy-loaded), null/stub segmenters |
| `sources.py` | camera / video file / synthetic sources, `CaptureThread` (keeps only the newest frame) |
| `pipeline.py` | filter chain + processing thread + outputs + preview |
| `outputs.py` | virtual camera (pyvirtualcam), video file recorder |
| `app.py` | `Controller`: ties settings, pipeline and virtual camera together; autosave |
| `web/` | FastAPI control panel + MJPEG preview |
| `cli.py` | `cambobulator ui / run / cameras / filters` |

### Why a local web UI instead of Tkinter?

* **It works the same on Windows, macOS and Linux.** Some Python builds (several Linux distros, some pyenv and
  Homebrew setups) ship without Tkinter, and Tk's image widgets are slow at pushing 720p video. A browser just
  displays an MJPEG stream, with no extra GUI toolkit to install.
* **It keeps processing separate from the UI.** The camera pipeline runs in its own threads. The UI is just a
  JSON API plus two `<img>` streams, so a slow or closed browser tab never stalls the virtual camera.
* **It is easy to test.** The API is tested with FastAPI's test client, and no display is needed.
* It listens on `127.0.0.1` only by default, so nobody else on the network can see your camera.

### Latency & performance

The capture thread keeps only the newest frame, so the processing loop never works on stale frames (extra
latency stays at about one frame). Measured on a 4-core cloud VM at 1280×720: MediaPipe segmentation takes
~10 ms per frame and the fade filter ~13 ms per frame with a captured plate (~25 ms while still learning
the plate without one, ~22 ms with shimmer/auto-fade). That gives a steady 30 fps. Mask processing runs at
reduced resolution (≤ 480 px wide). If your machine struggles, use 960×540 or lower **Edge feather**.

## Known limits

* The person mask is only as good as MediaPipe's selfie model: hair, fingers and fast motion can leave faint
  edges at full fade. Grow/feather hide most of this.
* Real camera, virtual camera and friendly camera names on Windows/macOS were not tested in CI (the build box
  has no webcam). The code paths follow the documented OpenCV / pyvirtualcam / pygrabber APIs, but expect to
  report the odd driver quirk.
* macOS shows camera indices, not names, in the camera picker.
* Only one virtual-camera output at a time (that's what the OBS driver supports).
* The motion detector behind auto-fade doesn't hear audio. "Talking" means visible mouth/head movement.
  Tune **Motion sensitivity** for your camera's noise level.
