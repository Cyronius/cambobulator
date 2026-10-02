"""Command-line entry point: ``cambobulator ui | run | cameras | filters | driver``."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from cambobulator import __version__
from cambobulator.config import Settings, default_config_path

log = logging.getLogger("cambobulator")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=None, help=f"settings file (default: {default_config_path()})")
    p.add_argument("--source", help="camera index, 'synthetic', or a video file path")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--fps", type=int)
    p.add_argument("--segmenter", choices=("auto", "none", "synthetic"), help="person segmentation backend")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cambobulator", description="A software-defined virtual webcam.")
    parser.add_argument("--version", action="version", version=f"cambobulator {__version__}")
    sub = parser.add_subparsers(dest="command")

    ui = sub.add_parser("ui", help="open the control panel in your browser (default)")
    _add_common(ui)
    ui.add_argument("--host", help="address to serve on (default 127.0.0.1)")
    ui.add_argument("--port", type=int)
    ui.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    ui.add_argument("--no-vcam", action="store_true", help="start with the virtual camera off")

    run = sub.add_parser("run", help="run headless with the saved settings")
    _add_common(run)
    run.add_argument("--no-vcam", action="store_true", help="don't open the virtual camera")
    run.add_argument("--record", type=Path, help="also write the output to a video file (.mp4/.avi)")
    run.add_argument("--max-frames", type=int, help="stop after this many frames")
    run.add_argument("--capture-background", action="store_true",
                     help="start with a background-capture countdown (step out of frame!)")

    cams = sub.add_parser("cameras", help="list cameras")
    cams.add_argument("-v", "--verbose", action="store_true")
    flt = sub.add_parser("filters", help="list the effect's parameters")
    flt.add_argument("-v", "--verbose", action="store_true")

    drv = sub.add_parser("driver", help="install or remove the Windows virtual camera driver")
    drv.add_argument("action", choices=("status", "install", "uninstall"))
    drv.add_argument("--force", action="store_true", help="install over a driver registered by OBS Studio")
    drv.add_argument("--apply", action="store_true", help=argparse.SUPPRESS)  # the elevated helper
    drv.add_argument("--log", type=Path, help=argparse.SUPPRESS)
    drv.add_argument("-v", "--verbose", action="store_true")
    return parser


def _settings_from_args(args: argparse.Namespace) -> tuple[Settings, Path]:
    path = args.config or default_config_path()
    settings = Settings.load(path)
    for name in ("source", "width", "height", "fps", "segmenter"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(settings, name, str(value) if name == "source" else value)
    return settings, path


def cmd_cameras(args: argparse.Namespace) -> int:
    from cambobulator.cameras import list_cameras

    cams = list_cameras()
    if not cams:
        print("No cameras found.")
        return 1
    for c in cams:
        note = "   (virtual camera: don't use as a source)" if c.is_virtual else ""
        print(f"{c.index}: {c.name}{note}")
    return 0


def cmd_filters(args: argparse.Namespace) -> int:
    from cambobulator.filters import available_filters, get_filter_class

    for info in available_filters():
        cls = get_filter_class(info["type"])
        print(f"{info['type']}  -  {info['label']}")
        if info["description"]:
            print(f"    {info['description']}")
        for p in cls.PARAMS:
            rng = f" [{p.min}..{p.max}]" if p.min is not None else ""
            choices = f" {list(p.choices)}" if p.choices else ""
            print(f"    {p.name} ({p.kind}{rng}{choices}) default={p.default}  {p.help}")
        for a in cls.ACTIONS:
            print(f"    action: {a.name}  -  {a.label}")
    return 0


def cmd_driver(args: argparse.Namespace) -> int:
    from cambobulator import driver

    if args.apply:  # we are the elevated helper; the caller reads --log
        lines, code = [], 0
        try:
            lines = driver.apply_install() if args.action == "install" else driver.apply_uninstall()
        except Exception as exc:
            lines, code = [str(exc)], 1
        if args.log:
            args.log.write_text("\n".join(lines), encoding="utf-8")
        return code
    try:
        if args.action == "install":
            driver.install(force=args.force)
            # So apps opened before Cambobulator's first run still see a usable format.
            s = Settings.load(default_config_path())
            driver.write_format_hint(s.width, s.height, s.fps)
        elif args.action == "uninstall":
            driver.uninstall()
    except driver.DriverError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Virtual camera driver: {driver.status().describe()}")
    return 0


def _make_controller(args: argparse.Namespace):
    from cambobulator.app import Controller

    settings, path = _settings_from_args(args)
    return Controller(settings, path)


def cmd_run(args: argparse.Namespace) -> int:
    from cambobulator.outputs import VideoFileOutput, VirtualCameraError
    from cambobulator.sources import SourceError

    controller = _make_controller(args)
    s = controller.settings
    try:
        controller.start(virtual_camera=not args.no_vcam, strict=True)
    except (SourceError, VirtualCameraError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        controller.close()
        return 2
    if args.record:
        controller.pipeline.add_output("record", VideoFileOutput(args.record, s.width, s.height, s.fps))
    if args.capture_background:
        from cambobulator.filters.fade import CAPTURE_DELAY

        controller.run_action(controller.pipeline.slots[0].id, "capture_background")
        print(f"Capturing the background in {CAPTURE_DELAY:.0f} s. Step out of frame!")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    where = controller.vcam_device or "no virtual camera"
    print(f"Running: {controller.pipeline.stats()['source']} -> "
          f"{', '.join(sl.filter.LABEL for sl in controller.pipeline.slots if sl.enabled) or 'no filters'} -> {where}")
    print("Press Ctrl+C to stop.")
    last_report = time.monotonic()
    exit_code = 0
    try:
        while not stop.is_set():
            stop.wait(0.2)
            st = controller.pipeline.stats()
            if st["source_error"]:
                print(f"\nError: {st['source_error']}", file=sys.stderr)
                exit_code = 2
                break
            if args.max_frames and st["frames"] >= args.max_frames:
                break
            if args.verbose and time.monotonic() - last_report > 5:
                last_report = time.monotonic()
                print(f"{st['fps']} fps, {st['process_ms']} ms/frame, {st['frames']} frames")
    finally:
        controller.close()
    return exit_code


def cmd_ui(args: argparse.Namespace) -> int:
    import uvicorn

    from cambobulator.web.server import create_app

    controller = _make_controller(args)
    s = controller.settings
    host = args.host or s.ui_host
    port = args.port or s.ui_port
    controller.start(virtual_camera=False if args.no_vcam else None)
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    print(f"Cambobulator control panel: {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        import webbrowser

        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        uvicorn.run(create_app(controller), host=host, port=port, log_level="warning", timeout_graceful_shutdown=2)
    finally:
        controller.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args = parser.parse_args(["ui", *(argv if argv is not None else sys.argv[1:])])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("comtypes").setLevel(logging.WARNING)  # narrates its code generation at INFO
    if getattr(args, "verbose", False):
        log.setLevel(logging.DEBUG)
    commands = {"ui": cmd_ui, "run": cmd_run, "cameras": cmd_cameras, "filters": cmd_filters, "driver": cmd_driver}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
