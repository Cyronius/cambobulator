"""The Windows virtual-camera driver: OBS's DirectShow module, vendored.

pyvirtualcam's "obs" backend only checks that the OBS Virtual Camera CLSID is
registered, then writes frames into the "OBSVirtualCamVideo" shared-memory
queue that the module reads. So a registered copy of OBS's module is all it
needs; OBS itself never has to be installed. The camera keeps OBS's name,
"OBS Virtual Camera", in video apps. See ``write_format_hint`` for the one
other thing OBS normally does for it.

Apps load the module into their own process whenever they list cameras, so it
is installed under Program Files (not writable by normal users) and registered
machine-wide. That takes one UAC prompt: the unelevated process re-runs itself
elevated (``driver install --apply``) to copy the files and run regsvr32.
"""

from __future__ import annotations

import filecmp
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

CLSID = "{A3FCE0F5-3493-419F-958A-ABA1250EC20B}"
VENDOR_DIR = Path(__file__).with_name("virtualcam")
DLLS = {64: "obs-virtualcam-module64.dll", 32: "obs-virtualcam-module32.dll"}
# The module shows placeholder.png from its own folder while nothing is sending frames.
FILES = (*DLLS.values(), "placeholder.png", "COPYING.txt", "SOURCE.txt")

ERROR_CANCELLED = 1223


class DriverError(RuntimeError):
    pass


def install_dir() -> Path:
    # ProgramW6432 is the real Program Files even from a 32-bit Python.
    base = os.environ.get("ProgramW6432") or os.environ.get("ProgramFiles") or r"C:\Program Files"
    return Path(base) / "Cambobulator" / "virtualcam"


def registered_path(bits: int = 64) -> Path | None:
    """The DLL the OBS Virtual Camera CLSID points at in the 64- or 32-bit registry view."""
    if sys.platform != "win32":
        return None
    import winreg

    view = winreg.KEY_WOW64_64KEY if bits == 64 else winreg.KEY_WOW64_32KEY
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, rf"CLSID\{CLSID}\InprocServer32", 0,
                            winreg.KEY_READ | view) as key:
            value, _ = winreg.QueryValueEx(key, "")
    except OSError:
        return None
    return Path(value) if value else None


@dataclass
class DriverStatus:
    path64: Path | None
    path32: Path | None

    @property
    def installed(self) -> bool:
        """Usable by a 64-bit Cambobulator (pyvirtualcam only needs the 64-bit registration)."""
        return self.path64 is not None and self.path64.is_file()

    @property
    def ours(self) -> bool:
        return self.path64 is not None and _same_path(self.path64.parent, install_dir())

    @property
    def current(self) -> bool:
        """Ours, and every installed file matches the vendored copy."""
        return self.installed and self.ours and all(
            (install_dir() / f).is_file() and filecmp.cmp(VENDOR_DIR / f, install_dir() / f, shallow=False)
            for f in FILES
        )

    def describe(self) -> str:
        if not self.installed:
            stale = f" (registered at {self.path64}, but that file is gone)" if self.path64 else ""
            return f"not installed{stale}"
        owner = "Cambobulator's copy" if self.ours else "provided by another program, probably OBS Studio"
        bits = "64-bit" + (" + 32-bit" if self.path32 and self.path32.is_file() else "")
        return f"installed, {owner}: {self.path64} ({bits})"


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def status() -> DriverStatus:
    return DriverStatus(registered_path(64), registered_path(32))


def is_installed() -> bool:
    return status().installed


def write_format_hint(width: int, height: int, fps: float, path: Path | None = None) -> None:
    """Tell the module which video format to offer when Cambobulator isn't sending yet.

    The module has no default format: an app that opens the camera before
    Cambobulator starts sees no formats at all (and usually a broken camera)
    unless this file exists. OBS writes the same file when its virtual camera
    starts. The interval is in 100 ns units.
    """
    if path is None:
        path = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "obs-virtualcam.txt"
    path.write_text(f"{int(width)}x{int(height)}x{round(10_000_000 / fps)}", encoding="ascii")


def _require_windows() -> None:
    if sys.platform != "win32":
        raise DriverError("Cambobulator only bundles a virtual-camera driver for Windows.")


# -- the unelevated side -------------------------------------------------------
def install(force: bool = False) -> DriverStatus:
    """Make sure a virtual-camera driver is registered. Shows a UAC prompt if it has to install.

    A registration by OBS Studio is left alone unless ``force``; Cambobulator's
    own copy is refreshed when the vendored files changed.
    """
    _require_windows()
    st = status()
    if st.current or (st.installed and not st.ours and not force):
        return st
    _run_elevated("install")
    st = status()
    if not st.installed:
        raise DriverError("The virtual camera driver did not register. Run 'cambobulator driver install' to retry.")
    return st


def uninstall() -> DriverStatus:
    _require_windows()
    st = status()
    if st.installed and not st.ours:
        raise DriverError(f"The registered virtual camera belongs to another program ({st.path64}); "
                          "Cambobulator leaves it alone.")
    if st.path64 is None and st.path32 is None and not install_dir().exists():
        return st
    _run_elevated("uninstall")
    return status()


def _self_command() -> list[str]:
    if getattr(sys, "frozen", False):  # PyInstaller exe
        return [sys.executable]
    return [sys.executable, "-m", "cambobulator"]


def _run_elevated(action: str) -> None:
    """Run ``cambobulator driver <action> --apply`` as administrator and wait for it."""
    import ctypes
    from ctypes import wintypes

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD), ("fMask", wintypes.ULONG), ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR), ("lpFile", wintypes.LPCWSTR), ("lpParameters", wintypes.LPCWSTR),
            ("lpDirectory", wintypes.LPCWSTR), ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", ctypes.c_void_p), ("lpClass", wintypes.LPCWSTR), ("hkeyClass", wintypes.HKEY),
            ("dwHotKey", wintypes.DWORD), ("hIconOrMonitor", wintypes.HANDLE), ("hProcess", wintypes.HANDLE),
        ]

    SEE_MASK_NOCLOSEPROCESS = 0x40
    SW_HIDE = 0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SHELLEXECUTEINFOW)]
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    fd, log_path = tempfile.mkstemp(prefix="cambobulator-driver-", suffix=".log")
    os.close(fd)
    exe, *args = _self_command()
    params = subprocess.list2cmdline([*args, "driver", action, "--apply", "--log", log_path])
    info = SHELLEXECUTEINFOW(cbSize=ctypes.sizeof(SHELLEXECUTEINFOW), fMask=SEE_MASK_NOCLOSEPROCESS,
                             lpVerb="runas", lpFile=exe, lpParameters=params, lpDirectory=os.getcwd(),
                             nShow=SW_HIDE)
    log.info("Asking Windows for permission to %s the virtual camera driver", action)
    try:
        if not shell32.ShellExecuteExW(ctypes.byref(info)):
            err = ctypes.get_last_error()
            if err == ERROR_CANCELLED:
                raise DriverError(f"Driver {action} cancelled at the Windows permission prompt.")
            raise DriverError(f"Could not start the driver {action}: {ctypes.FormatError(err)}")
        kernel32.WaitForSingleObject(info.hProcess, 0xFFFFFFFF)
        code = wintypes.DWORD()
        kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code))
        kernel32.CloseHandle(info.hProcess)
        details = Path(log_path).read_text(encoding="utf-8", errors="replace").strip()
        if details:
            log.info("Driver %s:\n%s", action, details)
        if code.value != 0:
            raise DriverError(f"Driver {action} failed (exit code {code.value}).\n{details}")
    finally:
        Path(log_path).unlink(missing_ok=True)


# -- the elevated side ---------------------------------------------------------
def _regsvr32(bits: int) -> Path:
    windir = Path(os.environ.get("SystemRoot") or r"C:\Windows")
    if bits == 32:
        return windir / "SysWOW64" / "regsvr32.exe"
    native = windir / "Sysnative"  # only exists for 32-bit processes on 64-bit Windows
    return (native if native.is_dir() else windir / "System32") / "regsvr32.exe"


def _regsvr(bits: int, dll: Path, unregister: bool = False) -> str | None:
    """Run regsvr32; returns an error message or None."""
    exe = _regsvr32(bits)
    if not exe.is_file():
        return f"{exe} not found"
    args = [str(exe), "/s", *(["/u"] if unregister else []), str(dll)]
    code = subprocess.run(args, check=False).returncode
    return None if code == 0 else f"{' '.join(args)} exited with {code}"


def apply_install() -> list[str]:
    """Copy the vendored files to Program Files and register both DLLs. Needs admin.

    Returns a log of what happened; raises DriverError if the 64-bit DLL could not be registered.
    """
    lines = []
    target = install_dir()
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src, dst = VENDOR_DIR / name, target / name
        if dst.is_file() and filecmp.cmp(src, dst, shallow=False):
            continue
        try:
            shutil.copyfile(src, dst)
        except PermissionError as exc:
            raise DriverError(f"Could not replace {dst}; close any app that is using the camera and retry. "
                              f"({exc})") from exc
        lines.append(f"copied {name} -> {dst}")
    for bits in (64, 32):
        err = _regsvr(bits, target / DLLS[bits])
        if err is None:
            lines.append(f"registered {bits}-bit {target / DLLS[bits]}")
        elif bits == 64:
            raise DriverError("\n".join([*lines, err]))
        else:
            lines.append(f"warning: 32-bit registration failed ({err}); 32-bit apps won't see the camera")
    return lines


def apply_uninstall() -> list[str]:
    """Unregister both DLLs and delete the installed files. Needs admin."""
    lines = []
    target = install_dir()
    for bits in (64, 32):
        dll = target / DLLS[bits]
        if dll.is_file():
            err = _regsvr(bits, dll, unregister=True)
            lines.append(err or f"unregistered {bits}-bit {dll}")
    if target.exists():
        try:
            shutil.rmtree(target)
            lines.append(f"deleted {target}")
            parent = target.parent
            if parent.name == "Cambobulator" and not any(parent.iterdir()):
                parent.rmdir()
        except OSError as exc:
            lines.append(f"could not delete {target} (an app may still be using it; reboot and delete it): {exc}")
    return lines
