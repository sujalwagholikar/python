"""
adb_controller.py
==================
Core Android control engine built on top of ADB (Android Debug Bridge).

This module wraps every "physical action" you can perform on a non-rooted
Android phone through ADB: tapping, swiping, typing, opening apps, sending
intents (open URL / dial / search), taking screenshots, toggling flashlight,
volume, WiFi, Bluetooth, installing/uninstalling apps, reading notifications,
etc.

IMPORTANT — read before use
----------------------------
1. ADB must be installed on this PC (Windows: platform-tools in PATH).
2. Your phone needs "USB debugging" enabled (Settings > About phone > tap
   Build number 7x > Developer options > USB debugging).
3. On first USB connection, your phone will show an "Allow USB debugging?"
   popup — tap Allow (and "always allow from this computer").
4. Without root, Python/ADB CANNOT:
      - Bypass or brute-force a lock-screen PIN/pattern/password/biometric.
        "Unlock the phone" only works if the phone has NO lock set, or if
        we send the PIN you configured (see AndroidController.unlock()).
      - Read the *content* of WhatsApp/other app messages (no accessibility
        scraping is wired up here) — but it CAN open WhatsApp and send a
        message via WhatsApp's own "send text" intent, which is the
        reliable, non-root way to automate WhatsApp sends.
      - Access data of another app sandboxed away from ADB shell.
   Everything else in this file works on a stock, non-rooted phone.
"""

from __future__ import annotations

import re
import shlex
import socket
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class ADBError(RuntimeError):
    """Raised when an adb command fails or no device is available."""


class DeviceNotConnected(ADBError):
    """Raised when no device is connected over USB or WiFi."""


# --------------------------------------------------------------------------- #
# Low level ADB wrapper
# --------------------------------------------------------------------------- #

@dataclass
class DeviceInfo:
    serial: str
    connection: str  # "usb" or "wifi"
    model: str = ""
    android_version: str = ""


class ADBBridge:
    """
    Thin wrapper around the `adb` command line tool.

    Handles:
      - locating a device over USB
      - falling back to (or explicitly using) a WiFi (adb tcpip) connection
      - running shell commands / pulling / pushing files
    """

    def __init__(self, adb_path: str = "adb", preferred: str = "auto"):
        """
        adb_path : path to the adb executable (default assumes it's on PATH)
        preferred: "usb", "wifi", or "auto" (try USB first, fall back to WiFi)
        """
        self.adb_path = adb_path
        self.preferred = preferred
        self.device: Optional[DeviceInfo] = None
        self._check_adb_installed()

    # -- setup ------------------------------------------------------------- #

    def _check_adb_installed(self) -> None:
        try:
            subprocess.run(
                [self.adb_path, "version"],
                capture_output=True, text=True, check=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise ADBError(
                "adb executable not found. Install Android platform-tools "
                "and make sure 'adb' is on your PATH. "
                "Download: https://developer.android.com/tools/releases/platform-tools"
            ) from e

    def _raw(self, args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
        cmd = [self.adb_path] + args
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def list_devices(self) -> list[tuple[str, str]]:
        """Return list of (serial, state) currently visible to adb."""
        result = self._raw(["devices"])
        devices = []
        for line in result.stdout.strip().splitlines()[1:]:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            serial, state = line.split("\t")
            devices.append((serial, state))
        return devices

    def _pick_serial(self) -> str:
        devices = [d for d in self.list_devices() if d[1] == "device"]
        if not devices:
            raise DeviceNotConnected(
                "No authorized device found. Checklist:\n"
                "  1. USB debugging enabled on phone\n"
                "  2. Cable/WiFi actually connected\n"
                "  3. You tapped 'Allow' on the phone's USB debugging popup\n"
                "  4. Run `adb devices` manually to check status "
                "(look for 'unauthorized' vs 'device')"
            )
        if self.preferred == "wifi":
            wifi = [d for d in devices if ":" in d[0]]
            if wifi:
                return wifi[0][0]
        if self.preferred == "usb":
            usb = [d for d in devices if ":" not in d[0]]
            if usb:
                return usb[0][0]
        # auto: prefer USB (no ":"), else first available
        usb_first = sorted(devices, key=lambda d: (":" in d[0]))
        return usb_first[0][0]

    def connect(self) -> DeviceInfo:
        """Connect (select) a device; must be called before use."""
        serial = self._pick_serial()
        conn_type = "wifi" if ":" in serial else "usb"
        model = self._shell_static(serial, "getprop ro.product.model").strip()
        version = self._shell_static(serial, "getprop ro.build.version.release").strip()
        self.device = DeviceInfo(serial=serial, connection=conn_type,
                                  model=model, android_version=version)
        return self.device

    def _shell_static(self, serial: str, command: str, timeout: int = 15) -> str:
        result = self._raw(["-s", serial, "shell", command], timeout=timeout)
        return result.stdout

    # -- WiFi pairing helpers ------------------------------------------------ #

    def enable_tcpip_mode(self, port: int = 5555) -> str:
        """
        Switch the currently-USB-connected device into TCP/IP (WiFi ADB) mode.
        Call this ONCE while plugged in via USB, then you can unplug and use
        WiFi from then on (until phone reboots / disconnects WiFi).
        Returns the phone's IP address to connect to.
        """
        if not self.device:
            self.connect()
        ip = self.get_device_ip()
        if not ip:
            raise ADBError(
                "Could not determine phone's WiFi IP address. "
                "Make sure the phone is connected to WiFi first."
            )
        self._raw(["-s", self.device.serial, "tcpip", str(port)])
        time.sleep(2)
        return ip

    def get_device_ip(self) -> Optional[str]:
        if not self.device:
            self.connect()
        out = self._shell_static(self.device.serial, "ip route")
        # typical line: "192.168.1.0/24 dev wlan0 ... src 192.168.1.42"
        match = re.search(r"src\s+(\d+\.\d+\.\d+\.\d+)", out)
        if match:
            return match.group(1)
        # fallback method
        out2 = self._shell_static(self.device.serial, "ip -f inet addr show wlan0")
        match2 = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", out2)
        return match2.group(1) if match2 else None

    def connect_wifi(self, ip: str, port: int = 5555, timeout: int = 10) -> DeviceInfo:
        """Connect to a phone already in tcpip mode over WiFi by IP address."""
        # quick reachability check first for a clear error message
        try:
            with socket.create_connection((ip, port), timeout=3):
                pass
        except OSError as e:
            raise DeviceNotConnected(
                f"Cannot reach {ip}:{port} — is the phone on the same WiFi "
                f"network and in tcpip mode? ({e})"
            )
        result = self._raw(["connect", f"{ip}:{port}"], timeout=timeout)
        if "connected" not in result.stdout.lower():
            raise DeviceNotConnected(f"adb connect failed: {result.stdout} {result.stderr}")
        self.preferred = "wifi"
        return self.connect()

    # -- shell / exec -------------------------------------------------------- #

    def require_device(self) -> DeviceInfo:
        if not self.device:
            self.connect()
        return self.device

    def shell(self, command: str, timeout: int = 30) -> str:
        dev = self.require_device()
        result = self._raw(["-s", dev.serial, "shell", command], timeout=timeout)
        if result.returncode != 0 and result.stderr.strip():
            raise ADBError(f"shell command failed: {command}\n{result.stderr}")
        return result.stdout

    def exec_out(self, command: str, timeout: int = 30) -> bytes:
        """Like shell() but returns raw binary stdout (for screencap etc)."""
        dev = self.require_device()
        cmd = [self.adb_path, "-s", dev.serial, "exec-out"] + shlex.split(command)
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return result.stdout

    def pull(self, remote_path: str, local_path: str, timeout: int = 60) -> None:
        dev = self.require_device()
        result = self._raw(["-s", dev.serial, "pull", remote_path, local_path], timeout=timeout)
        if result.returncode != 0:
            raise ADBError(f"pull failed: {result.stderr}")

    def push(self, local_path: str, remote_path: str, timeout: int = 60) -> None:
        dev = self.require_device()
        result = self._raw(["-s", dev.serial, "push", local_path, remote_path], timeout=timeout)
        if result.returncode != 0:
            raise ADBError(f"push failed: {result.stderr}")

    def install(self, apk_path: str, timeout: int = 180) -> str:
        dev = self.require_device()
        result = self._raw(["-s", dev.serial, "install", "-r", apk_path], timeout=timeout)
        return result.stdout + result.stderr

    def uninstall(self, package: str, timeout: int = 60) -> str:
        dev = self.require_device()
        result = self._raw(["-s", dev.serial, "uninstall", package], timeout=timeout)
        return result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# High level Android controller (the actual "actions")
# --------------------------------------------------------------------------- #

# Common package names — extend as needed for apps you use.
APP_PACKAGES = {
    "whatsapp": "com.whatsapp",
    "whatsapp business": "com.whatsapp.w4b",
    "telegram": "org.telegram.messenger",
    "gmail": "com.google.android.gm",
    "chrome": "com.android.chrome",
    "camera": "com.android.camera",  # varies by OEM, see open_camera()
    "settings": "com.android.settings",
    "youtube": "com.google.android.youtube",
    "youtube music": "com.google.android.apps.youtube.music",
    "instagram": "com.instagram.android",
    "facebook": "com.facebook.katana",
    "maps": "com.google.android.apps.maps",
    "google maps": "com.google.android.apps.maps",
    "gallery": "com.google.android.apps.photos",
    "photos": "com.google.android.apps.photos",
    "phone": "com.android.dialer",
    "dialer": "com.android.dialer",
    "messages": "com.google.android.apps.messaging",
    "sms": "com.google.android.apps.messaging",
    "contacts": "com.android.contacts",
    "playstore": "com.android.vending",
    "play store": "com.android.vending",
    "spotify": "com.spotify.music",
    "twitter": "com.twitter.android",
    "x": "com.twitter.android",
    "facebook messenger": "com.facebook.orca",
    "messenger": "com.facebook.orca",
    "netflix": "com.netflix.mediaclient",
    "amazon": "in.amazon.mShop.android.shopping",
    "calculator": "com.google.android.calculator",
    "calendar": "com.google.android.calendar",
    "google calendar": "com.google.android.calendar",
    "clock": "com.google.android.deskclock",
    "alarm": "com.google.android.deskclock",
    "files": "com.google.android.apps.nbu.files",
    "file manager": "com.google.android.apps.nbu.files",
    # extra common apps
    "drive": "com.google.android.apps.docs",
    "google drive": "com.google.android.apps.docs",
    "docs": "com.google.android.apps.docs.editors.docs",
    "sheets": "com.google.android.apps.docs.editors.sheets",
    "slides": "com.google.android.apps.docs.editors.slides",
    "keep": "com.google.android.keep",
    "google keep": "com.google.android.keep",
    "notes": "com.google.android.keep",
    "translate": "com.google.android.apps.translate",
    "google translate": "com.google.android.apps.translate",
    "play music": "com.google.android.music",
    "podcasts": "com.google.android.apps.podcasts",
    "news": "com.google.android.apps.magazines",
    "play games": "com.google.android.play.games",
    "wallet": "com.google.android.apps.walletnfcrel",
    "google wallet": "com.google.android.apps.walletnfcrel",
    "pay": "com.google.android.apps.nbu.paisa.user",
    "google pay": "com.google.android.apps.nbu.paisa.user",
    "gpay": "com.google.android.apps.nbu.paisa.user",
    "phonepe": "com.phonepe.app",
    "paytm": "net.one97.paytm",
    "uber": "com.ubercab",
    "ola": "com.olacabs.customer",
    "swiggy": "in.swiggy.android",
    "zomato": "com.application.zomato",
    "linkedin": "com.linkedin.android",
    "reddit": "com.reddit.frontpage",
    "pinterest": "com.pinterest",
    "snapchat": "com.snapchat.android",
    "discord": "com.discord",
    "slack": "com.Slack",
    "zoom": "us.zoom.videomeetings",
    "teams": "com.microsoft.teams",
    "microsoft teams": "com.microsoft.teams",
    "outlook": "com.microsoft.office.outlook",
    "word": "com.microsoft.office.word",
    "excel": "com.microsoft.office.excel",
    "powerpoint": "com.microsoft.office.powerpoint",
    "onedrive": "com.microsoft.skydrive",
    "dropbox": "com.dropbox.android",
    "signal": "org.thoughtcrime.securesms",
    "prime video": "com.amazon.avod.thirdpartyclient",
    "amazon prime video": "com.amazon.avod.thirdpartyclient",
    "hotstar": "in.startv.hotstar",
    "disney+ hotstar": "in.startv.hotstar",
    "vlc": "org.videolan.vlc",
    "chrome browser": "com.android.chrome",
    "firefox": "org.mozilla.firefox",
    "app store": "com.android.vending",
    "google assistant": "com.google.android.googlequicksearchbox",
    "assistant": "com.google.android.googlequicksearchbox",
    "google": "com.google.android.googlequicksearchbox",
    "google app": "com.google.android.googlequicksearchbox",
    "duo": "com.google.android.apps.tachyon",
    "google meet": "com.google.android.apps.tachyon",
    "meet": "com.google.android.apps.tachyon",
    "clock app": "com.google.android.deskclock",
    "weather": "com.google.android.apps.weather",
    "play movies": "com.google.android.videos",
}

# keyevent codes used for hardware-style key presses
KEYCODES = {
    "home": 3, "back": 4, "call": 5, "endcall": 6,
    "volume_up": 24, "volume_down": 25, "power": 26,
    "camera": 27, "menu": 82, "enter": 66, "delete": 67,
    "app_switch": 187, "notification": 83, "search": 84,
    "play_pause": 85, "next_track": 87, "prev_track": 88,
    "screenshot": 120, "sleep": 223, "wakeup": 224,
}


class AndroidController:
    """
    High-level, human-friendly control surface for an Android phone.
    Every method here is a single well-defined physical/system action.
    Build the natural-language layer (command_parser.py) on top of this.
    """

    def __init__(self, bridge: Optional[ADBBridge] = None,
                 adb_path: str = "adb", preferred: str = "auto"):
        self.bridge = bridge or ADBBridge(adb_path=adb_path, preferred=preferred)

    # -- connection management ------------------------------------------- #

    def connect(self) -> DeviceInfo:
        info = self.bridge.connect()
        print(f"[connected] {info.model} (Android {info.android_version}) "
              f"via {info.connection.upper()} — serial: {info.serial}")
        return info

    def connect_over_wifi(self, ip: Optional[str] = None, port: int = 5555) -> DeviceInfo:
        """
        Connect over WiFi. If `ip` is not given, assumes a device is
        currently plugged in via USB, switches it into tcpip mode, and
        auto-detects its IP.
        """
        if ip is None:
            ip = self.bridge.enable_tcpip_mode(port=port)
            print(f"[wifi setup] Phone switched to WiFi ADB at {ip}:{port}. "
                  f"You can now unplug the USB cable.")
        return self.bridge.connect_wifi(ip, port=port)

    def status(self) -> str:
        devices = self.bridge.list_devices()
        if not devices:
            return "No devices detected by adb."
        lines = [f"  {serial}\t{state}" for serial, state in devices]
        return "adb devices:\n" + "\n".join(lines)

    # -- screen / lock ------------------------------------------------------ #

    def is_screen_on(self) -> bool:
        out = self.bridge.shell("dumpsys power | grep 'mHoldingDisplaySuspendBlocker\\|Display Power'")
        return "state=ON" in out or "true" in out.lower()

    def wake_screen(self) -> None:
        self.bridge.shell(f"input keyevent {KEYCODES['wakeup']}")

    def sleep_screen(self) -> None:
        self.bridge.shell(f"input keyevent {KEYCODES['sleep']}")

    def unlock(self, pin: Optional[str] = None, swipe_only: bool = False) -> None:
        """
        Wake + unlock the screen.
        - If the phone has no lock (or a simple swipe lock), swipe_only=True
          (or omitting pin) is enough.
        - If a PIN lock is set, pass pin="1234" — this TYPES the PIN via
          ADB input, it does not "crack" or bypass anything. You must
          already know the PIN (it's your own phone).
        - Pattern/biometric locks cannot be automated without root/accessibility
          services; this will only handle swipe or numeric-PIN locks.
        """
        self.wake_screen()
        time.sleep(0.3)
        # swipe up to reveal PIN pad / dismiss simple lock
        self.swipe(540, 1800, 540, 800, duration_ms=200)
        time.sleep(0.3)
        if pin and not swipe_only:
            self.bridge.shell(f"input text {pin}")
            self.bridge.shell(f"input keyevent {KEYCODES['enter']}")

    def lock(self) -> None:
        self.bridge.shell(f"input keyevent {KEYCODES['power']}")

    def screenshot(self, local_path: str = "screenshot.png") -> str:
        """Take a screenshot and pull it to the local machine. Returns local path."""
        remote = "/sdcard/_adb_ctrl_screenshot.png"
        self.bridge.shell(f"screencap -p {remote}")
        self.bridge.pull(remote, local_path)
        self.bridge.shell(f"rm {remote}")
        print(f"[screenshot] saved to {local_path}")
        return local_path

    def screen_record(self, local_path: str = "recording.mp4", seconds: int = 10) -> str:
        remote = "/sdcard/_adb_ctrl_record.mp4"
        # screenrecord blocks for `seconds` or until Ctrl+C; --time-limit caps it
        self.bridge.shell(f"screenrecord --time-limit {seconds} {remote}", timeout=seconds + 15)
        self.bridge.pull(remote, local_path)
        self.bridge.shell(f"rm {remote}")
        print(f"[screen_record] saved {seconds}s recording to {local_path}")
        return local_path

    # -- basic input primitives ---------------------------------------------- #

    def tap(self, x: int, y: int) -> None:
        self.bridge.shell(f"input tap {x} {y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self.bridge.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def long_press(self, x: int, y: int, duration_ms: int = 800) -> None:
        # a "swipe" with same start/end and long duration acts as a long-press
        self.bridge.shell(f"input swipe {x} {y} {x} {y} {duration_ms}")

    def type_text(self, text: str) -> None:
        """
        Type arbitrary text (including spaces/punctuation) into whatever
        field is currently focused.
        """
        escaped = text.replace(" ", "%s") \
                       .replace("&", "\\&") \
                       .replace("<", "\\<") \
                       .replace(">", "\\>") \
                       .replace("(", "\\(") \
                       .replace(")", "\\)") \
                       .replace("'", "\\'") \
                       .replace('"', '\\"') \
                       .replace(";", "\\;") \
                       .replace("|", "\\|")
        self.bridge.shell(f"input text {escaped}")

    def press_key(self, key: str) -> None:
        key = key.lower().strip()
        if key not in KEYCODES:
            raise ValueError(f"Unknown key '{key}'. Known keys: {list(KEYCODES)}")
        self.bridge.shell(f"input keyevent {KEYCODES[key]}")

    def go_home(self) -> None:
        self.press_key("home")

    def go_back(self) -> None:
        self.press_key("back")

    def recent_apps(self) -> None:
        self.press_key("app_switch")

    def take_screenshot_key(self) -> None:
        self.press_key("screenshot")

    # -- volume / hardware toggles -------------------------------------------- #

    def volume_up(self, times: int = 1) -> None:
        for _ in range(times):
            self.press_key("volume_up")

    def volume_down(self, times: int = 1) -> None:
        for _ in range(times):
            self.press_key("volume_down")

    def mute(self) -> None:
        self.bridge.shell("input keyevent 164")  # KEYCODE_VOLUME_MUTE

    def flashlight(self, on: bool = True) -> None:
        """
        Toggle flashlight/torch. Uses the `cmd` service on Android 9+.
        No root required.
        """
        state = "true" if on else "false"
        try:
            self.bridge.shell(f"cmd flashlight set-torch {state}")
        except ADBError:
            # older-device fallback via a broadcast some ROMs support
            self.bridge.shell(
                f"am broadcast -a com.android.torch.TOGGLE --ez state {state}"
            )
        print(f"[flashlight] {'ON' if on else 'OFF'}")

    def set_wifi(self, on: bool) -> None:
        self.bridge.shell(f"svc wifi {'enable' if on else 'disable'}")

    def set_bluetooth(self, on: bool) -> None:
        self.bridge.shell(f"svc bluetooth {'enable' if on else 'disable'}")

    def set_airplane_mode(self, on: bool) -> None:
        self.bridge.shell(f"settings put global airplane_mode_on {1 if on else 0}")
        self.bridge.shell(
            f"am broadcast -a android.intent.action.AIRPLANE_MODE --ez state {'true' if on else 'false'}"
        )

    def set_brightness(self, level_0_to_255: int) -> None:
        level = max(0, min(255, level_0_to_255))
        self.bridge.shell(f"settings put system screen_brightness {level}")

    def battery_status(self) -> dict:
        out = self.bridge.shell("dumpsys battery")
        info = {}
        for line in out.splitlines():
            if ":" in line:
                k, _, v = line.strip().partition(":")
                info[k.strip()] = v.strip()
        return info

    # -- apps ---------------------------------------------------------------- #

    def resolve_package(self, app_name: str) -> str:
        key = app_name.lower().strip()
        if key in APP_PACKAGES:
            return APP_PACKAGES[key]
        # fall back to searching installed packages for a fuzzy name match
        installed = self.list_installed_packages()
        matches = [p for p in installed if key.replace(" ", "") in p.lower()]
        if matches:
            return matches[0]
        raise ValueError(
            f"Don't know the package name for '{app_name}'. "
            f"Add it to APP_PACKAGES, or call open_app_by_package() directly "
            f"if you know it (e.g. run list_installed_packages() to find it)."
        )

    def list_installed_packages(self) -> list[str]:
        out = self.bridge.shell("pm list packages")
        return [line.replace("package:", "").strip() for line in out.splitlines() if line.strip()]

    def open_app(self, app_name: str) -> None:
        package = self.resolve_package(app_name)
        self.open_app_by_package(package)

    def open_app_by_package(self, package: str) -> None:
        # monkey -p launches the app's default launcher activity — most reliable,
        # works even if we don't know the exact Activity class name.
        self.bridge.shell(f"monkey -p {package} -c android.intent.category.LAUNCHER 1")

    def close_app(self, app_name: str) -> None:
        package = self.resolve_package(app_name)
        self.bridge.shell(f"am force-stop {package}")

    def open_camera(self, take_photo: bool = False, delay_before_capture: float = 2.0) -> None:
        """Open the camera app via the standard CAMERA intent (works across OEMs)."""
        self.bridge.shell("am start -a android.media.action.STILL_IMAGE_CAMERA")
        if take_photo:
            time.sleep(delay_before_capture)
            self.press_key("camera")  # hardware camera keycode triggers shutter on most camera apps

    def open_video_camera(self) -> None:
        self.bridge.shell("am start -a android.media.action.VIDEO_CAMERA")

    # -- system intents (open url / search / dial / share) ------------------- #

    def open_url(self, url: str) -> None:
        if not re.match(r"^\w+://", url):
            url = "https://" + url
        self.bridge.shell(f'am start -a android.intent.action.VIEW -d "{url}"')

    def google_search(self, query: str) -> None:
        encoded = urllib.parse.quote(query)
        self.open_url(f"https://www.google.com/search?q={encoded}")

    def dial_number(self, number: str) -> None:
        """Opens the dialer with the number pre-filled (does NOT auto-call)."""
        self.bridge.shell(f'am start -a android.intent.action.DIAL -d "tel:{number}"')

    def call_number(self, number: str) -> None:
        """
        Actually places a call. Requires CALL_PHONE permission to already be
        granted to the shell (Android normally allows this via ADB).
        """
        self.bridge.shell(f'am start -a android.intent.action.CALL -d "tel:{number}"')

    def send_sms(self, number: str, message: str) -> None:
        """Opens the default SMS app with number + message pre-filled and sends it."""
        encoded_msg = urllib.parse.quote(message)
        self.bridge.shell(
            f'am start -a android.intent.action.SENDTO -d "sms:{number}" '
            f'--es sms_body "{encoded_msg}" --ez exit_on_sent true'
        )
        time.sleep(2.0)
        # Try to find and tap the send button by label first (works across SMS apps)
        for label in ("Send", "send", "Send message", "MMS send", "SMS send"):
            if self.tap_text(label):
                print(f"[sms] send button tapped (label: {label!r})")
                return
        # Fallback: KEYCODE_ENTER — Google Messages responds to this when field is focused
        self.bridge.shell(f"input keyevent {KEYCODES['enter']}")
        print("[sms] sent via Enter key (send button not found by label)")

    def open_whatsapp_chat(self, phone_number: str, message: str = "") -> None:
        """
        Opens a WhatsApp chat with the given phone number (international
        format, no + or spaces, e.g. "919876543210") with `message`
        pre-filled, using WhatsApp's official `wa.me` deep link — no root
        needed. Does NOT press send automatically by default; call
        send_whatsapp_message() to also auto-send.
        """
        encoded_msg = urllib.parse.quote(message)
        url = f"https://wa.me/{phone_number}?text={encoded_msg}"
        self.open_url(url)

    def send_whatsapp_message(self, phone_number: str, message: str, wait_seconds: float = 3.5) -> None:
        """
        Opens WhatsApp chat with message pre-filled, waits for the chat/UI
        to load, then taps the send button.

        Searches by multiple possible send-button labels/resource-ids
        (WhatsApp uses "Send", "send", and a resource-id containing "send")
        before falling back to a fixed coordinate.
        """
        self.open_whatsapp_chat(phone_number, message)
        time.sleep(wait_seconds)
        # WhatsApp's send button content-desc varies: "Send", "send", "Send message"
        for label in ("Send", "send", "Send message", "send_btn", "send message"):
            if self.tap_text(label):
                print(f"[whatsapp] message sent to {phone_number} (button: {label!r})")
                return
        # Last resort: keyevent enter (works if message field is still focused)
        self.bridge.shell(f"input keyevent {KEYCODES['enter']}")
        print(f"[whatsapp] sent via Enter key to {phone_number} — "
              f"verify it landed (send button not auto-found by label)")

    def open_telegram_chat(self, phone_number: str, message: str = "") -> None:
        """Opens a Telegram chat via its official t.me deep link, then types message."""
        # t.me/+ opens a chat by phone number; message must be typed separately
        url = f"https://t.me/+{phone_number}"
        self.open_url(url)
        if message:
            time.sleep(2.5)
            self.type_text(message)

    def send_telegram_message(self, phone_number: str, message: str, wait_seconds: float = 3.5) -> None:
        self.open_telegram_chat(phone_number, message)
        time.sleep(wait_seconds)
        for label in ("Send", "send", "Send message", "send message"):
            if self.tap_text(label):
                print(f"[telegram] message sent to {phone_number}")
                return
        # Fallback: Enter key
        self.bridge.shell(f"input keyevent {KEYCODES['enter']}")
        print(f"[telegram] sent via Enter key to {phone_number}")

    def send_signal_message(self, phone_number: str, message: str) -> None:
        """
        Signal doesn't support prefilled-message deep links the way
        WhatsApp/Telegram do; opens the app and leaves the rest to the
        UI-dump-based generic flow (open_app_and_do in the parser layer).
        """
        self.open_app("signal")
        time.sleep(2)
        print("[signal] app opened — Signal has no official prefill deep link; "
              "use the app UI or a chained open_app_and_do sequence.")

    def set_alarm(self, hour: int, minute: int = 0, label: str = "") -> None:
        """
        Uses Android's official ACTION_SET_ALARM intent — opens the default
        clock app with an alarm pre-filled at hour:minute (24h). Most clock
        apps set it immediately without further taps; some show a confirm
        screen first (normal Android behavior, not something ADB can/should
        bypass).
        """
        cmd = (
            f"am start -a android.intent.action.SET_ALARM "
            f"--ei android.intent.extra.alarm.HOUR {hour} "
            f"--ei android.intent.extra.alarm.MINUTES {minute} "
            f"--ez android.intent.extra.alarm.SKIP_UI true"
        )
        if label:
            encoded = label.replace('"', '\\"')
            cmd += f' --es android.intent.extra.alarm.MESSAGE "{encoded}"'
        self.bridge.shell(cmd)

    def set_timer(self, seconds: int, label: str = "") -> None:
        """Uses ACTION_SET_TIMER — starts a countdown timer for `seconds`."""
        cmd = (
            f"am start -a android.intent.action.SET_TIMER "
            f"--ei android.intent.extra.alarm.LENGTH {seconds} "
            f"--ez android.intent.extra.alarm.SKIP_UI true"
        )
        if label:
            encoded = label.replace('"', '\\"')
            cmd += f' --es android.intent.extra.alarm.MESSAGE "{encoded}"'
        self.bridge.shell(cmd)

    def open_settings_page(self, page: str) -> None:
        """
        Opens a specific system Settings screen by short name, using the
        official android.settings.* intent actions (all documented, no
        root). Unknown `page` values fall back to the main Settings screen.
        """
        pages = {
            "wifi": "android.settings.WIFI_SETTINGS",
            "bluetooth": "android.settings.BLUETOOTH_SETTINGS",
            "display": "android.settings.DISPLAY_SETTINGS",
            "sound": "android.settings.SOUND_SETTINGS",
            "apps": "android.settings.APPLICATION_SETTINGS",
            "battery": "android.intent.action.POWER_USAGE_SUMMARY",
            "storage": "android.settings.INTERNAL_STORAGE_SETTINGS",
            "location": "android.settings.LOCATION_SOURCE_SETTINGS",
            "security": "android.settings.SECURITY_SETTINGS",
            "accounts": "android.settings.SYNC_SETTINGS",
            "date_time": "android.settings.DATE_SETTINGS",
            "language": "android.settings.LOCALE_SETTINGS",
            "accessibility": "android.settings.ACCESSIBILITY_SETTINGS",
            "developer": "android.settings.APPLICATION_DEVELOPMENT_SETTINGS",
            "notifications": "android.settings.APP_NOTIFICATION_SETTINGS",
            "airplane": "android.settings.AIRPLANE_MODE_SETTINGS",
        }
        action = pages.get(page.lower().strip(), "android.settings.SETTINGS")
        self.bridge.shell(f"am start -a {action}")

    def share_text(self, text: str, app_package: Optional[str] = None) -> None:
        """Open the Android share sheet with `text`, optionally targeting one app."""
        encoded = urllib.parse.quote(text)
        cmd = (f'am start -a android.intent.action.SEND -t text/plain '
               f'--es android.intent.extra.TEXT "{encoded}"')
        if app_package:
            cmd += f" -p {app_package}"
        self.bridge.shell(cmd)

    # -- notifications / clipboard ------------------------------------------- #

    def get_notifications(self) -> str:
        """
        Dumps current notification state (best-effort text dump; Android
        doesn't expose a clean non-root notification-reading API via shell,
        but dumpsys notification gives a raw, parseable dump).
        """
        return self.bridge.shell("dumpsys notification --noredact")

    def clear_notifications(self) -> None:
        self.bridge.shell("service call notification 1")

    def set_clipboard(self, text: str) -> None:
        # requires a small helper: uses `am broadcast` to a clipboard-setting
        # service is not standard; simplest reliable route without root is
        # focusing a text field and using input text, OR using `adb shell`
        # with `cmd clipboard` (Android 10+ shell utility).
        escaped = text.replace('"', '\\"')
        try:
            self.bridge.shell(f'cmd clipboard set-clip {escaped}')
        except ADBError as e:
            raise ADBError(
                "Clipboard set failed — 'cmd clipboard' isn't available on "
                "this Android version. Not supported below Android 10."
            ) from e

    # -- media / misc ---------------------------------------------------------- #

    def play_pause_media(self) -> None:
        self.press_key("play_pause")

    def next_track(self) -> None:
        self.press_key("next_track")

    def prev_track(self) -> None:
        self.press_key("prev_track")

    def vibrate(self, ms: int = 300) -> None:
        self.bridge.shell(f"cmd vibrator vibrate {ms}")

    def get_current_app(self) -> str:
        """Returns the package name of the foreground app."""
        out = self.bridge.shell("dumpsys window | grep mCurrentFocus")
        match = re.search(r"([a-zA-Z0-9_.]+)/[a-zA-Z0-9_.]+\}", out)
        return match.group(1) if match else out.strip()

    def reboot(self) -> None:
        self.bridge.shell("reboot")

    # -- do not disturb / sound profile --------------------------------------- #

    def set_dnd(self, on: bool) -> None:
        """
        Toggle Do Not Disturb. Requires the Notification Policy Access
        permission to already be granted to the shell — Android grants
        this automatically for `cmd notification` calls from adb shell
        on most versions; if it silently fails, grant it once manually:
        Settings > Apps > Special access > Do Not Disturb access.
        """
        mode = "priority" if on else "off"
        self.bridge.shell(f"cmd notification set_dnd {mode}")

    def set_ringer_mode(self, mode: str) -> None:
        """mode: 'normal', 'silent', or 'vibrate'."""
        mode = mode.lower().strip()
        table = {"normal": "2", "vibrate": "1", "silent": "0"}
        if mode not in table:
            raise ValueError("ringer mode must be 'normal', 'silent', or 'vibrate'")
        self.bridge.shell(f"cmd audio set-ringer-mode {mode}")

    # -- rotation / display ---------------------------------------------------- #

    def set_auto_rotate(self, on: bool) -> None:
        self.bridge.shell(f"settings put system accelerometer_rotation {1 if on else 0}")

    def set_rotation(self, degrees: int) -> None:
        """degrees: 0, 90, 180, or 270. Auto-rotate should be off for this to stick."""
        table = {0: 0, 90: 1, 180: 2, 270: 3}
        if degrees not in table:
            raise ValueError("rotation degrees must be one of 0, 90, 180, 270")
        self.set_auto_rotate(False)
        self.bridge.shell(f"settings put system user_rotation {table[degrees]}")

    def set_screen_timeout(self, seconds: int) -> None:
        self.bridge.shell(f"settings put system screen_off_timeout {seconds * 1000}")

    # -- app management --------------------------------------------------------- #

    def clear_app_data(self, app_name: str) -> None:
        """Wipes an app's local data/cache, like Settings > App > Clear data."""
        package = self.resolve_package(app_name)
        self.bridge.shell(f"pm clear {package}")

    def force_stop_all_recent(self) -> None:
        """Best-effort: force-stop every currently backgrounded 3rd-party app."""
        installed = self.list_installed_packages()
        for pkg in installed:
            if not pkg.startswith(("com.android.", "com.google.android.gms",
                                    "com.google.android.gsf", "android")):
                try:
                    self.bridge.shell(f"am force-stop {pkg}")
                except ADBError:
                    pass

    def uninstall_app(self, app_name: str) -> str:
        package = self.resolve_package(app_name)
        return self.bridge.uninstall(package)

    def install_apk(self, local_apk_path: str) -> str:
        return self.bridge.install(local_apk_path)

    def open_app_settings(self, app_name: str) -> None:
        """Opens the system Settings page for a specific app (uninstall/perms/etc)."""
        package = self.resolve_package(app_name)
        self.bridge.shell(
            f'am start -a android.settings.APPLICATION_DETAILS_SETTINGS '
            f'-d "package:{package}"'
        )

    def split_screen_current(self) -> None:
        """
        Puts the current foreground app into split-screen (top half) via the
        recents/app_switch gesture equivalent. Behavior varies by OEM/Android
        version — this is best-effort, not guaranteed on every device/launcher.
        """
        self.press_key("app_switch")
        time.sleep(0.4)
        self.long_press(540, 1900, duration_ms=600)

    # -- scrolling helpers (built on swipe) -------------------------------------- #

    def scroll_up(self, amount: str = "medium") -> None:
        dist = {"small": 300, "medium": 700, "large": 1300}.get(amount, 700)
        self.swipe(540, 900, 540, 900 + dist, duration_ms=300)

    def scroll_down(self, amount: str = "medium") -> None:
        dist = {"small": 300, "medium": 700, "large": 1300}.get(amount, 700)
        self.swipe(540, 900 + dist, 540, 900, duration_ms=300)

    def scroll_left(self, amount: str = "medium") -> None:
        dist = {"small": 300, "medium": 700, "large": 1300}.get(amount, 700)
        self.swipe(200, 1200, 200 + dist, 1200, duration_ms=300)

    def scroll_right(self, amount: str = "medium") -> None:
        dist = {"small": 300, "medium": 700, "large": 1300}.get(amount, 700)
        self.swipe(200 + dist, 1200, 200, 1200, duration_ms=300)

    # -- text selection / editing helpers on the currently focused field -------- #

    def select_all_text(self) -> None:
        """
        Select-all in the currently focused text field. Uses `input
        keycombination` (Ctrl+A) where supported (Android 12+ with a
        physical/virtual keyboard event path); silently no-ops on older
        versions where that shell command doesn't exist — use
        clear_text_field() instead, which works everywhere.
        """
        try:
            self.bridge.shell("input keycombination 113 29")  # CTRL_LEFT + A
        except ADBError:
            pass

    def clear_text_field(self, max_chars: int = 250) -> None:
        """
        Clears the focused text field without relying on Ctrl+A (works on
        every Android version): moves the cursor to the end, then deletes
        backwards enough times to cover typical field lengths.
        """
        self.bridge.shell("input keyevent 123")  # MOVE_END
        # batch all DEL presses into one shell call instead of one-per-char
        dels = " ".join(["67"] * max_chars)
        self.bridge.shell(f"input keyevent {dels}")

    def paste_clipboard(self) -> None:
        """Paste into the focused field. Requires Android 12+ (input keycombination)."""
        try:
            self.bridge.shell("input keycombination 113 47")  # CTRL_LEFT + V
        except ADBError as e:
            raise ADBError(
                "Paste via key-combination isn't supported on this Android "
                "version. Use set_clipboard() + manual long-press-paste, or "
                "type_text() to type the content directly instead."
            ) from e

    def get_clipboard(self) -> str:
        try:
            return self.bridge.shell("cmd clipboard get-clip").strip()
        except ADBError as e:
            raise ADBError(
                "Clipboard get failed — 'cmd clipboard' isn't available on "
                "this Android version (needs Android 10+)."
            ) from e

    # -- screen content inspection (UI Automator dump) --------------------------- #

    def dump_ui(self) -> str:
        """
        Dumps the current screen's UI hierarchy as XML (uiautomator dump) —
        the standard, official, no-root way to see what's on screen: every
        visible element's text, content-description, class, and bounding
        box (bounds="[x1,y1][x2,y2]"). Used by find_text_on_screen()/
        tap_text() below, and useful on its own for debugging what a
        command actually sees.
        """
        remote = "/sdcard/_adb_ctrl_uidump.xml"
        self.bridge.shell(f"uiautomator dump {remote}")
        out = self.bridge.shell(f"cat {remote}")
        self.bridge.shell(f"rm {remote}")
        return out

    def find_text_on_screen(self, text: str) -> list[tuple[int, int]]:
        """
        Returns center (x, y) coordinates of every on-screen element whose
        visible text or content-description contains `text` (case-
        insensitive). Empty list if nothing matches. This is what lets
        commands like "tap the Send button" or "tap on Settings" work
        without you having to know pixel coordinates.

        Parses each XML node individually so attribute order doesn't matter
        (uiautomator dumps don't guarantee a fixed attribute sequence).
        """
        xml = self.dump_ui()
        needle = text.lower()
        results = []
        # Match every XML node tag (self-closing or open)
        for node in re.finditer(r'<node\b([^>]*?)/?\s*>', xml, re.DOTALL):
            attrs = node.group(1)

            def _attr(name: str) -> str:
                m = re.search(rf'{name}="([^"]*)"', attrs)
                return m.group(1) if m else ""

            label_text = _attr("text")
            desc = _attr("content-desc")
            resource_id = _attr("resource-id")
            bounds_raw = _attr("bounds")

            # Check text, content-desc, and resource-id for the needle
            if not (needle in label_text.lower()
                    or needle in desc.lower()
                    or needle in resource_id.lower()):
                continue

            bm = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_raw)
            if not bm:
                continue
            x1, y1, x2, y2 = int(bm.group(1)), int(bm.group(2)), int(bm.group(3)), int(bm.group(4))
            # Skip zero-size elements (invisible/hidden nodes)
            if x1 == x2 or y1 == y2:
                continue
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            results.append((cx, cy))
        return results

    def tap_text(self, text: str) -> bool:
        """
        Finds an on-screen element containing `text` and taps its center.
        Returns True if found and tapped, False if no match. Requires the
        target text to actually be visible/rendered on screen right now.
        """
        matches = self.find_text_on_screen(text)
        if not matches:
            return False
        x, y = matches[0]
        self.tap(x, y)
        return True


    # -- smart send (press Enter OR tap Send button — whichever works) --------- #

    def press_enter(self) -> None:
        """Press the Enter / Done / Go key (KEYCODE_ENTER). Works in most text fields."""
        self.bridge.shell(f"input keyevent {KEYCODES['enter']}")

    def smart_send(self, wait_before: float = 0.5) -> bool:
        """
        Intelligently submit whatever is in the currently focused field.
        Tries (in order):
          1. Tap a visible Send/Go/Search/Submit/Done button on screen.
          2. Press KEYCODE_ENTER.
        Returns True if a button was found and tapped, False if Enter was used.
        """
        time.sleep(wait_before)
        for label in ("Send", "Go", "Search", "Submit", "Done",
                      "Post", "send", "search", "OK", "Ok", "SEND"):
            if self.tap_text(label):
                return True
        self.press_enter()
        return False

    def type_and_send(self, text: str, wait_after_type: float = 0.3) -> None:
        """Type text into the focused field, then smart-send it."""
        self.type_text(text)
        time.sleep(wait_after_type)
        self.smart_send()

    # -- screen reading (OCR-free; returns visible text labels) --------------- #

    def read_screen_text(self) -> list[str]:
        """
        Returns a list of all non-empty visible text strings currently on screen
        (from the UI dump). Useful for checking what's on screen without OCR.
        """
        xml = self.dump_ui()
        texts = []
        for m in re.finditer(r'text="([^"]+)"', xml):
            val = m.group(1).strip()
            if val:
                texts.append(val)
        return texts

    def read_screen_summary(self) -> str:
        """Returns a short human-readable summary of visible text on screen."""
        items = self.read_screen_text()
        if not items:
            return "(nothing readable on screen)"
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = [x for x in items if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]
        return " | ".join(unique[:30])

    # -- device info ---------------------------------------------------------- #

    def get_device_info(self) -> dict:
        """Returns basic device info: model, Android version, resolution, etc."""
        model = self.bridge.shell("getprop ro.product.model").strip()
        version = self.bridge.shell("getprop ro.build.version.release").strip()
        sdk = self.bridge.shell("getprop ro.build.version.sdk").strip()
        manufacturer = self.bridge.shell("getprop ro.product.manufacturer").strip()
        resolution = self.bridge.shell("wm size").strip()
        density = self.bridge.shell("wm density").strip()
        ip = self.get_device_ip() or "unknown"
        return {
            "model": model,
            "manufacturer": manufacturer,
            "android_version": version,
            "sdk": sdk,
            "resolution": resolution,
            "density": density,
            "wifi_ip": ip,
        }

    def get_screen_resolution(self) -> tuple[int, int]:
        """Returns (width, height) of the physical display in pixels."""
        out = self.bridge.shell("wm size").strip()
        m = re.search(r'(\d+)x(\d+)', out)
        if m:
            return int(m.group(1)), int(m.group(2))
        return (1080, 2400)  # sane default

    # -- volume level (absolute) ---------------------------------------------- #

    def set_volume_level(self, stream: str, level: int) -> None:
        """
        Set an absolute volume level for a stream.
        stream: 'music', 'ring', 'alarm', 'notification', 'call'
        level: 0–15 (max varies by device; most streams go 0–15)
        Uses the `media volume` shell command (Android 9+).
        """
        stream_map = {
            "music": "music", "media": "music",
            "ring": "ring", "ringer": "ring",
            "alarm": "alarm",
            "notification": "notification",
            "call": "call", "voice": "call",
        }
        s = stream_map.get(stream.lower().strip(), "music")
        self.bridge.shell(f"media volume --stream {s} --set {level}")

    def get_volume_level(self, stream: str = "music") -> str:
        """Returns current volume info for the given stream (raw dumpsys output)."""
        out = self.bridge.shell("dumpsys audio")
        return out[:3000]  # truncated; full dump is large

    # -- hotspot / tethering -------------------------------------------------- #

    def set_wifi_hotspot(self, on: bool) -> None:
        """
        Toggle WiFi hotspot (tethering). Uses the svc command.
        Note: turning hotspot on typically turns WiFi client off on most devices.
        """
        state = "start" if on else "stop"
        self.bridge.shell(f"svc wifi {state}tethering")

    # -- display / dark mode -------------------------------------------------- #

    def set_dark_mode(self, on: bool) -> None:
        """Toggle dark mode (Android 10+). Uses the uimode night command."""
        mode = "yes" if on else "no"
        self.bridge.shell(f"cmd uimode night {mode}")

    def set_font_size(self, scale: float) -> None:
        """
        Set system font size scale. Normal=1.0, Large=1.15, Larger=1.3, Largest=1.45.
        """
        scale = round(max(0.85, min(1.6, scale)), 2)
        self.bridge.shell(f"settings put system font_scale {scale}")

    # -- status bar / notifications shade ------------------------------------- #

    def expand_notifications(self) -> None:
        """Pull down the notification shade."""
        self.bridge.shell("cmd statusbar expand-notifications")

    def collapse_notifications(self) -> None:
        """Collapse the notification shade."""
        self.bridge.shell("cmd statusbar collapse")

    def expand_quick_settings(self) -> None:
        """Pull down quick settings (two-finger pull or double-expand)."""
        self.bridge.shell("cmd statusbar expand-settings")

    # -- input method (keyboard) ---------------------------------------------- #

    def hide_keyboard(self) -> None:
        """Dismiss the soft keyboard if visible."""
        self.bridge.shell("input keyevent 111")  # KEYCODE_ESCAPE dismisses IME on most devices

    def show_keyboard(self) -> None:
        """Request soft keyboard to appear (works if a text field is focused)."""
        self.bridge.shell("input keyevent 120")  # KEYCODE_FUNCTION or use IME show

    # -- app info ------------------------------------------------------------- #

    def get_app_version(self, app_name: str) -> str:
        """Returns the installed version name for an app."""
        package = self.resolve_package(app_name)
        out = self.bridge.shell(f"dumpsys package {package} | grep versionName")
        return out.strip() or f"(version not found for {package})"

    def list_running_apps(self) -> list[str]:
        """Returns a list of currently running (foreground/background) app packages."""
        out = self.bridge.shell("dumpsys activity recents | grep 'packageName'")
        packages = re.findall(r'packageName=(\S+)', out)
        seen: set[str] = set()
        return [p for p in packages if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]

    # -- file operations (basic) ----------------------------------------------- #

    def list_files(self, path: str = "/sdcard/") -> list[str]:
        """List files in a directory on the device."""
        out = self.bridge.shell(f"ls {path}")
        return [f.strip() for f in out.splitlines() if f.strip()]

    def delete_file(self, path: str) -> None:
        """Delete a file from the device."""
        self.bridge.shell(f"rm -f {path}")

    # -- network info ---------------------------------------------------------- #

    def get_wifi_ssid(self) -> str:
        """Returns the SSID of the currently connected WiFi network."""
        out = self.bridge.shell("dumpsys wifi | grep 'mWifiInfo'")
        m = re.search(r'SSID: (.+?),', out)
        return m.group(1).strip() if m else "(not connected)"

    def ping(self, host: str = "8.8.8.8", count: int = 3) -> str:
        """Run a ping on the device and return results (tests device internet)."""
        out = self.bridge.shell(f"ping -c {count} {host}", timeout=15)
        return out.strip()


# Adjust this if your phone's WhatsApp "send" button isn't at this location.
# Default assumes a common ~1080x2400 portrait phone. Use screenshot() +
# an image viewer to find your device's exact coordinates if sends miss.
WHATSAPP_SEND_BUTTON_COORDS = (1000, 2150)
