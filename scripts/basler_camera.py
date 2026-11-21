"""Basler camera access through pypylon, and a simulated camera with the same interface.

Both expose:
    model, serial, pixel_format, max_value, gain_unit
    limits[name] -> (min, max, increment) for "black_level", "gain", "exposure_us"
    set(name, value) -> the value the camera actually took
    settings() -> {"black_level", "gain", "exposure_us"} read back from the camera
    start(), stop(), grab(timeout_s) -> Frame, close()

On connect the real camera is put in Mono12 with every automatic or nonlinear
feature off (auto exposure, auto gain, gamma, LUT, digital shift), so counts are
the sensor's raw output.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

SETTINGS = ("black_level", "gain", "exposure_us")


@dataclass
class Frame:
    image: np.ndarray  # uint16, raw counts
    block_id: int  # the camera's frame counter, to spot dropped frames
    timestamp: int  # ns


class GrabError(RuntimeError):
    """A frame arrived incomplete."""


def snap(value: float, lo: float, hi: float, inc: float | None) -> float:
    value = min(max(value, lo), hi)
    if inc:
        value = lo + round((value - lo) / inc) * inc
    return min(max(value, lo), hi)


class BaslerCamera:
    # Node names differ between ace USB and GigE models; the first one present is used.
    NODES = {
        "black_level": ("BlackLevel", "BlackLevelRaw"),
        "gain": ("Gain", "GainRaw"),
        "exposure_us": ("ExposureTime", "ExposureTimeAbs"),
    }
    # Applied in order, skipping any node this model lacks. Selectors come first.
    FIXED = (
        ("GainSelector", "All"),
        ("BlackLevelSelector", "All"),
        ("ExposureAuto", "Off"),
        ("GainAuto", "Off"),
        ("ExposureMode", "Timed"),
        ("LUTEnable", False),
        ("GammaEnable", False),
        ("Gamma", 1.0),
        ("DigitalShift", 0),
        ("AcquisitionFrameRateEnable", False),
        ("TriggerMode", "Off"),
    )

    def __init__(self, serial: str | None = None):
        from pypylon import pylon

        self._pylon = pylon
        tl = pylon.TlFactory.GetInstance()
        devices = list(tl.EnumerateDevices())
        if serial:
            devices = [d for d in devices if d.GetSerialNumber() == serial]
        if not devices:
            raise RuntimeError(
                "No Basler camera found. Check it is plugged into a USB 3 port and that "
                "pylon Viewer (or another program) is not holding it."
            )
        info = devices[0]
        self.cam = pylon.InstantCamera(tl.CreateDevice(info))
        self.cam.Open()
        self.model = info.GetModelName()
        self.serial = info.GetSerialNumber()
        self.nodes = self.cam.GetNodeMap()
        self._names = {}
        for key, candidates in self.NODES.items():
            for name in candidates:
                if self._node(name) is not None:
                    self._names[key] = name
                    break
            else:
                raise RuntimeError(f"{self.model} has none of the nodes {candidates}")
        self.gain_unit = "dB" if self._names["gain"] == "Gain" else "raw"
        self._configure()
        self.limits = {key: self._limits(key) for key in SETTINGS}

    def _node(self, name: str):
        from pypylon import genicam

        try:
            node = self.nodes.GetNode(name)
        except genicam.GenericException:
            return None
        if node is None or not genicam.IsAvailable(node):
            return None
        return node

    def _try_set(self, name: str, value) -> bool:
        from pypylon import genicam

        node = self._node(name)
        if node is None or not genicam.IsWritable(node):
            return False
        try:
            node.Value = value
            return True
        except genicam.GenericException:
            return False

    def _configure(self):
        node = self._node("PixelFormat")
        if node is None or "Mono12" not in node.Symbolics:
            raise RuntimeError(f"{self.model} does not offer the unpacked Mono12 pixel format")
        node.Value = "Mono12"
        self.pixel_format = "Mono12"
        self.max_value = 4095
        for name, value in self.FIXED:
            self._try_set(name, value)
        # Enough buffers that writing frames to disk doesn't make the camera drop any.
        self.cam.MaxNumBuffer.Value = 64

    def _limits(self, key: str) -> tuple[float, float, float | None]:
        node = self._node(self._names[key])
        try:
            inc = float(node.Inc)
        except Exception:
            inc = None
        return float(node.Min), float(node.Max), inc

    def get(self, key: str) -> float:
        return float(self._node(self._names[key]).Value)

    def set(self, key: str, value: float) -> float:
        from pypylon import genicam

        lo, hi, inc = self.limits[key]
        value = snap(value, lo, hi, inc)
        node = self._node(self._names[key])
        if isinstance(node.Value, int):
            value = int(round(value))
        try:
            node.Value = value
        except genicam.AccessException:
            # Some models only allow this while stopped.
            grabbing = self.cam.IsGrabbing()
            self.stop()
            node.Value = value
            if grabbing:
                self.start()
        return self.get(key)

    def settings(self) -> dict:
        return {key: self.get(key) for key in SETTINGS}

    def temperature(self) -> float | None:
        node = self._node("DeviceTemperature") or self._node("TemperatureAbs")
        try:
            return float(node.Value) if node is not None else None
        except Exception:
            return None

    def start(self):
        if not self.cam.IsGrabbing():
            self.cam.StartGrabbing(self._pylon.GrabStrategy_OneByOne)

    def stop(self):
        if self.cam.IsGrabbing():
            self.cam.StopGrabbing()

    def grab(self, timeout_s: float) -> Frame:
        res = self.cam.RetrieveResult(int(timeout_s * 1000), self._pylon.TimeoutHandling_ThrowException)
        try:
            if not res.GrabSucceeded():
                raise GrabError(res.GetErrorDescription())
            return Frame(res.Array.copy(), int(res.BlockID), int(res.TimeStamp))
        finally:
            res.Release()

    def close(self):
        try:
            self.stop()
        finally:
            self.cam.Close()


class SimulatedCamera:
    """A 12-bit camera with known bias, gain, read noise, dark current and full well.

    Defaults are close to an ace acA1920-40um at reduced size. Set `covered` to block
    the light, and `light` for the photoelectron rate (e-/s) at the brightest pixel.
    Each pixel's counts are
        floor(K * e + read noise + black level + fixed pattern), clipped to 0..4095
    with e ~ Poisson((light * pattern + dark) * t) capped at the full well and
    K = base_gain * 10**(gain_db / 20) counts/e-.
    """

    model = "Simulated acA1920-40um"
    serial = "SIM0001"
    pixel_format = "Mono12"
    max_value = 4095
    gain_unit = "dB"

    def __init__(
        self,
        shape: tuple[int, int] = (300, 480),
        *,
        base_gain: float = 0.126,
        read_noise_e: float = 6.9,
        adc_noise: float = 0.3,
        dark_e_per_s: float = 5.0,
        full_well: float = 32_800,
        dsnu: float = 0.3,
        light: float = 0.0,
        light_jitter: float = 0.002,
        max_fps: float = 40.0,
        gain_increment: float = 0.1,
        seed: int | None = None,
        realtime: bool = True,
    ):
        self.shape = shape
        self.base_gain = base_gain
        self.read_noise_e = read_noise_e
        self.adc_noise = adc_noise
        self.dark_e_per_s = dark_e_per_s
        self.full_well = full_well
        self.light = light
        self.light_jitter = light_jitter
        self.covered = True
        self.max_fps = max_fps
        self.realtime = realtime
        self.rng = np.random.default_rng(seed)
        self.limits = {"black_level": (0.0, 511.0, 1.0), "gain": (0.0, 24.0, gain_increment), "exposure_us": (21.0, 10e6, 1.0)}
        self._values = {"black_level": 0.0, "gain": 0.0, "exposure_us": 10_000.0}
        rows, cols = shape
        y, x = np.mgrid[0:rows, 0:cols]
        r = np.hypot(y - rows / 2, x - cols / 2) / (0.5 * math.hypot(rows, cols))
        # A structured scene: a radial falloff with rings, from about 5% to 100% brightness.
        self.pattern = (0.05 + 0.95 * (1 - r) ** 2 * (0.6 + 0.4 * np.cos(14 * r) ** 2)).astype(np.float64)
        self.fixed_pattern = self.rng.normal(0, dsnu, shape)
        self._grabbing = False
        self._block_id = 0
        self._next_time = 0.0

    def gain_counts_per_e(self, gain_db: float | None = None) -> float:
        gain_db = self._values["gain"] if gain_db is None else gain_db
        return self.base_gain * 10 ** (gain_db / 20)

    def read_noise_counts(self, gain_db: float | None = None) -> float:
        return math.hypot(self.read_noise_e * self.gain_counts_per_e(gain_db), self.adc_noise)

    def get(self, key: str) -> float:
        return self._values[key]

    def set(self, key: str, value: float) -> float:
        self._values[key] = snap(value, *self.limits[key])
        return self._values[key]

    def settings(self) -> dict:
        return dict(self._values)

    def temperature(self) -> float | None:
        return 35.0

    def start(self):
        self._grabbing = True
        self._next_time = time.monotonic()

    def stop(self):
        self._grabbing = False

    def render(self) -> np.ndarray:
        t = self._values["exposure_us"] * 1e-6
        flux = 0.0 if self.covered else self.light * (1 + self.light_jitter * self.rng.standard_normal())
        electrons = self.rng.poisson((flux * self.pattern + self.dark_e_per_s) * t)
        electrons = np.minimum(electrons, self.full_well)
        k = self.gain_counts_per_e()
        noise_e = self.rng.normal(0, self.read_noise_e, self.shape)
        counts = k * (electrons + noise_e) + self.rng.normal(0, self.adc_noise, self.shape)
        counts += self._values["black_level"] + self.fixed_pattern
        return np.clip(np.floor(counts), 0, self.max_value).astype(np.uint16)

    def grab(self, timeout_s: float) -> Frame:
        if not self._grabbing:
            raise RuntimeError("not grabbing")
        period = max(self._values["exposure_us"] * 1e-6, 1 / self.max_fps)
        if self.realtime:
            wait = self._next_time + period - time.monotonic()
            if wait > timeout_s:
                time.sleep(timeout_s)
                raise TimeoutError("grab timed out")
            if wait > 0:
                time.sleep(wait)
            self._next_time = max(self._next_time + period, time.monotonic() - period)
        self._block_id += 1
        return Frame(self.render(), self._block_id, time.monotonic_ns())

    def close(self):
        self.stop()


def camera_metadata(cam) -> dict:
    """Everything saved with a recording about the camera and its current settings."""
    return {
        **cam.settings(),
        "gain_unit": cam.gain_unit,
        "camera_model": cam.model,
        "camera_serial": cam.serial,
        "pixel_format": cam.pixel_format,
        "bit_depth": 12,
        "max_value": cam.max_value,
        "temperature_c": cam.temperature(),
    }


def open_camera(simulate: bool = False, serial: str | None = None):
    return SimulatedCamera() if simulate else BaslerCamera(serial)
