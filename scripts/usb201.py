"""Minimal pyusb driver for the MCC USB-201 DAQ.

Protocol from Warren Jasper's MCC Linux drivers (wjasper/Linux_Drivers, usb-20X.c).
Needs only pyusb and libusb-package, so it runs on macOS without MCC's own library.

The USB-201 is 12-bit, fixed +/-10 V (about 4.9 mV per code), at most 100 kS/s.
"""

from __future__ import annotations

import struct
import time

import libusb_package
import numpy as np
import usb.core
import usb.util

VID, PID = 0x09DB, 0x0113
IN_REQ, OUT_REQ = 0xC0, 0x40  # vendor request to device: device->host, host->device
AIN, AIN_SCAN_START, AIN_SCAN_STOP, AIN_SCAN_CLR_FIFO = 0x10, 0x11, 0x12, 0x15
CAL_MEMORY, STATUS, SERIAL = 0x30, 0x44, 0x48
AIN_SCAN_RUNNING, AIN_SCAN_OVERRUN = 1 << 1, 1 << 2
EP_IN = 0x81
PACKET = 64
BASE_CLOCK = 70e6
MAX_RATE = 100_000
N_CHANNELS = 8
LSB_VOLTS = 20.0 / 4096  # one ADC code


class ScanOverrun(RuntimeError):
    """The device FIFO filled because the host did not read fast enough."""


def pacer_period(rate: float) -> tuple[int, float]:
    """Timer period for a requested sample rate, and the rate it actually gives."""
    if not 0 < rate <= MAX_RATE:
        raise ValueError(f"rate must be in (0, {MAX_RATE}] Hz")
    period = int(round(BASE_CLOCK / rate - 1))
    return period, BASE_CLOCK / (period + 1)


def codes_to_volts(codes, slope: float, intercept: float) -> np.ndarray:
    """Apply the factory calibration, then map 0..4095 onto -10..+10 V."""
    cal = np.asarray(codes, dtype=float) * slope + intercept
    return (cal - 2048.0) * 10.0 / 2048.0


class USB201:
    def __init__(self):
        dev = usb.core.find(idVendor=VID, idProduct=PID, backend=libusb_package.get_libusb1_backend())
        if dev is None:
            raise RuntimeError("USB-201 not found. Is it plugged in?")
        self.dev = dev
        try:
            dev.set_configuration()
            usb.util.claim_interface(dev, 0)
        except usb.core.USBError as e:
            raise RuntimeError(
                "Can't open the USB-201. Another program is probably using it, "
                "such as an open daq_live.py window. Close that and try again."
            ) from e
        raw = bytes(dev.ctrl_transfer(IN_REQ, CAL_MEMORY, 0, 0, N_CHANNELS * 8))
        cal = struct.unpack(f"<{2 * N_CHANNELS}f", raw)
        self.cal = [(cal[2 * i], cal[2 * i + 1]) for i in range(N_CHANNELS)]  # (slope, intercept)
        self._leftover = b""
        self.stop_scan()

    # --- housekeeping -------------------------------------------------------
    def serial(self) -> str:
        return bytes(self.dev.ctrl_transfer(IN_REQ, SERIAL, 0, 0, 8)).decode()

    def status(self) -> int:
        return struct.unpack("<H", bytes(self.dev.ctrl_transfer(IN_REQ, STATUS, 0, 0, 2)))[0]

    def to_volts(self, codes, channel: int) -> np.ndarray:
        return codes_to_volts(codes, *self.cal[channel])

    def close(self):
        try:
            self.stop_scan()
        finally:
            usb.util.dispose_resources(self.dev)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- single reads -------------------------------------------------------
    def ain(self, channel: int) -> int:
        return struct.unpack("<H", bytes(self.dev.ctrl_transfer(IN_REQ, AIN, channel, 0, 2)))[0]

    # --- hardware-paced scans ----------------------------------------------
    def _start(self, channel: int, count: int, rate: float) -> float:
        period, actual = pacer_period(rate)
        self.stop_scan()
        # count u32 (0 = continuous), pacer_period u32, channel mask, options, trig source, trig mode
        payload = struct.pack("<IIBBBB", count, period, 1 << channel, 0, 0, 0)
        self.dev.ctrl_transfer(OUT_REQ, AIN_SCAN_START, 0, 0, payload)
        self._leftover = b""
        return actual

    def stop_scan(self):
        """Stop any scan, then empty the endpoint and FIFO so the next scan starts clean."""
        self.dev.ctrl_transfer(OUT_REQ, AIN_SCAN_STOP, 0, 0, None)
        try:
            self.dev.clear_halt(EP_IN)  # an overrun stalls the endpoint
        except usb.core.USBError:
            pass
        while True:
            try:
                if not self.dev.read(EP_IN, 4096, timeout=50):
                    break
            except usb.core.USBError:
                break
        self.dev.ctrl_transfer(OUT_REQ, AIN_SCAN_CLR_FIFO, 0, 0, None)

    def scan(self, channel: int, n: int, rate: float) -> tuple[np.ndarray, float]:
        """Fixed-length scan of one channel. Returns (raw codes, actual rate)."""
        actual = self._start(channel, n, rate)
        nbytes = 2 * n
        buf = bytearray()
        timeout_ms = int(1000 * n / actual) + 2000
        deadline = time.monotonic() + timeout_ms / 1000
        while len(buf) < nbytes:
            if time.monotonic() > deadline:
                raise TimeoutError(f"got {len(buf)} of {nbytes} bytes")
            buf.extend(self.dev.read(EP_IN, min(16384, nbytes - len(buf) + PACKET), timeout=timeout_ms))
        overrun = self.status() & AIN_SCAN_OVERRUN
        self.stop_scan()
        if overrun:
            raise ScanOverrun("AIn scan overrun")
        return np.frombuffer(bytes(buf[:nbytes]), dtype="<u2").copy(), actual

    def start_continuous(self, channel: int, rate: float) -> float:
        """Start a scan that runs until stop_scan(). Returns the actual rate."""
        return self._start(channel, 0, rate)

    def read_available(self, max_bytes: int = 4096, timeout_ms: int = 500) -> np.ndarray:
        """Raw codes from a running continuous scan; may be empty on timeout.

        Call often: the device FIFO only holds a few tens of ms at 100 kS/s.
        """
        try:
            data = bytes(self.dev.read(EP_IN, max_bytes, timeout=timeout_ms))
        except usb.core.USBTimeoutError:
            return np.empty(0, dtype="<u2")
        except usb.core.USBError as e:
            if self.status() & AIN_SCAN_OVERRUN:
                raise ScanOverrun("AIn scan overrun: host fell behind") from e
            raise
        data = self._leftover + data
        even = len(data) - len(data) % 2
        self._leftover = data[even:]
        return np.frombuffer(data[:even], dtype="<u2").copy()


if __name__ == "__main__":
    CH = 7
    with USB201() as d:
        print("serial:", d.serial())
        raw, rate = d.scan(CH, 10_000, 100_000)
        v = d.to_volts(raw, CH)
        print(f"ch{CH}: {raw.size} samples @ {rate:.0f} Hz, mean {v.mean()*1e3:.2f} mV, sd {v.std()*1e3:.2f} mV")
