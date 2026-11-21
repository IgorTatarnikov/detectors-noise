"""A simulated SiPM on a USB-201, with the same interface as usb201.USB201.

Lets sipm_daq_live.py and its tests run without hardware. The detector is modelled as
Poisson pulses, each an exponential decay of height H and time constant tau, sampled
instantaneously at the DAQ rate and rounded to the USB-201's 12-bit codes.

State the GUI (or a test) can change:
    bias_step   1-10, the detector's knob. Pulse height, dark rate and photon detection
                rise with it above breakdown.
    connected   False = the detector unplugged, leaving only the DAQ's own noise.
    covered     True = no light reaches the SiPM.
    light       photon rate on the SiPM when uncovered, in photons/s.
"""

from __future__ import annotations

import math
import time

import numpy as np

from usb201 import MAX_RATE, codes_to_volts, pacer_period

LSB_MV = 20_000 / 4096


class SipmModel:
    """Pulse height, rates and crosstalk against the bias knob."""

    def __init__(
        self,
        *,
        breakdown_step: float = 2.5,
        height_per_step_mv: float = 40.0,
        dark_rate_per_step2: float = 2e4,
        pde_per_step: float = 0.06,
        crosstalk_per_step: float = 0.02,
        tau_s: float = 50e-9,
        baseline_mv: float = 3.9,
        read_noise_mv: float = 0.16,
        ceiling_mv: float = 3600.0,
    ):
        self.breakdown_step = breakdown_step
        self.height_per_step_mv = height_per_step_mv
        self.dark_rate_per_step2 = dark_rate_per_step2
        self.pde_per_step = pde_per_step
        self.crosstalk_per_step = crosstalk_per_step
        self.tau_s = tau_s
        self.baseline_mv = baseline_mv
        self.read_noise_mv = read_noise_mv
        self.ceiling_mv = ceiling_mv

    def overvoltage(self, step: float) -> float:
        return max(0.0, step - self.breakdown_step)

    def height_mv(self, step: float) -> float:
        return self.height_per_step_mv * self.overvoltage(step)

    def dark_rate(self, step: float) -> float:
        return self.dark_rate_per_step2 * self.overvoltage(step) ** 2

    def pde(self, step: float) -> float:
        return min(0.5, self.pde_per_step * self.overvoltage(step))

    def crosstalk(self, step: float) -> float:
        return min(0.4, self.crosstalk_per_step * self.overvoltage(step))

    def h_eff_mv(self, step: float) -> float:
        """integral(h^2) / integral(h) for exponential pulses, with crosstalk doubling some."""
        p = self.crosstalk(step)
        return self.height_mv(step) / 2 * (1 + 3 * p) / (1 + p)


def simulate_mv(
    n: int,
    rate: float,
    pulse_rate: float,
    height_mv: float,
    rng: np.random.Generator,
    *,
    tau_s: float = 50e-9,
    crosstalk: float = 0.0,
    baseline_mv: float = 3.9,
    read_noise_mv: float = 0.16,
    ceiling_mv: float = math.inf,
) -> np.ndarray:
    """n instantaneous samples of a Poisson pulse train plus Gaussian noise, in mV.

    Pulses shorter than the sample spacing only affect the sample they precede, so each
    sample sees Poisson(pulse_rate * window) pulses of random age in the window before it.
    """
    window = min(10 * tau_s, 1 / rate)
    counts = rng.poisson(pulse_rate * window, n)
    total = int(counts.sum())
    signal = np.zeros(n)
    if total and height_mv > 0:
        index = np.repeat(np.arange(n), counts)
        ages = rng.uniform(0, window, total)
        heights = height_mv * (1 + (rng.random(total) < crosstalk))
        signal = np.bincount(index, heights * np.exp(-ages / tau_s), minlength=n)
    mv = baseline_mv + signal + rng.normal(0, read_noise_mv, n)
    return np.minimum(mv, ceiling_mv)


def mv_to_codes(mv: np.ndarray) -> np.ndarray:
    """Inverse of codes_to_volts with the identity calibration."""
    return np.clip(np.round(np.asarray(mv) / 1e3 * 2048 / 10 + 2048), 0, 4095).astype(np.uint16)


class SimulatedDAQ:
    cal = [(1.0, 0.0)] * 8

    def __init__(self, model: SipmModel | None = None, seed: int | None = None, realtime: bool = True):
        self.model = model or SipmModel()
        self.rng = np.random.default_rng(seed)
        self.realtime = realtime
        self.bias_step = 5.0
        self.connected = True
        self.covered = True
        self.light = 2e5
        self._rate = MAX_RATE
        self._running = False
        self._t_last = 0.0
        self._owed = 0.0

    def serial(self) -> str:
        return "SIMDAQ01"

    def to_volts(self, codes, channel: int) -> np.ndarray:
        return codes_to_volts(codes, *self.cal[channel])

    def pulse_rate(self) -> float:
        if not self.connected:
            return 0.0
        step = self.bias_step
        light = 0.0 if self.covered else self.light * self.model.pde(step)
        return self.model.dark_rate(step) + light

    def generate(self, n: int) -> np.ndarray:
        m = self.model
        if not self.connected:
            mv = m.baseline_mv + self.rng.normal(0, m.read_noise_mv, n)
        else:
            mv = simulate_mv(
                n,
                self._rate,
                self.pulse_rate(),
                m.height_mv(self.bias_step),
                self.rng,
                tau_s=m.tau_s,
                crosstalk=m.crosstalk(self.bias_step),
                baseline_mv=m.baseline_mv,
                read_noise_mv=m.read_noise_mv,
                ceiling_mv=m.ceiling_mv,
            )
        return mv_to_codes(mv)

    # --- the USB201 interface used by daq_live.Reader -------------------------
    def start_continuous(self, channel: int, rate: float) -> float:
        self._rate = pacer_period(rate)[1]
        self._running = True
        self._t_last = time.monotonic()
        self._owed = 0.0
        return self._rate

    def read_available(self, max_bytes: int = 4096, timeout_ms: int = 500) -> np.ndarray:
        if not self._running:
            return np.empty(0, dtype="<u2")
        max_n = max_bytes // 2
        if not self.realtime:
            return self.generate(max_n)
        deadline = time.monotonic() + timeout_ms / 1000
        while True:
            now = time.monotonic()
            self._owed += (now - self._t_last) * self._rate
            self._t_last = now
            n = int(min(self._owed, max_n))
            if n >= min(max_n, 256) or (n > 0 and now >= deadline):
                self._owed -= n
                return self.generate(n)
            if now >= deadline:
                return np.empty(0, dtype="<u2")
            time.sleep(0.002)

    def scan(self, channel: int, n: int, rate: float):
        self._rate = pacer_period(rate)[1]
        return self.generate(n), self._rate

    def stop_scan(self):
        self._running = False

    def close(self):
        self.stop_scan()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
