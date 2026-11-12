# SiPM acquisition on macOS — context notes

Summary of design decisions for acquiring data from a Thorlabs SiPM amplified
detector (PDA4x series, e.g. PDA43/PDA44/PDA45) with a Rigol DS1074Z
oscilloscope, controlled from Python on macOS.

## Hardware constraints

- **Thorlabs PDA4x SiPM detectors have no computer interface.** Bias level is
  set by a physical knob (steps 1–10); the TEC is self-regulating. Python can
  only *read* the analog output, not control the detector.
- **NI USB DAQs are not usable on modern macOS.** NI-DAQmx is Windows/Linux
  only; NI-DAQmx Base (the old macOS driver) is EOL and limited to macOS ≤ 10.14.
  The USB protocol is closed, so pyusb is not an option. NI hardware is
  x64-only, so a Windows-on-ARM VM on Apple silicon does not work either.
  Workarounds, if ever needed: run `nidaqmx` on a separate x86 Windows/Linux box
  and stream data over the network, or use a macOS-supported DAQ (e.g. LabJack).
- **Chosen digitiser: Rigol DS1074Z oscilloscope** via `pyvisa` + `pyvisa-py`.

## DS1074Z connection

- Prefer **LAN**: `TCPIP::<ip>::INSTR` (or `TCPIP::<ip>::5555::SOCKET`).
- USB-TMC also works: `brew install libusb`, `pip install pyusb`.
- Install: `pip install pyvisa pyvisa-py numpy scipy`.
- Use `pyvisa.ResourceManager("@py")`.

## DS1074Z limits relevant to SiPM pulses

- **Input is 1 MΩ only — no 50 Ω option.** If the detector output expects a
  50 Ω load, use a BNC feed-through 50 Ω terminator at the scope input.
- **70 MHz bandwidth**: ns-scale edges are attenuated; absolute pulse heights
  may be underestimated, but photon-number peaks usually still separate.
- **Sample rate is shared across channels**: 1 GSa/s (1 ch), 500 MSa/s (2 ch),
  250 MSa/s (3–4 ch). Use a single channel.
- **Memory**: 12 Mpts standard. Max sample rate ≈ min(1 GSa/s, memory / window).
- **Readout is slow** (seconds for deep memory) and the scope is not acquiring
  while transferring — captures are snapshots, not a continuous stream.
- `NORM` waveform mode returns screen data (1200 pts). `RAW` mode returns full
  memory but requires the scope to be stopped, and BYTE reads are capped at
  ~250k points per request, so read in chunks.

### Reading full memory (RAW, chunked)

```python
import numpy as np
import pyvisa

rm = pyvisa.ResourceManager("@py")
s = rm.open_resource("TCPIP::192.168.1.50::INSTR")
s.timeout = 20_000

s.write(":STOP")
s.write(":WAV:SOUR CHAN1")
s.write(":WAV:MODE RAW")
s.write(":WAV:FORM BYTE")

pre = [float(x) for x in s.query(":WAV:PRE?").split(",")]
npts = int(pre[2])
xinc, xorig, xref, yinc, yorig, yref = pre[4:10]

CHUNK = 250_000
chunks = []
for start in range(1, npts + 1, CHUNK):
    stop = min(start + CHUNK - 1, npts)
    s.write(f":WAV:STAR {start}")
    s.write(f":WAV:STOP {stop}")
    chunks.append(s.query_binary_values(":WAV:DATA?", datatype="B", container=np.array))

raw = np.concatenate(chunks).astype(float)
volts = (raw - yorig - yref) * yinc
t = np.arange(npts) * xinc + xorig
s.write(":RUN")
```

Set memory depth with `:ACQ:MDEP` before acquiring; use only as much as needed.

## Measurement goal

"Exposures" of 1 ms – 10 s, integrating the signal, against a **constant source**
or **room lights** (not a pulsed laser).

### Key insight: the source is stationary

The SiPM output is a Poisson stream of single-photon pulses plus dark counts.
The quantity of interest is the **photon rate**; an exposure of length T is
`rate × T`. Because the source doesn't change, gaps between captures don't
matter. So:

- Take **short captures at full sample rate** (1–10 ms at ~1 GSa/s) so every
  pulse is resolved.
- **Repeat N times** until total captured time gives the desired precision
  (Poisson noise ∝ 1/√counts).
- Derive the equivalent signal for any exposure T from the rate.

Avoid single long captures: at 1 s the sample interval is ~80 ns, at 10 s ~1 µs,
so ns-wide pulses are missed or aliased and the integral becomes noisy/biased.

### Counting vs integrating

- **Low light (pulses mostly separate):** count pulses with
  `scipy.signal.find_peaks`, threshold ≈ half the single-photon pulse height.
  Most precise.
- **Brighter light (pulses overlap):** use mean voltage
  (`integral = sum(volts - baseline) * xinc`, i.e. mean × duration).
  An RC low-pass at the input (τ ≫ pulse width, ≪ exposure, e.g. 10 µs–1 ms)
  preserves the mean and allows slow sampling.
- The scope's `:MEAS:ITEM? VAVG,CHAN1` is a fast way to get the mean, but only
  valid when data is well sampled (or low-passed). High Resolution acquisition
  mode may also help at slow timebases — validate against a well-sampled capture.

### Dark subtraction

Dark count rates are kcps to hundreds of kcps depending on model and bias.
Always take a **dark set** (source blocked, same settings) and subtract its rate
or integral — not just a DC baseline.

### Room-light caveats

- **Saturation:** room lighting is many orders of magnitude too bright for a
  single-photon detector. Check the max input power in the PDA4x manual; use
  strong ND filtering, an enclosure, or a pinhole/fibre.
- **Flicker:** UK mains lighting flickers at **100 Hz**; many LED fittings use
  kHz PWM. Make captures an integer number of 10 ms periods, or average many
  captures at random phase. A DC-driven LED / stabilised lamp avoids this.

## Suggested procedure (constant source)

1. Set memory depth so a 10 ms capture runs at full sample rate (single channel).
2. Acquire a dark set of N captures, then a light set of N captures.
3. Per capture: count pulses (low light) or compute mean voltage (bright).
4. Report dark-subtracted rate ± standard error; scale to required exposure
   times.

## Alternatives if requirements grow

- Continuous high-rate photon counting / long gapless acquisitions: a counter,
  time tagger, or Red Pitaya (TCP-controlled, OS-independent; see
  github.com/nybergflow/piccolo for a 4-SiPM Red Pitaya example).
- Simple slow intensity monitoring: RC low-pass + LabJack (native macOS + Python).
