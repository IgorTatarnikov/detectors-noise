"""The notebook and the slides run on a session in the app's format, and agree with noise_analysis.py.

Their analysis code stays inline so students can read it; this keeps it honest.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import noise_analysis as na  # noqa: E402
from recording import load_recording  # noqa: E402
from simulate_session import simulate_session  # noqa: E402

GAINS = [0, 1, 12, 24]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    path = tmp_path_factory.mktemp("live") / "practical"
    # An awkward gain step, as a real camera may have: 12 dB reads back as 11.99 dB
    cam = simulate_session(
        path, shape=(100, 140), frames=30, dark_frames=2, dark_exposures_ms=(0.04, 1, 100, 1000, 4000), gain_increment=0.0359
    )
    assert json.loads((path / "shot" / "light1" / "metadata.json").read_text())["gain"] != 12
    return path, cam


def run_code(chunks: list[str], session: Path) -> dict:
    """Run code chunks in one namespace, as a kernel would, with the data folder swapped in."""
    namespace: dict = {}
    plt.show = lambda *a, **k: plt.close("all")
    for code in chunks:
        code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith(("%", "#|")))
        code = code.replace("data_dir = 'data/live/practical/'", f"data_dir = '{session}/'")
        exec(compile(code, "<practical>", "exec"), namespace)
    plt.close("all")
    return namespace


def notebook_chunks() -> list[str]:
    nb = json.loads((ROOT / "noise_practical.ipynb").read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def slide_chunks() -> list[str]:
    text = (ROOT / "index.qmd").read_text()
    return re.findall(r"```\{python\}\n(.*?)```", text, re.S)


@pytest.fixture(scope="module", params=["notebook", "slides"])
def results(request, session):
    path, cam = session
    chunks = notebook_chunks() if request.param == "notebook" else slide_chunks()
    return run_code(chunks, path), path, cam


def test_no_two_frame_read_noise_method_left():
    for text in ((ROOT / "noise_practical.ipynb").read_text(), (ROOT / "index.qmd").read_text()):
        assert "frame_pair" not in text and "np.sqrt(2)" not in text


def test_read_noise_matches_module(results):
    ns, path, cam = results
    for i, g in enumerate(GAINS):
        stack, _ = load_recording(path / "read_noise" / f"{g}dB")
        expected = na.read_noise(stack[:, ::5, ::5]).read_noise
        assert ns["read_noise"][i] == pytest.approx(expected, rel=1e-4)
        assert ns["read_noise"][i] == pytest.approx((cam.read_noise_counts(g) ** 2 + 1 / 12) ** 0.5, rel=0.1)


def test_dark_counts_match_module(results):
    ns, path, cam = results
    for i, g in enumerate(GAINS):
        fit = ns["dark_count_results"][i][2]
        folders = sorted((path / "dark" / f"{g}dB").iterdir(), key=lambda f: json.loads((f / "metadata.json").read_text())["exposure_us"])
        exposures = [json.loads((f / "metadata.json").read_text())["exposure_us"] * 1e-6 for f in folders]
        means = [load_recording(f)[0].mean() for f in folders]
        assert fit.coef_[0] == pytest.approx(na.dark_counts(exposures, means).slope, rel=1e-3)


def test_shot_noise_matches_module_and_truth(results):
    ns, path, cam = results
    lit, _ = load_recording(path / "shot" / "light1")
    dark, _ = load_recording(path / "shot" / "light1_dark")
    expected = na.shot_noise(lit, dark).fit.slope
    assert ns["camera_gain"] == pytest.approx(expected, rel=1e-3)
    assert ns["camera_gain"] == pytest.approx(cam.gain_counts_per_e(12), rel=0.05)
    assert ns["saturated_value"] == 4095
