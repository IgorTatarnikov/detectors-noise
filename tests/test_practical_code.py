"""The notebook and the slides' gain plot run on a session in the app's format, and the notebook agrees with noise_analysis.py.

The notebook's analysis code stays inline so students can read it; this keeps it honest.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import tifffile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import noise_analysis as na  # noqa: E402
from recording import load_recording  # noqa: E402
from simulate_session import simulate_session  # noqa: E402
from summarise_session import main as summarise_session  # noqa: E402

GAINS = [0, 1, 12, 24]


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    path = tmp_path_factory.mktemp("live") / "practical"
    # An awkward gain step, as a real camera may have: 12 dB reads back as 11.99 dB
    cam = simulate_session(
        path, shape=(100, 140), frames=30, dark_frames=2, dark_exposures_ms=(0.04, 1, 100, 1000, 4000), gain_increment=0.0359
    )
    assert json.loads((path / "shot" / "12dB" / "metadata.json").read_text())["gain"] != 12
    summarise_session([str(path), str(path / "summary.json")])
    return path, cam


def run_code(chunks: list[str], session: Path) -> dict:
    """Run code chunks in one namespace, as a kernel would, with the data folder swapped in."""
    namespace: dict = {}
    plt.show = lambda *a, **k: plt.close("all")
    for code in chunks:
        code = "\n".join(line for line in code.splitlines() if not line.lstrip().startswith(("%", "#|")))
        code = code.replace("data_dir = 'data/live/practical/'", f"data_dir = '{session}/'")
        code = code.replace("results_path = 'results/practical.json'", f"results_path = '{session}/summary.json'")
        exec(compile(code, "<practical>", "exec"), namespace)
    plt.close("all")
    return namespace


def notebook_chunks() -> list[str]:
    nb = json.loads((ROOT / "noise_practical.ipynb").read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def slide_chunks() -> list[str]:
    text = (ROOT / "index.qmd").read_text()
    return re.findall(r"```\{python\}\n(.*?)```", text, re.S)


@pytest.fixture(scope="module")
def results(session):
    path, cam = session
    return run_code(notebook_chunks(), path), path, cam


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
        means = [(lambda st: st[na.steady_frames(st.mean(axis=(1, 2)))].mean())(load_recording(f)[0]) for f in folders]
        assert fit.coef_[0] == pytest.approx(na.dark_counts(exposures, means).slope, rel=1e-3)


def test_dark_counts_skip_an_offset_jump(results, capsys):
    ns = results[0]
    capsys.readouterr()
    frames = np.full((10, 4, 4), 127.2, dtype=np.float32)
    frames[:2] += 2  # the offset sat 2 counts higher for the first two frames
    assert ns["steady_frames"](frames).shape[0] == 8
    assert ns["steady_frames"](frames).mean() == pytest.approx(127.2)
    assert capsys.readouterr().out == ""


def test_notebook_read_noise_skips_an_offset_jump(results, tmp_path, capsys):
    ns, path, _ = results
    for gain in GAINS:  # a 2 count jump after the first frame, at one gain
        folder = tmp_path / f"{gain}dB"
        folder.mkdir()
        stack, _ = load_recording(path / "read_noise" / f"{gain}dB")
        if gain == 1:
            stack[0] += 2
        for i, frame in enumerate(stack):
            tifffile.imwrite(folder / f"frame_{i:04d}.tif", frame)
    capsys.readouterr()
    plt.show = lambda *a, **k: plt.close("all")
    noise, bias = ns["plot_readout_noise_multiple_gain"](GAINS, f"{tmp_path}/", 5)
    stack, _ = load_recording(path / "read_noise" / "1dB")
    assert bias[1] == pytest.approx(stack[1:].mean(), abs=1e-3)
    assert noise[1] == pytest.approx(na.read_noise(stack[1:, ::5, ::5]).read_noise, rel=1e-4)
    assert "jump" not in capsys.readouterr().out


def test_shot_noise_matches_module_and_truth(results):
    ns, path, cam = results
    assert len(ns["camera_gain"]) == len(GAINS)
    for i, g in enumerate(GAINS):
        rn, _ = load_recording(path / "read_noise" / f"{g}dB")
        lit, meta = load_recording(path / "shot" / f"{g}dB", step=4)
        result = na.shot_noise(lit, rn.mean(), meta["max_value"])
        assert ns["camera_gain"][i] == pytest.approx(result.fit.slope, rel=1e-3)
        assert ns["shot_results"][i]["intercept"] == pytest.approx(result.fit.intercept, rel=1e-2, abs=0.05)
        assert ns["camera_gain"][i] == pytest.approx(cam.gain_counts_per_e(meta["gain"]), rel=0.05)
    assert ns["saturated_value"] == 4095
    assert ns["gain_0db"] == ns["camera_gain"][0]


def test_slides_gain_plot_matches_notebook(results):
    ns, path, _ = results
    slides = run_code(slide_chunks(), path)
    assert slides["ax"].get_title() == "From a previous run"
    assert slides["camera_gain"] == pytest.approx(ns["camera_gain"], rel=1e-3)


def table_rows(markdown: str) -> list[list[str]]:
    rows = [line.strip().strip("|").split("|") for line in markdown.splitlines() if line.startswith("|")]
    return [[c.strip() for c in row] for row in rows if not set("".join(row)) <= set("-: ")]


def test_slide_tables_are_filled_from_the_session(session, capsys):
    path, cam = session
    text = (ROOT / "index.qmd").read_text()
    assert not table_rows(text), "tables typed into the slides can't hold the session's values"
    capsys.readouterr()
    run_code(slide_chunks(), path)
    out = capsys.readouterr().out
    assert out.count("\n: From a previous run\n") == 4  # one caption per table
    rows = table_rows(out)
    assert all(cell for row in rows for cell in row)
    by_header = {}
    for row in rows:
        if not re.fullmatch(r"\d+ dB", row[0]) and row[0] not in ("Read noise", "Saturation", "Dynamic range"):
            header = tuple(row)
            by_header[header] = []
        else:
            by_header[header].append(row[1:])
    read, dark, dr, spec = by_header.values()
    assert [len(x) for x in (read, dark, dr, spec)] == [4, 4, 4, 3]
    for i, g in enumerate(GAINS):
        expected = (cam.read_noise_counts(g) ** 2 + 1 / 12) ** 0.5
        assert float(read[i][1]) == pytest.approx(expected, rel=0.1)
        ratio = float(dr[i][0].removesuffix(":1"))
        assert float(dr[i][1]) == pytest.approx(20 * np.log10(ratio), abs=0.06)
        assert ratio == pytest.approx((4095 - float(read[i][0])) / float(read[i][1]), rel=0.01)
    assert float(spec[0][0].removesuffix(" e-")) == pytest.approx(expected_read_noise_e(cam), rel=0.1)
    assert spec[2][0] == f"{float(dr[0][1]):.1f} dB"


def expected_read_noise_e(cam) -> float:
    return (cam.read_noise_counts(0) ** 2 + 1 / 12) ** 0.5 / cam.gain_counts_per_e(0)


def test_committed_summary_has_every_number_the_slides_show(session):
    path, _ = session
    committed = json.loads((ROOT / "results" / "practical.json").read_text())
    simulated = json.loads((path / "summary.json").read_text())
    assert "Simulated" not in committed["camera_model"]
    assert [r["gain_db"] for r in committed["gains"]] == GAINS
    for row, expected in zip(committed["gains"], simulated["gains"]):
        assert row.keys() == expected.keys() and np.isfinite(list(row.values())).all()
