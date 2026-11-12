import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from daq_live import BlockAverager, RingBuffer  # noqa: E402
from usb201 import codes_to_volts, pacer_period  # noqa: E402


def test_ring_buffer_keeps_latest_in_order_across_wraps():
    rb = RingBuffer(5)
    rb.write([1, 2, 3])
    np.testing.assert_array_equal(rb.latest(), [1, 2, 3])
    rb.write([4, 5, 6, 7])
    np.testing.assert_array_equal(rb.latest(), [3, 4, 5, 6, 7])
    rb.write([8])
    np.testing.assert_array_equal(rb.latest(), [4, 5, 6, 7, 8])


def test_ring_buffer_write_larger_than_size():
    rb = RingBuffer(3)
    rb.write([1])
    rb.write(np.arange(10))
    np.testing.assert_array_equal(rb.latest(), [7, 8, 9])
    rb.write([10])
    np.testing.assert_array_equal(rb.latest(), [8, 9, 10])


def test_block_averager_carries_remainder_between_pushes():
    avg = BlockAverager(4)
    np.testing.assert_array_equal(avg.push([1, 1, 1]), [])
    np.testing.assert_array_equal(avg.push([1, 3, 3, 3, 3, 5]), [1, 3])
    np.testing.assert_array_equal(avg.push([5, 5, 5]), [5])


def test_pacer_period_matches_70mhz_clock():
    assert pacer_period(100_000) == (699, 100_000.0)
    with pytest.raises(ValueError):
        pacer_period(200_000)


def test_codes_to_volts_identity_calibration():
    np.testing.assert_allclose(codes_to_volts([0, 2048, 4096], 1.0, 0.0), [-10, 0, 10])
