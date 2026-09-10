"""Regression test: new extract CLI produces byte-identical activations to the golden H5."""
import glob
import os
import pathlib
import subprocess
import sys

import h5py
import numpy as np
import pytest

GOLDEN = pathlib.Path("tests/golden/extract_droid_smoke.h5")


def _leaf_groups(h: h5py.File) -> list[str]:
    """Return all HDF5 group paths that contain a 'last_token_attn' dataset."""
    out: list[str] = []
    h.visititems(
        lambda n, o: out.append(n)
        if isinstance(o, h5py.Group) and "last_token_attn" in o
        else None
    )
    return out


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden extract H5 not present (needs GPU run)")
def test_extract_droid_matches_golden(tmp_path: pathlib.Path) -> None:
    """Run extract over 2 episodes and confirm activations match the 20-episode golden."""
    matches = glob.glob("temp_data/swap_green_red_cube_20.h5")
    if not matches:
        pytest.skip("droid input H5 not present")
    out = tmp_path / "out.h5"
    env = {**os.environ, "OPENPI_DATA_HOME": "/scr2/yusenluo/openpi_robotv/.cache/openpi"}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "openpi.head_tuning.extract",
            "--setup",
            "droid",
            "--input-h5",
            matches[0],
            "--out-h5",
            str(out),
            "--task-prompt",
            "swap green red cube",
            "--max-episodes",
            "2",
        ],
        check=True,
        env=env,
    )
    with h5py.File(GOLDEN) as g, h5py.File(out) as o:
        okeys = sorted(_leaf_groups(o))
        gkeys = set(_leaf_groups(g))
        assert okeys, "new extract produced no frames"
        for k in okeys:
            assert k in gkeys, f"new key {k!r} not in golden"
            np.testing.assert_allclose(
                o[k]["last_token_attn"][:],
                g[k]["last_token_attn"][:],
                rtol=1e-5,
                atol=1e-5,
            )
