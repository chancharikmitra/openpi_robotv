import json, subprocess, sys, glob, pathlib

import pytest


def test_select_reproduces_golden(tmp_path):
    golden = json.loads(pathlib.Path("tests/golden/knn_swap_green_red_cube.json").read_text())
    matches = glob.glob("temp_data/*swap_green_red_cube*pi05_action.h5")
    if not matches:
        pytest.skip("activation H5 temp_data/*swap_green_red_cube*pi05_action.h5 not present")
    attn = matches[0]
    out = tmp_path / "heads.json"
    subprocess.run([sys.executable, "-m", "openpi.head_tuning.select",
                    "--attn-h5", attn, "--unit-mode", "head",
                    "--target", "20", "--out", str(out)], check=True)
    got = json.loads(out.read_text())
    assert got["best_k"] == golden["best_k"]
    assert abs(got["cv_mse"] - golden["cv_mse"]) < golden["cv_mse_tol"]
    assert [list(p) for p in got["trainable_head_indices"]] == golden["trainable_head_indices"]
