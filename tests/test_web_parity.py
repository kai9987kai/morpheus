"""The browser engine (web/tissue.js) must reproduce the Python engine step for step."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from morpheus import anatomy
from morpheus.tissue import C, VIS, Physics, init_params, seed_state, step

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_js_engine_matches_python(tmp_path):
    rng = np.random.default_rng(0)
    theta = init_params(rng)
    theta[-(32 * 18 + 18):] += rng.normal(0, 0.05, 32 * 18 + 18).astype(np.float32)
    physics = Physics(diffusion=0.1, fire_rate=0.5, gain=0.8)
    size, steps = 16, 12
    fires = (rng.random((steps, 1, size, size, 1)) < 0.5)
    s = seed_state(1, size, size)
    eps = np.zeros((1, size, size, VIS), np.float32)
    for t in range(steps):
        if t == 8:
            s = s * ~anatomy.wound("disc", np.random.default_rng(1), size)[None, :, :, None]
            eps = eps * ~anatomy.wound("disc", np.random.default_rng(1), size)[None, :, :, None]
        s, eps, _, _ = step(s, eps, theta, physics, fires[t])
    cut = anatomy.wound("disc", np.random.default_rng(1), size)
    spec = {"theta": theta.tolist(), "physics": physics.__dict__, "size": size, "steps": steps,
            "fires": fires[:, 0, :, :, 0].astype(int).reshape(steps, -1).tolist(),
            "cut_at": 8, "cut": cut.astype(int).ravel().tolist()}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    script = f"""
const T = require({json.dumps(str(ROOT / 'web' / 'tissue.js'))});
const spec = JSON.parse(require('fs').readFileSync({json.dumps(str(tmp_path / 'spec.json'))}, 'utf8'));
const t = new T.Tissue(spec.theta, spec.physics, spec.size);
for (let k = 0; k < spec.steps; k++) {{
  if (k === spec.cut_at) t.cut((y, x) => spec.cut[y * spec.size + x] === 1);
  t.step({{ fire: Uint8Array.from(spec.fires[k]) }});
}}
process.stdout.write(JSON.stringify(Array.from(t.s)));
"""
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, check=True).stdout
    js = np.asarray(json.loads(out), np.float32).reshape(1, size, size, C)
    assert np.abs(s).max() > 0.1
    np.testing.assert_allclose(js, s, atol=2e-4)
