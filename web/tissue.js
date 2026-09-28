/* Morpheus tissue engine for the browser (and Node, for the parity test).
 * A line-by-line port of src/morpheus/tissue.py `step`: same channels, same perception
 * kernels, same network, same gap-junction physics, same self-model error.
 * tests/test_web_parity.py runs both engines on the same inputs and compares states. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MorpheusTissue = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const C = 13, VIS = 5, HID = 32, N_IN = 4 * C + VIS, N_OUT = C + VIS;
  const ALPHA = 0, VOLT = 1, CLIP = 4.0, THRESH = 0.1;

  function unpack(theta) {
    let i = 0;
    const take = (n) => { const a = Float32Array.from(theta.slice(i, i + n)); i += n; return a; };
    return { W1: take(N_IN * HID), b1: take(HID), W2: take(HID * N_OUT), b2: take(N_OUT) };
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class Tissue {
    constructor(theta, physics, size) {
      this.p = unpack(theta);
      this.physics = Object.assign({ diffusion: 0.1, fire_rate: 0.5, gain: 1.0 }, physics || {});
      this.size = size || 32;
      const n = this.size * this.size;
      this.s = new Float32Array(n * C);
      this.eps = new Float32Array(n * VIS);
      this.lastEpsIn = new Float32Array(n * VIS);
      this.history = [];
      this.rand = mulberry32(1);
      this._new = new Float32Array(n * C);
      this._pre = new Uint8Array(n);
      this._x = new Float32Array(N_IN);
      this._h = new Float32Array(HID);
      this._o = new Float32Array(N_OUT);
      this.t = 0;
      this.seed();
    }

    idx(y, x) { return y * this.size + x; }

    seed() {
      this.s.fill(0); this.eps.fill(0); this.history = []; this.t = 0;
      const c = this.idx(this.size >> 1, this.size >> 1) * C;
      for (let k = 0; k < C; k++) this.s[c + k] = (k >= 2 && k < 5) ? 0 : 1;
    }

    get(s, y, x, k) {
      const S = this.size;
      if (y < 0 || x < 0 || y >= S || x >= S) return 0;
      return s[(y * S + x) * C + k];
    }

    aliveMask(s, out) {
      const S = this.size;
      for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
        let m = -Infinity;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const v = this.get(s, y + dy, x + dx, ALPHA);
          if (v > m) m = v;
        }
        out[y * S + x] = m > THRESH ? 1 : 0;
      }
      return out;
    }

    /* opts: {fire: Uint8Array|undefined, diffusion, epsIn: Float32Array|undefined, vclamp: {mask: Uint8Array, value}} */
    step(opts) {
      opts = opts || {};
      const S = this.size, n = S * S, s = this.s, nw = this._new, P = this.p;
      const ph = this.physics;
      const D = opts.diffusion !== undefined ? opts.diffusion : ph.diffusion;
      const epsIn = opts.epsIn || this.eps;
      this.lastEpsIn.set(epsIn);
      const pre = this.aliveMask(s, this._pre);
      const fire = opts.fire || null;
      const pred = new Float32Array(n * VIS);
      const fired = new Uint8Array(n);
      const x = this._x, h = this._h, o = this._o;
      for (let y = 0; y < S; y++) for (let xx = 0; xx < S; xx++) {
        const i = y * S + xx;
        for (let k = 0; k < C; k++) {
          const g = (dy, dx) => this.get(s, y + dy, xx + dx, k);
          const c0 = g(0, 0);
          x[k] = c0;
          x[C + k] = (g(-1, 1) + 2 * g(0, 1) + g(1, 1) - g(-1, -1) - 2 * g(0, -1) - g(1, -1)) / 8;
          x[2 * C + k] = (g(1, -1) + 2 * g(1, 0) + g(1, 1) - g(-1, -1) - 2 * g(-1, 0) - g(-1, 1)) / 8;
          x[3 * C + k] = (g(-1, -1) + g(-1, 1) + g(1, -1) + g(1, 1)
            + 2 * (g(-1, 0) + g(1, 0) + g(0, -1) + g(0, 1)) - 12 * c0) / 16;
        }
        for (let k = 0; k < VIS; k++) x[4 * C + k] = ph.gain * epsIn[i * VIS + k];
        for (let j = 0; j < HID; j++) {
          let a = P.b1[j];
          for (let k = 0; k < N_IN; k++) a += x[k] * P.W1[k * HID + j];
          h[j] = a > 0 ? a : 0;
        }
        for (let j = 0; j < N_OUT; j++) {
          let a = P.b2[j];
          for (let k = 0; k < HID; k++) a += h[k] * P.W2[k * N_OUT + j];
          o[j] = a;
        }
        const f = fire ? fire[i] : (this.rand() < ph.fire_rate ? 1 : 0);
        fired[i] = f;
        for (let k = 0; k < C; k++) nw[i * C + k] = s[i * C + k] + (f ? o[k] : 0);
        for (let k = 0; k < VIS; k++) pred[i * VIS + k] = o[C + k];
      }
      if (D) {
        for (let y = 0; y < S; y++) for (let xx = 0; xx < S; xx++) {
          const i = y * S + xx;
          if (!pre[i]) continue;
          const v = s[i * C + VOLT];
          let flux = 0;
          for (const [dy, dx] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
            const yy = y + dy, x2 = xx + dx;
            if (yy < 0 || x2 < 0 || yy >= S || x2 >= S) continue;
            const j = yy * S + x2;
            if (pre[j]) flux += s[j * C + VOLT] - v;
          }
          nw[i * C + VOLT] += D * flux;
        }
      }
      if (opts.vclamp) {
        for (let i = 0; i < n; i++) if (opts.vclamp.mask[i]) nw[i * C + VOLT] = opts.vclamp.value;
      }
      for (let i = 0; i < n * C; i++) nw[i] = nw[i] > CLIP ? CLIP : (nw[i] < -CLIP ? -CLIP : nw[i]);
      const post = this.aliveMask(nw, new Uint8Array(n));
      const eps = new Float32Array(n * VIS);
      for (let i = 0; i < n; i++) {
        const alive = pre[i] && post[i];
        if (!alive) for (let k = 0; k < C; k++) nw[i * C + k] = 0;
        for (let k = 0; k < VIS; k++) {
          eps[i * VIS + k] = alive ? (nw[i * C + k] - s[i * C + k]) - (fired[i] ? pred[i * VIS + k] : 0) : 0;
        }
      }
      this.history.push(this.eps);
      if (this.history.length > 16) this.history.shift();
      this.s = nw; this._new = s; this.eps = eps; this.t++;
      return this;
    }

    /* Comparator input for the next step under a named condition. */
    comparatorInput(mode, delay) {
      if (mode === "zero") return new Float32Array(this.eps.length);
      if (mode === "delayed") {
        const d = delay || 8, hlen = this.history.length;
        return hlen >= d ? this.history[hlen - d] : new Float32Array(this.eps.length);
      }
      return this.eps;
    }

    cut(maskFn) {
      const S = this.size;
      for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) if (maskFn(y, x)) {
        const i = y * S + x;
        for (let k = 0; k < C; k++) this.s[i * C + k] = 0;
        for (let k = 0; k < VIS; k++) this.eps[i * VIS + k] = 0;
      }
    }
  }

  return { Tissue, C, VIS, HID, N_IN, N_OUT, mulberry32 };
});
