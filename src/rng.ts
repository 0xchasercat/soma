/**
 * Seeded pseudo-random number generator based on mulberry32.
 *
 * Provides deterministic-but-divergent trajectories: pass an explicit seed
 * for reproducibility, omit for per-call randomness. Fork a child RNG with
 * fork() to avoid trampling the parent's sequence.
 */

/** Advance the mulberry32 state and return a float in [0, 1). */
function mulberry32Step(state: { s: number }): number {
  let t = (state.s += 0x6d2b79f5) | 0;
  t = Math.imul(t ^ (t >>> 15), 1 | t);
  t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

export class RNG {
  private readonly state: { s: number };
  /** Spare value from Box-Muller paired draw. */
  private gaussianSpare: number | null = null;

  constructor(seed?: number) {
    // Mix the seed so adjacent integers produce well-separated sequences.
    const raw = seed ?? (Math.random() * 0xffffffff) | 0;
    let s = raw | 0;
    // One warmup round of splitmix to spread low seeds.
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b) | 0;
    s = Math.imul(s ^ (s >>> 16), 0x45d9f3b) | 0;
    s ^= s >>> 16;
    this.state = { s };
  }

  /** Uniform float in [0, 1). */
  next(): number {
    return mulberry32Step(this.state);
  }

  /** Uniform integer in [0, n). */
  nextInt(n: number): number {
    return Math.floor(this.next() * n);
  }

  /** Uniform float in [lo, hi). */
  nextRange(lo: number, hi: number): number {
    return lo + this.next() * (hi - lo);
  }

  /**
   * Normal(mu, sigma) via Box-Muller transform.
   * Paired draws — consecutive calls consume one pair.
   */
  nextGaussian(mu = 0, sigma = 1): number {
    if (this.gaussianSpare !== null) {
      const s = this.gaussianSpare;
      this.gaussianSpare = null;
      return mu + sigma * s;
    }
    let u: number, v: number, s: number;
    do {
      u = this.next() * 2 - 1;
      v = this.next() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const mul = Math.sqrt((-2 * Math.log(s)) / s);
    this.gaussianSpare = v * mul;
    return mu + sigma * u * mul;
  }

  /**
   * Lognormal(mu, sigma): exp(Normal(mu, sigma)).
   * mu/sigma are the parameters of the underlying normal (not the mean/std of
   * the lognormal distribution itself).
   */
  nextLognormal(mu: number, sigma: number): number {
    return Math.exp(this.nextGaussian(mu, sigma));
  }

  /**
   * Create a deterministic child RNG seeded from current state + optional salt.
   * Does NOT advance the parent's sequence.
   */
  fork(salt = 0): RNG {
    // Read one value to derive child seed without disturbing the parent
    // (we snapshot state, step, restore).
    const snapshot = this.state.s;
    const derived = mulberry32Step(this.state) * 0xffffffff;
    this.state.s = snapshot; // restore
    return new RNG((derived ^ (salt * 0x9e3779b9)) | 0);
  }
}

/** Module-level default RNG (randomly seeded, shared). Use for quick one-offs. */
let _default: RNG | null = null;
export function defaultRNG(): RNG {
  if (!_default) _default = new RNG();
  return _default;
}
