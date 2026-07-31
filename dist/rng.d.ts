/**
 * Seeded pseudo-random number generator based on mulberry32.
 *
 * Provides deterministic-but-divergent trajectories: pass an explicit seed
 * for reproducibility, omit for per-call randomness. Fork a child RNG with
 * fork() to avoid trampling the parent's sequence.
 */
export declare class RNG {
    private readonly state;
    /** Spare value from Box-Muller paired draw. */
    private gaussianSpare;
    constructor(seed?: number);
    /** Uniform float in [0, 1). */
    next(): number;
    /** Uniform integer in [0, n). */
    nextInt(n: number): number;
    /** Uniform float in [lo, hi). */
    nextRange(lo: number, hi: number): number;
    /**
     * Normal(mu, sigma) via Box-Muller transform.
     * Paired draws — consecutive calls consume one pair.
     */
    nextGaussian(mu?: number, sigma?: number): number;
    /**
     * Lognormal(mu, sigma): exp(Normal(mu, sigma)).
     * mu/sigma are the parameters of the underlying normal (not the mean/std of
     * the lognormal distribution itself).
     */
    nextLognormal(mu: number, sigma: number): number;
    /**
     * Create a deterministic child RNG seeded from current state + optional salt.
     * Does NOT advance the parent's sequence.
     */
    fork(salt?: number): RNG;
}
export declare function defaultRNG(): RNG;
//# sourceMappingURL=rng.d.ts.map