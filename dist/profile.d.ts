import type { BehaviorProfile } from './types.js';
/** Sensible default profile — 65 wpm typist, moderate tremor, realistic precision. */
export declare const DEFAULT_PROFILE: Readonly<BehaviorProfile>;
/**
 * Preset profiles for different operator personas.
 * Use these or compose your own with mergeProfile().
 */
export declare const PROFILES: {
    /** Careful, deliberate typist — slow and precise. */
    readonly careful: {
        wpm: number;
        tremor: number;
        precision: number;
        mistakeRate: number;
        speed: number;
    };
    /** Average office worker. */
    readonly average: {
        wpm: number;
        tremor: number;
        precision: number;
        mistakeRate: number;
        speed: number;
        cadence?: number;
    };
    /** Fast, slightly sloppy power user. */
    readonly fast: {
        wpm: number;
        tremor: number;
        precision: number;
        mistakeRate: number;
        speed: number;
    };
    /** Mobile-style interaction — slower, less precise, different tremor. */
    readonly mobile: {
        wpm: number;
        tremor: number;
        precision: number;
        mistakeRate: number;
        speed: number;
    };
};
/** Merge partial overrides onto the default profile and clamp untrusted input. */
export declare function mergeProfile(partial?: Partial<BehaviorProfile>): BehaviorProfile;
/** Clamp all profile values to valid finite ranges. */
export declare function clampProfile(p: BehaviorProfile): BehaviorProfile;
//# sourceMappingURL=profile.d.ts.map