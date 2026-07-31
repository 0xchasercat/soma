import type { BehaviorProfile } from './types.js';

/** Sensible default profile — 65 wpm typist, moderate tremor, realistic precision. */
export const DEFAULT_PROFILE: Readonly<BehaviorProfile> = {
  wpm: 65,
  tremor: 0.3,
  precision: 0.75,
  mistakeRate: 0.02,
  speed: 1.0,
};

/**
 * Preset profiles for different operator personas.
 * Use these or compose your own with mergeProfile().
 */
export const PROFILES = {
  /** Careful, deliberate typist — slow and precise. */
  careful: {
    wpm: 40,
    tremor: 0.25,
    precision: 0.88,
    mistakeRate: 0.01,
    speed: 0.8,
  } satisfies BehaviorProfile,

  /** Average office worker. */
  average: {
    ...DEFAULT_PROFILE,
  } satisfies BehaviorProfile,

  /** Fast, slightly sloppy power user. */
  fast: {
    wpm: 100,
    tremor: 0.35,
    precision: 0.65,
    mistakeRate: 0.035,
    speed: 1.3,
  } satisfies BehaviorProfile,

  /** Mobile-style interaction — slower, less precise, different tremor. */
  mobile: {
    wpm: 30,
    tremor: 0.45,
    precision: 0.55,
    mistakeRate: 0.04,
    speed: 0.7,
  } satisfies BehaviorProfile,
} as const;

/** Merge partial overrides onto the default profile and clamp untrusted input. */
export function mergeProfile(partial?: Partial<BehaviorProfile>): BehaviorProfile {
  const p: BehaviorProfile = {
    wpm: Number.isFinite(partial?.wpm) ? partial!.wpm! : DEFAULT_PROFILE.wpm,
    tremor: Number.isFinite(partial?.tremor) ? partial!.tremor! : DEFAULT_PROFILE.tremor,
    precision: Number.isFinite(partial?.precision) ? partial!.precision! : DEFAULT_PROFILE.precision,
    mistakeRate: Number.isFinite(partial?.mistakeRate) ? partial!.mistakeRate! : DEFAULT_PROFILE.mistakeRate,
    speed: Number.isFinite(partial?.speed) ? partial!.speed! : DEFAULT_PROFILE.speed,
  };
  if (Number.isFinite(partial?.cadence)) p.cadence = partial!.cadence!;
  return clampProfile(p);
}

/** Clamp all profile values to valid finite ranges. */
export function clampProfile(p: BehaviorProfile): BehaviorProfile {
  const out: BehaviorProfile = {
    wpm: Math.max(10, Math.min(200, Number.isFinite(p.wpm) ? p.wpm : DEFAULT_PROFILE.wpm)),
    tremor: Math.max(0, Math.min(1, Number.isFinite(p.tremor) ? p.tremor : DEFAULT_PROFILE.tremor)),
    precision: Math.max(0, Math.min(1, Number.isFinite(p.precision) ? p.precision : DEFAULT_PROFILE.precision)),
    mistakeRate: Math.max(0, Math.min(0.15, Number.isFinite(p.mistakeRate) ? p.mistakeRate : DEFAULT_PROFILE.mistakeRate)),
    speed: Math.max(0.2, Math.min(3.0, Number.isFinite(p.speed) ? p.speed : DEFAULT_PROFILE.speed)),
  };
  if (Number.isFinite(p.cadence)) out.cadence = Math.max(0.25, Math.min(4, p.cadence!));
  return out;
}
