export interface Point2D {
    x: number;
    y: number;
}
/** A trajectory sample at a specific wall-clock timestamp. */
export interface TrajectoryPoint extends Point2D {
    /** Milliseconds since start of trajectory (monotonic). */
    tMs: number;
}
/** Axis-aligned bounding box of a DOM target element (viewport coordinates). */
export interface TargetBox {
    x: number;
    y: number;
    width: number;
    height: number;
}
/**
 * Behavioral persona driving all synthesis. All synthesis functions accept this
 * and derive humanizing parameters from it rather than hardcoding constants.
 */
export interface BehaviorProfile {
    /** Typing speed in words-per-minute (default: 65). */
    wpm: number;
    /**
     * Micro-tremor amplitude as a fraction of one CSS pixel (default: 0.3).
     * Controls autocorrelated jitter injected on every trajectory.
     * τ ≈ 30 ms autocorrelation matches physiological hand tremor (8–12 Hz band).
     */
    tremor: number;
    /**
     * Endpoint precision bias [0, 1] (default: 0.75).
     * 1 = click always lands near target centre; 0 = uniform random inside box.
     */
    precision: number;
    /**
     * Per-character typo injection rate (default: 0.02 ≈ 2%).
     * Mistakes are realistic QWERTY-adjacent key errors followed by Backspace.
     */
    mistakeRate: number;
    /**
     * Movement speed multiplier applied to Fitts MT (default: 1.0).
     * < 1 = faster; > 1 = slower/more deliberate.
     */
    speed: number;
    /** Optional cadence multiplier for inter-action timing (default: 1). */
    cadence?: number;
}
/**
 * A complete pointer movement plan ready to dispatch via CDP or similar.
 * Points are already 60 Hz-resampled and include autocorrelated tremor.
 */
export interface TrajectoryPlan {
    /** 60 Hz resampled trajectory. Drive by scheduling moves at point.tMs offsets. */
    points: TrajectoryPoint[];
    /** Total movement duration in ms (last point.tMs). */
    durationMs: number;
    /**
     * Settle delay to wait BEFORE issuing the first move (pre-move dwell).
     * Models the Fitts perceptual-motor preparation interval.
     * Drawn from Gaussian(150, 50) ms.
     */
    preMoveDelayMs: number;
    /** Final endpoint (where the click lands — sampled inside target box). */
    endpoint: Point2D;
}
/** A single key event ready to dispatch (via CDP Input.dispatchKeyEvent or equivalent). */
export interface KeyEvent {
    /** Key label (e.g. "a", "A", "Backspace", " "). */
    key: string;
    /** Physical code (e.g. "KeyA", "Backspace", "Space"). */
    code: string;
    /** Milliseconds from start of the keystroke sequence when keydown fires. */
    downMs: number;
    /** Duration the key is held (keyup - keydown), drawn from Gaussian(~80, 25) ms. */
    holdMs: number;
    /** True when the physical key requires Shift (uppercase or shifted punctuation). */
    shiftKey?: boolean;
    /** Modifier metadata for dispatchers that do not infer Shift from key. */
    modifiers?: {
        shift: boolean;
    };
    /** Explicit correction role and pause for mistake timing. */
    correction?: KeyCorrectionMetadata;
    /** Explicit pause after key dispatch, in addition to hold time. */
    pauseAfterMs?: number;
}
/** Optional metadata for a key correction sequence. */
export interface KeyCorrectionMetadata {
    /** The role this event plays in a mistake/correction sequence. */
    kind: 'mistake' | 'backspace' | 'retry';
    /** Explicit pause after this event before the next keydown. */
    pauseMs: number;
}
/** A complete keystroke sequence plan. */
export interface KeystrokePlan {
    /** Ordered sequence of key events (includes mistake + backspace events). */
    events: KeyEvent[];
    /** Total duration from first downMs to last (downMs + holdMs). */
    totalDurationMs: number;
}
/** One dispatched wheel event (via CDP Input.dispatchMouseEvent type=mouseWheel). */
export interface ScrollFrame {
    /** Milliseconds from scroll start. */
    tMs: number;
    /** Vertical delta for this frame in CSS pixels (positive = scroll down). */
    deltaY: number;
}
/** A complete inertial scroll plan. */
export interface ScrollPlan {
    frames: ScrollFrame[];
    durationMs: number;
    /** Peak |deltaY| across all frames (~200–260 px for a real flick). */
    peakDeltaY: number;
}
/**
 * Features extracted from a single captured/synthesized trajectory.
 * These are the evidence.pointer.* fields from the grimoire differential.
 * All candidates in behav.synthetic_motion_flag.yaml consume these.
 */
export interface TrajectoryFeatures {
    /** Mean absolute turning angle per unit arc length (rad/px). ~0 for straight line. */
    curvature: number;
    /**
     * Number of corrective sub-movements (accel sign-change count).
     * Human intermittent control: ≥ 1 for any non-trivial reach.
     */
    submovement_count: number;
    /** Speed-envelope classification. Humans produce 'bell'. */
    velocity_profile: 'bell' | 'linear' | 'flat' | 'unknown';
    /**
     * Goodness-of-fit (R²) of a single-lobe lognormal fit to the normalized
     * speed envelope. Sigma-Lognormal / BeCAPTCHA-Mouse neuromotor feature.
     * Human: > 0.7. Synthetic flat/linear: < 0.4.
     */
    sigma_lognormal_fit: number;
    /**
     * True if physiological jitter is present in the 8–12 Hz tremor band
     * of the detrended trajectory residual.
     */
    has_micro_tremor: boolean;
    /**
     * Lag-1 autocorrelation of the high-frequency residual per axis.
     * A real hand: ρ ≈ 0.4–0.7. White-noise jitter: ρ ≈ 0. The cside
     * cursor_v2 promotion cause — wrong autocorr structure caught a
     * motion-humanized agent 100% despite flawless fingerprint stealth.
     */
    jitter_autocorr: number;
    /**
     * Surrogate neural motion scorer output ∈ [0, 1].
     * 0 = human, 1 = bot. Uses trained ONNX model when available,
     * otherwise falls back to the heuristic weighted combination.
     * Documents this boundary in score/heuristic.ts.
     */
    neural_bot_score: number;
}
/** Features extracted from a captured keystroke sequence. */
export interface KeystrokeFeatures {
    /** Per-key HT array (keyup − keydown in ms). */
    hold_times: number[];
    /** Inter-key FT array (tDown[i+1] − tDown[i] in ms). */
    flight_times: number[];
    /** Coefficient of variation (σ/μ) of hold times. */
    hold_time_cv: number;
    /** Coefficient of variation (σ/μ) of flight times. */
    flight_time_cv: number;
    /**
     * Variance of mean FT across digraph classes
     * (same-hand / cross-hand / after-space / after-punct).
     * Near-zero = no context structure.
     */
    digraph_class_variance: number;
    /**
     * Kolmogorov-Smirnov distance between the session's smoothed empirical
     * CDF of FTs and a human reference CDF.
     */
    timing_cdf_distance: number;
    /** Number of Backspace/Delete correction events. */
    correction_count: number;
    /** Minimum flight time in ms (paste/HID-injection yields near-zero). */
    min_flight_time: number;
}
/** Interaction cadence features for a session. */
export interface CadenceFeatures {
    /** Coefficient of variation of inter-action intervals. */
    inter_action_interval_cv: number;
    /** Coefficient of variation of dwell times (think-time between actions). */
    dwell_time_cv: number;
    /** Actions per second over the session. */
    action_rate: number;
    /** Minimum observed latency from element-ready to action-fired (ms). */
    min_action_latency_ms: number;
}
/**
 * Normalized 12-dimensional feature vector fed into the neural scorer.
 * All values ∈ [0, 1] after normalization.
 *
 * Architecture: input(12) → FC(12,104) → ReLU → FC(104,64) → ReLU
 *             → FC(64,32) → ReLU → FC(32,1) → Sigmoid  ≈ 10.2k params
 */
export type ModelFeatureVector = [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number
];
/** Session-level identifier shared across all events in one page visit. */
export type SessionId = string;
/** Schema that originally produced a stored capture record. */
export type CaptureSourceSchema = 'soma.capture.v1' | 'soma.capture.v2';
/** Raw pointer movement captured by the extension content script. */
export interface RawMovement {
    id?: string;
    sessionId: SessionId;
    capturedAt: number;
    /** Absent only on imported pre-v2 records; the export envelope is the fallback. */
    sourceSchema?: CaptureSourceSchema;
    trustedEvents?: true;
    trajectory: TrajectoryPoint[];
    clickTarget: {
        tagName: string;
        isInteractive: boolean;
        boundingBox: TargetBox;
    };
    viewport: {
        width: number;
        height: number;
    };
}
export type RawKeyHand = 'left' | 'right' | 'none' | 'unknown';
export type RawKeyKind = 'character' | 'space' | 'punctuation' | 'correction' | 'modifier' | 'navigation' | 'control' | 'other';
export type RawDigraphClass = 'same-hand' | 'cross-hand' | 'after-space' | 'after-punctuation' | 'none';
/** Raw keystroke sequence. V2 stores only privacy-safe key classes, never printable key/code values. */
export interface RawKeystroke {
    id?: string;
    sessionId: SessionId;
    capturedAt: number;
    /** Absent only on imported pre-v2 records; the export envelope is the fallback. */
    sourceSchema?: CaptureSourceSchema;
    trustedEvents?: true;
    events: Array<{
        /** Present only on imported v1 records. V2 records never store key values. */
        key?: string;
        /** Present only on imported v1 records. V2 records never store physical codes. */
        code?: string;
        downMs: number;
        upMs: number;
        shiftKey?: boolean;
        trusted?: true;
        hand?: RawKeyHand;
        keyKind?: RawKeyKind;
        digraphClass?: RawDigraphClass;
    }>;
    charCount: number;
    charsCommitted?: number;
    inputTypes?: string[];
    fieldType: string;
    correctionCount: number;
}
/** Raw scroll gesture; v2 frames are normalized to CSS-pixel deltas. */
export interface RawScroll {
    id?: string;
    sessionId: SessionId;
    capturedAt: number;
    /** Absent only on imported pre-v2 records; the export envelope is the fallback. */
    sourceSchema?: CaptureSourceSchema;
    trustedEvents?: true;
    frames: Array<{
        tMs: number;
        deltaX: number;
        deltaY: number;
        deltaMode?: 0;
        isTrusted: boolean;
    }>;
    concurrentHoverEventCount: number;
    peakAbsDeltaY: number;
}
/** Privacy-safe event provenance captured for each user interaction. */
export interface RawInteraction {
    id?: string;
    sessionId: SessionId;
    capturedAt: number;
    sourceSchema: 'soma.capture.v2';
    eventType: 'pointerdown' | 'pointerup' | 'click' | 'keydown' | 'keyup' | 'beforeinput' | 'input' | 'wheel' | 'focus';
    isTrusted: boolean;
    userActivation: boolean;
    pointerType: 'mouse' | 'touch' | 'pen' | 'none' | 'unknown';
    sinceNavigationMs: number;
    /** Best-known monotonic time when the event target became actionable. */
    actionReadySinceNavigationMs?: number;
    /** Event time minus actionReadySinceNavigationMs. */
    actionLatencyMs?: number;
}
/** Recorder-derived evidence for one interaction session. */
export interface CapturedInteractionFeatures {
    event: {
        isTrusted: boolean;
        has_user_activation: boolean;
        pointerTypes: RawInteraction['pointerType'][];
        sequence: RawInteraction['eventType'][];
    };
    interaction: CadenceFeatures & {
        count: number;
    };
}
/** Aggregated extension export payload. v1 remains accepted for migration. */
export interface CaptureExport {
    schema: 'soma.capture.v1' | 'soma.capture.v2';
    schemaVersion?: 1 | 2;
    exportedAt: string;
    exportId?: string;
    producer?: string;
    userAgent: string;
    movements: RawMovement[];
    keystrokes: RawKeystroke[];
    scrolls: RawScroll[];
    interactions?: RawInteraction[];
}
/** Labeled trajectory sample for model training. 0 = human, 1 = bot. */
export interface LabeledSample {
    features: ModelFeatureVector;
    label: 0 | 1;
    /** Tag describing the generator (e.g. 'human', 'linear', 'white_noise', etc.). */
    source: string;
}
//# sourceMappingURL=types.d.ts.map