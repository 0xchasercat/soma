import type { BehaviorProfile, KeyEvent, KeystrokePlan, Point2D, ScrollPlan, TargetBox, TrajectoryPlan } from './types.js';
import { type FlowModel } from './pointer/flow-synthesis.js';
export type Delay = (milliseconds: number) => Promise<void>;
export type Clock = () => number;
export interface PointerMoveDriver {
    move(x: number, y: number): Promise<void>;
}
export interface PointerClickDriver extends PointerMoveDriver {
    /** Live hit-test performed after movement and immediately before pointer-down. */
    isInteractiveAt(x: number, y: number): Promise<boolean>;
    down(): Promise<void>;
    up(): Promise<void>;
}
export interface KeyboardDriver {
    down(event: KeyEvent): Promise<void>;
    up(event: KeyEvent): Promise<void>;
}
export interface ScrollDriver {
    wheel(deltaY: number): Promise<void>;
}
export interface DispatchedClick {
    trajectory: TrajectoryPlan;
    postArrivalSettleMs: number;
    clickHoldMs: number;
    clickTargetIsInteractive: true;
}
export declare const systemDelay: Delay;
/** Dispatch a trajectory at its absolute timestamps, including pre-move dwell. */
export declare function dispatchTrajectory(plan: TrajectoryPlan, driver: PointerMoveDriver, delay?: Delay, clock?: Clock): Promise<void>;
/** Dispatch keydown/keyup events on one absolute timeline, including overlap. */
export declare function dispatchKeystrokePlan(plan: KeystrokePlan, driver: KeyboardDriver, delay?: Delay, clock?: Clock): Promise<void>;
/** Dispatch wheel frames at their absolute timestamps. */
export declare function dispatchScrollPlan(plan: ScrollPlan, driver: ScrollDriver, delay?: Delay, clock?: Clock): Promise<void>;
/**
 * Stateful pointer convenience wrapper. `click()` refuses to dispatch a click
 * when the live endpoint no longer resolves to an interactive element.
 *
 * Pass `flowModel` to use the trained normalizing-flow synthesizer for all
 * pointer movements. When omitted the parametric Bezier path is used, which
 * is the safe default for environments that have not loaded the flow ONNX.
 */
export declare class HumanPointer {
    private readonly driver;
    private readonly profile?;
    private readonly delay;
    private readonly flowModel?;
    private position;
    constructor(driver: PointerClickDriver, initialPosition: Point2D, profile?: Partial<BehaviorProfile> | undefined, delay?: Delay, flowModel?: FlowModel | undefined);
    get currentPosition(): Point2D;
    moveTo(target: TargetBox, seed?: number): Promise<TrajectoryPlan>;
    click(target: TargetBox, seed?: number): Promise<DispatchedClick>;
}
//# sourceMappingURL=dispatch.d.ts.map