import type {
  BehaviorProfile,
  KeyEvent,
  KeystrokePlan,
  Point2D,
  ScrollPlan,
  TargetBox,
  TrajectoryPlan,
} from './types.js';
import { synthesizeClick, synthesizeMovement } from './pointer/trajectory.js';
import { type FlowModel, synthesizeClickFlow, synthesizeMovementFlow } from './pointer/flow-synthesis.js';

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

export const systemDelay: Delay = async (milliseconds) => {
  if (!Number.isFinite(milliseconds) || milliseconds <= 0) return;
  await new Promise<void>((resolve) => setTimeout(resolve, milliseconds));
};

/** Dispatch a trajectory at its absolute timestamps, including pre-move dwell. */
export async function dispatchTrajectory(
  plan: TrajectoryPlan,
  driver: PointerMoveDriver,
  delay: Delay = systemDelay,
  clock: Clock = () => performance.now(),
): Promise<void> {
  const startedAt = clock();
  for (const point of plan.points) {
    const dueAt = plan.preMoveDelayMs + point.tMs;
    await delay(Math.max(0, dueAt - (clock() - startedAt)));
    await driver.move(point.x, point.y);
  }
}

/** Dispatch keydown/keyup events on one absolute timeline, including overlap. */
export async function dispatchKeystrokePlan(
  plan: KeystrokePlan,
  driver: KeyboardDriver,
  delay: Delay = systemDelay,
  clock: Clock = () => performance.now(),
): Promise<void> {
  const timeline = plan.events.flatMap((event, index) => [
    { atMs: event.downMs, kind: 'down' as const, event, index },
    { atMs: event.downMs + event.holdMs, kind: 'up' as const, event, index },
  ]);
  timeline.sort((a, b) => a.atMs - b.atMs || (a.kind === b.kind ? a.index - b.index : a.kind === 'up' ? -1 : 1));

  const startedAt = clock();
  for (const item of timeline) {
    await delay(Math.max(0, item.atMs - (clock() - startedAt)));
    if (item.kind === 'down') await driver.down(item.event);
    else await driver.up(item.event);
  }
}

/** Dispatch wheel frames at their absolute timestamps. */
export async function dispatchScrollPlan(
  plan: ScrollPlan,
  driver: ScrollDriver,
  delay: Delay = systemDelay,
  clock: Clock = () => performance.now(),
): Promise<void> {
  const startedAt = clock();
  for (const frame of plan.frames) {
    await delay(Math.max(0, frame.tMs - (clock() - startedAt)));
    await driver.wheel(frame.deltaY);
  }
}

/**
 * Stateful pointer convenience wrapper. `click()` refuses to dispatch a click
 * when the live endpoint no longer resolves to an interactive element.
 *
 * Pass `flowModel` to use the trained normalizing-flow synthesizer for all
 * pointer movements. When omitted the parametric Bezier path is used, which
 * is the safe default for environments that have not loaded the flow ONNX.
 */
export class HumanPointer {
  private position: Point2D;

  constructor(
    private readonly driver: PointerClickDriver,
    initialPosition: Point2D,
    private readonly profile?: Partial<BehaviorProfile>,
    private readonly delay: Delay = systemDelay,
    private readonly flowModel?: FlowModel,
  ) {
    this.position = { ...initialPosition };
  }

  get currentPosition(): Point2D {
    return { ...this.position };
  }

  async moveTo(target: TargetBox, seed?: number): Promise<TrajectoryPlan> {
    const plan = this.flowModel
      ? await synthesizeMovementFlow(this.flowModel, this.position, target, this.profile, seed)
      : synthesizeMovement(this.position, target, this.profile, seed);
    await dispatchTrajectory(plan, this.driver, this.delay);
    this.position = { ...plan.endpoint };
    return plan;
  }

  async click(target: TargetBox, seed?: number): Promise<DispatchedClick> {
    let trajectory: TrajectoryPlan;
    let settleMs: number;
    let holdMs: number;
    if (this.flowModel) {
      const click = await synthesizeClickFlow(this.flowModel, this.position, target, this.profile, seed);
      trajectory = click.trajectory;
      settleMs = click.settleMs;
      holdMs = click.holdMs;
    } else {
      const click = synthesizeClick(this.position, target, this.profile, seed);
      trajectory = click.trajectory;
      settleMs = click.postArrivalSettleMs;
      holdMs = click.clickHoldMs;
    }
    await dispatchTrajectory(trajectory, this.driver, this.delay);
    this.position = { ...trajectory.endpoint };
    await this.delay(settleMs);

    const interactive = await this.driver.isInteractiveAt(this.position.x, this.position.y);
    if (!interactive) {
      throw new Error('Refusing click: live endpoint is not interactive');
    }

    await this.driver.down();
    await this.delay(holdMs);
    await this.driver.up();
    return {
      trajectory,
      postArrivalSettleMs: settleMs,
      clickHoldMs: holdMs,
      clickTargetIsInteractive: true,
    };
  }
}
