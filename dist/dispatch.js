import { synthesizeClick, synthesizeMovement } from './pointer/trajectory.js';
export const systemDelay = async (milliseconds) => {
    if (!Number.isFinite(milliseconds) || milliseconds <= 0)
        return;
    await new Promise((resolve) => setTimeout(resolve, milliseconds));
};
/** Dispatch a trajectory at its absolute timestamps, including pre-move dwell. */
export async function dispatchTrajectory(plan, driver, delay = systemDelay) {
    await delay(plan.preMoveDelayMs);
    let elapsed = 0;
    for (const point of plan.points) {
        await delay(Math.max(0, point.tMs - elapsed));
        await driver.move(point.x, point.y);
        elapsed = point.tMs;
    }
}
/** Dispatch keydown/keyup events on one absolute timeline, including overlap. */
export async function dispatchKeystrokePlan(plan, driver, delay = systemDelay) {
    const timeline = plan.events.flatMap((event, index) => [
        { atMs: event.downMs, kind: 'down', event, index },
        { atMs: event.downMs + event.holdMs, kind: 'up', event, index },
    ]);
    timeline.sort((a, b) => a.atMs - b.atMs || (a.kind === b.kind ? a.index - b.index : a.kind === 'up' ? -1 : 1));
    let elapsed = 0;
    for (const item of timeline) {
        await delay(Math.max(0, item.atMs - elapsed));
        if (item.kind === 'down')
            await driver.down(item.event);
        else
            await driver.up(item.event);
        elapsed = item.atMs;
    }
}
/** Dispatch wheel frames at their absolute timestamps. */
export async function dispatchScrollPlan(plan, driver, delay = systemDelay) {
    let elapsed = 0;
    for (const frame of plan.frames) {
        await delay(Math.max(0, frame.tMs - elapsed));
        await driver.wheel(frame.deltaY);
        elapsed = frame.tMs;
    }
}
/**
 * Stateful pointer convenience wrapper. `click()` refuses to dispatch a click
 * when the live endpoint no longer resolves to an interactive element.
 */
export class HumanPointer {
    driver;
    profile;
    delay;
    position;
    constructor(driver, initialPosition, profile, delay = systemDelay) {
        this.driver = driver;
        this.profile = profile;
        this.delay = delay;
        this.position = { ...initialPosition };
    }
    get currentPosition() {
        return { ...this.position };
    }
    async moveTo(target, seed) {
        const plan = synthesizeMovement(this.position, target, this.profile, seed);
        await dispatchTrajectory(plan, this.driver, this.delay);
        this.position = { ...plan.endpoint };
        return plan;
    }
    async click(target, seed) {
        const click = synthesizeClick(this.position, target, this.profile, seed);
        await dispatchTrajectory(click.trajectory, this.driver, this.delay);
        this.position = { ...click.trajectory.endpoint };
        await this.delay(click.postArrivalSettleMs);
        const interactive = await this.driver.isInteractiveAt(this.position.x, this.position.y);
        if (!interactive) {
            throw new Error('Refusing click: live endpoint is not interactive');
        }
        await this.driver.down();
        await this.delay(click.clickHoldMs);
        await this.driver.up();
        return { ...click, clickTargetIsInteractive: true };
    }
}
//# sourceMappingURL=dispatch.js.map