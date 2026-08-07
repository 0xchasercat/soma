export {};

/**
 * Content script for privacy-preserving behavioral capture.
 *
 * Only trusted browser events enter movement, keystroke, and scroll samples.
 * Keystroke content is never stored; only privacy-safe physical classifications are retained.
 */

interface CapturedMovement {
  sessionId: string;
  capturedAt: number;
  sourceSchema: 'soma.capture.v2';
  trustedEvents: true;
  trajectory: Array<{ x: number; y: number; tMs: number }>;
  clickTarget: {
    tagName: string;
    isInteractive: boolean;
    boundingBox: { x: number; y: number; width: number; height: number };
  };
  viewport: { width: number; height: number };
}

type KeyHand = 'left' | 'right' | 'none' | 'unknown';
type KeyKind = 'character' | 'space' | 'punctuation' | 'correction' | 'modifier' | 'navigation' | 'control' | 'other';
type DigraphClass = 'same-hand' | 'cross-hand' | 'after-space' | 'after-punctuation' | 'none';

interface CapturedKeyEvent {
  downMs: number;
  upMs: number;
  shiftKey: boolean;
  trusted: true;
  hand: KeyHand;
  keyKind: KeyKind;
  digraphClass: DigraphClass;
}

interface CapturedKeystroke {
  sessionId: string;
  capturedAt: number;
  sourceSchema: 'soma.capture.v2';
  trustedEvents: true;
  events: CapturedKeyEvent[];
  charCount: number;
  charsCommitted: number;
  inputTypes: string[];
  fieldType: string;
  correctionCount: number;
}

interface CapturedScrollFrame {
  tMs: number;
  deltaX: number;
  deltaY: number;
  /** Wheel deltas are always normalized to CSS pixels. */
  deltaMode: 0;
  isTrusted: true;
}

interface CapturedScroll {
  sessionId: string;
  capturedAt: number;
  sourceSchema: 'soma.capture.v2';
  trustedEvents: true;
  frames: CapturedScrollFrame[];
  concurrentHoverEventCount: number;
  peakAbsDeltaY: number;
}

type InteractionEventType = 'pointerdown' | 'pointerup' | 'click' | 'keydown' | 'keyup' | 'beforeinput' | 'input' | 'wheel' | 'focus';
type InteractionPointerType = 'mouse' | 'touch' | 'pen' | 'none' | 'unknown';

interface CapturedInteraction {
  sessionId: string;
  capturedAt: number;
  sourceSchema: 'soma.capture.v2';
  eventType: InteractionEventType;
  isTrusted: boolean;
  userActivation: boolean;
  pointerType: InteractionPointerType;
  sinceNavigationMs: number;
  actionReadySinceNavigationMs?: number;
  actionLatencyMs?: number;
}

type CaptureMessageType = 'store_movement' | 'store_keystroke' | 'store_scroll' | 'store_interaction';

const sessionId = `${Date.now()}_${Math.random().toString(36).slice(2)}`;
const sensitiveFieldPattern = /(?:pass(?:word|wd)?|secret|token|credential|auth|api[\s_-]*key|otp|one[\s_-]*time|user(?:name)?|login)/i;
const sensitiveAutocomplete = new Set([
  'current-password',
  'new-password',
  'one-time-code',
  'password',
  'username',
]);

// Writes are serialized while each event buffer is snapshotted synchronously.
// This prevents a slow service worker response from clearing a newer buffer.
let pendingWrites: Promise<void> = Promise.resolve();

function queueCapture(type: CaptureMessageType, data: CapturedMovement | CapturedKeystroke | CapturedScroll | CapturedInteraction): void {
  pendingWrites = pendingWrites
    .catch(() => undefined)
    .then(async () => {
      await chrome.runtime.sendMessage({ type, data });
    })
    .catch((error: unknown) => {
      console.error('[Soma] Failed to store capture:', error);
    });
}


function isElement(value: EventTarget | null): value is HTMLElement {
  return value instanceof HTMLElement;
}

function isCaptureField(field: HTMLElement): boolean {
  return field.tagName === 'INPUT' || field.tagName === 'TEXTAREA' || field.isContentEditable;
}

const CAPTURE_FIELD_SELECTOR = 'input,textarea,[contenteditable]:not([contenteditable="false"])';

function eventCaptureField(event: Event): HTMLElement | null {
  for (const candidate of event.composedPath()) {
    if (!(candidate instanceof HTMLElement)) continue;
    const field = candidate.closest<HTMLElement>(CAPTURE_FIELD_SELECTOR);
    if (field && isCaptureField(field)) return field;
  }
  return null;
}

function hasOpaqueCustomTarget(event: Event): boolean {
  const target = event.target;
  return target instanceof HTMLElement && target.tagName.includes('-') && target.shadowRoot === null &&
    event.composedPath()[0] === target;
}

function isSensitiveField(field: HTMLElement): boolean {
  if (!isCaptureField(field)) return true;

  if (field.tagName === 'INPUT') {
    const input = field as HTMLInputElement;
    const inputType = input.type.toLowerCase();
    if (inputType === 'password' || inputType === 'hidden') return true;
  }

  const attributes = [
    field.getAttribute('autocomplete'),
    field.getAttribute('name'),
    field.id,
    field.getAttribute('aria-label'),
    field.getAttribute('placeholder'),
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
  const autocomplete = (field.getAttribute('autocomplete') ?? '').toLowerCase().split(' ')[0];
  if (sensitiveAutocomplete.has(autocomplete) || sensitiveFieldPattern.test(attributes)) return true;

  const form = field.closest('form');
  if (form) {
    const formAttributes = [
      form.getAttribute('autocomplete'),
      form.getAttribute('name'),
      form.id,
      form.getAttribute('aria-label'),
    ]
      .filter(Boolean)
      .join(' ');
    if (sensitiveFieldPattern.test(formAttributes)) return true;
  }
  return false;
}

function resetTrajectory(): void {
  currentTrajectory = [];
  isCapturingTrajectory = false;
}

function resetKeystroke(): void {
  currentKeystrokes = [];
  keydownStates.clear();
  activeField = null;
  keystrokeCharCount = 0;
  keystrokeCorrectionCount = 0;
  keystrokeInputTypes = [];
  activeFieldLength = null;
  previousKeyContext = null;
}

function resetScroll(): void {
  currentScrollFrames = [];
  isCapturingScroll = false;
  scrollHoverCount = 0;
}

let currentTrajectory: Array<{ x: number; y: number; tMs: number }> = [];
let trajectoryStartTime = 0;
let isCapturingTrajectory = false;

let currentKeystrokes: CapturedKeyEvent[] = [];
let keystrokeStartTime = 0;
let keystrokeCharCount = 0;
let keystrokeCorrectionCount = 0;
let keystrokeInputTypes: string[] = [];
let activeField: HTMLElement | null = null;
let activeFieldLength: number | null = null;
let previousKeyContext: { hand: KeyHand; keyKind: KeyKind } | null = null;
const keydownStates = new Map<string, Omit<CapturedKeyEvent, 'upMs'>>();

let currentScrollFrames: CapturedScrollFrame[] = [];
let scrollStartTime = 0;
let scrollHoverCount = 0;
let isCapturingScroll = false;


const INTERACTIVE_SELECTOR = 'a,button,input,select,textarea,[role="button"],[role="link"],[onclick]';
const actionReadyTimes = new WeakMap<HTMLElement, number>();
function finite(value: number): boolean {
  return Number.isFinite(value);
}

function fieldLength(field: HTMLElement): number | null {
  if (field instanceof HTMLInputElement || field instanceof HTMLTextAreaElement) return Array.from(field.value).length;
  if (field.isContentEditable) return Array.from(field.textContent ?? '').length;
  return null;
}

function keyHand(code: string): KeyHand {
  if (/^(?:Key[QWERTASDFGZXCVB]|Digit[1-5]|Backquote|ShiftLeft|ControlLeft|AltLeft)$/.test(code)) return 'left';
  if (/^(?:Key[YUIOPHJKLNM]|Digit[6-9]|Digit0|Minus|Equal|BracketLeft|BracketRight|Backslash|Semicolon|Quote|Comma|Period|Slash|ShiftRight|ControlRight|AltRight)$/.test(code)) return 'right';
  if (code === 'Space') return 'none';
  return 'unknown';
}

function keyKind(key: string, code: string): KeyKind {
  if (key === 'Backspace' || key === 'Delete') return 'correction';
  if (key === ' ' || code === 'Space') return 'space';
  if (key.length === 1) return /[\p{L}\p{N}]/u.test(key) ? 'character' : 'punctuation';
  if (/^(?:Shift|Control|Alt|Meta|CapsLock|NumLock|ScrollLock)/.test(key)) return 'modifier';
  if (/^(?:Arrow|Home$|End$|PageUp$|PageDown$)/.test(key)) return 'navigation';
  if (/^(?:Enter|Tab|Escape)$/.test(key)) return 'control';
  return 'other';
}

function digraphClass(hand: KeyHand, kind: KeyKind): DigraphClass {
  if (!previousKeyContext) return 'none';
  if (previousKeyContext.keyKind === 'space') return 'after-space';
  if (previousKeyContext.keyKind === 'punctuation') return 'after-punctuation';
  if ((hand === 'left' || hand === 'right') && (previousKeyContext.hand === 'left' || previousKeyContext.hand === 'right')) {
    return hand === previousKeyContext.hand ? 'same-hand' : 'cross-hand';
  }
  return 'none';
}

function interactionPointerType(event: Event): InteractionPointerType {
  if (event instanceof PointerEvent) {
    return event.pointerType === 'mouse' || event.pointerType === 'touch' || event.pointerType === 'pen'
      ? event.pointerType
      : 'unknown';
  }
  if (event instanceof MouseEvent || event instanceof WheelEvent) return 'mouse';
  return 'none';
}

function captureInteraction(event: Event): void {
  if (hasOpaqueCustomTarget(event)) return;
  const captureField = eventCaptureField(event);
  if (captureField && isSensitiveField(captureField)) return;
  const sinceNavigationMs = Math.max(0, performance.now());
  let eventTarget: HTMLElement | null = null;
  for (const candidate of event.composedPath()) {
    if (!(candidate instanceof Element)) continue;
    eventTarget = interactiveAncestor(candidate);
    if (eventTarget) break;
  }
  const readyAt = eventTarget ? actionReadyTimes.get(eventTarget) ?? 0 : undefined;
  const interaction: CapturedInteraction = {
    sessionId,
    capturedAt: Date.now(),
    sourceSchema: 'soma.capture.v2',
    eventType: event.type as InteractionEventType,
    isTrusted: event.isTrusted,
    userActivation: event.isTrusted && navigator.userActivation?.isActive === true,
    pointerType: interactionPointerType(event),
    sinceNavigationMs,
    ...(readyAt === undefined ? {} : {
      actionReadySinceNavigationMs: readyAt,
      actionLatencyMs: Math.max(0, sinceNavigationMs - readyAt),
    }),
  };
  queueCapture('store_interaction', interaction);
}

function startTrajectoryCapture(e: MouseEvent): void {
  if (!e.isTrusted || isCapturingTrajectory || !finite(e.clientX) || !finite(e.clientY)) return;
  isCapturingTrajectory = true;
  trajectoryStartTime = performance.now();
  currentTrajectory = [{ x: e.clientX, y: e.clientY, tMs: 0 }];
}

function captureMouseMove(e: MouseEvent): void {
  if (!e.isTrusted || !finite(e.clientX) || !finite(e.clientY)) return;
  if (!isCapturingTrajectory) startTrajectoryCapture(e);
  if (!isCapturingTrajectory) return;
  currentTrajectory.push({
    x: e.clientX,
    y: e.clientY,
    tMs: Math.max(0, performance.now() - trajectoryStartTime),
  });
}

function interactiveAncestor(target: Element | null): HTMLElement | null {
  if (!(target instanceof HTMLElement)) return null;
  return target.closest<HTMLElement>(INTERACTIVE_SELECTOR);
}

function isActionable(element: HTMLElement): boolean {
  if (!element.matches(INTERACTIVE_SELECTOR) || element.hidden || element.getAttribute('aria-disabled') === 'true') return false;
  return !('disabled' in element) || (element as HTMLButtonElement | HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement).disabled !== true;
}

function updateActionReadyState(element: HTMLElement, atMs: number): void {
  if (isActionable(element)) {
    if (!actionReadyTimes.has(element)) actionReadyTimes.set(element, atMs);
  } else {
    actionReadyTimes.delete(element);
  }
}

function markActionableTree(node: Node, atMs: number): void {
  if (!(node instanceof HTMLElement)) return;
  updateActionReadyState(node, atMs);
  for (const element of Array.from(node.querySelectorAll<HTMLElement>(INTERACTIVE_SELECTOR))) updateActionReadyState(element, atMs);
}

function endTrajectoryCapture(e?: MouseEvent): void {
  if (e && (!e.isTrusted || e.button !== 0)) return;
  if (!isCapturingTrajectory) return;

  if (e && finite(e.clientX) && finite(e.clientY)) {
    currentTrajectory.push({
      x: e.clientX,
      y: e.clientY,
      tMs: Math.max(0, performance.now() - trajectoryStartTime),
    });
  }
  const trajectory = currentTrajectory.slice();
  const hit = e ? document.elementFromPoint(e.clientX, e.clientY) : null;
  const interactive = interactiveAncestor(hit);
  const target = interactive ?? (hit instanceof HTMLElement ? hit : null);
  resetTrajectory();
  // Preserve sparse but genuine trusted movements. Downstream measurement can
  // extract a finite feature vector from two temporally distinct samples.
  if (trajectory.length < 2) return;

  const rect = target?.getBoundingClientRect();
  const width = rect?.width ?? 0;
  const height = rect?.height ?? 0;
  const movement: CapturedMovement = {
    sessionId,
    capturedAt: Date.now(),
    sourceSchema: 'soma.capture.v2',
    trustedEvents: true,
    trajectory,
    clickTarget: {
      tagName: target?.tagName ?? 'unknown',
      isInteractive: interactive !== null,
      boundingBox: rect && finite(rect.left) && finite(rect.top) && finite(width) && finite(height)
        ? { x: rect.left, y: rect.top, width, height }
        : { x: 0, y: 0, width: 0, height: 0 },
    },
    viewport: { width: window.innerWidth, height: window.innerHeight },
  };
  queueCapture('store_movement', movement);
}

function startKeystrokeCapture(field: HTMLElement): void {
  if (!isCaptureField(field) || isSensitiveField(field)) {
    if (activeField) endKeystrokeCapture();
    return;
  }
  if (activeField === field) return;
  if (activeField) endKeystrokeCapture();
  activeField = field;
  activeFieldLength = fieldLength(field);
  keystrokeStartTime = performance.now();
  currentKeystrokes = [];
  keydownStates.clear();
  keystrokeCharCount = 0;
  keystrokeCorrectionCount = 0;
  keystrokeInputTypes = [];
  previousKeyContext = null;
}

function captureKeydown(e: KeyboardEvent): void {
  if (hasOpaqueCustomTarget(e) || e.ctrlKey || e.metaKey || e.altKey || !e.isTrusted || !activeField ||
      eventCaptureField(e) !== activeField || isSensitiveField(activeField) || keydownStates.has(e.code)) return;
  const hand = keyHand(e.code);
  const kind = keyKind(e.key, e.code);
  const downMs = Math.max(0, performance.now() - keystrokeStartTime);
  keydownStates.set(e.code, {
    downMs,
    shiftKey: e.shiftKey,
    trusted: true,
    hand,
    keyKind: kind,
    digraphClass: digraphClass(hand, kind),
  });
  if (kind === 'character' || kind === 'space' || kind === 'punctuation') previousKeyContext = { hand, keyKind: kind };
  if (kind === 'correction') keystrokeCorrectionCount++;
}

function captureKeyup(e: KeyboardEvent): void {
  if (hasOpaqueCustomTarget(e) || !e.isTrusted || !activeField || eventCaptureField(e) !== activeField || isSensitiveField(activeField)) return;
  const downState = keydownStates.get(e.code);
  if (!downState) return;
  currentKeystrokes.push({
    ...downState,
    upMs: Math.max(downState.downMs, performance.now() - keystrokeStartTime),
  });
  keydownStates.delete(e.code);
}

function captureInput(e: Event): void {
  if (hasOpaqueCustomTarget(e) || !e.isTrusted || !activeField || eventCaptureField(e) !== activeField || isSensitiveField(activeField)) return;
  const input = e as InputEvent;
  if (input.inputType) keystrokeInputTypes.push(input.inputType);
  const nextLength = fieldLength(activeField);
  const lengthDelta = nextLength !== null && activeFieldLength !== null ? nextLength - activeFieldLength : null;
  if (input.inputType.startsWith('insert')) {
    const dataLength = Array.from(input.data ?? '').length;
    keystrokeCharCount += Math.max(1, dataLength, lengthDelta ?? 0);
  } else if (input.inputType.startsWith('delete')) {
    keystrokeCharCount = Math.max(0, keystrokeCharCount + Math.min(-1, lengthDelta ?? -1));
  }
  activeFieldLength = nextLength;
}

function endKeystrokeCapture(): void {
  if (!activeField) return;
  const field = activeField;
  const events = currentKeystrokes.slice().sort((a, b) => a.downMs - b.downMs);
  const charsCommitted = Math.max(0, keystrokeCharCount);
  const inputTypes = keystrokeInputTypes.slice();
  const correctionCount = keystrokeCorrectionCount;
  const fieldType = field.tagName === 'INPUT'
    ? (field as HTMLInputElement).type
    : field.tagName === 'TEXTAREA' ? 'textarea' : 'contenteditable';
  resetKeystroke();
  if ((events.length === 0 && charsCommitted === 0 && inputTypes.length === 0) || isSensitiveField(field)) return;

  const keystroke: CapturedKeystroke = {
    sessionId,
    capturedAt: Date.now(),
    sourceSchema: 'soma.capture.v2',
    trustedEvents: true,
    events,
    charCount: charsCommitted,
    charsCommitted,
    inputTypes,
    fieldType,
    correctionCount,
  };
  queueCapture('store_keystroke', keystroke);
}

function wheelScale(deltaMode: number, axis: 'x' | 'y'): number | null {
  if (deltaMode === 0) return 1;
  if (deltaMode === 1) {
    const rootLineHeight = Number.parseFloat(getComputedStyle(document.documentElement).lineHeight);
    return finite(rootLineHeight) && rootLineHeight > 0 ? rootLineHeight : 16;
  }
  if (deltaMode === 2) return axis === 'x' ? window.innerWidth : window.innerHeight;
  return null;
}

function normalizeWheelDelta(delta: number, deltaMode: number, axis: 'x' | 'y'): number | null {
  if (!finite(delta)) return null;
  const scale = wheelScale(deltaMode, axis);
  if (scale === null || !finite(scale)) return null;
  const normalized = delta * scale;
  return finite(normalized) ? Math.max(-100000, Math.min(100000, normalized)) : null;
}

function startScrollCapture(): void {
  if (isCapturingScroll) return;
  isCapturingScroll = true;
  scrollStartTime = performance.now();
  currentScrollFrames = [];
  scrollHoverCount = 0;
}

function captureWheel(e: WheelEvent): void {
  if (!e.isTrusted) return;
  const deltaX = normalizeWheelDelta(e.deltaX, e.deltaMode, 'x');
  const deltaY = normalizeWheelDelta(e.deltaY, e.deltaMode, 'y');
  if (deltaX === null || deltaY === null) return;
  if (!isCapturingScroll) startScrollCapture();
  currentScrollFrames.push({
    tMs: Math.max(0, performance.now() - scrollStartTime),
    deltaX,
    deltaY,
    deltaMode: 0,
    isTrusted: true,
  });
}

function captureHoverEvent(e: MouseEvent): void {
  if (e.isTrusted && isCapturingScroll) scrollHoverCount++;
}

function endScrollCapture(): void {
  if (!isCapturingScroll) return;
  const frames = currentScrollFrames.slice();
  const hoverCount = scrollHoverCount;
  resetScroll();
  if (frames.length < 3) return;

  const scroll: CapturedScroll = {
    sessionId,
    capturedAt: Date.now(),
    sourceSchema: 'soma.capture.v2',
    trustedEvents: true,
    frames,
    concurrentHoverEventCount: hoverCount,
    peakAbsDeltaY: Math.max(...frames.map((frame) => Math.abs(frame.deltaY)), 0),
  };
  queueCapture('store_scroll', scroll);
}

function terminateTrajectoryOnDocumentExit(e: MouseEvent): void {
  if (!e.isTrusted || !isCapturingTrajectory) return;
  const related = e.relatedTarget;
  const enteredChildFrame = related instanceof HTMLIFrameElement || related instanceof HTMLFrameElement;
  const leftDocument = related === null || (related instanceof Node && related.ownerDocument !== document);
  if (enteredChildFrame || leftDocument) resetTrajectory();
}

const actionReadyObserver = new MutationObserver((mutations) => {
  const atMs = Math.max(0, performance.now());
  for (const mutation of mutations) {
    if (mutation.type === 'childList') {
      for (const node of Array.from(mutation.addedNodes)) markActionableTree(node, atMs);
    } else if (mutation.target instanceof HTMLElement) {
      updateActionReadyState(mutation.target, atMs);
    }
  }
});
actionReadyObserver.observe(document, {
  subtree: true,
  childList: true,
  attributes: true,
  attributeFilter: ['disabled', 'aria-disabled', 'hidden'],
});

// Pointer trajectories are frame-local trusted movement streams finalized by a click.
document.addEventListener('mousemove', captureMouseMove, { passive: true });
document.addEventListener('mouseup', endTrajectoryCapture, { passive: true });
document.addEventListener('mouseout', terminateTrajectoryOnDocumentExit, { passive: true });
document.addEventListener('keydown', captureKeydown, { passive: true });
document.addEventListener('keyup', captureKeyup, { passive: true });
document.addEventListener('input', captureInput, { passive: true });
document.addEventListener('focus', (e) => {
  if (hasOpaqueCustomTarget(e)) return;
  const field = eventCaptureField(e);
  if (e.isTrusted && field) startKeystrokeCapture(field);
}, true);
document.addEventListener('blur', (e) => {
  if (!hasOpaqueCustomTarget(e) && e.isTrusted && eventCaptureField(e) === activeField) endKeystrokeCapture();
}, true);
document.addEventListener('wheel', captureWheel, { passive: true });
document.addEventListener('mouseover', captureHoverEvent, { passive: true });
document.addEventListener('mouseout', captureHoverEvent, { passive: true });

for (const eventType of ['pointerdown', 'pointerup', 'click', 'keydown', 'keyup', 'beforeinput', 'input', 'wheel', 'focus'] as const) {
  document.addEventListener(eventType, captureInteraction, { capture: true, passive: true });
}

let scrollTimeout: number | undefined;
document.addEventListener('wheel', (e) => {
  if (!e.isTrusted) return;
  if (scrollTimeout !== undefined) window.clearTimeout(scrollTimeout);
  scrollTimeout = window.setTimeout(endScrollCapture, 500);
}, { passive: true });

function flushCaptureBuffers(): void {
  if (scrollTimeout !== undefined) window.clearTimeout(scrollTimeout);
  scrollTimeout = undefined;
  // A pointer trajectory is valid only when a trusted click finalized it.
  resetTrajectory();
  endKeystrokeCapture();
  endScrollCapture();
}

window.addEventListener('pagehide', flushCaptureBuffers, { passive: true });
window.addEventListener('beforeunload', flushCaptureBuffers, { passive: true });
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    flushCaptureBuffers();
  } else if (document.visibilityState === 'visible' && isElement(document.activeElement)) {
    startKeystrokeCapture(document.activeElement);
  }
});
