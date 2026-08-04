/**
 * Shared IndexedDB access for the Soma capture extension.
 *
 * The service worker and popup share the chrome-extension:// origin, so both
 * can open this database. Large exports must read here directly — chrome.runtime
 * messages are capped at 64 MiB.
 */

export const DB_NAME = 'soma_capture';
export const DB_VERSION = 3;
export const LEGACY_STORE_NAMES = ['movements', 'keystrokes', 'scrolls'] as const;
export const STORE_NAMES = [...LEGACY_STORE_NAMES, 'interactions'] as const;
export type StoreName = (typeof STORE_NAMES)[number];
export type LegacyStoreName = (typeof LEGACY_STORE_NAMES)[number];

export type CaptureStores = {
  movements: unknown[];
  keystrokes: unknown[];
  scrolls: unknown[];
  interactions: unknown[];
};

export type CaptureCounts = {
  movements: number;
  keystrokes: number;
  scrolls: number;
  interactions: number;
};

let db: IDBDatabase | null = null;
let opening: Promise<IDBDatabase> | null = null;

export function recordId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

const KEY_HANDS = new Set(['left', 'right', 'none', 'unknown']);
const KEY_KINDS = new Set(['character', 'space', 'punctuation', 'correction', 'modifier', 'navigation', 'control', 'other']);
const DIGRAPH_CLASSES = new Set(['same-hand', 'cross-hand', 'after-space', 'after-punctuation', 'none']);
const INTERACTION_TYPES = new Set(['pointerdown', 'pointerup', 'click', 'keydown', 'keyup', 'beforeinput', 'input', 'wheel', 'focus']);
const POINTER_TYPES = new Set(['mouse', 'touch', 'pen', 'none', 'unknown']);

function member(value: unknown, values: Set<string>): value is string {
  return typeof value === 'string' && values.has(value);
}

function isTrustedMovement(data: unknown): boolean {
  if (!isRecord(data) || data.trustedEvents !== true || !Array.isArray(data.trajectory)) return false;
  return data.trajectory.length >= 5 && data.trajectory.every((point) => (
    isRecord(point) && finite(point.x) && finite(point.y) && finite(point.tMs)
  ));
}

function isTrustedKeystroke(data: unknown): boolean {
  if (!isRecord(data) || data.trustedEvents !== true || !Array.isArray(data.events) ||
      !finite(data.charCount) || !finite(data.charsCommitted) || !Array.isArray(data.inputTypes)) return false;
  if (!data.inputTypes.every((inputType) => typeof inputType === 'string' && /^[A-Za-z][A-Za-z0-9]{0,63}$/.test(inputType))) return false;
  if (data.events.length === 0 && data.charsCommitted === 0 && data.inputTypes.length === 0) return false;
  let previousDownMs = -Infinity;
  return data.events.every((event) => {
    if (!isRecord(event) || 'key' in event || 'code' in event || event.trusted !== true ||
        !finite(event.downMs) || !finite(event.upMs) || event.upMs < event.downMs ||
        typeof event.shiftKey !== 'boolean' || !member(event.hand, KEY_HANDS) ||
        !member(event.keyKind, KEY_KINDS) || !member(event.digraphClass, DIGRAPH_CLASSES) ||
        event.downMs < previousDownMs) return false;
    previousDownMs = event.downMs;
    return true;
  });
}

function isTrustedScroll(data: unknown): boolean {
  if (!isRecord(data) || data.trustedEvents !== true || !Array.isArray(data.frames)) return false;
  return data.frames.length >= 3 && data.frames.every((frame) => (
    isRecord(frame) && frame.isTrusted === true && frame.deltaMode === 0 &&
    finite(frame.tMs) && finite(frame.deltaX) && finite(frame.deltaY)
  ));
}

function isInteraction(data: unknown): boolean {
  if (!isRecord(data) || !member(data.eventType, INTERACTION_TYPES) || typeof data.isTrusted !== 'boolean' ||
      typeof data.userActivation !== 'boolean' || !member(data.pointerType, POINTER_TYPES) ||
      !finite(data.sinceNavigationMs) || data.sinceNavigationMs < 0 || !finite(data.capturedAt) ||
      typeof data.sessionId !== 'string') return false;
  const hasReady = 'actionReadySinceNavigationMs' in data || 'actionLatencyMs' in data;
  if (!hasReady) return true;
  return finite(data.actionReadySinceNavigationMs) && data.actionReadySinceNavigationMs >= 0 &&
    data.actionReadySinceNavigationMs <= data.sinceNavigationMs && finite(data.actionLatencyMs) && data.actionLatencyMs >= 0 &&
    Math.abs(data.actionLatencyMs - (data.sinceNavigationMs - data.actionReadySinceNavigationMs)) < 1e-6;
}

export function isTrustedData(storeName: StoreName, data: unknown): boolean {
  if (storeName === 'movements') return isTrustedMovement(data);
  if (storeName === 'keystrokes') return isTrustedKeystroke(data);
  if (storeName === 'scrolls') return isTrustedScroll(data);
  return isInteraction(data);
}

function migrationId(storeName: LegacyStoreName, value: unknown, index: number): string {
  const session = isRecord(value) && typeof value.sessionId === 'string' ? value.sessionId : 'legacy';
  const capturedAt = isRecord(value) && finite(value.capturedAt) ? value.capturedAt : 0;
  return `legacy-${storeName}-${session}-${capturedAt}-${index}-${recordId()}`;
}

function migratedRecord(storeName: LegacyStoreName, value: Record<string, unknown>, sourceSchema: 'soma.capture.v1' | 'soma.capture.v2'): Record<string, unknown> {
  if (storeName !== 'keystrokes' || !Array.isArray(value.events)) return { ...value, sourceSchema };
  const events = value.events.map((candidate) => {
    if (!isRecord(candidate)) return candidate;
    const { key: _key, code: _code, ...event } = candidate;
    return {
      ...event,
      hand: member(event.hand, KEY_HANDS) ? event.hand : 'unknown',
      keyKind: member(event.keyKind, KEY_KINDS) ? event.keyKind : 'other',
      digraphClass: member(event.digraphClass, DIGRAPH_CLASSES) ? event.digraphClass : 'none',
    };
  });
  const charCount = finite(value.charCount) ? Math.max(0, value.charCount) : 0;
  return {
    ...value,
    sourceSchema,
    events,
    charCount,
    charsCommitted: finite(value.charsCommitted) ? Math.max(0, value.charsCommitted) : charCount,
    inputTypes: Array.isArray(value.inputTypes) ? value.inputTypes : [],
  };
}

function migrateStore(database: IDBDatabase, transaction: IDBTransaction, storeName: LegacyStoreName): void {
  const tempName = `${storeName}_legacy_v1`;
  if (database.objectStoreNames.contains(tempName)) database.deleteObjectStore(tempName);
  const temp = database.createObjectStore(tempName, { keyPath: 'id' });
  const old = transaction.objectStore(storeName);
  let index = 0;
  const cursorRequest = old.openCursor();
  cursorRequest.onerror = () => transaction.abort();
  cursorRequest.onsuccess = () => {
    const cursor = cursorRequest.result;
    if (!cursor) {
      database.deleteObjectStore(storeName);
      const destination = database.createObjectStore(storeName, { keyPath: 'id' });
      const tempCursorRequest = temp.openCursor();
      tempCursorRequest.onerror = () => transaction.abort();
      tempCursorRequest.onsuccess = () => {
        const tempCursor = tempCursorRequest.result;
        if (!tempCursor) {
          database.deleteObjectStore(tempName);
          return;
        }
        destination.put(tempCursor.value);
        tempCursor.continue();
      };
      return;
    }
    const value = cursor.value;
    if (isRecord(value)) {
      temp.put({ ...migratedRecord(storeName, value, 'soma.capture.v1'), id: migrationId(storeName, value, index++) });
    }
    cursor.continue();
  };
}

function tagExistingStore(transaction: IDBTransaction, storeName: LegacyStoreName): void {
  const request = transaction.objectStore(storeName).openCursor();
  request.onerror = () => transaction.abort();
  request.onsuccess = () => {
    const cursor = request.result;
    if (!cursor) return;
    if (isRecord(cursor.value)) {
      const sourceSchema = typeof cursor.value.id === 'string' && cursor.value.id.startsWith('legacy-')
        ? 'soma.capture.v1'
        : 'soma.capture.v2';
      cursor.update(migratedRecord(storeName, cursor.value, sourceSchema));
    }
    cursor.continue();
  };
}

export function openDB(): Promise<IDBDatabase> {
  if (db) return Promise.resolve(db);
  if (opening) return opening;
  opening = new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => {
      opening = null;
      reject(request.error ?? new Error('Unable to open capture database'));
    };
    request.onblocked = () => {
      opening = null;
      reject(new Error('Capture database upgrade is blocked by another tab'));
    };
    request.onupgradeneeded = (event) => {
      const database = request.result;
      const transaction = request.transaction;
      if (!transaction) {
        reject(new Error('Missing IndexedDB upgrade transaction'));
        return;
      }
      if (event.oldVersion < 2) {
        if (event.oldVersion === 1) {
          for (const storeName of LEGACY_STORE_NAMES) {
            if (database.objectStoreNames.contains(storeName)) migrateStore(database, transaction, storeName);
          }
        } else {
          for (const storeName of LEGACY_STORE_NAMES) database.createObjectStore(storeName, { keyPath: 'id' });
        }
      }
      if (event.oldVersion < 3) {
        if (event.oldVersion === 2) {
          for (const storeName of LEGACY_STORE_NAMES) tagExistingStore(transaction, storeName);
        }
        if (!database.objectStoreNames.contains('interactions')) {
          database.createObjectStore('interactions', { keyPath: 'id' });
        }
      }
    };
    request.onsuccess = () => {
      const opened = request.result;
      opened.onversionchange = () => {
        opened.close();
        if (db === opened) db = null;
      };
      opened.onerror = () => {
        if (db === opened) db = null;
      };
      db = opened;
      opening = null;
      resolve(opened);
    };
  });
  return opening;
}

export function withTransaction<T>(
  database: IDBDatabase,
  storeName: StoreName,
  mode: IDBTransactionMode,
  action: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let result: T;
    let requestError: DOMException | null = null;
    try {
      const transaction = database.transaction(storeName, mode);
      const request = action(transaction.objectStore(storeName));
      request.onsuccess = () => { result = request.result; };
      request.onerror = () => { requestError = request.error; };
      transaction.oncomplete = () => requestError ? reject(requestError) : resolve(result!);
      transaction.onabort = () => reject(transaction.error ?? requestError ?? new Error('IndexedDB transaction aborted'));
      transaction.onerror = () => { requestError = transaction.error ?? requestError; };
    } catch (error) {
      reject(error);
    }
  });
}

export async function storeData(storeName: StoreName, data: unknown): Promise<void> {
  if (!isTrustedData(storeName, data)) throw new Error('Rejected invalid capture record');
  const recordData = data as Record<string, unknown>;
  if (recordData.sourceSchema !== 'soma.capture.v2') throw new Error('Rejected capture with invalid source schema');
  const database = await openDB();
  const record = { ...recordData, id: recordId() };
  await withTransaction(database, storeName, 'readwrite', (store) => store.add(record));
}

export async function getAllData(): Promise<CaptureStores> {
  const database = await openDB();
  const [movements, keystrokes, scrolls, interactions] = await Promise.all(STORE_NAMES.map((storeName) => (
    withTransaction(database, storeName, 'readonly', (store) => store.getAll())
  )));
  return { movements, keystrokes, scrolls, interactions };
}

/** Counts only — safe for popup stats when the capture set exceeds the runtime message size limit. */
export async function getCounts(): Promise<CaptureCounts> {
  const database = await openDB();
  const [movements, keystrokes, scrolls, interactions] = await Promise.all(STORE_NAMES.map((storeName) => (
    withTransaction(database, storeName, 'readonly', (store) => store.count())
  )));
  return { movements, keystrokes, scrolls, interactions };
}

export async function clearAllData(): Promise<void> {
  const database = await openDB();
  await Promise.all(STORE_NAMES.map((storeName) => (
    withTransaction(database, storeName, 'readwrite', (store) => store.clear())
  )));
}
