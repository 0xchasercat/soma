export {};

/**
 * Background service worker: validates captured data and stores it in IndexedDB.
 * Records use a random id rather than a wall-clock key so same-millisecond
 * captures cannot overwrite one another.
 *
 * Full capture payloads are never returned over chrome.runtime messaging —
 * the popup reads IndexedDB directly so exports can exceed the 64 MiB limit.
 */

import {
  clearAllData,
  getCounts,
  isRecord,
  storeData,
  type StoreName,
} from './db.js';

const STORE_BY_MESSAGE: Record<string, StoreName> = {
  store_movement: 'movements',
  store_keystroke: 'keystrokes',
  store_scroll: 'scrolls',
  store_interaction: 'interactions',
};

chrome.runtime.onMessage.addListener((message: unknown, _sender, sendResponse) => {
  const request = isRecord(message) ? message : {};
  const type = typeof request.type === 'string' ? request.type : '';
  let operation: Promise<unknown> | null = null;

  const storeName = STORE_BY_MESSAGE[type];
  if (storeName) {
    operation = storeData(storeName, request.data).then(() => ({ ok: true }));
  } else if (type === 'get_counts') {
    operation = getCounts().then((counts) => ({ ok: true, ...counts }));
  } else if (type === 'clear_all_data') {
    operation = clearAllData().then(() => ({ ok: true }));
  } else if (type === 'get_all_data') {
    // Kept only as an explicit rejection so old popup builds fail clearly.
    operation = Promise.resolve({
      ok: false,
      error: 'Full capture export must read IndexedDB from the popup (runtime messages cap at 64 MiB)',
    });
  } else {
    return false;
  }

  operation.then(sendResponse).catch((error: unknown) => {
    sendResponse({ ok: false, error: error instanceof Error ? error.message : 'Capture operation failed' });
  });
  return true;
});

console.log('[Soma Capture] Background service worker loaded');
