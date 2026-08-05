export {};

/**
 * Popup script: display stats and export captured data.
 *
 * Stats and export read the shared extension IndexedDB directly. chrome.runtime
 * messages cannot carry captures larger than 64 MiB.
 */

import { clearAllData, getCounts, streamStore, STORE_NAMES } from '../db.js';

const movementCountEl = document.getElementById('movementCount')!;
const keystrokeCountEl = document.getElementById('keystrokeCount')!;
const scrollCountEl = document.getElementById('scrollCount')!;
const exportBtn = document.getElementById('exportBtn')!;
const refreshBtn = document.getElementById('refreshBtn')!;
const clearBtn = document.getElementById('clearBtn')!;

async function refreshStats() {
  const counts = await getCounts();
  movementCountEl.textContent = counts.movements.toString();
  keystrokeCountEl.textContent = counts.keystrokes.toString();
  scrollCountEl.textContent = counts.scrolls.toString();
}

/**
 * Export every capture store as one JSON file.
 *
 * The payload is assembled as an ARRAY of Blob parts, never a single string:
 * `JSON.stringify` on a full multi-10k-gesture capture exceeds V8's maximum
 * string length and throws `Invalid string length`. Each record is stringified
 * on its own and the surrounding JSON structure is emitted as literal
 * fragments, so no individual string is ever larger than one record and the
 * Blob does the concatenation off-heap.
 */
async function exportData() {
  exportBtn.setAttribute('disabled', 'true');
  const originalLabel = exportBtn.textContent;
  try {
    const exportedAt = new Date().toISOString();
    const parts: BlobPart[] = [];
    const header = {
      schema: 'soma.capture.v2' as const,
      schemaVersion: 2 as const,
      producer: 'soma-extension/0.2.0',
      exportId: crypto.randomUUID(),
      exportedAt,
      userAgent: navigator.userAgent,
    };

    // Open the object and write the scalar header fields, minus the closing brace.
    parts.push(JSON.stringify(header).slice(0, -1));

    for (const storeName of STORE_NAMES) {
      parts.push(`,${JSON.stringify(storeName)}:[`);
      let written = 0;
      await streamStore(storeName, 500, (batch) => {
        for (const record of batch) {
          parts.push(written === 0 ? JSON.stringify(record) : `,${JSON.stringify(record)}`);
          written++;
        }
        exportBtn.textContent = `Exporting ${storeName} ${written}…`;
      });
      parts.push(']');
    }
    parts.push('}');

    const blob = new Blob(parts, { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const timestamp = exportedAt.replace(/[:.]/g, '-').slice(0, -5);
    const a = document.createElement('a');
    a.href = url;
    a.download = `soma-capture-${timestamp}.json`;
    a.click();
    URL.revokeObjectURL(url);
  } finally {
    exportBtn.textContent = originalLabel;
    exportBtn.removeAttribute('disabled');
  }
}

async function clearData() {
  if (!confirm('Clear all captured data? This cannot be undone.')) return;
  await clearAllData();
  await refreshStats();
}

exportBtn.addEventListener('click', () => {
  void exportData().catch((error: unknown) => {
    console.error('[Soma] Export failed:', error);
    alert(error instanceof Error ? error.message : 'Export failed');
  });
});
refreshBtn.addEventListener('click', () => {
  void refreshStats().catch((error: unknown) => {
    console.error('[Soma] Refresh failed:', error);
  });
});
clearBtn.addEventListener('click', () => {
  void clearData().catch((error: unknown) => {
    console.error('[Soma] Clear failed:', error);
    alert(error instanceof Error ? error.message : 'Clear failed');
  });
});

// Initial stats load.
void refreshStats().catch((error: unknown) => {
  console.error('[Soma] Initial stats failed:', error);
});
