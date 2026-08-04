export {};

/**
 * Popup script: display stats and export captured data.
 *
 * Stats and export read the shared extension IndexedDB directly. chrome.runtime
 * messages cannot carry captures larger than 64 MiB.
 */

import { clearAllData, getAllData, getCounts } from '../db.js';

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

async function exportData() {
  exportBtn.setAttribute('disabled', 'true');
  try {
    const data = await getAllData();
    const exportedAt = new Date().toISOString();
    const exportPayload = {
      schema: 'soma.capture.v2' as const,
      schemaVersion: 2 as const,
      producer: 'soma-extension/0.2.0',
      exportId: crypto.randomUUID(),
      exportedAt,
      userAgent: navigator.userAgent,
      movements: data.movements,
      keystrokes: data.keystrokes,
      scrolls: data.scrolls,
      interactions: data.interactions,
    };

    // Compact JSON — pretty-print roughly doubles memory for multi‑10MB captures.
    const json = JSON.stringify(exportPayload);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const timestamp = exportedAt.replace(/[:.]/g, '-').slice(0, -5);
    const a = document.createElement('a');
    a.href = url;
    a.download = `soma-capture-${timestamp}.json`;
    a.click();
    URL.revokeObjectURL(url);
  } finally {
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
