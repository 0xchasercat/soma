export {};

/**
 * Popup script: display stats and export captured data.
 */

const movementCountEl = document.getElementById('movementCount')!;
const keystrokeCountEl = document.getElementById('keystrokeCount')!;
const scrollCountEl = document.getElementById('scrollCount')!;
const exportBtn = document.getElementById('exportBtn')!;
const refreshBtn = document.getElementById('refreshBtn')!;
const clearBtn = document.getElementById('clearBtn')!;

type CaptureData = {
  movements: unknown[];
  keystrokes: unknown[];
  scrolls: unknown[];
  interactions: unknown[];
  ok?: boolean;
  error?: string;
};

async function getCaptureData(): Promise<CaptureData> {
  const response = await chrome.runtime.sendMessage({ type: 'get_all_data' }) as CaptureData;
  if (response.ok === false || !Array.isArray(response.movements) || !Array.isArray(response.keystrokes) ||
      !Array.isArray(response.scrolls) || !Array.isArray(response.interactions)) {
    throw new Error(response.error ?? 'Unable to read captured data');
  }
  return response;
}

async function refreshStats() {
  const data = await getCaptureData();
  movementCountEl.textContent = data.movements.length.toString();
  keystrokeCountEl.textContent = data.keystrokes.length.toString();
  scrollCountEl.textContent = data.scrolls.length.toString();
}

async function exportData() {
  const data = await getCaptureData();
  const exportedAt = new Date().toISOString();
  const exportPayload = {
    schema: 'soma.capture.v2',
    schemaVersion: 2,
    producer: 'soma-extension/0.2.0',
    exportId: crypto.randomUUID(),
    exportedAt,
    userAgent: navigator.userAgent,
    movements: data.movements,
    keystrokes: data.keystrokes,
    scrolls: data.scrolls,
    interactions: data.interactions,
  };

  const json = JSON.stringify(exportPayload, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const timestamp = exportedAt.replace(/[:.]/g, '-').slice(0, -5);
  const a = document.createElement('a');
  a.href = url;
  a.download = `soma-capture-${timestamp}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

async function clearData() {
  if (!confirm('Clear all captured data? This cannot be undone.')) return;
  await chrome.runtime.sendMessage({ type: 'clear_all_data' });
  await refreshStats();
}

exportBtn.addEventListener('click', exportData);
refreshBtn.addEventListener('click', refreshStats);
clearBtn.addEventListener('click', clearData);

// Initial stats load.
refreshStats();
