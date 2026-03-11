const runBtn = document.getElementById('runBtn');
const statusEl = document.getElementById('status');
const liveStatusEl = document.getElementById('liveStatus');
const promptEl = document.getElementById('prompt');
const forwardToggleBtn = document.getElementById('forwardToggleBtn');

// Per-backend DOM elements
const backends = {
  baseline: {
    outputEl: document.getElementById('baselineOutput'),
    metricsCard: document.getElementById('baselineGatewayMetrics'),
    kvMetricsCard: document.getElementById('baselineKvMetrics'),
    sessions: new Map(),
    activeRequestId: null,
  },
  triattention: {
    outputEl: document.getElementById('triattentionOutput'),
    metricsCard: document.getElementById('triattentionGatewayMetrics'),
    kvMetricsCard: document.getElementById('triattentionKvMetrics'),
    sessions: new Map(),
    activeRequestId: null,
  },
};

let currentForwardBackend = 'triattention';

function setStatus(text, mode) {
  statusEl.textContent = text;
  statusEl.className = `status ${mode}`;
}

function setLiveStatus(text, mode) {
  liveStatusEl.textContent = text;
  liveStatusEl.className = `status ${mode}`;
}

function setMetric(backendName, field, value) {
  const card = backends[backendName]?.metricsCard;
  if (!card) return;
  const node = card.querySelector(`[data-field='${field}']`);
  if (node) node.textContent = value;
}

function setKvMetric(backendName, field, value) {
  const card = backends[backendName]?.kvMetricsCard;
  if (!card) return;
  const node = card.querySelector(`[data-field='${field}']`);
  if (node) node.textContent = value;
}

function resetKvMetrics(backendName) {
  for (const field of ['status', 'usage', 'used', 'capacity', 'blocks', 'blockSize', 'updated']) {
    setKvMetric(backendName, field, '-');
  }
}

function formatInt(value) {
  return Number.isFinite(value) ? Math.round(value).toLocaleString() : '-';
}

function resetMetrics(backendName) {
  for (const field of ['request', 'ttft', 'tps', 'total', 'tokens', 'finish']) {
    setMetric(backendName, field, '-');
  }
}

function resolveBackend(payload) {
  return payload.backend || null;
}

function ensureSession(backendName, requestId) {
  const b = backends[backendName];
  if (!b) return null;
  if (!b.sessions.has(requestId)) {
    b.sessions.set(requestId, {
      text: '',
      startedAt: performance.now(),
      firstTokenAt: null,
      tokenCount: 0,
      finishReason: '-',
      elapsedMs: null,
      done: false,
    });
  }
  return b.sessions.get(requestId);
}

function activateRequest(backendName, requestId) {
  const b = backends[backendName];
  if (!b) return;
  b.activeRequestId = requestId;
  const session = ensureSession(backendName, requestId);
  if (!session) return;

  b.outputEl.textContent = session.text;
  b.outputEl.scrollTop = b.outputEl.scrollHeight;
  setMetric(backendName, 'request', requestId);
  setMetric(backendName, 'tokens', session.tokenCount);
  setMetric(backendName, 'finish', session.finishReason);

  if (session.firstTokenAt !== null) {
    setMetric(backendName, 'ttft', (session.firstTokenAt - session.startedAt).toFixed(2));
  } else {
    setMetric(backendName, 'ttft', '-');
  }

  if (session.elapsedMs !== null) {
    setMetric(backendName, 'total', session.elapsedMs.toFixed(2));
  } else {
    setMetric(backendName, 'total', '-');
  }

  if (session.firstTokenAt !== null && session.tokenCount > 1) {
    const endAt = session.elapsedMs !== null ? session.startedAt + session.elapsedMs : performance.now();
    const decodeSec = (endAt - session.firstTokenAt) / 1000;
    if (decodeSec > 0) {
      setMetric(backendName, 'tps', (session.tokenCount / decodeSec).toFixed(2));
    }
  } else {
    setMetric(backendName, 'tps', '-');
  }
}

function handleRequestStarted(payload) {
  const backendName = resolveBackend(payload);
  const requestId = payload.request_id;
  if (!backendName || !requestId) return;

  const session = ensureSession(backendName, requestId);
  if (!session) return;
  session.startedAt = performance.now();
  session.firstTokenAt = null;
  session.tokenCount = 0;
  session.finishReason = '-';
  session.elapsedMs = null;
  session.done = false;
  session.text = '';

  activateRequest(backendName, requestId);
  setStatus(`Streaming...`, 'running');
}

function handleToken(payload) {
  const backendName = resolveBackend(payload);
  const requestId = payload.request_id;
  const text = payload.text || '';
  if (!backendName || !requestId || !text) return;

  const b = backends[backendName];
  if (!b) return;
  const session = ensureSession(backendName, requestId);
  if (!session) return;

  if (session.firstTokenAt === null) {
    session.firstTokenAt = performance.now();
  }
  session.tokenCount += 1;
  session.text += text;

  if (b.activeRequestId !== requestId) {
    activateRequest(backendName, requestId);
  } else {
    b.outputEl.textContent += text;
    b.outputEl.scrollTop = b.outputEl.scrollHeight;
    setMetric(backendName, 'tokens', session.tokenCount);
    setMetric(backendName, 'ttft', (session.firstTokenAt - session.startedAt).toFixed(2));
  }
}

function handleRequestFinish(payload) {
  const backendName = resolveBackend(payload);
  const requestId = payload.request_id;
  if (!backendName || !requestId) return;

  const session = ensureSession(backendName, requestId);
  if (!session) return;
  if (typeof payload.finish_reason === 'string' && payload.finish_reason) {
    session.finishReason = payload.finish_reason;
  }
  const b = backends[backendName];
  if (b && b.activeRequestId === requestId) {
    setMetric(backendName, 'finish', session.finishReason);
  }
}

function handleRequestDone(payload) {
  const backendName = resolveBackend(payload);
  const requestId = payload.request_id;
  if (!backendName || !requestId) return;

  const b = backends[backendName];
  if (!b) return;
  const session = ensureSession(backendName, requestId);
  if (!session) return;

  session.done = true;
  if (typeof payload.elapsed_ms === 'number') {
    session.elapsedMs = payload.elapsed_ms;
  } else {
    session.elapsedMs = performance.now() - session.startedAt;
  }
  if (typeof payload.token_count === 'number') {
    session.tokenCount = payload.token_count;
  }
  if (typeof payload.finish_reason === 'string' && payload.finish_reason) {
    session.finishReason = payload.finish_reason;
  }
  if (typeof payload.text === 'string' && payload.text) {
    session.text = payload.text;
  }

  activateRequest(backendName, requestId);

  // Check if both backends are done
  const otherName = backendName === 'baseline' ? 'triattention' : 'baseline';
  const otherB = backends[otherName];
  const otherActive = otherB?.activeRequestId;
  const otherSession = otherActive ? otherB.sessions.get(otherActive) : null;
  if (!otherSession || otherSession.done) {
    setStatus('Done', 'done');
  }
}

function handleRequestError(payload) {
  const backendName = resolveBackend(payload);
  const requestId = payload.request_id || 'unknown';
  if (!backendName) return;

  const b = backends[backendName];
  if (!b) return;
  const session = ensureSession(backendName, requestId);
  if (!session) return;

  const msg = payload.message || 'Unknown error';
  session.text += `\n\n[ERROR] ${msg}\n`;
  session.done = true;
  if (b.activeRequestId !== requestId) {
    activateRequest(backendName, requestId);
  } else {
    b.outputEl.textContent = session.text;
  }
  setStatus(`Error (${backendName}): ${requestId}`, 'error');
}

function startLiveFeed() {
  const source = new EventSource('/api/live/stream');

  source.onopen = () => {
    setLiveStatus('Live feed: connected', 'done');
  };

  source.onerror = () => {
    setLiveStatus('Live feed: reconnecting...', 'running');
  };

  source.addEventListener('connected', () => {
    setLiveStatus('Live feed: connected', 'done');
  });
  source.addEventListener('request_started', (event) => {
    handleRequestStarted(JSON.parse(event.data));
  });
  source.addEventListener('token', (event) => {
    handleToken(JSON.parse(event.data));
  });
  source.addEventListener('request_finish', (event) => {
    handleRequestFinish(JSON.parse(event.data));
  });
  source.addEventListener('request_done', (event) => {
    handleRequestDone(JSON.parse(event.data));
  });
  source.addEventListener('request_error', (event) => {
    handleRequestError(JSON.parse(event.data));
  });
}

async function refreshKvCache(backendName) {
  try {
    const response = await fetch(`/api/kv-cache?backend=${backendName}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    const usagePercent = Number(payload.usage_percent);
    const usedTokens = Number(payload.used_tokens_estimate);
    const capacityTokens = Number(payload.capacity_tokens_estimate);

    setKvMetric(backendName, 'status', payload.ok ? 'ok' : 'degraded');
    setKvMetric(backendName, 'usage', Number.isFinite(usagePercent) ? `${usagePercent.toFixed(2)}%` : '-');
    setKvMetric(backendName, 'used', formatInt(usedTokens));
    setKvMetric(backendName, 'capacity', formatInt(capacityTokens));
    setKvMetric(backendName, 'blocks', formatInt(Number(payload.num_gpu_blocks)));
    setKvMetric(backendName, 'blockSize', formatInt(Number(payload.block_size_tokens)));
    setKvMetric(backendName, 'updated', new Date().toLocaleTimeString());
  } catch {
    setKvMetric(backendName, 'status', 'unreachable');
    setKvMetric(backendName, 'updated', new Date().toLocaleTimeString());
  }
}

function startKvPolling() {
  refreshKvCache('baseline');
  refreshKvCache('triattention');
  setInterval(() => {
    refreshKvCache('baseline');
    refreshKvCache('triattention');
  }, 2000);
}

async function updateForwardToggle() {
  try {
    const resp = await fetch('/api/forward-toggle');
    if (resp.ok) {
      const data = await resp.json();
      currentForwardBackend = data.backend;
      forwardToggleBtn.textContent = currentForwardBackend;
    }
  } catch {
    // ignore
  }
}

async function toggleForward() {
  const newBackend = currentForwardBackend === 'triattention' ? 'baseline' : 'triattention';
  try {
    const resp = await fetch('/api/forward-toggle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ backend: newBackend }),
    });
    if (resp.ok) {
      const data = await resp.json();
      currentForwardBackend = data.backend;
      forwardToggleBtn.textContent = currentForwardBackend;
    }
  } catch {
    // ignore
  }
}

async function runStreamingChat() {
  const prompt = promptEl.value.trim();
  if (!prompt) {
    setStatus('Prompt is required', 'error');
    return;
  }

  runBtn.disabled = true;
  resetMetrics('baseline');
  resetMetrics('triattention');
  setStatus('Submitting request...', 'running');

  const body = {
    model: 'gateway-auto',
    messages: [{ role: 'user', content: prompt }],
    max_tokens: Number(document.getElementById('maxTokens').value),
    temperature: Number(document.getElementById('temperature').value),
    top_p: Number(document.getElementById('topP').value),
    stream: true,
    seed: Number(document.getElementById('seed').value),
  };

  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    // Drain stream; rendering comes from the live feed subscription.
    const reader = response.body.getReader();
    while (true) {
      const { done } = await reader.read();
      if (done) break;
    }
  } catch (error) {
    setStatus(`Request failed: ${error.message}`, 'error');
  } finally {
    runBtn.disabled = false;
  }
}

// Initialize
resetMetrics('baseline');
resetMetrics('triattention');
resetKvMetrics('baseline');
resetKvMetrics('triattention');
startLiveFeed();
startKvPolling();
updateForwardToggle();
runBtn.addEventListener('click', runStreamingChat);
forwardToggleBtn.addEventListener('click', toggleForward);
