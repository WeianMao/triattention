const runBtn = document.getElementById('runBtn');
const statusEl = document.getElementById('status');
const liveStatusEl = document.getElementById('liveStatus');
const promptEl = document.getElementById('prompt');
const outputEl = document.getElementById('output');
const metricsCard = document.getElementById('gatewayMetrics');
const kvMetricsCard = document.getElementById('kvMetrics');

const sessions = new Map();
let activeRequestId = null;

function setStatus(text, mode) {
  statusEl.textContent = text;
  statusEl.className = `status ${mode}`;
}

function setLiveStatus(text, mode) {
  liveStatusEl.textContent = text;
  liveStatusEl.className = `status ${mode}`;
}

function setMetric(field, value) {
  const node = metricsCard.querySelector(`[data-field='${field}']`);
  if (node) node.textContent = value;
}

function setKvMetric(field, value) {
  const node = kvMetricsCard.querySelector(`[data-field='${field}']`);
  if (node) node.textContent = value;
}

function resetKvMetrics() {
  for (const field of ['status', 'usage', 'used', 'capacity', 'blocks', 'blockSize', 'updated']) {
    setKvMetric(field, '-');
  }
}

function formatInt(value) {
  return Number.isFinite(value) ? Math.round(value).toLocaleString() : '-';
}

function resetMetrics() {
  for (const field of ['request', 'ttft', 'tps', 'total', 'tokens', 'finish']) {
    setMetric(field, '-');
  }
}

function ensureSession(requestId) {
  if (!sessions.has(requestId)) {
    sessions.set(requestId, {
      text: '',
      startedAt: performance.now(),
      firstTokenAt: null,
      tokenCount: 0,
      finishReason: '-',
      elapsedMs: null,
      done: false,
    });
  }
  return sessions.get(requestId);
}

function activateRequest(requestId) {
  activeRequestId = requestId;
  const session = ensureSession(requestId);
  outputEl.textContent = session.text;
  outputEl.scrollTop = outputEl.scrollHeight;
  setMetric('request', requestId);
  setMetric('tokens', session.tokenCount);
  setMetric('finish', session.finishReason);

  if (session.firstTokenAt !== null) {
    setMetric('ttft', (session.firstTokenAt - session.startedAt).toFixed(2));
  } else {
    setMetric('ttft', '-');
  }

  if (session.elapsedMs !== null) {
    setMetric('total', session.elapsedMs.toFixed(2));
  } else {
    setMetric('total', '-');
  }

  if (session.firstTokenAt !== null && session.tokenCount > 1) {
    const endAt = session.elapsedMs !== null ? session.startedAt + session.elapsedMs : performance.now();
    const decodeSec = (endAt - session.firstTokenAt) / 1000;
    if (decodeSec > 0) {
      setMetric('tps', (session.tokenCount / decodeSec).toFixed(2));
    }
  } else {
    setMetric('tps', '-');
  }
}

function handleRequestStarted(payload) {
  const requestId = payload.request_id;
  if (!requestId) return;
  const session = ensureSession(requestId);
  session.startedAt = performance.now();
  session.firstTokenAt = null;
  session.tokenCount = 0;
  session.finishReason = '-';
  session.elapsedMs = null;
  session.done = false;
  session.text = '';

  activateRequest(requestId);
  setStatus(`Streaming request ${requestId}`, 'running');
}

function handleToken(payload) {
  const requestId = payload.request_id;
  const text = payload.text || '';
  if (!requestId || !text) return;

  const session = ensureSession(requestId);
  if (session.firstTokenAt === null) {
    session.firstTokenAt = performance.now();
  }
  session.tokenCount += 1;
  session.text += text;

  if (activeRequestId !== requestId) {
    activateRequest(requestId);
  } else {
    outputEl.textContent += text;
    outputEl.scrollTop = outputEl.scrollHeight;
    setMetric('tokens', session.tokenCount);
    setMetric('ttft', (session.firstTokenAt - session.startedAt).toFixed(2));
  }
}

function handleRequestFinish(payload) {
  const requestId = payload.request_id;
  if (!requestId) return;
  const session = ensureSession(requestId);
  if (typeof payload.finish_reason === 'string' && payload.finish_reason) {
    session.finishReason = payload.finish_reason;
  }
  if (activeRequestId === requestId) {
    setMetric('finish', session.finishReason);
  }
}

function handleRequestDone(payload) {
  const requestId = payload.request_id;
  if (!requestId) return;
  const session = ensureSession(requestId);
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

  if (activeRequestId !== requestId) {
    activateRequest(requestId);
  } else {
    activateRequest(activeRequestId);
  }
  setStatus(`Done: ${requestId}`, 'done');
}

function handleRequestError(payload) {
  const requestId = payload.request_id || 'unknown';
  const session = ensureSession(requestId);
  const msg = payload.message || 'Unknown error';
  session.text += `\n\n[ERROR] ${msg}\n`;
  session.done = true;
  if (activeRequestId !== requestId) {
    activateRequest(requestId);
  } else {
    outputEl.textContent = session.text;
  }
  setStatus(`Error: ${requestId}`, 'error');
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

async function refreshKvCache() {
  try {
    const response = await fetch('/api/kv-cache');
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    const usagePercent = Number(payload.usage_percent);
    const usedTokens = Number(payload.used_tokens_estimate);
    const capacityTokens = Number(payload.capacity_tokens_estimate);

    setKvMetric('status', payload.ok ? 'ok' : 'degraded');
    setKvMetric('usage', Number.isFinite(usagePercent) ? `${usagePercent.toFixed(2)}%` : '-');
    setKvMetric('used', formatInt(usedTokens));
    setKvMetric('capacity', formatInt(capacityTokens));
    setKvMetric('blocks', formatInt(Number(payload.num_gpu_blocks)));
    setKvMetric('blockSize', formatInt(Number(payload.block_size_tokens)));
    setKvMetric('updated', new Date().toLocaleTimeString());
  } catch {
    setKvMetric('status', 'unreachable');
    setKvMetric('updated', new Date().toLocaleTimeString());
  }
}

function startKvPolling() {
  refreshKvCache();
  setInterval(refreshKvCache, 2000);
}

async function runStreamingChat() {
  const prompt = promptEl.value.trim();
  if (!prompt) {
    setStatus('Prompt is required', 'error');
    return;
  }

  runBtn.disabled = true;
  resetMetrics();
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

resetMetrics();
resetKvMetrics();
startLiveFeed();
startKvPolling();
runBtn.addEventListener('click', runStreamingChat);
