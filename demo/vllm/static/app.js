const runBtn = document.getElementById('runBtn');
const statusEl = document.getElementById('status');
const liveStatusEl = document.getElementById('liveStatus');
const promptEl = document.getElementById('prompt');
const forwardToggleBtn = document.getElementById('forwardToggleBtn');

const backends = {
  baseline: {
    outputEl: document.getElementById('baselineOutput'),
    toolsEl: document.getElementById('baselineTools'),
    metricsCard: document.getElementById('baselineGatewayMetrics'),
    kvMetricsCard: document.getElementById('baselineKvMetrics'),
    sessionLabelEl: document.getElementById('baselineSessionLabel'),
    phaseBadgeEl: document.getElementById('baselinePhaseBadge'),
    sessions: new Map(),
    activeSessionId: null,
  },
  triattention: {
    outputEl: document.getElementById('triattentionOutput'),
    toolsEl: document.getElementById('triattentionTools'),
    metricsCard: document.getElementById('triattentionGatewayMetrics'),
    kvMetricsCard: document.getElementById('triattentionKvMetrics'),
    sessionLabelEl: document.getElementById('triattentionSessionLabel'),
    phaseBadgeEl: document.getElementById('triattentionPhaseBadge'),
    sessions: new Map(),
    activeSessionId: null,
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

function resetMetrics(backendName) {
  for (const field of ['session', 'request', 'ttft', 'tps', 'total', 'tokens', 'finish']) {
    setMetric(backendName, field, '-');
  }
}

function formatInt(value) {
  return Number.isFinite(value) ? Math.round(value).toLocaleString() : '-';
}

function clip(value, limit = 18) {
  if (!value) return '-';
  return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function resolveBackend(payload) {
  return payload.backend || null;
}

function resolveSessionId(payload) {
  return payload.session_id || payload.request_id || null;
}

function ensureSession(backendName, sessionId) {
  const backend = backends[backendName];
  if (!backend || !sessionId) return null;
  if (!backend.sessions.has(sessionId)) {
    backend.sessions.set(sessionId, {
      id: sessionId,
      text: '',
      startedAt: performance.now(),
      firstTokenAt: null,
      tokenCount: 0,
      finishReason: '-',
      elapsedMs: null,
      done: false,
      phase: 'idle',
      latestRequestId: '-',
      tools: new Map(),
      toolOrder: [],
      routeKind: null,
    });
  }
  return backend.sessions.get(sessionId);
}

function setPhase(backendName, session, phase) {
  session.phase = phase;
  const badge = backends[backendName]?.phaseBadgeEl;
  if (!badge) return;
  badge.textContent = phase;
  badge.className = `phase-badge ${phase.replaceAll(/\s+/g, '-').toLowerCase()}`;
}

function ensureTool(session, toolCallId, toolName = 'tool') {
  if (!session.tools.has(toolCallId)) {
    session.tools.set(toolCallId, {
      id: toolCallId,
      name: toolName,
      status: 'running',
      argumentsFull: '',
      resultSummary: '',
      resultText: '',
      origin: '',
      updatedAt: Date.now(),
    });
    session.toolOrder.push(toolCallId);
  }
  const tool = session.tools.get(toolCallId);
  if (toolName && (!tool.name || tool.name === 'tool')) {
    tool.name = toolName;
  }
  return tool;
}

function renderToolTimeline(backendName, session) {
  const backend = backends[backendName];
  if (!backend) return;
  if (!session || session.toolOrder.length === 0) {
    backend.toolsEl.className = 'tool-timeline empty-state';
    backend.toolsEl.textContent = 'No tool activity yet.';
    return;
  }

  backend.toolsEl.className = 'tool-timeline';
  backend.toolsEl.innerHTML = session.toolOrder
    .map((toolId) => {
      const tool = session.tools.get(toolId);
      if (!tool) return '';
      const args = tool.argumentsFull ? escapeHtml(tool.argumentsFull) : '<span class="muted">No arguments</span>';
      const result = tool.resultSummary
        ? `<div class="tool-result"><span class="tool-subtitle">Result</span><p>${escapeHtml(tool.resultSummary)}</p></div>`
        : '';
      return `
        <article class="tool-card ${escapeHtml(tool.status)}">
          <div class="tool-card-header">
            <strong>${escapeHtml(tool.name || 'tool')}</strong>
            <span class="tool-status ${escapeHtml(tool.status)}">${escapeHtml(tool.status)}</span>
          </div>
          <div class="tool-meta">ID: ${escapeHtml(tool.id)}${tool.origin ? ` · ${escapeHtml(tool.origin)}` : ''}</div>
          <div class="tool-args">
            <span class="tool-subtitle">Arguments</span>
            <pre>${args}</pre>
          </div>
          ${result}
        </article>
      `;
    })
    .join('');
}

function activateSession(backendName, sessionId) {
  const backend = backends[backendName];
  if (!backend) return;
  const session = ensureSession(backendName, sessionId);
  if (!session) return;

  backend.activeSessionId = sessionId;
  backend.outputEl.textContent = session.text;
  backend.outputEl.scrollTop = backend.outputEl.scrollHeight;
  backend.sessionLabelEl.textContent = clip(sessionId, 22);
  setMetric(backendName, 'session', clip(sessionId, 22));
  setMetric(backendName, 'request', clip(session.latestRequestId, 22));
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
    setMetric(backendName, 'tps', decodeSec > 0 ? (session.tokenCount / decodeSec).toFixed(2) : '-');
  } else {
    setMetric(backendName, 'tps', '-');
  }

  setPhase(backendName, session, session.phase || 'idle');
  renderToolTimeline(backendName, session);
}

function clearSessionsForNewManualRun() {
  for (const backendName of Object.keys(backends)) {
    backends[backendName].sessions.clear();
    backends[backendName].activeSessionId = null;
    backends[backendName].outputEl.textContent = '';
    backends[backendName].toolsEl.className = 'tool-timeline empty-state';
    backends[backendName].toolsEl.textContent = 'No tool activity yet.';
    backends[backendName].sessionLabelEl.textContent = '-';
    setPhase(backendName, { phase: 'idle' }, 'idle');
    resetMetrics(backendName);
  }
}

function handleRequestStarted(payload) {
  const backendName = resolveBackend(payload);
  const sessionId = resolveSessionId(payload);
  if (!backendName || !sessionId) return;

  const session = ensureSession(backendName, sessionId);
  if (!session) return;
  session.startedAt = performance.now();
  session.done = false;
  session.elapsedMs = null;
  session.finishReason = '-';
  session.latestRequestId = payload.request_id || session.latestRequestId;
  session.routeKind = payload.route_kind || session.routeKind;
  if (!session.text && payload.route_kind === 'chat') {
    session.firstTokenAt = null;
    session.tokenCount = 0;
  }
  setPhase(backendName, session, payload.phase || (payload.route_kind === 'completions' ? 'tooling' : 'streaming'));
  activateSession(backendName, sessionId);
  setStatus('Streaming...', 'running');
}

function handleTextDelta(payload) {
  const backendName = resolveBackend(payload);
  const sessionId = resolveSessionId(payload);
  const text = payload.text || '';
  if (!backendName || !sessionId || !text) return;

  const backend = backends[backendName];
  const session = ensureSession(backendName, sessionId);
  if (!backend || !session) return;

  if (session.firstTokenAt === null) {
    session.firstTokenAt = performance.now();
  }
  session.tokenCount = typeof payload.token_count === 'number' ? payload.token_count : session.tokenCount + 1;
  session.text += text;
  session.latestRequestId = payload.request_id || session.latestRequestId;
  setPhase(backendName, session, 'writing');

  if (backend.activeSessionId !== sessionId) {
    activateSession(backendName, sessionId);
    return;
  }

  backend.outputEl.textContent = session.text;
  backend.outputEl.scrollTop = backend.outputEl.scrollHeight;
  setMetric(backendName, 'tokens', session.tokenCount);
  setMetric(backendName, 'request', clip(session.latestRequestId, 22));
  setMetric(backendName, 'ttft', (session.firstTokenAt - session.startedAt).toFixed(2));
}

function handleToolCallStarted(payload) {
  const backendName = resolveBackend(payload);
  const sessionId = resolveSessionId(payload);
  if (!backendName || !sessionId || !payload.tool_call_id) return;

  const session = ensureSession(backendName, sessionId);
  if (!session) return;
  const tool = ensureTool(session, payload.tool_call_id, payload.tool_name);
  tool.status = 'running';
  tool.argumentsFull = payload.arguments_full || tool.argumentsFull;
  tool.origin = payload.origin || tool.origin;
  tool.updatedAt = Date.now();
  session.latestRequestId = payload.request_id || session.latestRequestId;
  setPhase(backendName, session, 'calling-tool');
  activateSession(backendName, sessionId);
}

function handleToolCallDelta(payload) {
  const backendName = resolveBackend(payload);
  const sessionId = resolveSessionId(payload);
  if (!backendName || !sessionId || !payload.tool_call_id) return;

  const session = ensureSession(backendName, sessionId);
  if (!session) return;
  const tool = ensureTool(session, payload.tool_call_id, payload.tool_name);
  tool.status = 'running';
  if (typeof payload.arguments_full === 'string' && payload.arguments_full) {
    tool.argumentsFull = payload.arguments_full;
  } else if (typeof payload.arguments_delta === 'string') {
    tool.argumentsFull += payload.arguments_delta;
  }
  tool.origin = payload.origin || tool.origin;
  tool.updatedAt = Date.now();
  setPhase(backendName, session, 'calling-tool');
  activateSession(backendName, sessionId);
}

function handleToolCallFinished(payload) {
  const backendName = resolveBackend(payload);
  const sessionId = resolveSessionId(payload);
  if (!backendName || !sessionId || !payload.tool_call_id) return;

  const session = ensureSession(backendName, sessionId);
  if (!session) return;
  const tool = ensureTool(session, payload.tool_call_id, payload.tool_name);
  tool.status = 'completed';
  tool.argumentsFull = payload.arguments_full || tool.argumentsFull;
  tool.origin = payload.origin || tool.origin;
  tool.updatedAt = Date.now();
  setPhase(backendName, session, 'tool-finished');
  activateSession(backendName, sessionId);
}

function handleToolResult(payload) {
  const backendName = resolveBackend(payload);
  const sessionId = resolveSessionId(payload);
  if (!backendName || !sessionId || !payload.tool_call_id) return;

  const session = ensureSession(backendName, sessionId);
  if (!session) return;
  const tool = ensureTool(session, payload.tool_call_id, payload.tool_name);
  tool.status = 'result';
  tool.resultSummary = payload.result_summary || tool.resultSummary;
  tool.resultText = payload.result_text || tool.resultText;
  tool.origin = payload.origin || tool.origin;
  tool.updatedAt = Date.now();
  setPhase(backendName, session, 'tool-returned');
  activateSession(backendName, sessionId);
}

function handleRequestFinish(payload) {
  const backendName = resolveBackend(payload);
  const sessionId = resolveSessionId(payload);
  if (!backendName || !sessionId) return;

  const session = ensureSession(backendName, sessionId);
  if (!session) return;
  if (typeof payload.finish_reason === 'string' && payload.finish_reason) {
    session.finishReason = payload.finish_reason;
  }
  session.latestRequestId = payload.request_id || session.latestRequestId;
  activateSession(backendName, sessionId);
}

function handleRequestDone(payload) {
  const backendName = resolveBackend(payload);
  const sessionId = resolveSessionId(payload);
  if (!backendName || !sessionId) return;

  const session = ensureSession(backendName, sessionId);
  if (!session) return;

  session.done = true;
  session.latestRequestId = payload.request_id || session.latestRequestId;
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

  if (session.text) {
    setPhase(backendName, session, 'done');
  } else if (session.toolOrder.length > 0) {
    setPhase(backendName, session, 'tool-finished');
  } else {
    setPhase(backendName, session, 'done');
  }
  activateSession(backendName, sessionId);

  const otherName = backendName === 'baseline' ? 'triattention' : 'baseline';
  const otherBackend = backends[otherName];
  const otherSession = otherBackend?.activeSessionId ? otherBackend.sessions.get(otherBackend.activeSessionId) : null;
  if (!otherSession || otherSession.done) {
    setStatus('Done', 'done');
  }
}

function handleRequestError(payload) {
  const backendName = resolveBackend(payload);
  const sessionId = resolveSessionId(payload);
  if (!backendName || !sessionId) return;

  const backend = backends[backendName];
  const session = ensureSession(backendName, sessionId);
  if (!backend || !session) return;

  const msg = payload.message || 'Unknown error';
  const kvInsufficient = msg.toLowerCase().includes('kv cache insufficient');
  const displayMsg = kvInsufficient ? 'KV cache不足，网关已终止该 backend 请求。' : msg;
  if (session.text) {
    session.text += `\n\n[ERROR] ${displayMsg}\n`;
  } else {
    session.text = `[ERROR] ${displayMsg}\n`;
  }
  session.done = true;
  session.finishReason = kvInsufficient ? 'kv-cache-insufficient' : 'error';
  session.latestRequestId = payload.request_id || session.latestRequestId;
  setPhase(backendName, session, 'error');
  activateSession(backendName, sessionId);
  setStatus(`Error (${backendName}): ${clip(session.latestRequestId, 22)}`, 'error');
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
  source.addEventListener('text_delta', (event) => {
    handleTextDelta(JSON.parse(event.data));
  });
  source.addEventListener('tool_call_started', (event) => {
    handleToolCallStarted(JSON.parse(event.data));
  });
  source.addEventListener('tool_call_delta', (event) => {
    handleToolCallDelta(JSON.parse(event.data));
  });
  source.addEventListener('tool_call_finished', (event) => {
    handleToolCallFinished(JSON.parse(event.data));
  });
  source.addEventListener('tool_result', (event) => {
    handleToolResult(JSON.parse(event.data));
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
  clearSessionsForNewManualRun();
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

for (const backendName of Object.keys(backends)) {
  resetMetrics(backendName);
  resetKvMetrics(backendName);
  backends[backendName].toolsEl.className = 'tool-timeline empty-state';
}

startLiveFeed();
startKvPolling();
updateForwardToggle();
runBtn.addEventListener('click', runStreamingChat);
forwardToggleBtn.addEventListener('click', toggleForward);
