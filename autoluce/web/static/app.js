let allCampaigns = [];
let currentFilter = 'all';

const TOKEN_KEY = 'autoluce_web_token';
const ROOT_ORDER = ['research', 'benchmarks'];
const ROOT_LABEL = { research: 'Research', benchmarks: 'Benchmark archive' };
const PALETTE = ['#5c7cfa', '#40c057', '#fab005', '#fa5252', '#7950f2', '#15aabf', '#fd7e14'];

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || '';
}

function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

function authHeaders() {
  const token = getToken();
  return token ? { 'Authorization': `Bearer ${token}` } : {};
}

async function apiFetch(path) {
  const res = await fetch(path, { headers: authHeaders() });
  if (res.status === 401) {
    showTokenDialog('Authentication required.');
    throw new Error('Unauthorized');
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res;
}

function showTokenDialog(message) {
  const dialog = document.getElementById('token-dialog');
  dialog.querySelector('#token-error').textContent = message || '';
  dialog.classList.remove('hidden');
  document.getElementById('token-input').focus();
}

function hideTokenDialog() {
  document.getElementById('token-dialog').classList.add('hidden');
}

async function loadCampaigns() {
  const status = document.getElementById('status');
  try {
    const res = await apiFetch('/api/campaigns');
    allCampaigns = await res.json();
    status.textContent = `${allCampaigns.length} campaigns loaded`;
    renderBoard();
  } catch (err) {
    if (err.message !== 'Unauthorized') {
      status.textContent = 'Error loading campaigns: ' + err.message;
    }
  }
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function metricUnit(metric) {
  return metric && metric.endsWith('_tok_s') ? ' tok/s' : '';
}

function objectiveMetric(c) {
  return (c.objective && c.objective.metric) || 'score';
}

// --- board: grouped by source root -> family -------------------------------

function groupByTree(campaigns) {
  const roots = {};
  for (const c of campaigns) {
    const families = roots[c.root] || (roots[c.root] = {});
    (families[c.family] || (families[c.family] = [])).push(c);
  }
  return roots;
}

function renderBoard() {
  const board = document.getElementById('board');
  const filtered = currentFilter === 'all'
    ? allCampaigns
    : allCampaigns.filter(c => c.status === currentFilter);
  const roots = groupByTree(filtered);

  let html = '';
  for (const root of ROOT_ORDER) {
    const families = roots[root];
    if (!families) continue;
    const names = Object.keys(families).sort();
    if (!names.length) continue;
    html += `<section class="root-section"><h2 class="root-title">${ROOT_LABEL[root] || root}</h2>`;
    for (const family of names) {
      const items = families[family].slice().sort((a, b) => (a.variant || a.name).localeCompare(b.variant || b.name));
      html += `<div class="family">
        <div class="family-head"><span class="family-name">${escapeHtml(family)}</span><span class="family-count">${items.length}</span></div>
        ${renderFamilyChart(items)}
        <div class="cards">${items.map(renderCard).join('')}</div>
      </div>`;
    }
    html += `</section>`;
  }
  board.innerHTML = html || '<p class="empty">No campaigns match this filter.</p>';
}

function renderCard(c) {
  const title = c.variant || c.name;
  const subtitle = c.variant ? c.name : `${c.system.model || '?'} · ${c.system.backend || '?'}`;
  const metric = objectiveMetric(c);
  return `<div class="card" data-id="${escapeHtml(c.id)}">
    <div class="card-title">${escapeHtml(title)}</div>
    <div class="card-meta">${escapeHtml(subtitle)}</div>
    <div class="badges">
      <span class="badge stage">${escapeHtml(c.stage)}</span>
      <span class="badge ${c.status}">${escapeHtml(c.status)}</span>
      <span class="badge metric">${escapeHtml(metric)}</span>
    </div>
    ${renderSparkline(c.sparkline)}
    <div class="stats">
      <span title="evidence">ev ${c.evidence_count}</span>
      <span title="frontier">fr ${c.frontier_count}</span>
      <span title="references">ref ${c.reference_count}</span>
    </div>
  </div>`;
}

function renderSparkline(values) {
  if (!values || values.length < 2) return '';
  const w = 120, h = 26;
  const min = Math.min(...values), max = Math.max(...values);
  const range = Math.max(max - min, 1e-9);
  const x = i => 1 + (i / (values.length - 1)) * (w - 2);
  const y = v => h - 1 - ((v - min) / range) * (h - 2);
  const d = values.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ');
  return `<svg class="sparkline" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <path d="${d}" fill="none" stroke="#5c7cfa" stroke-width="1.5"/>
    <circle cx="${x(values.length - 1).toFixed(1)}" cy="${y(values[values.length - 1]).toFixed(1)}" r="2" fill="#5c7cfa"/>
  </svg>`;
}

// Overlay one metric across the campaigns of a family that share it. This is
// where a family's variants (e.g. the normal-kv-prefill KV-cache sweep) become
// comparable on one axis -- the W&B-style moment, honest because variants in a
// family share metric and direction.
function renderFamilyChart(campaigns) {
  const byMetric = {};
  for (const c of campaigns) {
    if (!c.sparkline || c.sparkline.length === 0) continue;
    const m = objectiveMetric(c);
    (byMetric[m] = byMetric[m] || []).push(c);
  }
  const metric = Object.keys(byMetric).find(m => byMetric[m].length >= 2);
  if (!metric) return '';
  return renderOverlay(metric, byMetric[metric]);
}

function renderOverlay(metric, campaigns) {
  const unit = metricUnit(metric);
  const w = 540, h = 168;
  const pad = { top: 12, right: 14, bottom: 26, left: 54 };
  const iw = w - pad.left - pad.right, ih = h - pad.top - pad.bottom;
  const all = campaigns.flatMap(c => c.sparkline);
  const min = Math.min(...all), max = Math.max(...all);
  const vrange = Math.max(max - min, 1e-9);
  const maxLen = Math.max(...campaigns.map(c => c.sparkline.length));
  const xSpan = Math.max(maxLen - 1, 1);
  const x = i => pad.left + (i / xSpan) * iw;
  const y = v => pad.top + ih - ((v - min) / vrange) * ih;

  let yAxis = '';
  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {
    const v = min + (vrange * i / yTicks);
    yAxis += `<line x1="${pad.left}" y1="${y(v).toFixed(1)}" x2="${w - pad.right}" y2="${y(v).toFixed(1)}" stroke="#2a2e36" stroke-dasharray="2"/>`;
    yAxis += `<text x="${pad.left - 6}" y="${(y(v) + 3).toFixed(1)}" text-anchor="end" fill="#9aa0a6" font-size="9">${v.toFixed(0)}</text>`;
  }

  let lines = '';
  let legend = '';
  campaigns.forEach((c, idx) => {
    const col = PALETTE[idx % PALETTE.length];
    const vals = c.sparkline;
    const d = vals.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ');
    lines += `<path d="${d}" fill="none" stroke="${col}" stroke-width="2"/>`;
    legend += `<span class="legend-item"><span class="legend-swatch" style="background:${col}"></span>${escapeHtml(c.variant || c.name)}</span>`;
  });

  return `<div class="family-chart">
    <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet">${yAxis}${lines}</svg>
    <div class="legend">${legend}</div>
  </div>`;
}

// --- detail drawer ----------------------------------------------------------

async function showDetail(id) {
  const detail = document.getElementById('detail');
  const body = document.getElementById('detail-body');
  body.innerHTML = '<p>Loading...</p>';
  detail.classList.remove('hidden');
  try {
    const res = await apiFetch('/api/campaigns/' + encodeURIComponent(id));
    const data = await res.json();
    renderDetail(data);
  } catch (err) {
    body.innerHTML = '<p>Error: ' + escapeHtml(err.message) + '</p>';
  }
}

function renderPlot(plot) {
  if (!plot || !plot.points.length) {
    return '<p class="empty">No numeric progression data.</p>';
  }
  const width = 560;
  const height = 220;
  const pad = { top: 20, right: 20, bottom: 40, left: 60 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;

  const points = plot.points;
  const indexMin = points[0].index;
  const indexMax = points[points.length - 1].index;
  const indexRange = Math.max(indexMax - indexMin, 1);
  const valueMin = plot.min;
  const valueMax = plot.max;
  const valueRange = Math.max(valueMax - valueMin, 1e-9);

  const x = i => pad.left + ((i - indexMin) / indexRange) * innerW;
  const y = v => pad.top + innerH - ((v - valueMin) / valueRange) * innerH;

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(p.index)} ${y(p.value)}`).join(' ');
  const dots = points.map(p => `<circle cx="${x(p.index)}" cy="${y(p.value)}" r="3" title="#${p.index + 1}\n${p.value.toFixed(2)} ${plot.unit}\n${escapeHtml(p.evidence_id)}"></circle>`).join('');

  const yTicks = 5;
  let yAxis = '';
  for (let i = 0; i <= yTicks; i++) {
    const v = valueMin + (valueRange * i / yTicks);
    const yp = y(v);
    yAxis += `<line x1="${pad.left}" y1="${yp}" x2="${width - pad.right}" y2="${yp}" stroke="#2a2e36" stroke-dasharray="2"/>`;
    yAxis += `<text x="${pad.left - 8}" y="${yp + 4}" text-anchor="end" fill="#9aa0a6" font-size="10">${v.toFixed(1)}</text>`;
  }

  const xAxis = `
    <text x="${pad.left}" y="${height - 10}" fill="#9aa0a6" font-size="10">#${indexMin + 1}</text>
    <text x="${width - pad.right}" y="${height - 10}" text-anchor="end" fill="#9aa0a6" font-size="10">#${indexMax + 1}</text>
  `;

  return `
    <div class="plot">
      <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet">
        ${yAxis}
        <path d="${pathD}" fill="none" stroke="#5c7cfa" stroke-width="2"/>
        ${dots}
        ${xAxis}
      </svg>
      <div class="plot-caption">${escapeHtml(plot.metric)} progression${plot.unit ? ' (' + escapeHtml(plot.unit) + ')' : ''} · ${points.length} points</div>
    </div>
  `;
}

function renderDetail(data) {
  const campaign = data.campaign;
  const state = data.state;
  const evidence = (state && state.evidence) || [];
  const frontier = new Set(state && state.frontier ? state.frontier : []);
  document.getElementById('detail-title').textContent = campaign.name;

  const evidenceRows = evidence.map((e, i) => {
    const gates = Object.entries(e.gates || {}).map(([k, v]) => `${k}: ${v}`).join(', ');
    const metrics = Object.entries(e.metrics || {}).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`).join(', ');
    const onFrontier = frontier.has(e.evidence_id) ? ' ⭐' : '';
    return `<tr>
      <td>${escapeHtml(e.evidence_id)}${onFrontier}</td>
      <td>#${i + 1}</td>
      <td>${escapeHtml(metrics)}</td>
      <td>${escapeHtml(gates)}</td>
    </tr>`;
  }).join('');

  const refs = state && state.references
    ? state.references.map(r => `<li>${escapeHtml(r.kind)}${r.locator ? ' · ' + escapeHtml(r.locator) : ''}</li>`).join('')
    : '';
  const refCount = state && state.references ? state.references.length : (campaign.reference ? 1 : 0);

  document.getElementById('detail-body').innerHTML = `
    <p><span class="badge ${data.status}">${escapeHtml(data.status)}</span> <span class="badge stage">${escapeHtml(campaign.lifecycle_stage)}</span></p>
    <p class="empty">${escapeHtml(data.path)}</p>

    <h3>Objective</h3>
    <p>${escapeHtml(campaign.objective.metric)} — ${escapeHtml(campaign.objective.direction)}</p>

    <h3>Progression</h3>
    ${renderPlot(data.plot)}

    <h3>System</h3>
    <pre>${escapeHtml(JSON.stringify(campaign.system, null, 2))}</pre>

    <h3>Constraints</h3>
    <pre>${escapeHtml(JSON.stringify(campaign.constraints, null, 2))}</pre>

    <h3>References (${refCount})</h3>
    ${refs ? `<ul>${refs}</ul>` : '<p class="empty">No references attached.</p>'}

    <h3>Evidence (${evidence.length})</h3>
    ${evidenceRows ? `<table><thead><tr><th>ID</th><th>#</th><th>Metrics</th><th>Gates</th></tr></thead><tbody>${evidenceRows}</tbody></table>` : '<p class="empty">No evidence yet.</p>'}

    <h3>Promotion</h3>
    <p>${state && state.promotion ? escapeHtml(state.promotion) : '<span class="empty">No promoted evidence</span>'}</p>
  `;
}

document.getElementById('close-detail').addEventListener('click', () => {
  document.getElementById('detail').classList.add('hidden');
});

document.getElementById('board').addEventListener('click', e => {
  const card = e.target.closest('.card');
  if (card && card.dataset.id) showDetail(card.dataset.id);
});

document.querySelectorAll('.filter').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter = btn.dataset.filter;
    renderBoard();
  });
});

document.getElementById('token-save').addEventListener('click', async () => {
  const input = document.getElementById('token-input');
  const token = input.value.trim();
  if (!token) return;
  setToken(token);
  hideTokenDialog();
  await loadCampaigns();
});

document.getElementById('token-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('token-save').click();
});

loadCampaigns();
