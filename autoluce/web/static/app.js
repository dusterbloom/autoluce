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
      const cls = items.length === 1 ? "family family--single" : "family";
      html += `<div class="${cls}">
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
  const unit = metricUnit(metric).trim();
  const sp = c.sparkline || [];
  const latest = sp.length ? sp[sp.length - 1] : null;
  const distinct = sp.length >= 2 && new Set(sp).size >= 2;   // a genuine trend, not 1pt or duplicates
  const valueBlock = latest !== null
    ? `<div class="metric"><span class="metric-value">${latest.toFixed(latest >= 100 ? 0 : 1)}</span>${unit ? `<span class="metric-unit">${escapeHtml(unit)}</span>` : ''}</div>`
    : `<div class="metric metric-empty">no measurement</div>`;
  return `<div class="card" data-id="${escapeHtml(c.id)}">
    <div class="card-title">${escapeHtml(title)}</div>
    <div class="card-meta">${escapeHtml(subtitle)}</div>
    <div class="card-badges">
      <span class="badge ${c.status}">${escapeHtml(c.status)}</span>
      <span class="badge stage">${escapeHtml(c.stage)}</span>
    </div>
    ${valueBlock}
    ${distinct ? renderSparkline(sp) : ''}
    <div class="stats"><span>ev ${c.evidence_count}</span><span>fr ${c.frontier_count}</span><span>ref ${c.reference_count}</span></div>
  </div>`;
}

// A real temporal trend only when there are >=2 distinct measurements. Otherwise
// no chart -- an honest blank beats a flat line that pretends to be progression.
function renderSparkline(values) {
  const w = 120, h = 30;
  const min = Math.min(...values), max = Math.max(...values);
  const range = Math.max(max - min, 1e-9);
  const x = i => 1 + (i / (values.length - 1)) * (w - 2);
  const y = v => h - 2 - ((v - min) / range) * (h - 4);
  const d = values.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ');
  const area = `M ${x(0).toFixed(1)} ${h} ` + values.map((v, i) => `L ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' ') + ` L ${x(values.length - 1).toFixed(1)} ${h} Z`;
  return `<svg class="sparkline" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <path d="${area}" fill="rgba(111,133,255,0.15)"/>
    <path d="${d}" fill="none" stroke="#6f85ff" stroke-width="1.5"/>
  </svg>`;
}

// Cross-sectional comparison: one row per variant, dot at its latest objective
// value. This is the honest view for a family of variants each measured ~once
// (e.g. a KV-cache quant sweep) -- never a fake time-series line.
function renderFamilyChart(campaigns) {
  const byMetric = {};
  for (const c of campaigns) {
    const sp = c.sparkline || [];
    if (!sp.length) continue;
    (byMetric[objectiveMetric(c)] = byMetric[objectiveMetric(c)] || []).push(c);
  }
  const metric = Object.keys(byMetric).find(m => byMetric[m].length >= 2);
  if (!metric) return '';
  return renderLollipop(metric, byMetric[metric]);
}

function renderLollipop(metric, campaigns) {
  const unit = metricUnit(metric).trim();
  const rows = campaigns.map(c => ({
    label: c.variant || c.name,
    value: (c.sparkline[c.sparkline.length - 1]),
    status: c.status,
  }));
  const vals = rows.map(r => r.value);
  const min = Math.min(...vals), max = Math.max(...vals);
  const range = Math.max(max - min, 1e-9);
  rows.sort((a, b) => b.value - a.value);  // best (maximize) on top
  const body = rows.map(r => {
    const pct = ((r.value - min) / range) * 100;
    return `<div class="lol-row">
      <span class="lol-label">${escapeHtml(r.label)}</span>
      <span class="lol-track"><span class="lol-dot ${r.status}" style="left:${pct.toFixed(1)}%"></span></span>
      <span class="lol-value">${r.value.toFixed(r.value >= 100 ? 0 : 1)}</span>
    </div>`;
  }).join('');
  return `<div class="lollipop">
    <div class="lol-title">${escapeHtml(metric)}${unit ? ` <span class="lol-unit">${escapeHtml(unit)}</span>` : ''}<span class="lol-hint">higher is better</span></div>
    ${body}
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

// --- realtime: poll a cheap generation, reload only when it moves -----------
// (visibility-aware; uses the authenticated apiFetch, so it works through the
// token-gated coordinator too. EventSource can't send the bearer header.)
let lastGeneration = null;

async function checkVersion() {
  if (document.visibilityState !== 'visible') return;
  try {
    const data = await (await apiFetch('/api/version')).json();
    if (lastGeneration !== null && data.generation !== lastGeneration) {
      loadCampaigns();
    }
    lastGeneration = data.generation;
  } catch (_) { /* transient; the next tick retries */ }
}

setInterval(checkVersion, 5000);
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') checkVersion();
});

loadCampaigns();
