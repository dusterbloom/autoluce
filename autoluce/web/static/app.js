const stages = ['observe', 'discover', 'explore', 'compare', 'explain', 'promote'];
let allCampaigns = [];
let currentFilter = 'all';

const TOKEN_KEY = 'autoluce_web_token';

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
  const error = document.getElementById('token-error');
  dialog.classList.remove('hidden');
  error.textContent = message || '';
  document.getElementById('token-input').focus();
}

function hideTokenDialog() {
  document.getElementById('token-dialog').classList.add('hidden');
}

async function tryLoadCampaigns() {
  if (!getToken()) {
    showTokenDialog('Enter your coordinator token to view campaigns.');
    return;
  }
  await loadCampaigns();
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
  return String(str).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderBoard() {
  stages.forEach(stage => {
    const col = document.querySelector(`.column[data-stage="${stage}"]`);
    col.querySelectorAll('.card, .empty').forEach(el => el.remove());
  });

  const filtered = currentFilter === 'all'
    ? allCampaigns
    : allCampaigns.filter(c => c.status === currentFilter);

  stages.forEach(stage => {
    const col = document.querySelector(`.column[data-stage="${stage}"]`);
    const items = filtered.filter(c => c.stage === stage);
    if (items.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'empty';
      empty.textContent = 'No campaigns';
      col.appendChild(empty);
      return;
    }
    items.forEach(c => {
      const card = document.createElement('div');
      card.className = 'card';
      card.dataset.id = c.id;
      card.innerHTML = `
        <div class="card-title">${escapeHtml(c.name)}</div>
        <div class="card-meta">${escapeHtml(c.system.model || '?')} · ${escapeHtml(c.system.backend || '?')} · ${escapeHtml(c.system.hardware || '?')}</div>
        <span class="badge ${c.status}">${c.status}</span>
        <span class="badge">${c.objective.metric}</span>
        <div class="stats">
          <span>ev ${c.evidence_count}</span>
          <span>fr ${c.frontier_count}</span>
          <span>ref ${c.reference_count}</span>
        </div>
      `;
      card.addEventListener('click', () => showDetail(c.id));
      col.appendChild(card);
    });
  });
}

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

  // Y-axis ticks
  const yTicks = 5;
  let yAxis = '';
  for (let i = 0; i <= yTicks; i++) {
    const v = valueMin + (valueRange * i / yTicks);
    const yp = y(v);
    yAxis += `<line x1="${pad.left}" y1="${yp}" x2="${width - pad.right}" y2="${yp}" stroke="#2a2e36" stroke-dasharray="2"/>`;
    yAxis += `<text x="${pad.left - 8}" y="${yp + 4}" text-anchor="end" fill="#9aa0a6" font-size="10">${v.toFixed(1)}</text>`;
  }

  // X-axis labels (first and last evidence index)
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
  const evidence = data.evidence || [];
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
    <p><span class="badge ${data.status}">${data.status}</span> <span class="badge">${campaign.lifecycle_stage}</span></p>
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

tryLoadCampaigns();
