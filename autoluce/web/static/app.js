const TOKEN_KEY = 'autoluce_web_token';
const ROOT_ORDER = ['research', 'benchmarks'];
const ROOT_LABELS = { research: 'Research', benchmarks: 'Benchmark archive' };

let allCampaigns = [];
let currentFilter = 'all';
let lastGeneration = null;
let detailTrigger = null;

const board = document.getElementById('board');
const status = document.getElementById('status');
const boardSummary = document.getElementById('board-summary');
const tokenDialog = document.getElementById('token-dialog');
const detail = document.getElementById('detail');
const detailBackdrop = document.getElementById('detail-backdrop');

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || '';
}

function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function apiFetch(path) {
  const response = await fetch(path, { headers: authHeaders() });
  if (response.status === 401) {
    showTokenDialog('Authentication required.');
    throw new Error('Unauthorized');
  }
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  return response;
}

function showTokenDialog(message) {
  document.getElementById('token-error').textContent = message || '';
  tokenDialog.hidden = false;
  document.getElementById('token-input').focus();
}

function hideTokenDialog() {
  tokenDialog.hidden = true;
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[char]));
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function numberValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatNumber(value) {
  const number = numberValue(value);
  if (number === null) return '—';
  const fractionDigits = Math.abs(number) >= 100 ? 0 : Math.abs(number) >= 10 ? 1 : 2;
  return number.toLocaleString(undefined, { maximumFractionDigits: fractionDigits });
}

function formatPercent(value) {
  const number = numberValue(value);
  if (number === null) return '—';
  return `${number >= 0 ? '+' : ''}${number.toFixed(1)}%`;
}

function formatContext(value) {
  const number = numberValue(value);
  if (number === null) return escapeHtml(value || 'context');
  if (number >= 1024) return `${formatNumber(number / 1024)}K context`;
  return `${formatNumber(number)} context`;
}

function formatLabel(row) {
  // Prefer an explicit label (e.g. "DSpark"/"AR"); fall back to a context length.
  return row && row.label ? escapeHtml(row.label) : formatContext(row ? row.context : null);
}

function metricUnit(metric) {
  return metric && metric.endsWith('_tok_s') ? 'tok/s' : '';
}

function objectiveMetric(campaign) {
  return campaign && campaign.objective && campaign.objective.metric || 'objective';
}

function engineName(runtime) {
  const value = String(runtime || 'unknown');
  if (/dflash|lucebox/i.test(value)) return 'dFlash';
  if (/llama/i.test(value)) return 'llama.cpp';
  return value || 'unknown';
}

function campaignEngine(campaign) {
  return engineName(campaign && campaign.system && campaign.system.runtime);
}

function engineBadge(engine) {
  if (engine === 'dFlash') return 'badge--dflash';
  if (engine === 'llama.cpp') return 'badge--llama';
  return '';
}

function statusBadge(statusValue) {
  return ['planned', 'measured', 'promoted'].includes(statusValue) ? `badge--${statusValue}` : '';
}

function descriptionForCampaign(campaign) {
  const system = campaign.system || {};
  if (campaign.variant) return campaign.name || '';
  return [system.model, system.backend].filter(Boolean).join(' · ') || campaign.path || '';
}

function comparisonValues(row) {
  const candidate = engineName(row.engine_candidate);
  const reference = engineName(row.engine_reference);
  const candidateValue = numberValue(row.value_candidate);
  const referenceValue = numberValue(row.value_reference);
  const reportedDelta = numberValue(row.delta_pct);

  if (candidate === 'dFlash') {
    return {
      context: formatLabel(row),
      dflash: candidateValue,
      llama: referenceValue,
      dflashLabel: candidate,
      llamaLabel: reference,
      delta: reportedDelta,
    };
  }
  if (reference === 'dFlash') {
    const delta = candidateValue !== null && candidateValue !== 0 && referenceValue !== null
      ? ((referenceValue - candidateValue) / candidateValue) * 100
      : reportedDelta === null ? null : -reportedDelta;
    return {
      context: formatLabel(row),
      dflash: referenceValue,
      llama: candidateValue,
      dflashLabel: reference,
      llamaLabel: candidate,
      delta,
    };
  }
  return {
    context: formatContext(row.context),
    dflash: candidateValue,
    llama: referenceValue,
    dflashLabel: candidate,
    llamaLabel: reference,
    delta: reportedDelta,
  };
}

function comparisonLead(rows) {
  const values = rows.map(comparisonValues).filter(row => row.delta !== null);
  if (!values.length) return { text: 'No reported gap', className: 'is-neutral' };
  const best = values.reduce((winner, row) => Math.abs(row.delta) > Math.abs(winner.delta) ? row : winner);
  if (best.delta > 0) return { text: `dFlash ${formatPercent(best.delta)} faster`, className: '' };
  if (best.delta < 0) return { text: `dFlash ${formatPercent(Math.abs(best.delta))} behind`, className: 'is-loss' };
  return { text: 'Even performance', className: 'is-neutral' };
}

function comparisonMagnitude(rows) {
  return asArray(rows).map(comparisonValues).reduce((largest, row) => {
    return row.delta === null ? largest : Math.max(largest, Math.abs(row.delta));
  }, -1);
}

function renderComparisonRows(rows) {
  return asArray(rows).slice().sort((left, right) => Number(left.context) - Number(right.context)).map(row => {
    const values = comparisonValues(row);
    const deltaClass = values.delta === null || values.delta === 0 ? 'is-neutral' : values.delta < 0 ? 'is-loss' : '';
    return `<div class="comparison-row">
      <span class="comparison-context">${values.context}</span>
      <span class="compare-value"><span class="compare-key">${escapeHtml(values.dflashLabel)}</span><strong>${formatNumber(values.dflash)}</strong></span>
      <span class="compare-value"><span class="compare-key">${escapeHtml(values.llamaLabel)}</span><strong>${formatNumber(values.llama)}</strong></span>
      <span class="compare-value compare-gap"><span class="compare-key">dFlash gap</span><strong class="delta ${deltaClass}">${formatPercent(values.delta)}</strong></span>
    </div>`;
  }).join('');
}

function renderComparisonCard(campaign) {
  const rows = asArray(campaign.head_to_head);
  const lead = comparisonLead(rows);
  const title = campaign.variant || campaign.name || 'Untitled campaign';
  const metric = rows[0] && rows[0].metric || objectiveMetric(campaign);
  const unit = metricUnit(metric);
  return `<article class="comparison-card">
    <h2 class="visually-hidden">${escapeHtml(title)}</h2>
    <button class="comparison-open open-detail" type="button" data-id="${escapeHtml(campaign.id)}" aria-label="Open ${escapeHtml(title)} details">
      <span class="comparison-card__head">
        <span>
          <span class="campaign-title">${escapeHtml(title)}</span>
          <span class="comparison-subtitle">${escapeHtml(descriptionForCampaign(campaign))} · ${escapeHtml(metric)}${unit ? ` · ${unit}` : ''}</span>
        </span>
        <span class="comparison-hero ${lead.className}">${escapeHtml(lead.text)}</span>
      </span>
      <span class="comparison-rows">${renderComparisonRows(rows)}</span>
    </button>
  </article>`;
}

function hasMeaningfulSparkline(values) {
  const finite = asArray(values).map(numberValue).filter(value => value !== null);
  return finite.length >= 2 && new Set(finite).size >= 2;
}

function renderSparkline(values) {
  const points = values.map(numberValue).filter(value => value !== null);
  const width = 180;
  const height = 32;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = Math.max(max - min, Number.EPSILON);
  const x = index => 1 + index / (points.length - 1) * (width - 2);
  const y = value => height - 2 - (value - min) / range * (height - 4);
  const line = points.map((value, index) => `${index ? 'L' : 'M'} ${x(index).toFixed(1)} ${y(value).toFixed(1)}`).join(' ');
  const area = `M ${x(0).toFixed(1)} ${height} ${points.map((value, index) => `L ${x(index).toFixed(1)} ${y(value).toFixed(1)}`).join(' ')} L ${x(points.length - 1).toFixed(1)} ${height} Z`;
  return `<svg class="sparkline" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="Objective trend across ${points.length} evidence points">
    <path d="${area}" fill="#eeeafd"></path>
    <path d="${line}" fill="none" stroke="#5946b2" stroke-width="1.75" vector-effect="non-scaling-stroke"></path>
  </svg>`;
}

function renderCampaignCard(campaign) {
  const title = campaign.variant || campaign.name || 'Untitled campaign';
  const metric = objectiveMetric(campaign);
  const values = asArray(campaign.sparkline);
  const latest = values.length ? numberValue(values[values.length - 1]) : null;
  const unit = metricUnit(metric);
  const trend = hasMeaningfulSparkline(values) ? renderSparkline(values) : '';
  const statusValue = campaign.status || 'planned';
  return `<article class="campaign-card">
    <h4><button class="campaign-open open-detail" type="button" data-id="${escapeHtml(campaign.id)}" aria-label="Open ${escapeHtml(title)} details">
      <span class="campaign-topline">
        <span>
          <span class="campaign-title">${escapeHtml(title)}</span>
          <span class="campaign-model">${escapeHtml(descriptionForCampaign(campaign))}</span>
        </span>
        <span class="badge ${engineBadge(campaignEngine(campaign))}">${escapeHtml(campaignEngine(campaign))}</span>
      </span>
      <span class="metric-block">
        <span class="metric-label">${escapeHtml(metric)}</span>
        ${latest === null
          ? '<span class="metric-empty">No measurement</span>'
          : `<span class="metric-value">${formatNumber(latest)}${unit ? `<span>${unit}</span>` : ''}</span>`}
      </span>
      ${trend}
      <span class="campaign-footer">
        <span class="badges"><span class="badge ${statusBadge(statusValue)}">${escapeHtml(statusValue)}</span><span class="badge">${escapeHtml(campaign.stage || '—')}</span></span>
        <span>${Number(campaign.evidence_count) || 0} evidence</span>
      </span>
    </button></h4>
  </article>`;
}

function groupCampaigns(campaigns) {
  return campaigns.reduce((roots, campaign) => {
    const root = campaign.root || 'research';
    const family = campaign.family || 'Other';
    if (!roots[root]) roots[root] = {};
    if (!roots[root][family]) roots[root][family] = [];
    roots[root][family].push(campaign);
    return roots;
  }, {});
}

function renderCampaignTree(campaigns) {
  if (!campaigns.length) {
    return '<p class="empty-state"><strong>No evidence campaigns match this filter.</strong><br>Head-to-head comparisons remain above when available.</p>';
  }
  const roots = groupCampaigns(campaigns);
  const rootNames = [...ROOT_ORDER.filter(root => roots[root]), ...Object.keys(roots).filter(root => !ROOT_ORDER.includes(root)).sort()];
  return rootNames.map(root => {
    const families = roots[root];
    const familyCards = Object.keys(families).sort().map(family => {
      const items = families[family].slice().sort((left, right) => {
        const leftName = left.variant || left.name || '';
        const rightName = right.variant || right.name || '';
        return leftName.localeCompare(rightName);
      });
      return `<section class="family" aria-labelledby="family-${escapeHtml(root)}-${escapeHtml(family)}">
        <div class="family-header">
          <h3 id="family-${escapeHtml(root)}-${escapeHtml(family)}">${escapeHtml(family)}</h3>
          <span class="family-count">${items.length} campaign${items.length === 1 ? '' : 's'}</span>
        </div>
        <div class="campaign-grid">${items.map(renderCampaignCard).join('')}</div>
      </section>`;
    }).join('');
    return `<section class="tree-section" aria-labelledby="root-${escapeHtml(root)}">
      <h2 id="root-${escapeHtml(root)}" class="root-heading">${escapeHtml(ROOT_LABELS[root] || root)} <span>campaign evidence</span></h2>
      <div class="family-grid">${familyCards}</div>
    </section>`;
  }).join('');
}

function renderBoard() {
  const filtered = currentFilter === 'all'
    ? allCampaigns
    : allCampaigns.filter(campaign => campaign.status === currentFilter);
  const comparisons = filtered.filter(campaign => asArray(campaign.head_to_head).length);
  const evidenceCampaigns = filtered.filter(campaign => !asArray(campaign.head_to_head).length);
  comparisons.sort((left, right) => comparisonMagnitude(right.head_to_head) - comparisonMagnitude(left.head_to_head));

  const comparisonContent = comparisons.length
    ? `<div class="comparison-grid">${comparisons.map(renderComparisonCard).join('')}</div>`
    : '<p class="empty-state"><strong>No head-to-head comparison matches this filter.</strong><br>Campaigns without a paired dFlash and llama.cpp result are listed below.</p>';
  board.innerHTML = `<section aria-labelledby="comparisons-heading">
    <div class="section-heading">
      <div><p class="eyebrow">First, the result</p><h1 id="comparisons-heading">dFlash vs llama.cpp</h1></div>
      <p>Positive gaps mean dFlash is faster. Every row compares the same context and metric.</p>
    </div>
    ${comparisonContent}
  </section>
  <section aria-labelledby="campaigns-heading">
    <div class="section-heading">
      <div><p class="eyebrow">Then, the evidence</p><h2 id="campaigns-heading">Campaigns</h2></div>
      <p>Unpaired campaigns, grouped by their research or benchmark directory.</p>
    </div>
    ${renderCampaignTree(evidenceCampaigns)}
  </section>`;
  boardSummary.textContent = `${filtered.length} campaign${filtered.length === 1 ? '' : 's'} shown · ${comparisons.length} comparison${comparisons.length === 1 ? '' : 's'}`;
  board.setAttribute('aria-busy', 'false');
}

async function loadCampaigns() {
  board.setAttribute('aria-busy', 'true');
  try {
    const response = await apiFetch('/api/campaigns');
    const data = await response.json();
    allCampaigns = asArray(data);
    renderBoard();
    status.textContent = `${allCampaigns.length} campaigns · refreshed ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
    return true;
  } catch (error) {
    board.setAttribute('aria-busy', 'false');
    if (error.message !== 'Unauthorized') status.textContent = `Unable to load campaigns: ${error.message}`;
    return false;
  }
}

function closeDetail() {
  detail.hidden = true;
  detailBackdrop.hidden = true;
  document.body.classList.remove('drawer-open');
  if (detailTrigger) detailTrigger.focus();
  detailTrigger = null;
}

function jsonBlock(value) {
  return escapeHtml(JSON.stringify(value || {}, null, 2));
}

function renderReference(reference) {
  if (reference && typeof reference === 'object') {
    const kind = reference.kind || 'reference';
    const location = reference.locator || reference.path || reference.name || '';
    return `<li><strong>${escapeHtml(kind)}</strong>${location ? ` <span class="reference-kind">· ${escapeHtml(location)}</span>` : ''}</li>`;
  }
  return `<li>${escapeHtml(reference)}</li>`;
}

function formatMap(values) {
  const entries = Object.entries(values || {});
  if (!entries.length) return '—';
  return entries.map(([key, value]) => `${key}: ${typeof value === 'number' ? formatNumber(value) : String(value)}`).join(' · ');
}

function renderEvidence(evidence, frontier) {
  if (!evidence.length) return '<p class="empty-state">No evidence has been recorded.</p>';
  const rows = evidence.map((item, index) => {
    const id = item.evidence_id || `evidence ${index + 1}`;
    const isFrontier = frontier.has(id);
    return `<tr>
      <td data-label="Evidence"><span class="evidence-id">${escapeHtml(id)}</span>${isFrontier ? '<span class="frontier-mark">Frontier</span>' : ''}</td>
      <td data-label="Metrics">${escapeHtml(formatMap(item.metrics))}</td>
      <td data-label="Gates">${escapeHtml(formatMap(item.gates))}</td>
    </tr>`;
  }).join('');
  return `<table class="evidence-table"><thead><tr><th>Evidence</th><th>Metrics</th><th>Gates</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function renderDetail(data) {
  const campaign = data.campaign || {};
  const state = data.state || {};
  const objective = campaign.objective || {};
  const evidence = asArray(state.evidence);
  const references = asArray(state.references);
  const comparisons = asArray(state.comparisons).filter(item => item && item.kind === 'head_to_head');
  const frontier = new Set(asArray(state.frontier));
  const system = campaign.system || {};
  const title = campaign.name || data.campaign_id || 'Campaign details';
  const promotion = state.promotion || data.promotion;

  document.getElementById('detail-title').textContent = title;
  document.getElementById('detail-kicker').textContent = `${data.status || 'campaign'} · ${data.stage || campaign.lifecycle_stage || '—'}`;
  document.getElementById('detail-body').innerHTML = `
    <div class="detail-summary">
      <p class="detail-path">${escapeHtml(data.path || '')}</p>
      <span class="badges"><span class="badge ${statusBadge(data.status)}">${escapeHtml(data.status || '—')}</span><span class="badge">${escapeHtml(data.stage || campaign.lifecycle_stage || '—')}</span></span>
    </div>
    <section class="detail-section">
      <h3>Objective</h3>
      <dl class="definition-list">
        <dt>Metric</dt><dd>${escapeHtml(objective.metric || '—')} ${metricUnit(objective.metric) ? `(${metricUnit(objective.metric)})` : ''}</dd>
        <dt>Direction</dt><dd>${escapeHtml(objective.direction || '—')}</dd>
        <dt>Promoted evidence</dt><dd>${promotion ? escapeHtml(promotion) : '—'}</dd>
      </dl>
    </section>
    <section class="detail-section">
      <h3>System</h3>
      <dl class="definition-list">
        <dt>Engine</dt><dd>${escapeHtml(campaignEngine(campaign))}</dd>
        <dt>Model</dt><dd>${escapeHtml(system.model || '—')}</dd>
        <dt>Backend</dt><dd>${escapeHtml(system.backend || '—')}</dd>
        <dt>Runtime</dt><dd>${escapeHtml(system.runtime || '—')}</dd>
      </dl>
    </section>
    <section class="detail-section">
      <h3>Constraints</h3>
      <pre class="data-block">${jsonBlock(campaign.constraints)}</pre>
    </section>
    ${comparisons.length ? `<section class="detail-section"><h3>dFlash vs llama.cpp</h3><div class="detail-comparison">${renderComparisonRows(comparisons)}</div></section>` : ''}
    <section class="detail-section">
      <h3>Evidence (${evidence.length})</h3>
      ${renderEvidence(evidence, frontier)}
    </section>
    <section class="detail-section">
      <h3>References (${references.length})</h3>
      ${references.length ? `<ul class="reference-list">${references.map(renderReference).join('')}</ul>` : '<p class="empty-state">No references attached.</p>'}
    </section>
    <section class="detail-section">
      <details><summary>Full campaign and state record</summary><pre class="data-block">${jsonBlock({ campaign, state })}</pre></details>
    </section>`;
}

async function showDetail(id, trigger) {
  detailTrigger = trigger || document.activeElement;
  detailBackdrop.hidden = false;
  detail.hidden = false;
  document.body.classList.add('drawer-open');
  document.getElementById('detail-title').textContent = 'Loading campaign…';
  document.getElementById('detail-kicker').textContent = 'Campaign';
  document.getElementById('detail-body').innerHTML = '<p class="empty-state">Loading campaign details…</p>';
  document.getElementById('close-detail').focus();
  try {
    const response = await apiFetch(`/api/campaigns/${encodeURIComponent(id)}`);
    renderDetail(await response.json());
  } catch (error) {
    document.getElementById('detail-body').innerHTML = `<p class="empty-state">Unable to load campaign details: ${escapeHtml(error.message)}</p>`;
  }
}

document.getElementById('board').addEventListener('click', event => {
  const trigger = event.target.closest('.open-detail');
  if (trigger && trigger.dataset.id) showDetail(trigger.dataset.id, trigger);
});

document.getElementById('close-detail').addEventListener('click', closeDetail);
detailBackdrop.addEventListener('click', closeDetail);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !detail.hidden) closeDetail();
});

document.querySelectorAll('.filter').forEach(button => {
  button.addEventListener('click', () => {
    currentFilter = button.dataset.filter;
    document.querySelectorAll('.filter').forEach(filter => {
      const selected = filter === button;
      filter.classList.toggle('is-active', selected);
      filter.setAttribute('aria-pressed', String(selected));
    });
    renderBoard();
  });
});

document.getElementById('token-save').addEventListener('click', async () => {
  const input = document.getElementById('token-input');
  const token = input.value.trim();
  if (!token) {
    document.getElementById('token-error').textContent = 'Enter a token to continue.';
    input.focus();
    return;
  }
  setToken(token);
  hideTokenDialog();
  if (await loadCampaigns()) await checkVersion();
});

document.getElementById('token-input').addEventListener('keydown', event => {
  if (event.key === 'Enter') document.getElementById('token-save').click();
});

async function checkVersion() {
  if (document.visibilityState !== 'visible') return;
  try {
    const data = await (await apiFetch('/api/version')).json();
    if (lastGeneration !== null && data.generation !== lastGeneration) await loadCampaigns();
    lastGeneration = data.generation;
  } catch (_) {
    // A later visible poll retries transient failures. A 401 has already opened the token dialog.
  }
}

async function initialize() {
  if (await loadCampaigns()) await checkVersion();
}

setInterval(checkVersion, 5000);
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') checkVersion();
});

initialize();
