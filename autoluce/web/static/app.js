const TOKEN_KEY = 'autoluce_web_token';
const ROOT_ORDER = ['research', 'benchmarks'];
const ROOT_LABELS = { research: 'Research', benchmarks: 'Benchmark archive' };

let allCampaigns = [];
let currentFilter = 'all';
let lastGeneration = null;
let detailTrigger = null;
let currentView = 'board';
let catalog = { series: [], comparisons: [], facets: {} };
let catalogLoaded = false;
let exploreInitialized = false;
let exploreFilters = {
  engines: new Set(), metrics: new Set(), outcomes: new Set(), families: new Set(),
  models: new Set(), backends: new Set(), contexts: new Set(),
};
let plotMetric = '';
let hiddenSeries = new Set();
let lineageTrigger = null;
let comparisonSortDescending = true;

const board = document.getElementById('board');
const boardToolbar = document.getElementById('board-toolbar');
const explore = document.getElementById('explore');
const status = document.getElementById('status');
const boardSummary = document.getElementById('board-summary');
const tokenDialog = document.getElementById('token-dialog');
const detail = document.getElementById('detail');
const detailBackdrop = document.getElementById('detail-backdrop');
const lineage = document.getElementById('lineage');
const lineageBackdrop = document.getElementById('lineage-backdrop');

const EXPLORE_FACETS = [
  ['engines', 'Engine'], ['metrics', 'Metric'], ['outcomes', 'Outcome'], ['families', 'Family'],
  ['models', 'Model'], ['backends', 'Backend'], ['contexts', 'Context'],
];
const FACET_FIELDS = { engines: 'engine', metrics: 'metric', families: 'family', models: 'model', backends: 'backend' };
const SERIES_COLORS = ['#5946b2', '#0c7a43', '#c15b28', '#176a9d', '#a23e7b', '#67721c', '#8a4f1d', '#3e5c9a'];

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

function exploreFacetValues(key) {
  return asArray(catalog.facets && catalog.facets[key]);
}

function formatExploreContext(value) {
  const number = numberValue(value);
  if (number === null) return value === null || value === undefined ? '—' : String(value);
  if (number >= 1024) return `${formatNumber(number / 1024)}K`;
  return formatNumber(number);
}

function formatExploreFacet(key, value) {
  return key === 'contexts' ? formatExploreContext(value) : String(value);
}

function facetValueFromDataset(key, value) {
  return key === 'contexts' ? Number(value) : value;
}

function resetExploreFilters(nextFacets) {
  EXPLORE_FACETS.forEach(([key]) => {
    const values = asArray(nextFacets[key]);
    const previousValues = exploreFacetValues(key);
    const previousSelection = exploreFilters[key];
    const selectedEverything = previousSelection.size === previousValues.length
      && previousValues.every(value => previousSelection.has(value));
    exploreFilters[key] = new Set(selectedEverything || !exploreInitialized
      ? values
      : values.filter(value => previousSelection.has(value)));
  });
  exploreInitialized = true;
  const selectedMetrics = exploreFilters.metrics;
  if (!selectedMetrics.has(plotMetric)) {
    plotMetric = selectedMetrics.has('prefill_tok_s')
      ? 'prefill_tok_s'
      : asArray(nextFacets.metrics).find(metric => selectedMetrics.has(metric)) || '';
  }
}

function isExploreFilterActive(key) {
  return exploreFilters[key].size !== exploreFacetValues(key).length;
}

function matchesExploreFacet(key, value) {
  return !isExploreFilterActive(key) || exploreFilters[key].has(value);
}

function seriesMatchesOutcome(series) {
  if (!isExploreFilterActive('outcomes')) return true;
  return asArray(catalog.comparisons).some(comparison => (
    comparison.campaign_locator === series.campaign_locator
      && comparison.metric === series.metric
      && exploreFilters.outcomes.has(comparison.outcome)
  ));
}

function seriesMatchesContext(series) {
  if (!isExploreFilterActive('contexts')) return true;
  return asArray(series.points).some(point => (
    typeof point.x === 'number' && exploreFilters.contexts.has(point.x)
  ));
}

function filteredExploreSeries() {
  return asArray(catalog.series).filter(series => (
    matchesExploreFacet('engines', series.engine)
      && matchesExploreFacet('metrics', series.metric)
      && matchesExploreFacet('families', series.family)
      && matchesExploreFacet('models', series.model)
      && matchesExploreFacet('backends', series.backend)
      && seriesMatchesOutcome(series)
      && seriesMatchesContext(series)
  ));
}

function plotPoints(series) {
  return asArray(series.points).filter(point => (
    numberValue(point.y) !== null
      && (!isExploreFilterActive('contexts')
        || (typeof point.x === 'number' && exploreFilters.contexts.has(point.x)))
  ));
}

function comparisonMatchesSeriesFacet(comparison, key) {
  if (!isExploreFilterActive(key)) return true;
  return asArray(catalog.series).some(series => (
    series.campaign_locator === comparison.campaign_locator
      && series.metric === comparison.metric
      && exploreFilters[key].has(series[key])
  ));
}

function comparisonMatchesContext(comparison) {
  if (!isExploreFilterActive('contexts')) return true;
  return typeof comparison.context === 'number' && exploreFilters.contexts.has(comparison.context);
}

function filteredExploreComparisons() {
  return asArray(catalog.comparisons).filter(comparison => {
    const engineMatches = !isExploreFilterActive('engines')
      || exploreFilters.engines.has(comparison.engine_candidate)
      || exploreFilters.engines.has(comparison.engine_reference);
    return engineMatches
      && matchesExploreFacet('metrics', comparison.metric)
      && matchesExploreFacet('families', comparison.family)
      && matchesExploreFacet('outcomes', comparison.outcome)
      && comparisonMatchesSeriesFacet(comparison, 'models')
      && comparisonMatchesSeriesFacet(comparison, 'backends')
      && comparisonMatchesContext(comparison);
  });
}

function facetCount(key, value) {
  if (key === 'outcomes') {
    return asArray(catalog.comparisons).filter(comparison => comparison.outcome === value).length;
  }
  if (key === 'contexts') {
    return asArray(catalog.series).reduce((total, series) => (
      total + asArray(series.points).filter(point => point.x === value).length
    ), 0);
  }
  return asArray(catalog.series).filter(series => series[FACET_FIELDS[key]] === value).length;
}

function renderExploreFacetGroup(key, label) {
  const values = exploreFacetValues(key);
  if (!values.length) return '';
  const chips = values.map(value => {
    const selected = exploreFilters[key].has(value);
    return `<button class="explore-chip ${selected ? 'is-active' : ''}" type="button" data-explore-facet="${escapeHtml(key)}" data-explore-value="${escapeHtml(String(value))}" aria-pressed="${selected}">
      <span>${escapeHtml(formatExploreFacet(key, value))}</span><small>${facetCount(key, value)}</small>
    </button>`;
  }).join('');
  return `<fieldset class="explore-filter-group explore-filter-group--${escapeHtml(key)}"><legend>${escapeHtml(label)}</legend><div class="explore-chips">${chips}</div></fieldset>`;
}

function stableColor(seriesId) {
  const hash = [...String(seriesId)].reduce((total, char) => ((total * 31) + char.charCodeAt(0)) >>> 0, 0);
  return SERIES_COLORS[hash % SERIES_COLORS.length];
}

function pointKey(point) {
  return typeof point.x === 'number' ? `number:${point.x}` : `label:${String(point.x)}`;
}

function isNumericSeries(points) {
  return points.length > 0 && points.every(point => typeof point.x === 'number' && Number.isFinite(point.x));
}

function renderExploreChart(series) {
  const visible = series.filter(item => !hiddenSeries.has(item.series_id));
  if (!visible.length) {
    return '<p class="empty-state">Every plotted series is hidden. Use its legend item to show it again.</p>';
  }

  const allPoints = visible.flatMap(item => plotPoints(item));
  if (!allPoints.length) return '<p class="empty-state">No finite values remain for this plot.</p>';

  const numericX = [...new Set(allPoints.filter(point => typeof point.x === 'number').map(point => point.x))].sort((a, b) => a - b);
  const labelsX = [...new Set(allPoints.filter(point => typeof point.x !== 'number').map(point => String(point.x)))];
  const axisValues = [
    ...numericX.map(value => ({ key: `number:${value}`, label: formatExploreContext(value) })),
    ...labelsX.map(value => ({ key: `label:${value}`, label: value })),
  ];
  const width = 840;
  const height = 280;
  const margin = { top: 18, right: 24, bottom: 54, left: 68 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xAt = index => axisValues.length === 1
    ? margin.left + (plotWidth / 2)
    : margin.left + (index / (axisValues.length - 1)) * plotWidth;
  const xPositions = new Map(axisValues.map((axis, index) => [axis.key, xAt(index)]));
  const values = allPoints.map(point => Number(point.y));
  const rawMin = Math.min(...values);
  const rawMax = Math.max(...values);
  const padding = rawMin === rawMax ? Math.max(Math.abs(rawMin) * 0.12, 1) : (rawMax - rawMin) * 0.1;
  const yMin = rawMin - padding;
  const yMax = rawMax + padding;
  const yAt = value => margin.top + ((yMax - value) / (yMax - yMin)) * plotHeight;
  const yTicks = Array.from({ length: 5 }, (_, index) => yMin + ((yMax - yMin) * index / 4));
  const grid = yTicks.map(value => {
    const y = yAt(value);
    return `<g><line class="explore-grid-line" x1="${margin.left}" x2="${width - margin.right}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"></line><text class="explore-axis-label" x="${margin.left - 11}" y="${(y + 4).toFixed(1)}" text-anchor="end">${escapeHtml(formatNumber(value))}</text></g>`;
  }).join('');
  const xLabels = axisValues.map((axis, index) => {
    const x = xAt(index);
    return `<text class="explore-axis-label explore-axis-label--x" x="${x.toFixed(1)}" y="${height - 32}" text-anchor="end" transform="rotate(-34 ${x.toFixed(1)} ${height - 32})">${escapeHtml(axis.label)}</text>`;
  }).join('');
  const paths = visible.map(item => {
    const points = plotPoints(item);
    const color = stableColor(item.series_id);
    const sorted = isNumericSeries(points) ? points.slice().sort((left, right) => left.x - right.x) : points;
    const line = isNumericSeries(sorted) && sorted.length > 1
      ? `M ${sorted.map(point => `${xPositions.get(pointKey(point)).toFixed(1)} ${yAt(Number(point.y)).toFixed(1)}`).join(' L ')}`
      : '';
    const label = `${item.label || item.series_id}, ${item.metric}`;
    const lineMarkup = line ? `<path class="explore-series-line" d="${line}" stroke="${color}"></path>
      <path class="explore-series-hit" d="${line}" data-lineage-series="${escapeHtml(item.series_id)}" tabindex="0" role="button" aria-label="Open lineage for ${escapeHtml(label)}"></path>` : '';
    const pointsMarkup = sorted.map(point => {
      const x = xPositions.get(pointKey(point));
      const y = yAt(Number(point.y));
      return `<circle class="explore-chart-point" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="4.5" fill="${color}" data-lineage-series="${escapeHtml(item.series_id)}" tabindex="0" role="button" aria-label="Open lineage for ${escapeHtml(label)}"><title>${escapeHtml(`${formatExploreContext(point.x)}: ${formatNumber(point.y)}`)}</title></circle>`;
    }).join('');
    return `<g>${lineMarkup}${pointsMarkup}</g>`;
  }).join('');
  return `<svg class="explore-chart" viewBox="0 0 ${width} ${height}" role="group" aria-label="${escapeHtml(plotMetric)} comparison plot. Click or press Enter on a line or point to open its lineage.">
    <rect class="explore-plot-surface" x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}"></rect>
    ${grid}
    <line class="explore-axis-line" x1="${margin.left}" x2="${width - margin.right}" y1="${height - margin.bottom}" y2="${height - margin.bottom}"></line>
    ${xLabels}
    <text class="explore-axis-title" x="${margin.left}" y="12">${escapeHtml(plotMetric)}${series[0] && series[0].unit ? ` · ${escapeHtml(series[0].unit)}` : ''}</text>
    ${paths}
  </svg>`;
}

function renderSeriesLegend(series) {
  if (!series.length) return '';
  return `<aside class="explore-legend-panel" aria-label="Series legend">
    <p class="explore-legend-title">Series legend <span>${series.length}</span></p>
    <p class="explore-legend-hint">Colors identify lines. Select a label to show or hide it.</p>
    <div class="explore-legend">${series.map(item => {
    const hidden = hiddenSeries.has(item.series_id);
    return `<button class="explore-legend-toggle ${hidden ? 'is-hidden' : ''}" type="button" data-series-toggle="${escapeHtml(item.series_id)}" aria-pressed="${!hidden}" title="Show or hide ${escapeHtml(item.label || item.series_id)}">
      <span class="legend-swatch" style="--series-color: ${stableColor(item.series_id)}"></span><span>${escapeHtml(item.label || item.series_id)}</span>
    </button>`;
  }).join('')}</div>
  </aside>`;
}

function outcomeClass(outcome) {
  return ['win', 'loss', 'parity'].includes(outcome) ? `outcome--${outcome}` : 'outcome--parity';
}

function sortedExploreComparisons() {
  const rows = filteredExploreComparisons().slice();
  return rows.sort((left, right) => {
    const leftValue = numberValue(left.delta_pct);
    const rightValue = numberValue(right.delta_pct);
    if (leftValue === null) return 1;
    if (rightValue === null) return -1;
    return comparisonSortDescending ? rightValue - leftValue : leftValue - rightValue;
  });
}

function renderExploreComparisons(rows) {
  if (!rows.length) return '<p class="empty-state">No head-to-head rows match these filters.</p>';
  const body = rows.map(row => {
    const index = asArray(catalog.comparisons).indexOf(row);
    const unit = metricUnit(row.metric);
    return `<tr class="explore-comparison-row" data-lineage-comparison="${index}" tabindex="0" role="button" aria-label="Open lineage for ${escapeHtml(row.campaign_name || 'comparison')}">
      <td data-label="Campaign"><strong>${escapeHtml(row.campaign_name || '—')}</strong><span>${escapeHtml(row.family || '—')}</span></td>
      <td data-label="Metric">${escapeHtml(row.metric || '—')}<span>${escapeHtml(row.label || formatExploreContext(row.context))}</span></td>
      <td data-label="Candidate"><strong>${escapeHtml(engineName(row.engine_candidate))}</strong><span>${formatNumber(row.value_candidate)}${unit ? ` ${escapeHtml(unit)}` : ''}</span></td>
      <td data-label="Reference"><strong>${escapeHtml(engineName(row.engine_reference))}</strong><span>${formatNumber(row.value_reference)}${unit ? ` ${escapeHtml(unit)}` : ''}</span></td>
      <td data-label="Delta"><strong class="delta ${numberValue(row.delta_pct) === null || numberValue(row.delta_pct) === 0 ? 'is-neutral' : numberValue(row.delta_pct) < 0 ? 'is-loss' : ''}">${formatPercent(row.delta_pct)}</strong></td>
      <td data-label="Outcome"><span class="outcome-chip ${outcomeClass(row.outcome)}">${escapeHtml(row.outcome || 'parity')}</span></td>
      <td data-label="Lineage"><span class="row-lineage-link">Trace lineage <span aria-hidden="true">→</span></span></td>
    </tr>`;
  }).join('');
  return `<div class="explore-table-wrap"><table class="explore-table"><thead><tr>
    <th>Campaign</th><th>Metric / run</th><th>Candidate</th><th>Reference</th>
    <th aria-sort="${comparisonSortDescending ? 'descending' : 'ascending'}"><button class="table-sort" type="button" data-sort-delta aria-label="Sort by delta, currently ${comparisonSortDescending ? 'highest first' : 'lowest first'}">Delta ${comparisonSortDescending ? '↓' : '↑'}</button></th><th>Outcome</th><th>Lineage</th>
  </tr></thead><tbody>${body}</tbody></table></div>`;
}

function renderExplore() {
  const selectedMetrics = exploreFacetValues('metrics').filter(metric => exploreFilters.metrics.has(metric));
  const filteredSeries = filteredExploreSeries();
  const plotSeries = filteredSeries.filter(series => series.metric === plotMetric && plotPoints(series).length);
  const comparisons = sortedExploreComparisons();
  const metricOptions = selectedMetrics.map(metric => `<option value="${escapeHtml(metric)}" ${metric === plotMetric ? 'selected' : ''}>${escapeHtml(metric)}</option>`).join('');
  const metricControl = metricOptions
    ? `<label class="plot-metric-control" for="plot-metric">Plot metric<select id="plot-metric">${metricOptions}</select></label>`
    : '<p class="explore-muted">Select at least one metric to compose a plot.</p>';
  explore.innerHTML = `<section class="explore-view" aria-labelledby="explore-heading">
    <div class="section-heading explore-heading">
      <div><p class="eyebrow">Compose the evidence</p><h1 id="explore-heading">Explore runs</h1></div>
      <p>OR within each filter; AND across filters. Select a line, point, or comparison to trace it back to evidence and code.</p>
    </div>
    <div class="explore-filter-bar" aria-label="Catalog filters">
      ${EXPLORE_FACETS.map(([key, label]) => renderExploreFacetGroup(key, label)).join('')}
    </div>
    <div class="explore-summary">${filteredSeries.length} series · ${comparisons.length} head-to-head row${comparisons.length === 1 ? '' : 's'} · ${plotSeries.length} series in this plot</div>
    <article class="explore-card" aria-labelledby="plot-heading">
      <div class="explore-card-heading"><div><p class="eyebrow">Overlay</p><h2 id="plot-heading">Same metric, shared view</h2></div>${metricControl}</div>
      <div class="explore-plot-layout">
        <div class="explore-chart-region">
          <p class="explore-trace-hint">Click a line or point to trace its lineage to evidence and code.</p>
          ${plotMetric ? renderExploreChart(plotSeries) : ''}
        </div>
        ${renderSeriesLegend(plotSeries)}
      </div>
    </article>
    <section class="explore-comparisons" aria-labelledby="explore-comparisons-heading">
      <div class="explore-card-heading"><div><p class="eyebrow">Head to head</p><h2 id="explore-comparisons-heading">Filtered comparisons</h2></div><p class="explore-muted">Select any row to trace its campaign, evidence, revision, and binary.</p></div>
      ${renderExploreComparisons(comparisons)}
    </section>
  </section>`;
  explore.setAttribute('aria-busy', 'false');
  if (currentView === 'explore') {
    status.textContent = `${filteredSeries.length} series · ${comparisons.length} comparisons · refreshed ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
  }
}

async function loadCatalog() {
  explore.setAttribute('aria-busy', 'true');
  try {
    const response = await apiFetch('/api/catalog');
    const nextCatalog = await response.json();
    resetExploreFilters(nextCatalog.facets || {});
    catalog = {
      series: asArray(nextCatalog.series), comparisons: asArray(nextCatalog.comparisons), facets: nextCatalog.facets || {},
    };
    hiddenSeries = new Set([...hiddenSeries].filter(seriesId => catalog.series.some(series => series.series_id === seriesId)));
    catalogLoaded = true;
    if (currentView === 'explore') renderExplore();
    return true;
  } catch (error) {
    explore.setAttribute('aria-busy', 'false');
    if (error.message !== 'Unauthorized' && currentView === 'explore') {
      status.textContent = `Unable to load catalog: ${error.message}`;
    }
    return false;
  }
}

function copyableLineageValue(value) {
  if (!value) return '<span class="lineage-missing">Not recorded</span>';
  return `<button class="copy-value" type="button" data-copy-value="${escapeHtml(String(value))}" title="Copy ${escapeHtml(String(value))}">${escapeHtml(String(value))}</button>`;
}

function openLineage(data, trigger) {
  const evidenceIds = asArray(data.evidenceIds).filter(Boolean);
  lineageTrigger = trigger || document.activeElement;
  lineageBackdrop.hidden = false;
  lineage.hidden = false;
  document.body.classList.add('drawer-open');
  document.getElementById('lineage-kicker').textContent = data.kind;
  document.getElementById('lineage-title').textContent = data.title || 'Run lineage';
  document.getElementById('lineage-body').innerHTML = `<p class="lineage-intro">Follow this result from campaign through the recorded run and evidence to the code and binary used.</p>
    <ol class="lineage-breadcrumb">
      <li><span>Campaign</span><strong>${escapeHtml(data.campaignName || '—')}</strong><small>${escapeHtml(data.campaignLocator || 'No campaign locator')}</small></li>
      <li><span>Run</span><strong>${escapeHtml(data.runLabel || '—')}</strong><small>${escapeHtml(data.runDetail || '')}</small></li>
      <li><span>Evidence</span><div class="lineage-values">${evidenceIds.length ? evidenceIds.map(copyableLineageValue).join('') : '<span class="lineage-missing">Not recorded</span>'}</div></li>
      <li><span>Code / binary</span><dl class="lineage-code"><dt>Git revision</dt><dd>${copyableLineageValue(data.gitRev)}</dd><dt>Binary SHA-256</dt><dd>${copyableLineageValue(data.binarySha)}</dd></dl></li>
    </ol>
    <p id="lineage-copy-status" class="lineage-copy-status" role="status" aria-live="polite"></p>`;
  document.getElementById('close-lineage').focus();
}

function openSeriesLineage(seriesId, trigger) {
  const series = asArray(catalog.series).find(item => item.series_id === seriesId);
  if (!series) return;
  const lineageData = series.lineage || {};
  openLineage({
    kind: 'Series lineage', title: series.label || series.series_id,
    campaignName: series.campaign_name, campaignLocator: lineageData.campaign_locator || series.campaign_locator,
    runLabel: series.label || series.series_id, runDetail: [series.engine, series.metric, series.model, series.backend].filter(Boolean).join(' · '),
    evidenceIds: lineageData.evidence_ids || asArray(series.points).map(point => point.evidence_id),
    gitRev: lineageData.git_rev, binarySha: lineageData.binary_sha256,
  }, trigger);
}

function openComparisonLineage(index, trigger) {
  const comparison = asArray(catalog.comparisons)[Number(index)];
  if (!comparison) return;
  const candidateSeries = asArray(catalog.series).find(series => (
    series.campaign_locator === comparison.campaign_locator
      && series.metric === comparison.metric
      && series.engine === comparison.engine_candidate
  )) || asArray(catalog.series).find(series => (
    series.campaign_locator === comparison.campaign_locator
      && series.engine === comparison.engine_candidate
  ));
  const candidateLineage = candidateSeries && candidateSeries.lineage ? candidateSeries.lineage : {};
  const candidatePoint = candidateSeries && asArray(candidateSeries.points).find(point => (
    point.evidence_id === comparison.candidate_evidence_id
  ));
  openLineage({
    kind: 'Comparison lineage', title: comparison.campaign_name || 'Head-to-head comparison',
    campaignName: comparison.campaign_name, campaignLocator: comparison.campaign_locator,
    runLabel: `${engineName(comparison.engine_candidate)} vs ${engineName(comparison.engine_reference)}`,
    runDetail: `${comparison.metric || 'metric'} · ${comparison.label || formatExploreContext(comparison.context)}`,
    evidenceIds: comparison.candidate_evidence_id
      ? [comparison.candidate_evidence_id]
      : candidatePoint && candidatePoint.evidence_id ? [candidatePoint.evidence_id] : [],
    gitRev: comparison.git_rev || candidateLineage.git_rev,
    binarySha: comparison.binary_sha256 || candidateLineage.binary_sha256,
  }, trigger);
}

function closeLineage() {
  lineage.hidden = true;
  lineageBackdrop.hidden = true;
  document.body.classList.remove('drawer-open');
  if (lineageTrigger) lineageTrigger.focus();
  lineageTrigger = null;
}

async function copyLineageValue(value, trigger) {
  let copied = false;
  try {
    await navigator.clipboard.writeText(value);
    copied = true;
  } catch (_) {
    const input = document.createElement('textarea');
    input.value = value;
    input.setAttribute('readonly', '');
    input.style.position = 'fixed';
    input.style.opacity = '0';
    document.body.append(input);
    input.select();
    copied = document.execCommand('copy');
    input.remove();
  }
  const message = document.getElementById('lineage-copy-status');
  if (message) message.textContent = copied ? 'Copied to clipboard.' : 'Unable to copy; select the value manually.';
  if (!copied && trigger) trigger.focus();
}

function setView(view) {
  currentView = view;
  const exploring = view === 'explore';
  board.hidden = exploring;
  boardToolbar.hidden = exploring;
  explore.hidden = !exploring;
  document.querySelectorAll('.view-toggle').forEach(button => {
    const selected = button.dataset.view === view;
    button.classList.toggle('is-active', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  if (exploring) {
    renderExplore();
    if (!catalogLoaded) loadCatalog();
  } else {
    status.textContent = `${allCampaigns.length} campaigns · refreshed ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
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
document.getElementById('close-lineage').addEventListener('click', closeLineage);
lineageBackdrop.addEventListener('click', closeLineage);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !lineage.hidden) {
    closeLineage();
    return;
  }
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

document.querySelectorAll('.view-toggle').forEach(button => {
  button.addEventListener('click', () => setView(button.dataset.view));
});

explore.addEventListener('click', event => {
  const chip = event.target.closest('[data-explore-facet]');
  if (chip) {
    const key = chip.dataset.exploreFacet;
    const value = facetValueFromDataset(key, chip.dataset.exploreValue);
    if (exploreFilters[key].has(value)) exploreFilters[key].delete(value);
    else exploreFilters[key].add(value);
    if (key === 'metrics' && !exploreFilters.metrics.has(plotMetric)) {
      plotMetric = exploreFacetValues('metrics').find(metric => exploreFilters.metrics.has(metric)) || '';
    }
    renderExplore();
    return;
  }
  const seriesToggle = event.target.closest('[data-series-toggle]');
  if (seriesToggle) {
    const seriesId = seriesToggle.dataset.seriesToggle;
    if (hiddenSeries.has(seriesId)) hiddenSeries.delete(seriesId);
    else hiddenSeries.add(seriesId);
    renderExplore();
    return;
  }
  if (event.target.closest('[data-sort-delta]')) {
    comparisonSortDescending = !comparisonSortDescending;
    renderExplore();
    return;
  }
  const seriesTarget = event.target.closest('[data-lineage-series]');
  if (seriesTarget) {
    openSeriesLineage(seriesTarget.dataset.lineageSeries, seriesTarget);
    return;
  }
  const comparisonTarget = event.target.closest('[data-lineage-comparison]');
  if (comparisonTarget) openComparisonLineage(comparisonTarget.dataset.lineageComparison, comparisonTarget);
});

explore.addEventListener('change', event => {
  if (event.target.id === 'plot-metric') {
    plotMetric = event.target.value;
    renderExplore();
  }
});

explore.addEventListener('keydown', event => {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  const target = event.target.closest('[data-lineage-series], [data-lineage-comparison]');
  if (!target) return;
  event.preventDefault();
  if (target.dataset.lineageSeries) openSeriesLineage(target.dataset.lineageSeries, target);
  else openComparisonLineage(target.dataset.lineageComparison, target);
});

lineage.addEventListener('click', event => {
  const copyButton = event.target.closest('[data-copy-value]');
  if (copyButton) copyLineageValue(copyButton.dataset.copyValue, copyButton);
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
  if (await loadCampaigns()) {
    await loadCatalog();
    await checkVersion();
  }
});

document.getElementById('token-input').addEventListener('keydown', event => {
  if (event.key === 'Enter') document.getElementById('token-save').click();
});

async function checkVersion() {
  if (document.visibilityState !== 'visible') return;
  try {
    const data = await (await apiFetch('/api/version')).json();
    if (lastGeneration !== null && data.generation !== lastGeneration) {
      await Promise.all([loadCampaigns(), loadCatalog()]);
    }
    lastGeneration = data.generation;
  } catch (_) {
    // A later visible poll retries transient failures. A 401 has already opened the token dialog.
  }
}

async function initialize() {
  if (await loadCampaigns()) {
    await loadCatalog();
    await checkVersion();
  }
}

setInterval(checkVersion, 5000);
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') checkVersion();
});

// Deep-link the active view (#explore) so it is bookmarkable/screenshot-addressable.
window.addEventListener('hashchange', () => setView(location.hash === '#explore' ? 'explore' : 'board'));
if (location.hash === '#explore') setView('explore');

initialize();
