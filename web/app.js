// ================================================================
// 动画评分排名 — 层次贝叶斯 — 纯静态前端
// ================================================================

const PAGE_SIZE = 50;
let allItems = [];
let filteredItems = [];
let shownCount = 0;

// ---- 加载数据 ----
async function init() {
  try {
    const resp = await fetch('data/rankings.json');
    allItems = await resp.json();
    document.getElementById('total-count').textContent = allItems.length.toLocaleString();
    applyFilters();
  } catch (e) {
    document.getElementById('table-body').innerHTML =
      '<tr><td colspan="7" class="loading">数据加载失败。请确认 web/data/rankings.json 存在。</td></tr>';
  }

  try {
    const s = await fetch('data/stats.json');
    const stats = await s.json();
    document.getElementById('stat-users').textContent = stats.n_users.toLocaleString();
    document.getElementById('stat-subjects').textContent = stats.n_subjects.toLocaleString();
    document.getElementById('stat-ratings').textContent = (stats.n_ratings / 1e4).toFixed(0) + '万';
    document.getElementById('stat-mu_q').textContent = stats.mu_q.toFixed(1);
    document.getElementById('stat-tau_q').textContent = stats.tau_q.toFixed(2);
    document.getElementById('stat-tau_b').textContent = stats.tau_b.toFixed(2);
    document.getElementById('stat-noise').textContent = stats.user_noise_median.toFixed(2);
    document.getElementById('stat-loss').textContent = (stats.loss_final / 1e6).toFixed(1) + 'M';
  } catch (e) { /* stats optional */ }
}

// ---- 筛选 & 排序 ----
function applyFilters() {
  const query = document.getElementById('search').value.toLowerCase().trim();
  const sortBy = document.getElementById('sort-by').value;

  let items = allItems;

  // 搜索
  if (query) {
    items = items.filter(it =>
      (it.name && it.name.toLowerCase().includes(query)) ||
      (it.name_cn && it.name_cn.toLowerCase().includes(query)) ||
      String(it.id).includes(query)
    );
  }

  // 排序
  if (sortBy === 'hb_rank') {
    items.sort((a, b) => a.hb_rank - b.hb_rank);
  } else if (sortBy === 'imdb_rank') {
    items.sort((a, b) => a.imdb_rank - b.imdb_rank);
  } else if (sortBy === 'delta') {
    items.sort((a, b) => b.delta - a.delta);
  } else if (sortBy === '-delta') {
    items.sort((a, b) => a.delta - b.delta);
  } else if (sortBy === 'score') {
    items.sort((a, b) => (b.score || 0) - (a.score || 0));
  } else if (sortBy === 'date') {
    items.sort((a, b) => (b.date || '').localeCompare(a.date || ''));
  }

  filteredItems = items;
  shownCount = 0;
  document.getElementById('filtered-count').textContent = items.length;
  renderMore();
}

// ---- 渲染 ----
function renderMore() {
  const tbody = document.getElementById('table-body');
  const showDelta = document.getElementById('show-delta').checked;
  const deltaHeader = document.getElementById('delta-header');

  if (shownCount === 0) {
    tbody.innerHTML = '';
    deltaHeader.style.display = showDelta ? '' : 'none';
  }

  const batch = filteredItems.slice(shownCount, shownCount + PAGE_SIZE);
  if (batch.length === 0) {
    document.getElementById('load-more').style.display = 'none';
    return;
  }

  const deltaCells = document.querySelectorAll('.col-delta');
  batch.forEach(it => {
    const tr = document.createElement('tr');
    tr.innerHTML =
      '<td class="col-rank">' + it.hb_rank + '</td>' +
      '<td class="col-title">' + escapeHtml(it.name_cn || it.name || '(无名)') + '</td>' +
      '<td class="col-year">' + (it.date ? it.date.slice(0, 4) : '-') + '</td>' +
      '<td class="col-score">' + (it.score != null ? it.score.toFixed(1) : '-') + '</td>' +
      '<td class="col-hb">' + it.hb_score.toFixed(2) + '</td>' +
      '<td class="col-imdb">' + it.imdb_rank + '</td>' +
      '<td class="col-delta">' + deltaHtml(it.delta) + '</td>';
    tr.addEventListener('click', () => showDetail(it));
    tbody.appendChild(tr);
  });

  // 切换 delta 列可见性
  document.querySelectorAll('.col-delta').forEach(td => {
    td.style.display = showDelta ? '' : 'none';
  });

  shownCount += batch.length;
  document.getElementById('shown-count').textContent = shownCount;

  const btn = document.getElementById('load-more');
  if (shownCount < filteredItems.length) {
    btn.style.display = 'inline-block';
  } else {
    btn.style.display = 'none';
  }
}

function deltaHtml(delta) {
  if (delta > 0) return '<span class="delta-up">+' + delta + '</span>';
  if (delta < 0) return '<span class="delta-down">' + delta + '</span>';
  return '<span class="delta-zero">-</span>';
}

// ---- 详情弹窗 ----
function showDetail(item) {
  document.getElementById('detail-title').textContent =
    (item.name_cn || item.name || '无名作品');

  const link = document.getElementById('detail-link');
  link.href = item.url || '#';
  link.style.display = item.url ? '' : 'none';

  const d = item.delta || 0;
  const deltaStr = d > 0 ? '↑' + d : (d < 0 ? '↓' + Math.abs(d) : '-');

  let explain = '';
  if (item.delta > 200) {
    explain = '这部作品在 IMDb 排名中被低估，因为它的评分者整体偏严格。层次贝叶斯模型纠正了这种偏差后，排名大幅上升。';
  } else if (item.delta > 50) {
    explain = '层次贝叶斯模型认为这部作品值得更高的排名——它的评分者虽然整体评分苛刻，但对这部作品评价很高。';
  } else if (item.delta < -50) {
    explain = '这部作品在 IMDb 排名中受益于评分者普遍手松。层次贝叶斯纠正后排名下降。';
  } else {
    explain = '两种排名方式对这部作品的评价基本一致。';
  }

  document.getElementById('detail-body').innerHTML =
    '<div class="detail-row"><span class="detail-label">Bangumi 评分</span><span class="detail-value">' + (item.score != null ? item.score.toFixed(1) : '-') + '</span></div>' +
    '<div class="detail-row"><span class="detail-label">HB 排名</span><span class="detail-value">#' + item.hb_rank + ' (' + item.hb_score.toFixed(2) + ')</span></div>' +
    '<div class="detail-row"><span class="detail-label">IMDb 排名</span><span class="detail-value">#' + item.imdb_rank + ' (' + item.imdb_score.toFixed(2) + ')</span></div>' +
    '<div class="detail-row"><span class="detail-label">排名变化</span><span class="detail-value">' + deltaStr + '</span></div>' +
    '<div class="detail-row"><span class="detail-label">年份</span><span class="detail-value">' + (item.date ? item.date.slice(0, 4) : '-') + '</span></div>' +
    '<div class="detail-explain">' + explain + '</div>';

  document.getElementById('detail-overlay').classList.remove('hidden');
}

function closeDetail() {
  document.getElementById('detail-overlay').classList.add('hidden');
}

// ---- 事件绑定 ----
document.getElementById('search').addEventListener('input', debounce(applyFilters, 200));
document.getElementById('sort-by').addEventListener('change', applyFilters);
document.getElementById('show-delta').addEventListener('change', function() {
  document.getElementById('delta-header').style.display = this.checked ? '' : 'none';
  const deltaCells = document.querySelectorAll('.col-delta');
  deltaCells.forEach(td => { td.style.display = this.checked ? '' : 'none'; });
});
document.getElementById('load-more').addEventListener('click', renderMore);
document.querySelector('.close-btn').addEventListener('click', closeDetail);
document.getElementById('detail-overlay').addEventListener('click', function(e) {
  if (e.target === this) closeDetail();
});
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeDetail();
});

// ---- 工具 ----
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function debounce(fn, ms) {
  let timer;
  return function() {
    clearTimeout(timer);
    timer = setTimeout(fn, ms);
  };
}

// ---- 统计面板切换 ----
document.getElementById('toggle-stats').addEventListener('click', function(e) {
  e.preventDefault();
  var panel = document.getElementById('stats-panel');
  panel.classList.toggle('hidden');
  this.textContent = panel.classList.contains('hidden') ? '查看统计参数' : '隐藏统计参数';
});

// ---- 启动 ----
init();
