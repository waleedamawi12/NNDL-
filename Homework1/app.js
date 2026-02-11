// app.js
// Titanic EDA web app (browser-only, GitHub Pages deployable)
//
// Reuse note:
// - To adapt to other split datasets (train/test), swap DEFAULT URLs, SCHEMA, and the columns used in charts/stats.
// - The merge pattern stays the same: parse both CSVs client-side, add `source`, concatenate.

/* ----------------------------- Config / Schema ----------------------------- */

const DEFAULT_TRAIN_URL = "./train.csv";
const DEFAULT_TEST_URL = "./test.csv";

// Titanic schema (swap here for reuse)
const SCHEMA = {
  target: "Survived", // train only
  id: "PassengerId",  // exclude from stats/corr
  features: ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
  // For numeric computations / correlation / histograms:
  numeric: ["Age", "SibSp", "Parch", "Fare"],
  // For categorical counts / bar charts:
  categorical: ["Sex", "Pclass", "Embarked"]
};

/* ----------------------------- DOM helpers ----------------------------- */

const $ = (sel) => document.querySelector(sel);

const els = {
  trainUrl: $("#trainUrl"),
  testUrl: $("#testUrl"),
  trainFile: $("#trainFile"),
  testFile: $("#testFile"),

  btnLoad: $("#btnLoad"),
  btnRun: $("#btnRun"),
  btnExportCsv: $("#btnExportCsv"),
  btnExportJson: $("#btnExportJson"),

  loadStatus: $("#loadStatus"),

  overviewKvs: $("#overviewKvs"),
  previewTable: $("#previewTable"),
  missingTable: $("#missingTable"),
  numericStatsTable: $("#numericStatsTable"),
  categoricalTable: $("#categoricalTable"),
  groupBySurvivedTable: $("#groupBySurvivedTable"),

  canvases: {
    missing: $("#missingChart"),
    barSex: $("#barSex"),
    barPclass: $("#barPclass"),
    barEmbarked: $("#barEmbarked"),
    histAge: $("#histAge"),
    histFare: $("#histFare"),
    corr: $("#corrHeatmap")
  }
};

function setStatus(kind, text) {
  // kind: "ok" | "warn" | "bad" | "info"
  const cls = kind === "ok" ? "ok" : kind === "warn" ? "warn" : kind === "bad" ? "bad" : "";
  els.loadStatus.innerHTML = `<span class="pill ${cls}">${escapeHtml(text)}</span>`;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmt(num, digits = 3) {
  if (num === null || num === undefined || Number.isNaN(num)) return "";
  if (!Number.isFinite(num)) return String(num);
  return Number(num).toFixed(digits);
}

function downloadText(filename, text, mime = "text/plain;charset=utf-8") {
  const blob = new Blob([text], { type: mime });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(a.href);
    a.remove();
  }, 0);
}

/* ----------------------------- Global state ----------------------------- */

let trainRows = null;
let testRows = null;
let mergedRows = null;

let charts = {
  missing: null,
  barSex: null,
  barPclass: null,
  barEmbarked: null,
  histAge: null,
  histFare: null,
  corr: null
};

let latestSummary = null;

/* ----------------------------- CSV Loading ----------------------------- */

function parseCsvFromFile(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      quotes: true,
      complete: (res) => {
        if (res.errors && res.errors.length) {
          reject(new Error(res.errors.map(e => e.message).join("; ")));
          return;
        }
        resolve(res.data);
      },
      error: (err) => reject(err)
    });
  });
}

function parseCsvFromUrl(url) {
  return new Promise((resolve, reject) => {
    Papa.parse(url, {
      download: true,
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
      quotes: true,
      complete: (res) => {
        if (res.errors && res.errors.length) {
          reject(new Error(res.errors.map(e => e.message).join("; ")));
          return;
        }
        resolve(res.data);
      },
      error: (err) => reject(err)
    });
  });
}

async function loadTrainAndTest() {
  const trainFile = els.trainFile.files?.[0] || null;
  const testFile = els.testFile.files?.[0] || null;

  // Default URLs (user-editable)
  const trainUrl = (els.trainUrl.value || DEFAULT_TRAIN_URL).trim();
  const testUrl = (els.testUrl.value || DEFAULT_TEST_URL).trim();

  setStatus("info", "loading…");

  try {
    const [train, test] = await Promise.all([
      trainFile ? parseCsvFromFile(trainFile) : parseCsvFromUrl(trainUrl),
      testFile ? parseCsvFromFile(testFile) : parseCsvFromUrl(testUrl)
    ]);

    // Light validation
    if (!Array.isArray(train) || !train.length) throw new Error("Train CSV parsed but has no rows.");
    if (!Array.isArray(test) || !test.length) throw new Error("Test CSV parsed but has no rows.");

    trainRows = train.map(r => ({ ...r, source: "train" }));
    testRows = test.map(r => ({ ...r, source: "test" }));

    // Ensure Survived exists for test rows as null (for consistent columns)
    for (const r of testRows) {
      if (!(SCHEMA.target in r)) r[SCHEMA.target] = null;
    }

    mergedRows = [...trainRows, ...testRows];

    setStatus("ok", `loaded ✓ (train=${trainRows.length}, test=${testRows.length})`);

    // Enable buttons
    els.btnRun.disabled = false;
    els.btnExportCsv.disabled = false; // merged available immediately
    els.btnExportJson.disabled = true; // summary generated after Run EDA

    renderOverview();
    renderPreviewTable(mergedRows, 10);

  } catch (err) {
    console.error(err);
    setStatus("bad", "load failed");
    alert(
      "Could not load/parse CSVs.\n\n" +
      "Common fixes:\n" +
      "• Ensure train.csv and test.csv are committed to the repo root\n" +
      "• Check the URL fields (default: ./train.csv and ./test.csv)\n" +
      "• If using file override, ensure you selected the right files\n\n" +
      "Details: " + (err?.message || String(err))
    );
  }
}

/* ----------------------------- Overview / Preview ----------------------------- */

function getAllColumns(rows) {
  const set = new Set();
  for (const r of rows) {
    for (const k of Object.keys(r)) set.add(k);
  }
  return Array.from(set);
}

function renderOverview() {
  if (!mergedRows) return;

  const cols = getAllColumns(mergedRows);
  const trainN = trainRows?.length ?? 0;
  const testN = testRows?.length ?? 0;

  const kvs = [
    ["Rows (merged)", mergedRows.length],
    ["Rows (train)", trainN],
    ["Rows (test)", testN],
    ["Columns", cols.length]
  ];

  els.overviewKvs.innerHTML = kvs.map(([k, v]) => `
    <div class="kv">
      <div class="k">${escapeHtml(k)}</div>
      <div class="v">${escapeHtml(v)}</div>
    </div>
  `).join("");
}

function renderPreviewTable(rows, limit = 10) {
  const table = els.previewTable;
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");

  const cols = getAllColumns(rows);
  const viewCols = [
    SCHEMA.id,
    "source",
    SCHEMA.target,
    ...SCHEMA.features
  ].filter((c, i, arr) => arr.indexOf(c) === i && cols.includes(c));

  const slice = rows.slice(0, limit);

  thead.innerHTML = `<tr>${viewCols.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr>`;
  tbody.innerHTML = slice.map(r => `
    <tr>${viewCols.map(c => `<td>${escapeHtml(r?.[c] ?? "")}</td>`).join("")}</tr>
  `).join("");
}

/* ----------------------------- Missingness ----------------------------- */

function isMissing(v) {
  // PapaParse dynamicTyping gives null for empty fields sometimes; also handle "" and undefined.
  return v === null || v === undefined || (typeof v === "string" && v.trim() === "");
}

function computeMissingness(rows) {
  const cols = getAllColumns(rows);
  const n = rows.length;

  const out = cols.map(col => {
    let miss = 0;
    for (const r of rows) if (isMissing(r[col])) miss++;
    return {
      column: col,
      missing: miss,
      total: n,
      pct: n ? (miss / n) * 100 : 0
    };
  });

  out.sort((a, b) => b.pct - a.pct);
  return out;
}

function renderMissingTable(miss) {
  const t = els.missingTable;
  const thead = t.querySelector("thead");
  const tbody = t.querySelector("tbody");

  thead.innerHTML = `<tr>
    <th>Column</th><th>Missing</th><th>Total</th><th>% Missing</th>
  </tr>`;

  tbody.innerHTML = miss.map(m => `
    <tr>
      <td>${escapeHtml(m.column)}</td>
      <td>${escapeHtml(m.missing)}</td>
      <td>${escapeHtml(m.total)}</td>
      <td>${escapeHtml(fmt(m.pct, 2))}</td>
    </tr>
  `).join("");
}

/* ----------------------------- Stats ----------------------------- */

function toNumber(v) {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function mean(arr) {
  const xs = arr.filter(x => x !== null);
  if (!xs.length) return null;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function median(arr) {
  const xs = arr.filter(x => x !== null).slice().sort((a, b) => a - b);
  if (!xs.length) return null;
  const mid = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
}

function std(arr) {
  // Sample std dev
  const xs = arr.filter(x => x !== null);
  if (xs.length < 2) return null;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) ** 2;
  return Math.sqrt(s / (xs.length - 1));
}

function computeNumericStats(rows, numericCols) {
  const res = [];
  for (const c of numericCols) {
    const vals = rows.map(r => toNumber(r[c])).filter(v => v !== null);
    res.push({
      column: c,
      count: vals.length,
      mean: mean(vals),
      median: median(vals),
      std: std(vals),
      min: vals.length ? Math.min(...vals) : null,
      max: vals.length ? Math.max(...vals) : null
    });
  }
  return res;
}

function renderNumericStatsTable(stats) {
  const t = els.numericStatsTable;
  const thead = t.querySelector("thead");
  const tbody = t.querySelector("tbody");

  thead.innerHTML = `<tr>
    <th>Column</th><th>N</th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th>
  </tr>`;

  tbody.innerHTML = stats.map(s => `
    <tr>
      <td>${escapeHtml(s.column)}</td>
      <td>${escapeHtml(s.count)}</td>
      <td>${escapeHtml(fmt(s.mean, 3))}</td>
      <td>${escapeHtml(fmt(s.median, 3))}</td>
      <td>${escapeHtml(fmt(s.std, 3))}</td>
      <td>${escapeHtml(fmt(s.min, 3))}</td>
      <td>${escapeHtml(fmt(s.max, 3))}</td>
    </tr>
  `).join("");
}

function computeCategoricalCounts(rows, catCols, topK = 10) {
  // Returns [{column, counts:[{key,count}...]}]
  const out = [];
  for (const c of catCols) {
    const map = new Map();
    for (const r of rows) {
      const v = r[c];
      if (isMissing(v)) continue;
      const key = String(v);
      map.set(key, (map.get(key) || 0) + 1);
    }
    const counts = Array.from(map.entries())
      .map(([key, count]) => ({ key, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, topK);

    out.push({ column: c, counts });
  }
  return out;
}

function renderCategoricalTable(catStats) {
  // Compact table: column | value | count
  const t = els.categoricalTable;
  const thead = t.querySelector("thead");
  const tbody = t.querySelector("tbody");

  thead.innerHTML = `<tr><th>Column</th><th>Value</th><th>Count</th></tr>`;

  const rows = [];
  for (const cs of catStats) {
    if (!cs.counts.length) {
      rows.push(`<tr><td>${escapeHtml(cs.column)}</td><td></td><td>0</td></tr>`);
      continue;
    }
    for (let i = 0; i < cs.counts.length; i++) {
      const { key, count } = cs.counts[i];
      rows.push(`<tr>
        <td>${escapeHtml(i === 0 ? cs.column : "")}</td>
        <td>${escapeHtml(key)}</td>
        <td>${escapeHtml(count)}</td>
      </tr>`);
    }
  }
  tbody.innerHTML = rows.join("");
}

function computeGroupBySurvivedMeans(train, numericCols) {
  const groups = new Map(); // survived => array rows
  for (const r of train) {
    const y = r[SCHEMA.target];
    if (y === null || y === undefined || (y !== 0 && y !== 1)) continue;
    if (!groups.has(y)) groups.set(y, []);
    groups.get(y).push(r);
  }
  const ys = Array.from(groups.keys()).sort((a, b) => a - b);

  const res = [];
  for (const col of numericCols) {
    const row = { column: col };
    for (const y of ys) {
      const vals = groups.get(y).map(r => toNumber(r[col])).filter(v => v !== null);
      row[`mean_${y}`] = mean(vals);
      row[`median_${y}`] = median(vals);
      row[`n_${y}`] = vals.length;
    }
    res.push(row);
  }
  return { ys, res };
}

function renderGroupBySurvivedTable(grouped) {
  const t = els.groupBySurvivedTable;
  const thead = t.querySelector("thead");
  const tbody = t.querySelector("tbody");

  if (!grouped.ys.length) {
    thead.innerHTML = `<tr><th>Train-only grouping unavailable (no Survived labels found)</th></tr>`;
    tbody.innerHTML = "";
    return;
  }

  const ys = grouped.ys;
  const headCells = [
    "<th>Column</th>",
    ...ys.flatMap(y => [`<th>Survived=${y} (N)</th>`, `<th>Mean</th>`, `<th>Median</th>`])
  ];

  thead.innerHTML = `<tr>${headCells.join("")}</tr>`;

  tbody.innerHTML = grouped.res.map(r => {
    const cells = [`<td>${escapeHtml(r.column)}</td>`];
    for (const y of ys) {
      cells.push(`<td>${escapeHtml(r[`n_${y}`] ?? "")}</td>`);
      cells.push(`<td>${escapeHtml(fmt(r[`mean_${y}`], 3))}</td>`);
      cells.push(`<td>${escapeHtml(fmt(r[`median_${y}`], 3))}</td>`);
    }
    return `<tr>${cells.join("")}</tr>`;
  }).join("");
}

/* ----------------------------- Charts helpers ----------------------------- */

function destroyChart(key) {
  if (charts[key]) {
    charts[key].destroy();
    charts[key] = null;
  }
}

function makeBarChart(canvas, labels, data, title) {
  return new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: title, data }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        title: { display: true, text: title }
      },
      scales: {
        x: { ticks: { color: "#cbd5e1" }, grid: { color: "rgba(36,48,68,.45)" } },
        y: { ticks: { color: "#cbd5e1" }, grid: { color: "rgba(36,48,68,.45)" }, beginAtZero: true }
      }
    }
  });
}

function makeHistogram(values, binCount = 20) {
  const xs = values.filter(v => v !== null && Number.isFinite(v));
  if (!xs.length) return { labels: [], counts: [] };

  const min = Math.min(...xs);
  const max = Math.max(...xs);
  if (min === max) {
    return { labels: [`${fmt(min, 2)}`], counts: [xs.length] };
  }

  const width = (max - min) / binCount;
  const bins = Array.from({ length: binCount }, () => 0);

  for (const v of xs) {
    let idx = Math.floor((v - min) / width);
    if (idx === binCount) idx = binCount - 1;
    bins[idx]++;
  }

  const labels = bins.map((_, i) => {
    const a = min + i * width;
    const b = a + width;
    return `${fmt(a, 1)}–${fmt(b, 1)}`;
  });

  return { labels, counts: bins };
}

// Correlation (Pearson) with pairwise deletion
function pearson(x, y) {
  const pairs = [];
  for (let i = 0; i < x.length; i++) {
    const a = x[i], b = y[i];
    if (a === null || b === null) continue;
    if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
    pairs.push([a, b]);
  }
  if (pairs.length < 3) return null;

  const xs = pairs.map(p => p[0]);
  const ys = pairs.map(p => p[1]);
  const mx = mean(xs);
  const my = mean(ys);

  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < xs.length; i++) {
    const a = xs[i] - mx;
    const b = ys[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  const den = Math.sqrt(dx * dy);
  return den === 0 ? null : num / den;
}

function computeCorrelationMatrix(rows, cols) {
  const colVals = {};
  for (const c of cols) colVals[c] = rows.map(r => toNumber(r[c]));
  const m = {};
  for (const a of cols) {
    m[a] = {};
    for (const b of cols) {
      m[a][b] = pearson(colVals[a], colVals[b]);
    }
  }
  return m;
}

// Simple red-white-blue-ish mapping without external libs.
// We avoid hardcoding chart theme colors, but we do need correlation coloring.
// This maps -1..1 to a visible gradient.
function corrToRgba(r) {
  if (r === null || r === undefined || Number.isNaN(r)) return "rgba(148,163,184,0.25)"; // slate-ish
  const v = Math.max(-1, Math.min(1, r));
  // Two-sided interpolation
  if (v >= 0) {
    // 0..1 : light -> strong
    const a = 0.25 + 0.55 * v;
    return `rgba(96,165,250,${a})`; // blue-ish
  } else {
    const a = 0.25 + 0.55 * (-v);
    return `rgba(248,113,113,${a})`; // red-ish
  }
}

// Correlation "heatmap" with Chart.js scatter rectangles.
// Each point is a cell; x,y are categories; pointStyle 'rect' approximates a heatmap.
function makeCorrHeatmap(canvas, cols, corr) {
  const points = [];
  for (let yi = 0; yi < cols.length; yi++) {
    for (let xi = 0; xi < cols.length; xi++) {
      const y = cols[yi];
      const x = cols[xi];
      const r = corr[y][x];
      points.push({
        x,
        y,
        rValue: r,
        backgroundColor: corrToRgba(r),
        borderColor: "rgba(36,48,68,.65)"
      });
    }
  }

  return new Chart(canvas, {
    type: "scatter",
    data: {
      datasets: [{
        label: "Correlation",
        data: points,
        pointStyle: "rect",
        pointRadius: 14,
        pointHoverRadius: 16,
        parsing: false,
        backgroundColor: (ctx) => ctx.raw?.backgroundColor || "rgba(148,163,184,0.25)",
        borderColor: (ctx) => ctx.raw?.borderColor || "rgba(36,48,68,.65)",
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      plugins: {
        title: { display: true, text: "Correlation heatmap (Pearson)" },
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const p = ctx.raw;
              const rv = p?.rValue;
              return ` ${p.y} vs ${p.x}: ${rv === null ? "n/a" : fmt(rv, 3)}`;
            }
          }
        }
      },
      scales: {
        x: {
          type: "category",
          labels: cols,
          ticks: { color: "#cbd5e1", maxRotation: 45, minRotation: 45 },
          grid: { display: false }
        },
        y: {
          type: "category",
          labels: cols,
          ticks: { color: "#cbd5e1" },
          grid: { display: false }
        }
      }
    }
  });
}

/* ----------------------------- EDA Runner ----------------------------- */

function runEda() {
  if (!mergedRows || !trainRows || !testRows) {
    alert("Load data first.");
    return;
  }

  // Missingness
  const missing = computeMissingness(mergedRows);
  renderMissingTable(missing);

  destroyChart("missing");
  charts.missing = makeBarChart(
    els.canvases.missing,
    missing.map(m => m.column),
    missing.map(m => Number(fmt(m.pct, 2))),
    "% Missing per column"
  );

  // Stats
  const numericStats = computeNumericStats(mergedRows, SCHEMA.numeric);
  renderNumericStatsTable(numericStats);

  const catStats = computeCategoricalCounts(mergedRows, SCHEMA.categorical, 10);
  renderCategoricalTable(catStats);

  const grouped = computeGroupBySurvivedMeans(trainRows, SCHEMA.numeric);
  renderGroupBySurvivedTable(grouped);

  // Visualizations (categorical bars)
  const sexCounts = computeCategoricalCounts(mergedRows, ["Sex"], 20)[0]?.counts || [];
  const pclassCounts = computeCategoricalCounts(mergedRows, ["Pclass"], 20)[0]?.counts || [];
  const embCounts = computeCategoricalCounts(mergedRows, ["Embarked"], 20)[0]?.counts || [];

  destroyChart("barSex");
  charts.barSex = makeBarChart(
    els.canvases.barSex,
    sexCounts.map(d => d.key),
    sexCounts.map(d => d.count),
    "Counts: Sex"
  );

  destroyChart("barPclass");
  charts.barPclass = makeBarChart(
    els.canvases.barPclass,
    pclassCounts.map(d => d.key),
    pclassCounts.map(d => d.count),
    "Counts: Pclass"
  );

  destroyChart("barEmbarked");
  charts.barEmbarked = makeBarChart(
    els.canvases.barEmbarked,
    embCounts.map(d => d.key),
    embCounts.map(d => d.count),
    "Counts: Embarked"
  );

  // Histograms
  const ageVals = mergedRows.map(r => toNumber(r.Age));
  const fareVals = mergedRows.map(r => toNumber(r.Fare));

  const ageHist = makeHistogram(ageVals, 20);
  destroyChart("histAge");
  charts.histAge = makeBarChart(els.canvases.histAge, ageHist.labels, ageHist.counts, "Histogram: Age");

  const fareHist = makeHistogram(fareVals, 20);
  destroyChart("histFare");
  charts.histFare = makeBarChart(els.canvases.histFare, fareHist.labels, fareHist.counts, "Histogram: Fare");

  // Correlation heatmap
  const corrCols = SCHEMA.numeric; // keep it simple; add more numeric cols if desired
  const corr = computeCorrelationMatrix(mergedRows, corrCols);

  destroyChart("corr");
  charts.corr = makeCorrHeatmap(els.canvases.corr, corrCols, corr);

  // Build JSON summary for export
  latestSummary = {
    schema: SCHEMA,
    shape: {
      merged_rows: mergedRows.length,
      train_rows: trainRows.length,
      test_rows: testRows.length,
      columns: getAllColumns(mergedRows)
    },
    missingness: missing,
    numeric_stats: numericStats,
    categorical_counts: catStats,
    group_by_survived_numeric_means: grouped,
    correlation: corr,
    generated_at: new Date().toISOString()
  };

  els.btnExportJson.disabled = false;
  setStatus("ok", "EDA ready ✓");
}

/* ----------------------------- Export ----------------------------- */

function exportMergedCsv() {
  if (!mergedRows) {
    alert("No merged dataset to export. Load first.");
    return;
  }
  try {
    const csv = Papa.unparse(mergedRows, { quotes: true });
    downloadText("titanic_merged_train_test.csv", csv, "text/csv;charset=utf-8");
  } catch (err) {
    console.error(err);
    alert("CSV export failed: " + (err?.message || String(err)));
  }
}

function exportJsonSummary() {
  if (!latestSummary) {
    alert("Run EDA first to generate a summary JSON.");
    return;
  }
  try {
    const json = JSON.stringify(latestSummary, null, 2);
    downloadText("titanic_eda_summary.json", json, "application/json;charset=utf-8");
  } catch (err) {
    console.error(err);
    alert("JSON export failed: " + (err?.message || String(err)));
  }
}

/* ----------------------------- Events / Init ----------------------------- */

els.btnLoad.addEventListener("click", loadTrainAndTest);
els.btnRun.addEventListener("click", runEda);
els.btnExportCsv.addEventListener("click", exportMergedCsv);
els.btnExportJson.addEventListener("click", exportJsonSummary);

// Auto-fill defaults (in case HTML was edited)
els.trainUrl.value = els.trainUrl.value || DEFAULT_TRAIN_URL;
els.testUrl.value = els.testUrl.value || DEFAULT_TEST_URL;

// Optional: attempt auto-load on first paint (nice for GitHub Pages demo)
// If files are missing, it will alert with actionable guidance.
window.addEventListener("DOMContentLoaded", () => {
  // Comment out the next line if you want strictly manual load.
  loadTrainAndTest();
});
