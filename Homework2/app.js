// app.js
// Browser-only Titanic EDA (GitHub Pages deployable)
//
// Uses:
// - PapaParse for robust CSV parsing (handles commas inside quotes correctly)
// - Chart.js for charts
//
// Reuse note:
// To reuse this for other datasets split into train/test, change SCHEMA below
// and adjust feature lists / target name as needed.

(() => {
  "use strict";

  // -----------------------------
  // Schema (swap here for other datasets)
  // -----------------------------
  const SCHEMA = {
    target: "Survived", // label (train only)
    features: ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
    identifier: "PassengerId", // exclude from stats/corr
  };

  // -----------------------------
  // State
  // -----------------------------
  const state = {
    train: null,
    test: null,
    merged: null,
    summary: null,
    charts: {
      missing: null,
      sex: null,
      pclass: null,
      embarked: null,
      ageHist: null,
      fareHist: null,
      corr: null,
    },
  };

  // -----------------------------
  // DOM
  // -----------------------------
  const $ = (id) => document.getElementById(id);

  const el = {
    trainFile: $("trainFile"),
    testFile: $("testFile"),
    btnLoad: $("btnLoad"),
    btnRun: $("btnRun"),
    btnReset: $("btnReset"),
    loadStatus: $("loadStatus"),

    kpiRows: $("kpiRows"),
    kpiCols: $("kpiCols"),
    kpiSplit: $("kpiSplit"),

    previewRows: $("previewRows"),
    previewTable: $("previewTable"),

    missingChart: $("missingChart"),
    missingNote: $("missingNote"),

    numericStatsTable: $("numericStatsTable"),
    numericStatsBySurvivedTable: $("numericStatsBySurvivedTable"),
    catCountsTable: $("catCountsTable"),
    catCountsBySurvivedTable: $("catCountsBySurvivedTable"),

    sexBar: $("sexBar"),
    pclassBar: $("pclassBar"),
    embarkedBar: $("embarkedBar"),
    ageHist: $("ageHist"),
    fareHist: $("fareHist"),
    corrHeatmap: $("corrHeatmap"),

    btnExportCSV: $("btnExportCSV"),
    btnExportJSON: $("btnExportJSON"),
    exportStatus: $("exportStatus"),
  };

  // -----------------------------
  // UI helpers
  // -----------------------------
  function escapeHtml(str) {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function setStatus(text) {
    el.loadStatus.innerHTML = `<b>Status:</b> ${escapeHtml(text)}`;
  }

  function setExportStatus(text) {
    el.exportStatus.innerHTML = `<b>Export:</b> ${escapeHtml(text)}`;
  }

  function alertUser(msg) {
    window.alert(msg);
  }

  // -----------------------------
  // Parsing (PapaParse)
  // -----------------------------
  function parseCsvFile(file) {
    // PapaParse configured to correctly handle:
    // - commas inside quoted strings (e.g., Name column)
    // - numeric typing where possible
    return new Promise((resolve, reject) => {
      Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        quotes: true,        // important for comma-in-quotes correctness
        dynamicTyping: true, // convert numeric fields where possible
        complete: (results) => {
          if (results.errors && results.errors.length) {
            console.error("PapaParse errors:", results.errors);
            reject(new Error(results.errors[0].message || "CSV parse error"));
            return;
          }
          resolve(results.data);
        },
        error: (err) => reject(err),
      });
    });
  }

  // -----------------------------
  // Merge train/test (add source column)
  // -----------------------------
  function normalizeRows(rows, sourceLabel) {
    return rows.map((r) => ({ ...r, source: sourceLabel }));
  }

  function mergeDatasets(trainRows, testRows) {
    // Train has target column, test typically does not.
    // We keep all columns and add `source` to distinguish them.
    return normalizeRows(trainRows, "train").concat(normalizeRows(testRows, "test"));
  }

  // -----------------------------
  // Overview
  // -----------------------------
  function getColumns(rows) {
    const s = new Set();
    for (const r of rows) Object.keys(r).forEach((k) => s.add(k));
    return Array.from(s);
  }

  function computeSplit(rows) {
    let train = 0, test = 0;
    for (const r of rows) (r.source === "train" ? train++ : test++);
    return { train, test };
  }

  function renderKPIs(rows) {
    const cols = getColumns(rows);
    const split = computeSplit(rows);
    el.kpiRows.textContent = String(rows.length);
    el.kpiCols.textContent = String(cols.length);
    el.kpiSplit.textContent = `${split.train} / ${split.test}`;
  }

  function renderPreviewTable(rows) {
    const n = Math.max(1, Number(el.previewRows.value || 12));
    const sample = rows.slice(0, n);
    const cols = getColumns(sample);

    const thead = el.previewTable.querySelector("thead");
    const tbody = el.previewTable.querySelector("tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    const trh = document.createElement("tr");
    for (const c of cols) {
      const th = document.createElement("th");
      th.textContent = c;
      trh.appendChild(th);
    }
    thead.appendChild(trh);

    for (const r of sample) {
      const tr = document.createElement("tr");
      for (const c of cols) {
        const td = document.createElement("td");
        const v = r[c];
        td.textContent = v === null || v === undefined || v === "" ? "—" : String(v);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }

  // -----------------------------
  // Missing values
  // -----------------------------
  function isMissing(v) {
    return v === null || v === undefined || v === "";
  }

  function missingness(rows) {
    const cols = getColumns(rows);
    const total = rows.length;
    const counts = Object.fromEntries(cols.map((c) => [c, 0]));

    for (const r of rows) {
      for (const c of cols) {
        if (isMissing(r[c])) counts[c] += 1;
      }
    }

    const pct = cols.map((c) => ({
      col: c,
      missing: counts[c],
      pct: total ? (counts[c] / total) * 100 : 0,
    }));

    pct.sort((a, b) => b.pct - a.pct);
    return pct;
  }

  function destroyChart(ref) {
    if (ref && typeof ref.destroy === "function") ref.destroy();
    return null;
  }

  function renderMissingChart(rows) {
    const miss = missingness(rows);
    const labels = miss.map((d) => d.col);
    const values = miss.map((d) => Number(d.pct.toFixed(2)));

    state.charts.missing = destroyChart(state.charts.missing);

    state.charts.missing = new Chart(el.missingChart, {
      type: "bar",
      data: {
        labels,
        datasets: [{ label: "Missing (%)", data: values }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, max: 100, ticks: { callback: (v) => `${v}%` } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => `${ctx.raw}% missing` } },
        },
      },
    });

    const top = miss.slice(0, 5).map((d) => `${d.col}: ${d.pct.toFixed(1)}%`).join(" • ");
    el.missingNote.textContent = miss.length ? `Top missing columns: ${top}` : "No data loaded.";
  }

  // -----------------------------
  // Stats (numeric + categorical)
  // -----------------------------
  function toNumberOrNull(v) {
    if (isMissing(v)) return null;
    const num = Number(v);
    return Number.isFinite(num) ? num : null;
  }

  function mean(arr) {
    if (!arr.length) return null;
    let s = 0;
    for (const x of arr) s += x;
    return s / arr.length;
  }

  function median(arr) {
    if (!arr.length) return null;
    const a = [...arr].sort((x, y) => x - y);
    const mid = Math.floor(a.length / 2);
    return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
  }

  function std(arr) {
    if (arr.length < 2) return null;
    const m = mean(arr);
    let s2 = 0;
    for (const x of arr) s2 += (x - m) ** 2;
    return Math.sqrt(s2 / (arr.length - 1));
  }

  function roundMaybe(v, digits = 4) {
    if (v === null || v === undefined) return "—";
    if (typeof v !== "number" || !Number.isFinite(v)) return "—";
    const factor = 10 ** digits;
    return String(Math.round(v * factor) / factor);
  }

  function numericStats(rows, numericCols) {
    return numericCols.map((c) => {
      const vals = rows.map((r) => toNumberOrNull(r[c])).filter((v) => v !== null);
      return { feature: c, n: vals.length, mean: mean(vals), median: median(vals), std: std(vals) };
    });
  }

  function renderNumericStatsTable(tableEl, stats) {
    const thead = tableEl.querySelector("thead");
    const tbody = tableEl.querySelector("tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    const header = ["Feature", "N", "Mean", "Median", "Std"];
    const trh = document.createElement("tr");
    header.forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      trh.appendChild(th);
    });
    thead.appendChild(trh);

    for (const s of stats) {
      const tr = document.createElement("tr");
      [s.feature, String(s.n), roundMaybe(s.mean), roundMaybe(s.median), roundMaybe(s.std)].forEach((c) => {
        const td = document.createElement("td");
        td.textContent = c;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
  }

  function categoricalCounts(rows, catCols, topK = 10) {
    const result = [];
    for (const c of catCols) {
      const map = new Map();
      let nonMissing = 0;
      for (const r of rows) {
        const v = r[c];
        if (isMissing(v)) continue;
        nonMissing++;
        const key = String(v);
        map.set(key, (map.get(key) || 0) + 1);
      }
      const counts = Array.from(map.entries())
        .map(([value, n]) => ({ value, n, pct: nonMissing ? (n / nonMissing) * 100 : 0 }))
        .sort((a, b) => b.n - a.n)
        .slice(0, topK);
      result.push({ feature: c, total: nonMissing, counts });
    }
    return result;
  }

  function renderCategoricalCountsTable(tableEl, catSummary) {
    const thead = tableEl.querySelector("thead");
    const tbody = tableEl.querySelector("tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    const header = ["Feature", "Value", "N", "% (non-missing)"];
    const trh = document.createElement("tr");
    header.forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      trh.appendChild(th);
    });
    thead.appendChild(trh);

    for (const f of catSummary) {
      if (!f.counts.length) {
        const tr = document.createElement("tr");
        [f.feature, "—", "0", "—"].forEach((x) => {
          const td = document.createElement("td");
          td.textContent = x;
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
        continue;
      }
      for (let i = 0; i < f.counts.length; i++) {
        const row = f.counts[i];
        const tr = document.createElement("tr");
        const cells = [i === 0 ? f.feature : "", row.value, String(row.n), `${row.pct.toFixed(2)}%`];
        cells.forEach((c) => {
          const td = document.createElement("td");
          td.textContent = c;
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      }
    }
  }

  function splitTrain(rows) {
    return rows.filter((r) => r.source === "train");
  }

  function groupBySurvived(trainRows) {
    const g0 = [];
    const g1 = [];
    for (const r of trainRows) {
      const y = r[SCHEMA.target];
      if (y === 0) g0.push(r);
      else if (y === 1) g1.push(r);
    }
    return { 0: g0, 1: g1 };
  }

  function renderGroupedNumericTable(tableEl, stats0, stats1) {
    const thead = tableEl.querySelector("thead");
    const tbody = tableEl.querySelector("tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    const header = ["Feature", "Mean(0)", "Median(0)", "Std(0)", "N(0)", "Mean(1)", "Median(1)", "Std(1)", "N(1)"];
    const trh = document.createElement("tr");
    header.forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      trh.appendChild(th);
    });
    thead.appendChild(trh);

    const m0 = new Map(stats0.map((s) => [s.feature, s]));
    const m1 = new Map(stats1.map((s) => [s.feature, s]));
    const features = Array.from(new Set([...m0.keys(), ...m1.keys()]));

    for (const f of features) {
      const a = m0.get(f) || { mean: null, median: null, std: null, n: 0 };
      const b = m1.get(f) || { mean: null, median: null, std: null, n: 0 };
      const tr = document.createElement("tr");
      [
        f,
        roundMaybe(a.mean), roundMaybe(a.median), roundMaybe(a.std), String(a.n),
        roundMaybe(b.mean), roundMaybe(b.median), roundMaybe(b.std), String(b.n),
      ].forEach((c) => {
        const td = document.createElement("td");
        td.textContent = c;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
  }

  function renderGroupedCategoricalTable(tableEl, cat0, cat1) {
    const thead = tableEl.querySelector("thead");
    const tbody = tableEl.querySelector("tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    const header = ["Feature", "Value", "N(0)", "%(0)", "N(1)", "%(1)"];
    const trh = document.createElement("tr");
    header.forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      trh.appendChild(th);
    });
    thead.appendChild(trh);

    const toMap = (arr) => {
      const m = new Map();
      for (const f of arr) {
        const vm = new Map();
        for (const c of f.counts) vm.set(c.value, { n: c.n, pct: c.pct });
        m.set(f.feature, vm);
      }
      return m;
    };

    const m0 = toMap(cat0);
    const m1 = toMap(cat1);
    const features = Array.from(new Set([...m0.keys(), ...m1.keys()]));

    for (const feature of features) {
      const vm0 = m0.get(feature) || new Map();
      const vm1 = m1.get(feature) || new Map();
      const values = Array.from(new Set([...vm0.keys(), ...vm1.keys()])).sort();

      if (!values.length) {
        const tr = document.createElement("tr");
        [feature, "—", "0", "—", "0", "—"].forEach((x) => {
          const td = document.createElement("td");
          td.textContent = x;
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
        continue;
      }

      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        const a = vm0.get(v) || { n: 0, pct: 0 };
        const b = vm1.get(v) || { n: 0, pct: 0 };
        const tr = document.createElement("tr");
        const cells = [
          i === 0 ? feature : "",
          v,
          String(a.n),
          a.n ? `${a.pct.toFixed(2)}%` : "—",
          String(b.n),
          b.n ? `${b.pct.toFixed(2)}%` : "—",
        ];
        cells.forEach((c) => {
          const td = document.createElement("td");
          td.textContent = c;
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      }
    }
  }

  // -----------------------------
  // Visualizations
  // -----------------------------
  function computeSurvivalRateByCategory(trainRows, feature) {
    const map = new Map(); // value -> {n, survived}
    for (const r of trainRows) {
      const v = r[feature];
      const y = r[SCHEMA.target];
      if (isMissing(v)) continue;
      if (y !== 0 && y !== 1) continue;

      const key = String(v);
      if (!map.has(key)) map.set(key, { n: 0, survived: 0 });
      const obj = map.get(key);
      obj.n += 1;
      obj.survived += (y === 1 ? 1 : 0);
    }
    const out = Array.from(map.entries()).map(([value, obj]) => ({
      value,
      n: obj.n,
      rate: obj.n ? (obj.survived / obj.n) * 100 : 0,
    }));

    out.sort((a, b) => {
      const an = Number(a.value), bn = Number(b.value);
      if (Number.isFinite(an) && Number.isFinite(bn)) return an - bn;
      return a.value.localeCompare(b.value);
    });

    return out;
  }

  function renderRateBarChart(canvasEl, existingChartRef, data) {
    const labels = data.map((d) => `${d.value}`);
    const rates = data.map((d) => Number(d.rate.toFixed(2)));
    const counts = data.map((d) => d.n);

    destroyChart(existingChartRef);

    return new Chart(canvasEl, {
      type: "bar",
      data: { labels, datasets: [{ label: "Survival rate (%)", data: rates }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true, max: 100, ticks: { callback: (v) => `${v}%` } } },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => `${ctx.raw}% survival (n=${counts[ctx.dataIndex]})` } },
        },
      },
    });
  }

  function histogram(rows, col, bins = 20) {
    const vals = rows.map((r) => toNumberOrNull(r[col])).filter((v) => v !== null);
    if (!vals.length) return { labels: [], counts: [] };

    const min = Math.min(...vals);
    const max = Math.max(...vals);
    if (min === max) return { labels: [`${min}`], counts: [vals.length] };

    const width = (max - min) / bins;
    const counts = new Array(bins).fill(0);

    for (const v of vals) {
      let idx = Math.floor((v - min) / width);
      if (idx >= bins) idx = bins - 1;
      if (idx < 0) idx = 0;
      counts[idx] += 1;
    }

    const labels = counts.map((_, i) => {
      const a = min + i * width;
      const b = a + width;
      return `${a.toFixed(1)}–${b.toFixed(1)}`;
    });

    return { labels, counts };
  }

  function renderHistogram(canvasEl, existingChartRef, hist, title) {
    destroyChart(existingChartRef);

    return new Chart(canvasEl, {
      type: "bar",
      data: { labels: hist.labels, datasets: [{ label: title, data: hist.counts }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { y: { beginAtZero: true } },
        plugins: { legend: { display: false } },
      },
    });
  }

  // Correlation (Pearson) on numeric features (train only)
  function pearson(x, y) {
    const n = x.length;
    if (n < 2) return null;
    const mx = mean(x);
    const my = mean(y);
    let num = 0, dx2 = 0, dy2 = 0;
    for (let i = 0; i < n; i++) {
      const dx = x[i] - mx;
      const dy = y[i] - my;
      num += dx * dy;
      dx2 += dx * dx;
      dy2 += dy * dy;
    }
    const den = Math.sqrt(dx2 * dy2);
    return den ? num / den : null;
  }

  function correlationMatrix(trainRows, numericCols) {
    const matrix = [];
    for (let i = 0; i < numericCols.length; i++) {
      const row = [];
      for (let j = 0; j < numericCols.length; j++) {
        const a = numericCols[i];
        const b = numericCols[j];
        const xs = [];
        const ys = [];
        for (const r of trainRows) {
          const va = toNumberOrNull(r[a]);
          const vb = toNumberOrNull(r[b]);
          if (va === null || vb === null) continue;
          xs.push(va);
          ys.push(vb);
        }
        row.push(pearson(xs, ys));
      }
      matrix.push(row);
    }
    return matrix;
  }

  function corrColor(c) {
    if (c === null || c === undefined || !Number.isFinite(c)) return "rgba(255,255,255,0.10)";
    const v = Math.max(-1, Math.min(1, c));
    const hue = v >= 0 ? 220 : 0; // blue positive, red negative
    const alpha = 0.15 + 0.65 * Math.abs(v);
    return `hsla(${hue}, 90%, 60%, ${alpha})`;
  }

  function renderCorrelationBubbleMatrix(trainRows) {
    const colsPresent = new Set(getColumns(trainRows));
    const numericCols = [SCHEMA.target, "Pclass", "Age", "SibSp", "Parch", "Fare"].filter((c) => colsPresent.has(c));
    if (numericCols.length < 2) {
      state.charts.corr = destroyChart(state.charts.corr);
      return;
    }

    const M = correlationMatrix(trainRows, numericCols);
    const points = [];
    for (let i = 0; i < numericCols.length; i++) {
      for (let j = 0; j < numericCols.length; j++) {
        const c = M[i][j];
        points.push({
          x: j,
          y: i,
          r: c === null ? 0 : 3 + 14 * Math.abs(c),
          _corr: c,
          backgroundColor: corrColor(c),
          borderColor: "rgba(255,255,255,0.10)",
          borderWidth: 1,
        });
      }
    }

    state.charts.corr = destroyChart(state.charts.corr);

    state.charts.corr = new Chart(el.corrHeatmap, {
      type: "bubble",
      data: {
        datasets: [{
          data: points,
          parsing: false,
          backgroundColor: (ctx) => ctx.raw.backgroundColor,
          borderColor: (ctx) => ctx.raw.borderColor,
          borderWidth: (ctx) => ctx.raw.borderWidth,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            min: -0.5,
            max: numericCols.length - 0.5,
            ticks: { callback: (v) => numericCols[v] ?? "", autoSkip: false, maxRotation: 0 },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
          y: {
            min: -0.5,
            max: numericCols.length - 0.5,
            ticks: { callback: (v) => numericCols[v] ?? "", autoSkip: false },
            grid: { color: "rgba(255,255,255,0.06)" },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => {
                const p = items[0].raw;
                return `${numericCols[p.y]} vs ${numericCols[p.x]}`;
              },
              label: (item) => {
                const c = item.raw._corr;
                return `corr = ${c === null ? "—" : c.toFixed(4)}`;
              },
            },
          },
        },
      },
    });
  }

  // -----------------------------
  // Export
  // -----------------------------
  function downloadBlob(filename, blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function exportMergedCSV() {
    if (!state.merged || !state.merged.length) {
      alertUser("No merged dataset available. Load & Merge first.");
      return;
    }
    try {
      const csv = Papa.unparse(state.merged, { quotes: true });
      downloadBlob("titanic_merged.csv", new Blob([csv], { type: "text/csv;charset=utf-8" }));
      setExportStatus("merged CSV downloaded");
    } catch (e) {
      console.error(e);
      alertUser("Export failed (CSV). Check console for details.");
      setExportStatus("CSV export failed");
    }
  }

  function buildSummary(rows) {
    // Lightweight summary for JSON export
    const cols = getColumns(rows);
    const split = computeSplit(rows);
    const missing = missingness(rows);

    const numericFeatures = ["Pclass", "Age", "SibSp", "Parch", "Fare"].filter((c) => cols.includes(c));
    const categoricalFeatures = ["Sex", "Embarked"].filter((c) => cols.includes(c));

    const overallNumeric = numericStats(rows, numericFeatures);
    const overallCat = categoricalCounts(rows, categoricalFeatures);

    const trainRows = splitTrain(rows);
    const grouped = groupBySurvived(trainRows);

    const trainNumeric0 = numericStats(grouped[0], numericFeatures);
    const trainNumeric1 = numericStats(grouped[1], numericFeatures);

    const trainCat0 = categoricalCounts(grouped[0], categoricalFeatures);
    const trainCat1 = categoricalCounts(grouped[1], categoricalFeatures);

    return {
      schema: SCHEMA,
      shape: { rows: rows.length, cols: cols.length },
      split,
      columns: cols,
      missing,
      stats: {
        overall: { numeric: overallNumeric, categorical: overallCat },
        trainBySurvived: {
          "0": { numeric: trainNumeric0, categorical: trainCat0 },
          "1": { numeric: trainNumeric1, categorical: trainCat1 },
        },
      },
    };
  }

  function exportSummaryJSON() {
    if (!state.summary) {
      alertUser("No EDA summary available. Run EDA first.");
      return;
    }
    try {
      const json = JSON.stringify(state.summary, null, 2);
      downloadBlob("titanic_eda_summary.json", new Blob([json], { type: "application/json;charset=utf-8" }));
      setExportStatus("JSON summary downloaded");
    } catch (e) {
      console.error(e);
      alertUser("Export failed (JSON). Check console for details.");
      setExportStatus("JSON export failed");
    }
  }

  // -----------------------------
  // Pipeline: Load & Merge, then Run EDA
  // -----------------------------
  async function loadAndMerge() {
    const trainFile = el.trainFile.files?.[0];
    const testFile = el.testFile.files?.[0];

    if (!trainFile || !testFile) {
      alertUser("Please upload both train.csv and test.csv.");
      return;
    }

    setStatus("parsing CSV files…");
    setExportStatus("idle");

    try {
      const [trainRows, testRows] = await Promise.all([
        parseCsvFile(trainFile),
        parseCsvFile(testFile),
      ]);

      if (!trainRows?.length || !testRows?.length) throw new Error("One of the CSV files looks empty.");

      state.train = trainRows;
      state.test = testRows;
      state.merged = mergeDatasets(trainRows, testRows);

      renderKPIs(state.merged);
      renderPreviewTable(state.merged);

      setStatus("loaded & merged ✅");
    } catch (err) {
      console.error(err);
      alertUser(`Load failed: ${err.message || err}`);
      setStatus("load failed");
    }
  }

  function runEDA() {
    if (!state.merged || !state.merged.length) {
      alertUser("No merged dataset available. Load & Merge first.");
      return;
    }

    try {
      setStatus("running EDA…");

      // Missing values chart
      renderMissingChart(state.merged);

      // Stats tables
      const cols = getColumns(state.merged);
      const numericFeatures = ["Pclass", "Age", "SibSp", "Parch", "Fare"].filter((c) => cols.includes(c));
      const categoricalFeatures = ["Sex", "Embarked"].filter((c) => cols.includes(c));

      renderNumericStatsTable(el.numericStatsTable, numericStats(state.merged, numericFeatures));
      renderCategoricalCountsTable(el.catCountsTable, categoricalCounts(state.merged, categoricalFeatures));

      const trainRows = splitTrain(state.merged);
      const grouped = groupBySurvived(trainRows);

      renderGroupedNumericTable(
        el.numericStatsBySurvivedTable,
        numericStats(grouped[0], numericFeatures),
        numericStats(grouped[1], numericFeatures),
      );

      renderGroupedCategoricalTable(
        el.catCountsBySurvivedTable,
        categoricalCounts(grouped[0], categoricalFeatures),
        categoricalCounts(grouped[1], categoricalFeatures),
      );

      // Visuals
      state.charts.sex = renderRateBarChart(el.sexBar, state.charts.sex, computeSurvivalRateByCategory(trainRows, "Sex"));
      state.charts.pclass = renderRateBarChart(el.pclassBar, state.charts.pclass, computeSurvivalRateByCategory(trainRows, "Pclass"));
      state.charts.embarked = renderRateBarChart(el.embarkedBar, state.charts.embarked, computeSurvivalRateByCategory(trainRows, "Embarked"));

      state.charts.ageHist = renderHistogram(el.ageHist, state.charts.ageHist, histogram(state.merged, "Age", 20), "Age");
      state.charts.fareHist = renderHistogram(el.fareHist, state.charts.fareHist, histogram(state.merged, "Fare", 24), "Fare");

      renderCorrelationBubbleMatrix(trainRows);

      // Summary for JSON export
      state.summary = buildSummary(state.merged);

      setStatus("EDA complete ✅");
      setExportStatus("ready");
    } catch (err) {
      console.error(err);
      alertUser(`EDA failed: ${err.message || err}`);
      setStatus("EDA failed");
    }
  }

  // -----------------------------
  // Reset
  // -----------------------------
  function resetAll() {
    state.train = null;
    state.test = null;
    state.merged = null;
    state.summary = null;

    state.charts.missing = destroyChart(state.charts.missing);
    state.charts.sex = destroyChart(state.charts.sex);
    state.charts.pclass = destroyChart(state.charts.pclass);
    state.charts.embarked = destroyChart(state.charts.embarked);
    state.charts.ageHist = destroyChart(state.charts.ageHist);
    state.charts.fareHist = destroyChart(state.charts.fareHist);
    state.charts.corr = destroyChart(state.charts.corr);

    el.trainFile.value = "";
    el.testFile.value = "";

    el.kpiRows.textContent = "—";
    el.kpiCols.textContent = "—";
    el.kpiSplit.textContent = "—";

    el.missingNote.textContent = "—";

    for (const t of [
      el.previewTable,
      el.numericStatsTable,
      el.numericStatsBySurvivedTable,
      el.catCountsTable,
      el.catCountsBySurvivedTable,
    ]) {
      t.querySelector("thead").innerHTML = "";
      t.querySelector("tbody").innerHTML = "";
    }

    setStatus("waiting for files");
    setExportStatus("idle");
  }

  // -----------------------------
  // Events
  // -----------------------------
  function init() {
    el.btnLoad.addEventListener("click", loadAndMerge);
    el.btnRun.addEventListener("click", runEDA);
    el.btnReset.addEventListener("click", resetAll);

    el.previewRows.addEventListener("change", () => {
      if (state.merged?.length) renderPreviewTable(state.merged);
    });

    el.btnExportCSV.addEventListener("click", exportMergedCSV);
    el.btnExportJSON.addEventListener("click", exportSummaryJSON);

    // Simple guidance status when files are selected
    const maybeStatus = () => {
      const ok = !!el.trainFile.files?.[0] && !!el.testFile.files?.[0];
      setStatus(ok ? "files selected — click Load & Merge" : "waiting for files");
    };
    el.trainFile.addEventListener("change", maybeStatus);
    el.testFile.addEventListener("change", maybeStatus);

    resetAll();
  }

  init();
})();
