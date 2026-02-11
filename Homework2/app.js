// app.js
// Titanic TF.js — Browser-only Shallow Classifier
// Runs fully client-side: fetch CSVs or use file inputs, preprocess, train, evaluate ROC/AUC with threshold slider,
// predict for test set, export submissions and artifacts.
//
// IMPORTANT (CSV parsing):
// Kaggle Titanic CSV has quoted fields containing commas, e.g. "Braund, Mr. Owen Harris".
// This parser handles quotes properly (no external libraries needed).

/* -----------------------------
   Repo auto-load defaults
------------------------------ */
const DEFAULT_TRAIN_URL = "https://waleedamawi12.github.io/NNDL-/Homework1/train.csv";
const DEFAULT_TEST_URL  = "https://waleedamawi12.github.io/NNDL-/Homework1/test.csv";

/* -----------------------------
   Schema (swap point)
   Target: Survived (0/1)
   Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
   Identifier (exclude): PassengerId
------------------------------ */
const SCHEMA = {
  id: "PassengerId",
  target: "Survived",
  features: ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
  categorical: ["Pclass", "Sex", "Embarked"],
  numeric: ["Age", "SibSp", "Parch", "Fare"],
};

/* -----------------------------
   UI helpers
------------------------------ */
const $ = (id) => document.getElementById(id);

const ui = {
  trainFile: $("trainFile"),
  testFile: $("testFile"),
  btnLoad: $("btnLoad"),
  btnPreprocess: $("btnPreprocess"),
  btnBuild: $("btnBuild"),
  btnTrain: $("btnTrain"),
  btnEval: $("btnEval"),
  btnPredict: $("btnPredict"),
  btnExportPre: $("btnExportPre"),
  btnSaveModel: $("btnSaveModel"),

  loadStatus: $("loadStatus"),
  trainShape: $("trainShape"),
  testShape: $("testShape"),
  trainMissing: $("trainMissing"),
  testMissing: $("testMissing"),

  trainTable: $("trainTable"),
  testTable: $("testTable"),

  edaSex: $("edaSex"),
  edaPclass: $("edaPclass"),

  optFamily: $("optFamily"),
  optAlone: $("optAlone"),

  featSummary: $("featSummary"),
  modelSummary: $("modelSummary"),

  fitVis: $("fitVis"),

  rocVis: $("rocVis"),
  cmWrap: $("cmWrap"),
  thSlider: $("thSlider"),
  thText: $("thText"),
  aucText: $("aucText"),
  metricText: $("metricText"),

  predStatus: $("predStatus"),
};

function setPill(el, label, value) {
  // el is a span.pill that contains <strong>Label</strong> <span>value</span> or similar
  el.innerHTML = `<strong>${escapeHtml(label)}</strong> <span class="muted">${escapeHtml(value)}</span>`;
}

function alertErr(msg, err) {
  console.error(msg, err);
  alert(`${msg}${err ? `\n\n${err.message || err}` : ""}`);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

/* -----------------------------
   CSV parsing (handles quotes + commas)
   - No external libs
   - Works for Kaggle Titanic format
------------------------------ */
function parseCSV(text) {
  // Returns array of rows (arrays), robust to quoted commas/newlines.
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const next = text[i + 1];

    if (inQuotes) {
      if (c === '"' && next === '"') {
        // escaped quote
        field += '"';
        i++;
      } else if (c === '"') {
        inQuotes = false;
      } else {
        field += c;
      }
    } else {
      if (c === '"') {
        inQuotes = true;
      } else if (c === ",") {
        row.push(field);
        field = "";
      } else if (c === "\n") {
        row.push(field);
        field = "";
        // Handle CRLF: remove trailing \r if present
        if (row.length === 1 && row[0] === "" && rows.length === 0) continue; // skip empty first line
        rows.push(row.map(v => v.endsWith("\r") ? v.slice(0, -1) : v));
        row = [];
      } else {
        field += c;
      }
    }
  }
  // last field
  row.push(field);
  rows.push(row.map(v => v.endsWith("\r") ? v.slice(0, -1) : v));
  // Remove possible trailing empty row
  while (rows.length && rows[rows.length - 1].every(v => v === "")) rows.pop();
  return rows;
}

function rowsToObjects(rows) {
  // rows: [ [header...], [val...], ...]
  const header = rows[0];
  const out = [];
  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    if (r.length === 1 && r[0] === "") continue;
    const obj = {};
    for (let j = 0; j < header.length; j++) {
      obj[header[j]] = r[j] ?? "";
    }
    out.push(obj);
  }
  return out;
}

async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result || ""));
    fr.onerror = () => reject(fr.error || new Error("File read failed"));
    fr.readAsText(file);
  });
}

async function loadCSV({ url, file }) {
  const text = file
    ? await readFileAsText(file)
    : await (await fetch(url, { cache: "no-store" })).text();

  const rows = parseCSV(text);
  if (!rows || rows.length < 2) throw new Error("CSV parse produced no rows");
  const objs = rowsToObjects(rows);
  return { rows, data: objs, header: rows[0] };
}

/* -----------------------------
   Data inspection utilities
------------------------------ */
function shapeOf(data) {
  const rows = data.length;
  const cols = rows ? Object.keys(data[0]).length : 0;
  return { rows, cols };
}

function missingnessTop(data, topK = 5) {
  if (!data.length) return [];
  const cols = Object.keys(data[0]);
  const counts = Object.fromEntries(cols.map(c => [c, 0]));
  const n = data.length;

  for (const r of data) {
    for (const c of cols) {
      const v = r[c];
      if (v === undefined || v === null || String(v).trim() === "") counts[c] += 1;
    }
  }
  const arr = cols.map(c => ({ col: c, pct: (counts[c] / n) * 100 }))
    .sort((a,b) => b.pct - a.pct)
    .slice(0, topK);
  return arr;
}

function renderTable(tableEl, data, maxRows = 10) {
  const thead = tableEl.querySelector("thead");
  const tbody = tableEl.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";
  if (!data.length) return;

  const cols = Object.keys(data[0]);

  const trh = document.createElement("tr");
  for (const c of cols) {
    const th = document.createElement("th");
    th.textContent = c;
    trh.appendChild(th);
  }
  thead.appendChild(trh);

  for (let i = 0; i < Math.min(maxRows, data.length); i++) {
    const tr = document.createElement("tr");
    for (const c of cols) {
      const td = document.createElement("td");
      const v = data[i][c];
      td.textContent = v === "" ? "NULL" : String(v);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

/* -----------------------------
   Quick EDA (tfjs-vis bars)
------------------------------ */
function safeNumber(x) {
  const v = Number(x);
  return Number.isFinite(v) ? v : NaN;
}

function groupSurvivalRateBy(data, groupCol) {
  // Only works on train with target present.
  const map = new Map(); // key -> {surv, total}
  for (const r of data) {
    const key = String(r[groupCol] ?? "Unknown");
    const y = safeNumber(r[SCHEMA.target]);
    if (!Number.isFinite(y)) continue;
    const cur = map.get(key) || { surv: 0, total: 0 };
    cur.total += 1;
    cur.surv += (y >= 0.5) ? 1 : 0;
    map.set(key, cur);
  }
  const labels = Array.from(map.keys());
  const values = labels.map(k => (map.get(k).surv / Math.max(1, map.get(k).total)));
  return { labels, values };
}

function renderEDABars(container, title, labels, values) {
  container.innerHTML = "";
  const surface = tfvis.visor().surface({ name: title, tab: "EDA" });
  // Put it into our div instead of visor? tfjs-vis is visor-centric.
  // We'll still show inside tfjs-vis visor, and also show a simple list in the div.
  const ul = document.createElement("div");
  ul.className = "muted";
  ul.style.fontSize = "12px";
  ul.innerHTML = labels.map((l,i)=>`${escapeHtml(l)}: <span class="mono">${(values[i]*100).toFixed(1)}%</span>`).join("<br/>");
  container.appendChild(ul);

  tfvis.render.barchart(
    surface,
    labels.map((l,i)=>({ index: l, value: values[i] })),
    { xLabel: groupLabel(title), yLabel: "Survival rate", height: 260 }
  );
}
function groupLabel(title){ return title.replace("Survival by ", ""); }

/* -----------------------------
   Preprocessing
------------------------------ */
const state = {
  rawTrain: null,
  rawTest: null,

  // Preprocess artifacts:
  artifacts: null,  // { ageMedian, embarkedMode, numMeanStd, oneHotMaps, featureOrder }
  XTrain: null,
  yTrain: null,
  XVal: null,
  yVal: null,

  XTest: null,
  testPassengerIds: null,

  model: null,

  // Eval cached:
  valProbs: null,  // Float32Array
  valLabels: null, // Int32Array
  roc: null,       // {fpr, tpr, auc}
};

function mode(values) {
  const m = new Map();
  for (const v of values) {
    if (v === null || v === undefined) continue;
    const s = String(v).trim();
    if (s === "") continue;
    m.set(s, (m.get(s) || 0) + 1);
  }
  let best = null, bestN = -1;
  for (const [k, n] of m.entries()) {
    if (n > bestN) { best = k; bestN = n; }
  }
  return best ?? "";
}

function median(nums) {
  const a = nums.filter(n => Number.isFinite(n)).slice().sort((x,y)=>x-y);
  if (!a.length) return NaN;
  const mid = Math.floor(a.length / 2);
  return (a.length % 2 === 0) ? (a[mid-1] + a[mid]) / 2 : a[mid];
}

function meanStd(nums) {
  const a = nums.filter(n => Number.isFinite(n));
  const n = a.length || 1;
  const mean = a.reduce((s,v)=>s+v,0) / n;
  const varr = a.reduce((s,v)=>s + (v-mean)*(v-mean), 0) / n;
  const std = Math.sqrt(varr) || 1;
  return { mean, std };
}

function buildOneHotMap(values) {
  // stable sorted categories
  const cats = Array.from(new Set(values.map(v => String(v ?? "").trim()))).filter(v => v !== "");
  cats.sort();
  const map = new Map();
  cats.forEach((c, i) => map.set(c, i));
  return { cats, map };
}

function oneHot(value, oh) {
  const out = new Array(oh.cats.length).fill(0);
  const key = String(value ?? "").trim();
  const idx = oh.map.get(key);
  if (idx !== undefined) out[idx] = 1;
  // Unknown category -> all zeros
  return out;
}

function preprocess({ addFamily, addAlone }) {
  if (!state.rawTrain || !state.rawTest) throw new Error("Load train/test first.");

  // Merge-like behavior for consistent one-hot categories: use both train+test.
  const train = state.rawTrain;
  const test = state.rawTest;

  // --- Imputation values from TRAIN only (typical ML hygiene)
  const agesTrain = train.map(r => safeNumber(r.Age));
  const ageMedian = median(agesTrain);

  const embarkedMode = mode(train.map(r => r.Embarked));

  // Fill and derive features (FamilySize / IsAlone) on both train/test
  function normalizeRow(r) {
    const out = { ...r };

    // numeric
    let Age = safeNumber(out.Age);
    if (!Number.isFinite(Age)) Age = ageMedian;

    let Fare = safeNumber(out.Fare);
    if (!Number.isFinite(Fare)) Fare = NaN; // will impute via mean later if needed

    let SibSp = safeNumber(out.SibSp);
    let Parch = safeNumber(out.Parch);
    let Pclass = String(out.Pclass ?? "").trim();
    let Sex = String(out.Sex ?? "").trim();
    let Embarked = String(out.Embarked ?? "").trim();
    if (!Embarked) Embarked = embarkedMode;

    const base = {
      Pclass,
      Sex,
      Age,
      SibSp: Number.isFinite(SibSp) ? SibSp : 0,
      Parch: Number.isFinite(Parch) ? Parch : 0,
      Fare, // may be NaN
      Embarked,
    };

    // optional engineered features
    if (addFamily || addAlone) {
      const fam = base.SibSp + base.Parch + 1;
      if (addFamily) base.FamilySize = fam;
      if (addAlone) base.IsAlone = (fam === 1) ? 1 : 0;
    }

    return base;
  }

  const trainN = train.map(normalizeRow);
  const testN = test.map(normalizeRow);

  // Fare impute: use TRAIN mean
  const fareTrain = trainN.map(r => safeNumber(r.Fare));
  const fareMeanStd = meanStd(fareTrain);
  for (const r of trainN) if (!Number.isFinite(r.Fare)) r.Fare = fareMeanStd.mean;
  for (const r of testN)  if (!Number.isFinite(r.Fare)) r.Fare = fareMeanStd.mean;

  // Standardize Age/Fare (and engineered numeric if present?) Spec says standardize Age/Fare only.
  const ageMS = meanStd(trainN.map(r => r.Age));
  const fareMS = meanStd(trainN.map(r => r.Fare));

  for (const r of trainN) {
    r.Age_z = (r.Age - ageMS.mean) / ageMS.std;
    r.Fare_z = (r.Fare - fareMS.mean) / fareMS.std;
  }
  for (const r of testN) {
    r.Age_z = (r.Age - ageMS.mean) / ageMS.std;
    r.Fare_z = (r.Fare - fareMS.mean) / fareMS.std;
  }

  // Build one-hot maps using BOTH train+test for stability (avoids missing categories at inference)
  const all = trainN.concat(testN);

  const ohPclass = buildOneHotMap(all.map(r => r.Pclass));
  const ohSex = buildOneHotMap(all.map(r => r.Sex));
  const ohEmb = buildOneHotMap(all.map(r => r.Embarked));

  // Feature order:
  // One-hot: Pclass, Sex, Embarked
  // Numeric: Age_z, SibSp, Parch, Fare_z
  // Optional engineered: FamilySize, IsAlone (as numeric 0/1)
  const featureOrder = [];
  const oneHotMaps = { Pclass: ohPclass, Sex: ohSex, Embarked: ohEmb };

  // Expand one-hot names for export/inspection
  for (const c of ohPclass.cats) featureOrder.push(`Pclass=${c}`);
  for (const c of ohSex.cats) featureOrder.push(`Sex=${c}`);
  for (const c of ohEmb.cats) featureOrder.push(`Embarked=${c}`);

  featureOrder.push("Age_z", "SibSp", "Parch", "Fare_z");
  if (addFamily) featureOrder.push("FamilySize");
  if (addAlone) featureOrder.push("IsAlone");

  function featurizeRow(r) {
    const feats = [];
    feats.push(...oneHot(r.Pclass, ohPclass));
    feats.push(...oneHot(r.Sex, ohSex));
    feats.push(...oneHot(r.Embarked, ohEmb));
    feats.push(r.Age_z, r.SibSp, r.Parch, r.Fare_z);
    if (addFamily) feats.push(r.FamilySize);
    if (addAlone) feats.push(r.IsAlone);
    return feats.map(v => Number.isFinite(v) ? v : 0);
  }

  // Build train X/y
  const X = [];
  const y = [];
  for (let i = 0; i < train.length; i++) {
    const row = train[i];
    const yn = safeNumber(row[SCHEMA.target]);
    if (!Number.isFinite(yn)) continue;
    X.push(featurizeRow(trainN[i]));
    y.push(yn >= 0.5 ? 1 : 0);
  }

  // Build test X and PassengerId
  const Xtest = [];
  const pids = [];
  for (let i = 0; i < test.length; i++) {
    const pid = test[i][SCHEMA.id];
    pids.push(pid);
    Xtest.push(featurizeRow(testN[i]));
  }

  // Stratified split 80/20
  const idx0 = [];
  const idx1 = [];
  for (let i = 0; i < y.length; i++) (y[i] === 1 ? idx1 : idx0).push(i);

  function shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
  }
  shuffle(idx0); shuffle(idx1);

  const valFrac = 0.2;
  const n0v = Math.max(1, Math.floor(idx0.length * valFrac));
  const n1v = Math.max(1, Math.floor(idx1.length * valFrac));
  const valIdx = idx0.slice(0, n0v).concat(idx1.slice(0, n1v));
  const trainIdx = idx0.slice(n0v).concat(idx1.slice(n1v));
  shuffle(valIdx); shuffle(trainIdx);

  const XTrain = trainIdx.map(i => X[i]);
  const yTrain = trainIdx.map(i => y[i]);
  const XVal = valIdx.map(i => X[i]);
  const yVal = valIdx.map(i => y[i]);

  // Convert to tensors
  const XTrainT = tf.tensor2d(XTrain);
  const yTrainT = tf.tensor2d(yTrain, [yTrain.length, 1]);
  const XValT = tf.tensor2d(XVal);
  const yValT = tf.tensor2d(yVal, [yVal.length, 1]);

  const XTestT = tf.tensor2d(Xtest);

  const artifacts = {
    ageMedian,
    embarkedMode,
    ageMeanStd: ageMS,
    fareMeanStd: fareMS,
    oneHotCategories: {
      Pclass: ohPclass.cats,
      Sex: ohSex.cats,
      Embarked: ohEmb.cats,
    },
    featureOrder,
    addFamily,
    addAlone,
    schema: SCHEMA,
  };

  // Cleanup prior tensors
  disposeStateTensors();

  state.artifacts = artifacts;
  state.XTrain = XTrainT;
  state.yTrain = yTrainT;
  state.XVal = XValT;
  state.yVal = yValT;
  state.XTest = XTestT;
  state.testPassengerIds = pids;

  // Reset eval cache
  state.valProbs = null;
  state.valLabels = null;
  state.roc = null;

  return {
    nTrain: XTrain.length,
    nVal: XVal.length,
    nFeatures: XTrain[0]?.length || 0,
    featureOrder,
  };
}

function disposeStateTensors() {
  // Dispose tensors in state if they exist
  const keys = ["XTrain","yTrain","XVal","yVal","XTest"];
  for (const k of keys) {
    if (state[k]) {
      state[k].dispose();
      state[k] = null;
    }
  }
}

/* -----------------------------
   Model
------------------------------ */
function buildModel(inputDim) {
  if (!Number.isFinite(inputDim) || inputDim <= 0) throw new Error("Invalid inputDim.");
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [inputDim] }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({
    optimizer: tf.train.adam(),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

function modelSummaryText(model) {
  const lines = [];
  lines.push("Model: Dense(16, relu) -> Dense(1, sigmoid)");
  lines.push(`Params: ${model.countParams()}`);
  // Show layer-wise
  model.layers.forEach((l, i) => {
    const cfg = l.getConfig();
    lines.push(`Layer ${i}: ${l.getClassName()} ${cfg.units ? `(units=${cfg.units}, act=${cfg.activation})` : ""}`);
  });
  return lines.join("\n");
}

/* -----------------------------
   Training (with early stopping)
------------------------------ */
async function trainModel() {
  if (!state.model) throw new Error("Build model first.");
  if (!state.XTrain || !state.yTrain || !state.XVal || !state.yVal) throw new Error("Preprocess first.");

  const model = state.model;

  // Clear tfjs-vis tab for fit curves
  tfvis.visor().open();
  const fitContainer = { name: "Training", tab: "Training" };

  let bestVal = Infinity;
  let bestWeights = null;
  let badEpochs = 0;
  const patience = 5;

  const callbacks = tfvis.show.fitCallbacks(
    fitContainer,
    ["loss", "acc", "val_loss", "val_acc"],
    { callbacks: ["onEpochEnd"] }
  );

  const history = await model.fit(state.XTrain, state.yTrain, {
    epochs: 50,
    batchSize: 32,
    validationData: [state.XVal, state.yVal],
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        // tfjs-vis live update
        await callbacks.onEpochEnd(epoch, logs);

        const v = logs?.val_loss;
        if (Number.isFinite(v)) {
          if (v < bestVal - 1e-6) {
            bestVal = v;
            badEpochs = 0;
            // Save best weights (clone)
            if (bestWeights) bestWeights.forEach(w => w.dispose());
            bestWeights = model.getWeights().map(w => w.clone());
          } else {
            badEpochs += 1;
            if (badEpochs >= patience) {
              model.stopTraining = true;
            }
          }
        }
      }
    }
  });

  // Restore best weights if we captured any
  if (bestWeights) {
    model.setWeights(bestWeights);
    bestWeights.forEach(w => w.dispose());
  }

  return history;
}

/* -----------------------------
   Metrics: ROC, AUC, confusion matrix, PRF1
------------------------------ */
function computeRocAuc(labels, probs, nThresholds = 200) {
  // labels: Int32Array 0/1; probs: Float32Array
  // Compute ROC by sweeping thresholds from 0..1.
  const tpr = [];
  const fpr = [];
  const thresholds = [];
  for (let i = 0; i <= nThresholds; i++) thresholds.push(i / nThresholds);

  const P = labels.reduce((s,y)=>s + (y===1?1:0), 0) || 1;
  const N = labels.length - P || 1;

  for (const th of thresholds) {
    let TP=0, FP=0, TN=0, FN=0;
    for (let i = 0; i < labels.length; i++) {
      const y = labels[i];
      const p = probs[i];
      const yhat = (p >= th) ? 1 : 0;
      if (y===1 && yhat===1) TP++;
      else if (y===0 && yhat===1) FP++;
      else if (y===0 && yhat===0) TN++;
      else if (y===1 && yhat===0) FN++;
    }
    const TPR = TP / (TP + FN || 1);
    const FPR = FP / (FP + TN || 1);
    tpr.push(TPR);
    fpr.push(FPR);
  }

  // Sort by FPR increasing to compute AUC by trapezoidal rule
  const pts = fpr.map((x,i)=>({x, y:tpr[i]})).sort((a,b)=>a.x-b.x);
  let auc = 0;
  for (let i = 1; i < pts.length; i++) {
    const x1 = pts[i-1].x, y1 = pts[i-1].y;
    const x2 = pts[i].x,   y2 = pts[i].y;
    auc += (x2 - x1) * (y1 + y2) / 2;
  }
  return { fpr: pts.map(p=>p.x), tpr: pts.map(p=>p.y), auc };
}

function confusionAndScores(labels, probs, th) {
  let TP=0, FP=0, TN=0, FN=0;
  for (let i = 0; i < labels.length; i++) {
    const y = labels[i];
    const p = probs[i];
    const yhat = (p >= th) ? 1 : 0;
    if (y===1 && yhat===1) TP++;
    else if (y===0 && yhat===1) FP++;
    else if (y===0 && yhat===0) TN++;
    else if (y===1 && yhat===0) FN++;
  }
  const precision = TP / (TP + FP || 1);
  const recall = TP / (TP + FN || 1);
  const f1 = (2 * precision * recall) / (precision + recall || 1);
  const acc = (TP + TN) / (TP + TN + FP + FN || 1);
  return { TP, FP, TN, FN, precision, recall, f1, acc };
}

function renderRoc(roc) {
  ui.rocVis.innerHTML = "";
  const surface = tfvis.visor().surface({ name: "ROC Curve", tab: "Metrics" });
  // Also show in our div a minimal text hint
  ui.rocVis.innerHTML = `<div class="muted" style="font-size:12px">
    ROC rendered in tfjs-vis tab <span class="mono">Metrics</span>.
  </div>`;

  const series = roc.fpr.map((x,i)=>({ x, y: roc.tpr[i] }));
  tfvis.render.linechart(
    surface,
    { values: series, series: ["ROC"] },
    { xLabel: "FPR", yLabel: "TPR", height: 280 }
  );
}

function renderConfusion(cm) {
  ui.cmWrap.innerHTML = "";
  const surface = tfvis.visor().surface({ name: "Confusion Matrix", tab: "Metrics" });
  const values = [
    [cm.TN, cm.FP],
    [cm.FN, cm.TP],
  ];
  tfvis.render.confusionMatrix(surface, { values, tickLabels: ["0", "1"] });

  // Also show a compact HTML version inside the page
  ui.cmWrap.innerHTML = `
    <div class="mono" style="font-size:12px; line-height:1.4">
      TN=${cm.TN}  FP=${cm.FP}<br/>
      FN=${cm.FN}  TP=${cm.TP}
    </div>`;
}

async function evaluateModel() {
  if (!state.model) throw new Error("Build/train model first.");
  if (!state.XVal || !state.yVal) throw new Error("Preprocess first.");

  // Predict probabilities on validation set
  const probsT = state.model.predict(state.XVal);
  const probs = await probsT.data();
  probsT.dispose();

  const labelsT = state.yVal;
  const labelsData = await labelsT.data();

  // Ensure typed arrays
  const probsArr = Float32Array.from(probs);
  const labelsArr = Int32Array.from(Array.from(labelsData).map(v => (v >= 0.5 ? 1 : 0)));

  state.valProbs = probsArr;
  state.valLabels = labelsArr;

  const roc = computeRocAuc(labelsArr, probsArr, 250);
  state.roc = roc;

  ui.aucText.textContent = roc.auc.toFixed(4);
  renderRoc(roc);

  // Update CM at current threshold
  const th = Number(ui.thSlider.value);
  updateThresholdUI(th);

  return roc;
}

function updateThresholdUI(th) {
  ui.thText.textContent = th.toFixed(2);
  if (!state.valProbs || !state.valLabels) {
    ui.metricText.textContent = "Run Evaluate to compute metrics.";
    return;
  }
  const cm = confusionAndScores(state.valLabels, state.valProbs, th);
  renderConfusion(cm);
  ui.metricText.textContent =
    `Accuracy: ${(cm.acc*100).toFixed(2)}%\n` +
    `Precision: ${cm.precision.toFixed(4)}\n` +
    `Recall: ${cm.recall.toFixed(4)}\n` +
    `F1: ${cm.f1.toFixed(4)}`;
}

/* -----------------------------
   Prediction & Export
------------------------------ */
function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
}

function toCSV(headers, rows) {
  // Escape fields that contain commas, quotes, or newlines
  const esc = (v) => {
    const s = String(v ?? "");
    if (/[",\n\r]/.test(s)) return `"${s.replaceAll('"','""')}"`;
    return s;
  };
  const lines = [];
  lines.push(headers.map(esc).join(","));
  for (const r of rows) lines.push(r.map(esc).join(","));
  return lines.join("\n");
}

async function predictAndExport() {
  if (!state.model) throw new Error("Build/train model first.");
  if (!state.XTest || !state.testPassengerIds) throw new Error("Preprocess first.");

  ui.predStatus.textContent = "running...";

  const probsT = state.model.predict(state.XTest);
  const probs = await probsT.data();
  probsT.dispose();

  const th = Number(ui.thSlider.value);
  const preds = Array.from(probs).map(p => (p >= th ? 1 : 0));

  // submission.csv
  const subRows = state.testPassengerIds.map((pid, i) => [pid, preds[i]]);
  const subCsv = toCSV([SCHEMA.id, SCHEMA.target], subRows);
  downloadText("submission.csv", subCsv);

  // probabilities.csv
  const probRows = state.testPassengerIds.map((pid, i) => [pid, Number(probs[i]).toFixed(6)]);
  const probCsv = toCSV([SCHEMA.id, "Survived_Prob"], probRows);
  downloadText("probabilities.csv", probCsv);

  ui.predStatus.textContent = "done";
}

async function saveModelDownloads() {
  if (!state.model) throw new Error("No model to save.");
  await state.model.save("downloads://titanic-tfjs");
}

function exportPreprocessJSON() {
  if (!state.artifacts) throw new Error("Run preprocess first.");
  const json = JSON.stringify(state.artifacts, null, 2);
  const blob = new Blob([json], { type: "application/json;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "preprocess_summary.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
}

/* -----------------------------
   Load flow
------------------------------ */
async function loadAllData() {
  try {
    setPill(ui.loadStatus, "Status", "loading...");
    // Determine overrides
    const trainFile = ui.trainFile.files?.[0] || null;
    const testFile = ui.testFile.files?.[0] || null;

    const train = await loadCSV({ url: DEFAULT_TRAIN_URL, file: trainFile });
    const test = await loadCSV({ url: DEFAULT_TEST_URL, file: testFile });

    state.rawTrain = train.data;
    state.rawTest = test.data;

    // Render previews
    const shTr = shapeOf(state.rawTrain);
    const shTe = shapeOf(state.rawTest);
    ui.trainShape.textContent = `${shTr.rows} / ${shTr.cols}`;
    ui.testShape.textContent = `${shTe.rows} / ${shTe.cols}`;

    const missTr = missingnessTop(state.rawTrain, 5).map(x => `${x.col}:${x.pct.toFixed(1)}%`).join(" | ");
    const missTe = missingnessTop(state.rawTest, 5).map(x => `${x.col}:${x.pct.toFixed(1)}%`).join(" | ");
    ui.trainMissing.textContent = missTr || "—";
    ui.testMissing.textContent = missTe || "—";

    renderTable(ui.trainTable, state.rawTrain, 10);
    renderTable(ui.testTable, state.rawTest, 10);

    // EDA charts from train
    const sex = groupSurvivalRateBy(state.rawTrain, "Sex");
    const pcl = groupSurvivalRateBy(state.rawTrain, "Pclass");
    renderEDABars(ui.edaSex, "Survival by Sex", sex.labels, sex.values);
    renderEDABars(ui.edaPclass, "Survival by Pclass", pcl.labels, pcl.values);

    // Enable next steps
    ui.btnPreprocess.disabled = false;
    setPill(ui.loadStatus, "Status", `loaded (train+test)`);
  } catch (err) {
    setPill(ui.loadStatus, "Status", "error");
    alertErr("Failed to load CSVs. Check GitHub Pages URLs or upload files.", err);
  }
}

/* -----------------------------
   Wiring
------------------------------ */
ui.btnLoad.addEventListener("click", async () => {
  await loadAllData();
});

ui.btnPreprocess.addEventListener("click", () => {
  try {
    const addFamily = ui.optFamily.checked;
    const addAlone = ui.optAlone.checked;
    const info = preprocess({ addFamily, addAlone });

    ui.featSummary.textContent =
      `Train: ${info.nTrain} rows | Val: ${info.nVal} rows\n` +
      `Features: ${info.nFeatures}\n` +
      `Order: ${info.featureOrder.join(", ")}`;

    ui.btnBuild.disabled = false;
    ui.btnExportPre.disabled = false;

    // If a model exists, invalidate it (feature dim may have changed)
    if (state.model) {
      state.model.dispose();
      state.model = null;
      ui.modelSummary.textContent = "—";
      ui.btnTrain.disabled = true;
      ui.btnEval.disabled = true;
      ui.btnPredict.disabled = true;
      ui.btnSaveModel.disabled = true;
    }
  } catch (err) {
    alertErr("Preprocess failed.", err);
  }
});

ui.btnBuild.addEventListener("click", () => {
  try {
    if (!state.XTrain) throw new Error("Preprocess first.");
    const inputDim = state.XTrain.shape[1];

    if (state.model) state.model.dispose();
    state.model = buildModel(inputDim);

    ui.modelSummary.textContent = modelSummaryText(state.model);

    ui.btnTrain.disabled = false;
    ui.btnEval.disabled = true;
    ui.btnPredict.disabled = true;
    ui.btnSaveModel.disabled = false;
  } catch (err) {
    alertErr("Model build failed.", err);
  }
});

ui.btnTrain.addEventListener("click", async () => {
  try {
    ui.btnTrain.disabled = true;
    await trainModel();
    ui.btnEval.disabled = false;
    ui.btnPredict.disabled = false;
    ui.btnTrain.disabled = false;
  } catch (err) {
    ui.btnTrain.disabled = false;
    alertErr("Training failed.", err);
  }
});

ui.btnEval.addEventListener("click", async () => {
  try {
    ui.btnEval.disabled = true;
    await evaluateModel();
    ui.btnEval.disabled = false;
  } catch (err) {
    ui.btnEval.disabled = false;
    alertErr("Evaluation failed.", err);
  }
});

ui.thSlider.addEventListener("input", () => {
  const th = Number(ui.thSlider.value);
  updateThresholdUI(th);
});

ui.btnPredict.addEventListener("click", async () => {
  try {
    await predictAndExport();
  } catch (err) {
    ui.predStatus.textContent = "error";
    alertErr("Prediction/export failed.", err);
  }
});

ui.btnExportPre.addEventListener("click", () => {
  try {
    exportPreprocessJSON();
  } catch (err) {
    alertErr("Export preprocess JSON failed.", err);
  }
});

ui.btnSaveModel.addEventListener("click", async () => {
  try {
    await saveModelDownloads();
  } catch (err) {
    alertErr("Model save failed.", err);
  }
});

/* -----------------------------
   Auto-load on page open
------------------------------ */
window.addEventListener("DOMContentLoaded", async () => {
  // Auto-load default CSVs immediately
  await loadAllData();

  // Initialize slider display
  updateThresholdUI(Number(ui.thSlider.value));
});
