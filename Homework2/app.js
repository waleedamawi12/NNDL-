// app.js (ADD Sigmoid Gate feature-importance + gated model option)
//
// NOTE: This file is your existing app.js PLUS the sigmoid gate additions.
// If you already have the earlier app.js, replace it fully with this one to avoid missing functions.

const DEFAULT_TRAIN_URL = "https://waleedamawi12.github.io/NNDL-/Homework1/train.csv";
const DEFAULT_TEST_URL  = "https://waleedamawi12.github.io/NNDL-/Homework1/test.csv";

const SCHEMA = {
  id: "PassengerId",
  target: "Survived",
  features: ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
  categorical: ["Pclass", "Sex", "Embarked"],
  numeric: ["Age", "SibSp", "Parch", "Fare"],
};

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

  thSlider: $("thSlider"),
  thText: $("thText"),
  aucText: $("aucText"),
  metricText: $("metricText"),
  rocVis: $("rocVis"),
  cmWrap: $("cmWrap"),

  predStatus: $("predStatus"),

  // NEW: Sigmoid gate UI
  btnGateAnalyze: $("btnGateAnalyze"),
  fiBody: $("fiBody"),
  fiTop3: $("fiTop3"),
};

function escapeHtml(s) {
  return String(s)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}
function setPill(el, label, value) {
  el.innerHTML = `<strong>${escapeHtml(label)}</strong> <span class="muted">${escapeHtml(value)}</span>`;
}
function alertErr(msg, err) {
  console.error(msg, err);
  alert(`${msg}${err ? `\n\n${err.message || err}` : ""}`);
}

/* -----------------------------
   CSV parsing (handles quoted commas)
------------------------------ */
function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const next = text[i + 1];

    if (inQuotes) {
      if (c === '"' && next === '"') { field += '"'; i++; }
      else if (c === '"') inQuotes = false;
      else field += c;
    } else {
      if (c === '"') inQuotes = true;
      else if (c === ",") { row.push(field); field = ""; }
      else if (c === "\n") {
        row.push(field); field = "";
        rows.push(row.map(v => v.endsWith("\r") ? v.slice(0,-1) : v));
        row = [];
      } else field += c;
    }
  }
  row.push(field);
  rows.push(row.map(v => v.endsWith("\r") ? v.slice(0,-1) : v));
  while (rows.length && rows[rows.length - 1].every(v => v === "")) rows.pop();
  return rows;
}
function rowsToObjects(rows) {
  const header = rows[0];
  const out = [];
  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    if (r.length === 1 && r[0] === "") continue;
    const obj = {};
    for (let j = 0; j < header.length; j++) obj[header[j]] = r[j] ?? "";
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
  return { rows, data: rowsToObjects(rows), header: rows[0] };
}

/* -----------------------------
   Inspection
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
  for (const r of data) for (const c of cols) {
    const v = r[c];
    if (v === undefined || v === null || String(v).trim() === "") counts[c] += 1;
  }
  return cols.map(c => ({ col: c, pct: (counts[c] / n) * 100 }))
    .sort((a,b) => b.pct - a.pct)
    .slice(0, topK);
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
  const map = new Map();
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
  const values = labels.map(k => map.get(k).surv / Math.max(1, map.get(k).total));
  return { labels, values };
}
function renderEDABars(container, title, labels, values) {
  container.innerHTML = "";
  tfvis.visor().open();
  const surface = tfvis.visor().surface({ name: title, tab: "EDA" });

  const ul = document.createElement("div");
  ul.className = "muted";
  ul.style.fontSize = "12px";
  ul.innerHTML = labels.map((l,i)=>`${escapeHtml(l)}: <span class="mono">${(values[i]*100).toFixed(1)}%</span>`).join("<br/>");
  container.appendChild(ul);

  tfvis.render.barchart(
    surface,
    labels.map((l,i)=>({ index: l, value: values[i] })),
    { xLabel: title.replace("Survival by ", ""), yLabel: "Survival rate", height: 260 }
  );
}

/* -----------------------------
   State
------------------------------ */
const state = {
  rawTrain: null,
  rawTest: null,

  artifacts: null,
  featureNames: null, // NEW: human-readable feature names (matches X columns)

  XTrain: null, yTrain: null,
  XVal: null, yVal: null,
  XTest: null,
  testPassengerIds: null,

  model: null,

  // NEW: gate model used for interpretability
  gateModel: null,

  valProbs: null,
  valLabels: null,
  roc: null,
};

function disposeStateTensors() {
  const keys = ["XTrain","yTrain","XVal","yVal","XTest"];
  for (const k of keys) {
    if (state[k]) { state[k].dispose(); state[k] = null; }
  }
}

/* -----------------------------
   Preprocessing
------------------------------ */
function mode(values) {
  const m = new Map();
  for (const v of values) {
    const s = String(v ?? "").trim();
    if (!s) continue;
    m.set(s, (m.get(s) || 0) + 1);
  }
  let best = "", bestN = -1;
  for (const [k,n] of m.entries()) if (n > bestN) { best = k; bestN = n; }
  return best;
}
function median(nums) {
  const a = nums.filter(n => Number.isFinite(n)).slice().sort((x,y)=>x-y);
  if (!a.length) return NaN;
  const mid = Math.floor(a.length/2);
  return (a.length % 2 === 0) ? (a[mid-1]+a[mid])/2 : a[mid];
}
function meanStd(nums) {
  const a = nums.filter(n => Number.isFinite(n));
  const n = a.length || 1;
  const mean = a.reduce((s,v)=>s+v,0) / n;
  const v = a.reduce((s,x)=>s+(x-mean)*(x-mean),0) / n;
  const std = Math.sqrt(v) || 1;
  return { mean, std };
}
function buildOneHotMap(values) {
  const cats = Array.from(new Set(values.map(v => String(v ?? "").trim()))).filter(v => v !== "");
  cats.sort();
  const map = new Map();
  cats.forEach((c,i)=>map.set(c,i));
  return { cats, map };
}
function oneHot(value, oh) {
  const out = new Array(oh.cats.length).fill(0);
  const key = String(value ?? "").trim();
  const idx = oh.map.get(key);
  if (idx !== undefined) out[idx] = 1;
  return out;
}

function preprocess({ addFamily, addAlone }) {
  if (!state.rawTrain || !state.rawTest) throw new Error("Load train/test first.");
  const train = state.rawTrain;
  const test = state.rawTest;

  const ageMedian = median(train.map(r => safeNumber(r.Age)));
  const embarkedMode = mode(train.map(r => r.Embarked));

  function normalizeRow(r) {
    let Age = safeNumber(r.Age);
    if (!Number.isFinite(Age)) Age = ageMedian;

    let Fare = safeNumber(r.Fare);
    let SibSp = safeNumber(r.SibSp);
    let Parch = safeNumber(r.Parch);

    let Pclass = String(r.Pclass ?? "").trim();
    let Sex = String(r.Sex ?? "").trim();
    let Embarked = String(r.Embarked ?? "").trim();
    if (!Embarked) Embarked = embarkedMode;

    const base = {
      Pclass, Sex, Embarked,
      Age,
      Fare: Number.isFinite(Fare) ? Fare : NaN,
      SibSp: Number.isFinite(SibSp) ? SibSp : 0,
      Parch: Number.isFinite(Parch) ? Parch : 0,
    };

    if (addFamily || addAlone) {
      const fam = base.SibSp + base.Parch + 1;
      if (addFamily) base.FamilySize = fam;
      if (addAlone) base.IsAlone = (fam === 1) ? 1 : 0;
    }
    return base;
  }

  const trainN = train.map(normalizeRow);
  const testN  = test.map(normalizeRow);

  const fareMS = meanStd(trainN.map(r => safeNumber(r.Fare)));
  for (const r of trainN) if (!Number.isFinite(r.Fare)) r.Fare = fareMS.mean;
  for (const r of testN)  if (!Number.isFinite(r.Fare)) r.Fare = fareMS.mean;

  const ageMS = meanStd(trainN.map(r => r.Age));
  for (const r of trainN) { r.Age_z = (r.Age - ageMS.mean) / ageMS.std; r.Fare_z = (r.Fare - fareMS.mean) / fareMS.std; }
  for (const r of testN)  { r.Age_z = (r.Age - ageMS.mean) / ageMS.std; r.Fare_z = (r.Fare - fareMS.mean) / fareMS.std; }

  const all = trainN.concat(testN);
  const ohPclass = buildOneHotMap(all.map(r => r.Pclass));
  const ohSex    = buildOneHotMap(all.map(r => r.Sex));
  const ohEmb    = buildOneHotMap(all.map(r => r.Embarked));

  // Feature names (for FI table)
  const featureNames = [];
  for (const c of ohSex.cats) featureNames.push(`Sex_${c}`);          // e.g. Sex_male, Sex_female
  for (const c of ohPclass.cats) featureNames.push(`Pclass_${c}`);    // Pclass_1, Pclass_2, Pclass_3
  for (const c of ohEmb.cats) featureNames.push(`Embarked_${c}`);     // Embarked_S/C/Q
  featureNames.push("Age_std", "SibSp", "Parch", "Fare_std");
  if (addFamily) featureNames.push("FamilySize");
  if (addAlone) featureNames.push("IsAlone");

  // IMPORTANT: Keep feature order stable and aligned with names
  function featurizeRow(r) {
    const feats = [];
    feats.push(...oneHot(r.Sex, ohSex));
    feats.push(...oneHot(r.Pclass, ohPclass));
    feats.push(...oneHot(r.Embarked, ohEmb));
    feats.push(r.Age_z, r.SibSp, r.Parch, r.Fare_z);
    if (addFamily) feats.push(r.FamilySize);
    if (addAlone) feats.push(r.IsAlone);
    return feats.map(v => Number.isFinite(v) ? v : 0);
  }

  const X = [];
  const y = [];
  for (let i = 0; i < train.length; i++) {
    const yn = safeNumber(train[i][SCHEMA.target]);
    if (!Number.isFinite(yn)) continue;
    X.push(featurizeRow(trainN[i]));
    y.push(yn >= 0.5 ? 1 : 0);
  }

  const Xtest = [];
  const pids = [];
  for (let i = 0; i < test.length; i++) {
    pids.push(test[i][SCHEMA.id]);
    Xtest.push(featurizeRow(testN[i]));
  }

  // Stratified 80/20
  const idx0 = [], idx1 = [];
  for (let i = 0; i < y.length; i++) (y[i] === 1 ? idx1 : idx0).push(i);
  const shuffle = (a) => { for (let i=a.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; } };
  shuffle(idx0); shuffle(idx1);

  const valFrac = 0.2;
  const n0v = Math.max(1, Math.floor(idx0.length * valFrac));
  const n1v = Math.max(1, Math.floor(idx1.length * valFrac));
  const valIdx = idx0.slice(0,n0v).concat(idx1.slice(0,n1v));
  const trIdx  = idx0.slice(n0v).concat(idx1.slice(n1v));
  shuffle(valIdx); shuffle(trIdx);

  const XTrain = trIdx.map(i => X[i]);
  const yTrain = trIdx.map(i => y[i]);
  const XVal   = valIdx.map(i => X[i]);
  const yVal   = valIdx.map(i => y[i]);

  disposeStateTensors();

  state.XTrain = tf.tensor2d(XTrain);
  state.yTrain = tf.tensor2d(yTrain, [yTrain.length, 1]);
  state.XVal   = tf.tensor2d(XVal);
  state.yVal   = tf.tensor2d(yVal, [yVal.length, 1]);
  state.XTest  = tf.tensor2d(Xtest);

  state.testPassengerIds = pids;
  state.featureNames = featureNames;

  state.artifacts = {
    ageMedian,
    embarkedMode,
    ageMeanStd: ageMS,
    fareMeanStd: fareMS,
    oneHotCategories: { Sex: ohSex.cats, Pclass: ohPclass.cats, Embarked: ohEmb.cats },
    featureNames,
    addFamily,
    addAlone,
    schema: SCHEMA,
  };

  state.valProbs = null;
  state.valLabels = null;
  state.roc = null;

  return { nTrain: XTrain.length, nVal: XVal.length, nFeatures: featureNames.length, featureNames };
}

/* -----------------------------
   Model (base) + Sigmoid Gate model
------------------------------ */
function modelSummaryText(model) {
  const lines = [];
  lines.push("Model: Dense(16, relu) -> Dense(1, sigmoid)");
  lines.push(`Params: ${model.countParams()}`);
  model.layers.forEach((l,i)=>{
    const cfg = l.getConfig();
    lines.push(`Layer ${i}: ${l.getClassName()} ${cfg.units ? `(units=${cfg.units}, act=${cfg.activation})` : ""}`);
  });
  return lines.join("\n");
}

function buildBaseModel(inputDim) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [inputDim] }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({ optimizer: tf.train.adam(), loss: "binaryCrossentropy", metrics: ["accuracy"] });
  return model;
}

// ✅ Sigmoid Gate (Mask) Model:
// gate(x) = sigmoid(W2 * relu(W1*x))  -> same dimension as x
// x_gated = x ⊙ gate(x)
// Then classifier on x_gated.
function buildGateModel(inputDim) {
  const input = tf.input({ shape: [inputDim] });

  // "Axis alignment" hidden layer (k->k): keep k dims to interpret gate per-feature
  const gateHidden = tf.layers.dense({ units: inputDim, activation: "relu", name: "gate_hidden" }).apply(input);

  // Sigmoid gate output (k dims): values in (0,1)
  const gate = tf.layers.dense({ units: inputDim, activation: "sigmoid", name: "gate" }).apply(gateHidden);

  // Hadamard product: x ⊙ gate
  const gated = tf.layers.multiply({ name: "gated_input" }).apply([input, gate]);

  // Classifier head (shallow)
  const h = tf.layers.dense({ units: 16, activation: "relu", name: "clf_hidden" }).apply(gated);
  const out = tf.layers.dense({ units: 1, activation: "sigmoid", name: "clf_out" }).apply(h);

  const model = tf.model({ inputs: input, outputs: out, name: "titanic_gate_model" });
  model.compile({ optimizer: tf.train.adam(), loss: "binaryCrossentropy", metrics: ["accuracy"] });

  // Also create a small model to output the gate for any X (for feature importance).
  const gateOutModel = tf.model({ inputs: input, outputs: gate, name: "gate_out_model" });

  return { model, gateOutModel };
}

/* -----------------------------
   Training with early stopping
------------------------------ */
async function trainModel(model) {
  let bestVal = Infinity;
  let bestWeights = null;
  let badEpochs = 0;
  const patience = 5;

  tfvis.visor().open();
  const fitContainer = { name: "Training", tab: "Training" };
  const callbacks = tfvis.show.fitCallbacks(
    fitContainer,
    ["loss", "acc", "val_loss", "val_acc"],
    { callbacks: ["onEpochEnd"] }
  );

  await model.fit(state.XTrain, state.yTrain, {
    epochs: 50,
    batchSize: 32,
    validationData: [state.XVal, state.yVal],
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        await callbacks.onEpochEnd(epoch, logs);

        const v = logs?.val_loss;
        if (Number.isFinite(v)) {
          if (v < bestVal - 1e-6) {
            bestVal = v;
            badEpochs = 0;
            if (bestWeights) bestWeights.forEach(w => w.dispose());
            bestWeights = model.getWeights().map(w => w.clone());
          } else {
            badEpochs += 1;
            if (badEpochs >= patience) model.stopTraining = true;
          }
        }
      }
    }
  });

  if (bestWeights) {
    model.setWeights(bestWeights);
    bestWeights.forEach(w => w.dispose());
  }
}

/* -----------------------------
   Metrics (ROC/AUC + threshold)
------------------------------ */
function computeRocAuc(labels, probs, nThresholds = 250) {
  const thresholds = [];
  for (let i=0;i<=nThresholds;i++) thresholds.push(i/nThresholds);

  const P = labels.reduce((s,y)=>s+(y===1?1:0),0) || 1;
  const N = labels.length - P || 1;

  const pts = [];
  for (const th of thresholds) {
    let TP=0, FP=0, TN=0, FN=0;
    for (let i=0;i<labels.length;i++){
      const y = labels[i];
      const p = probs[i];
      const yhat = (p >= th) ? 1 : 0;
      if (y===1 && yhat===1) TP++;
      else if (y===0 && yhat===1) FP++;
      else if (y===0 && yhat===0) TN++;
      else FN++;
    }
    const TPR = TP / (TP + FN || 1);
    const FPR = FP / (FP + TN || 1);
    pts.push({x:FPR, y:TPR});
  }
  pts.sort((a,b)=>a.x-b.x);

  let auc = 0;
  for (let i=1;i<pts.length;i++){
    const x1=pts[i-1].x, y1=pts[i-1].y;
    const x2=pts[i].x,   y2=pts[i].y;
    auc += (x2-x1)*(y1+y2)/2;
  }
  return { fpr: pts.map(p=>p.x), tpr: pts.map(p=>p.y), auc };
}

function confusionAndScores(labels, probs, th) {
  let TP=0, FP=0, TN=0, FN=0;
  for (let i=0;i<labels.length;i++){
    const y=labels[i], p=probs[i];
    const yhat = (p >= th) ? 1 : 0;
    if (y===1 && yhat===1) TP++;
    else if (y===0 && yhat===1) FP++;
    else if (y===0 && yhat===0) TN++;
    else FN++;
  }
  const precision = TP/(TP+FP||1);
  const recall = TP/(TP+FN||1);
  const f1 = (2*precision*recall)/(precision+recall||1);
  const acc = (TP+TN)/(TP+TN+FP+FN||1);
  return { TP,FP,TN,FN, precision, recall, f1, acc };
}

function renderRoc(roc) {
  ui.rocVis.innerHTML = `<div class="muted" style="font-size:12px">
    ROC is rendered in tfjs-vis tab <span class="mono">Metrics</span>.
  </div>`;
  tfvis.visor().open();
  const surface = tfvis.visor().surface({ name: "ROC Curve", tab: "Metrics" });
  const series = roc.fpr.map((x,i)=>({x, y: roc.tpr[i]}));
  tfvis.render.linechart(surface, { values: series, series: ["ROC"] }, { xLabel:"FPR", yLabel:"TPR", height:280 });
}

function renderConfusion(cm) {
  ui.cmWrap.innerHTML = `<div class="mono" style="font-size:12px; line-height:1.4">
    TN=${cm.TN}  FP=${cm.FP}<br/>FN=${cm.FN}  TP=${cm.TP}
  </div>`;
  tfvis.visor().open();
  const surface = tfvis.visor().surface({ name: "Confusion Matrix", tab: "Metrics" });
  const values = [[cm.TN, cm.FP],[cm.FN, cm.TP]];
  tfvis.render.confusionMatrix(surface, { values, tickLabels:["0","1"] });
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

async function evaluateModel() {
  const probsT = state.model.predict(state.XVal);
  const probs = Float32Array.from(await probsT.data());
  probsT.dispose();

  const labelsData = await state.yVal.data();
  const labels = Int32Array.from(Array.from(labelsData).map(v => (v >= 0.5 ? 1 : 0)));

  state.valProbs = probs;
  state.valLabels = labels;

  const roc = computeRocAuc(labels, probs);
  state.roc = roc;
  ui.aucText.textContent = roc.auc.toFixed(4);

  renderRoc(roc);
  updateThresholdUI(Number(ui.thSlider.value));
}

/* -----------------------------
   Sigmoid Gate Feature Importance
------------------------------ */
function renderFeatureImportance(scores, featureNames) {
  // scores: array of numbers in [0,1] (gate average)
  const rows = featureNames.map((name,i)=>({ name, score: scores[i] || 0 }));
  rows.sort((a,b)=>b.score-a.score);

  const maxScore = Math.max(...rows.map(r=>r.score), 1e-9);

  ui.fiBody.innerHTML = "";
  const top3 = rows.slice(0,3).map(r=>r.name).join(", ");
  ui.fiTop3.textContent = `Top 3 most important features: ${top3}`;

  rows.forEach((r, idx) => {
    const tr = document.createElement("tr");

    const tdName = document.createElement("td");
    tdName.textContent = r.name;

    const tdBar = document.createElement("td");
    tdBar.innerHTML = `
      <div class="barWrap">
        <div class="bar" style="width:${(r.score/maxScore)*100}%"></div>
        <div class="barLabel">${r.score.toFixed(4)}</div>
      </div>
    `;

    const tdRank = document.createElement("td");
    tdRank.textContent = String(idx + 1);

    tr.appendChild(tdName);
    tr.appendChild(tdBar);
    tr.appendChild(tdRank);
    ui.fiBody.appendChild(tr);
  });
}

async function analyzeGateImportance() {
  if (!state.gateModel) throw new Error("No gate model. Build & train first.");
  if (!state.featureNames) throw new Error("No feature names. Preprocess first.");

  // Gate importance = average gate value over validation set (or train set).
  // This makes interpretation straightforward: higher gate => feature kept more often.
  const gateOut = state.gateModel.gateOutModel.predict(state.XVal);
  const gateMat = await gateOut.array(); // shape [n, k]
  gateOut.dispose();

  const n = gateMat.length;
  const k = gateMat[0]?.length || 0;
  const avg = new Array(k).fill(0);

  for (let i=0;i<n;i++){
    for (let j=0;j<k;j++) avg[j] += gateMat[i][j];
  }
  for (let j=0;j<k;j++) avg[j] /= Math.max(1,n);

  renderFeatureImportance(avg, state.featureNames);
}

/* -----------------------------
   Predict & Export
------------------------------ */
function downloadText(filename, text) {
  const blob = new Blob([text], { type:"text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
}
function toCSV(headers, rows) {
  const esc = (v) => {
    const s = String(v ?? "");
    if (/[",\n\r]/.test(s)) return `"${s.replaceAll('"','""')}"`;
    return s;
  };
  return [headers.map(esc).join(","), ...rows.map(r=>r.map(esc).join(","))].join("\n");
}
async function predictAndExport() {
  ui.predStatus.textContent = "running...";
  const probsT = state.model.predict(state.XTest);
  const probs = await probsT.data();
  probsT.dispose();

  const th = Number(ui.thSlider.value);
  const preds = Array.from(probs).map(p => (p >= th ? 1 : 0));

  downloadText("submission.csv", toCSV([SCHEMA.id, SCHEMA.target], state.testPassengerIds.map((pid,i)=>[pid, preds[i]])));
  downloadText("probabilities.csv", toCSV([SCHEMA.id, "Survived_Prob"], state.testPassengerIds.map((pid,i)=>[pid, Number(probs[i]).toFixed(6)])));

  ui.predStatus.textContent = "done";
}
async function saveModelDownloads() {
  if (!state.model) throw new Error("No model to save.");
  await state.model.save("downloads://titanic-tfjs");
}
function exportPreprocessJSON() {
  if (!state.artifacts) throw new Error("Run preprocess first.");
  const json = JSON.stringify(state.artifacts, null, 2);
  const blob = new Blob([json], { type:"application/json;charset=utf-8" });
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
    const trainFile = ui.trainFile.files?.[0] || null;
    const testFile  = ui.testFile.files?.[0] || null;

    const train = await loadCSV({ url: DEFAULT_TRAIN_URL, file: trainFile });
    const test  = await loadCSV({ url: DEFAULT_TEST_URL, file: testFile });

    state.rawTrain = train.data;
    state.rawTest  = test.data;

    const shTr = shapeOf(state.rawTrain);
    const shTe = shapeOf(state.rawTest);
    ui.trainShape.textContent = `${shTr.rows} / ${shTr.cols}`;
    ui.testShape.textContent  = `${shTe.rows} / ${shTe.cols}`;

    ui.trainMissing.textContent = missingnessTop(state.rawTrain,5).map(x=>`${x.col}:${x.pct.toFixed(1)}%`).join(" | ") || "—";
    ui.testMissing.textContent  = missingnessTop(state.rawTest,5).map(x=>`${x.col}:${x.pct.toFixed(1)}%`).join(" | ") || "—";

    renderTable($("trainTable"), state.rawTrain, 10);
    renderTable($("testTable"), state.rawTest, 10);

    const sex = groupSurvivalRateBy(state.rawTrain, "Sex");
    const pcl = groupSurvivalRateBy(state.rawTrain, "Pclass");
    renderEDABars(ui.edaSex, "Survival by Sex", sex.labels, sex.values);
    renderEDABars(ui.edaPclass, "Survival by Pclass", pcl.labels, pcl.values);

    ui.btnPreprocess.disabled = false;
    setPill(ui.loadStatus, "Status", "loaded (train+test)");
  } catch (err) {
    setPill(ui.loadStatus, "Status", "error");
    alertErr("Failed to load CSVs. Check GitHub Pages URLs or upload files.", err);
  }
}

/* -----------------------------
   Wire buttons
------------------------------ */
ui.btnLoad.addEventListener("click", loadAllData);

ui.btnPreprocess.addEventListener("click", () => {
  try {
    const info = preprocess({ addFamily: ui.optFamily.checked, addAlone: ui.optAlone.checked });

    ui.featSummary.textContent =
      `Train: ${info.nTrain} | Val: ${info.nVal}\n` +
      `Features: ${info.nFeatures}\n` +
      `Names: ${info.featureNames.join(", ")}`;

    // Invalidate models
    if (state.model) { state.model.dispose(); state.model = null; }
    if (state.gateModel?.model) { state.gateModel.model.dispose(); state.gateModel = null; }

    ui.modelSummary.textContent = "—";
    ui.btnTrain.disabled = true;
    ui.btnEval.disabled = true;
    ui.btnPredict.disabled = true;
    ui.btnSaveModel.disabled = true;

    ui.btnBuild.disabled = false;
    ui.btnExportPre.disabled = false;

    // Gate FI disabled until model trained
    ui.btnGateAnalyze.disabled = true;
    ui.fiBody.innerHTML = `<tr><td colspan="3" class="muted">Train a model, then click “Analyze Feature Importance”.</td></tr>`;
    ui.fiTop3.textContent = "Top 3 most important features: —";
  } catch (err) {
    alertErr("Preprocess failed.", err);
  }
});

ui.btnBuild.addEventListener("click", () => {
  try {
    if (!state.XTrain) throw new Error("Preprocess first.");
    const k = state.XTrain.shape[1];

    // Build BOTH: base model (used for prediction) and gate model (used for FI + prediction)
    // To match the homework, we will actually TRAIN the gate model and USE it as the model.
    // That way feature importance matches the trained predictor.
    if (state.model) state.model.dispose();
    if (state.gateModel?.model) state.gateModel.model.dispose();

    state.gateModel = buildGateModel(k);
    state.model = state.gateModel.model;

    ui.modelSummary.textContent =
      modelSummaryText(state.model) +
      `\n\n(Interpretability) Gate: sigmoid mask of size k=${k} applied via element-wise product (x ⊙ gate(x)).`;

    ui.btnTrain.disabled = false;
    ui.btnEval.disabled = true;
    ui.btnPredict.disabled = true;
    ui.btnSaveModel.disabled = false;
    ui.btnGateAnalyze.disabled = true;
  } catch (err) {
    alertErr("Model build failed.", err);
  }
});

ui.btnTrain.addEventListener("click", async () => {
  try {
    ui.btnTrain.disabled = true;
    await trainModel(state.model);
    ui.btnTrain.disabled = false;

    ui.btnEval.disabled = false;
    ui.btnPredict.disabled = false;

    // Gate FI now available
    ui.btnGateAnalyze.disabled = false;
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

ui.thSlider.addEventListener("input", () => updateThresholdUI(Number(ui.thSlider.value)));

ui.btnPredict.addEventListener("click", async () => {
  try { await predictAndExport(); }
  catch (err) { ui.predStatus.textContent = "error"; alertErr("Prediction/export failed.", err); }
});

ui.btnExportPre.addEventListener("click", () => {
  try { exportPreprocessJSON(); }
  catch (err) { alertErr("Export preprocess JSON failed.", err); }
});

ui.btnSaveModel.addEventListener("click", async () => {
  try { await saveModelDownloads(); }
  catch (err) { alertErr("Model save failed.", err); }
});

// ✅ NEW: Feature importance button
ui.btnGateAnalyze?.addEventListener("click", async () => {
  try {
    ui.btnGateAnalyze.disabled = true;
    await analyzeGateImportance();
    ui.btnGateAnalyze.disabled = false;
  } catch (err) {
    ui.btnGateAnalyze.disabled = false;
    alertErr("Sigmoid gate analysis failed.", err);
  }
});

/* -----------------------------
   Auto-load on start
------------------------------ */
window.addEventListener("DOMContentLoaded", async () => {
  await loadAllData();
  updateThresholdUI(Number(ui.thSlider.value));
});
