```javascript
// app.js
// Browser-only Titanic shallow classifier using TensorFlow.js + tfjs-vis
//
// Reusable note (schema swap):
// Change SCHEMA.target/features/identifier and update preprocessOneHot() mapping if using another dataset.
// This file supports loading via file input. (Fetch-based loading could be added for repo-hosted CSVs.)

(() => {
  "use strict";

  // ----------------------------
  // Schema (swap here for other datasets)
  // ----------------------------
  const SCHEMA = {
    identifier: "PassengerId",
    target: "Survived",
    features: ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
  };

  // ----------------------------
  // DOM
  // ----------------------------
  const $ = (id) => document.getElementById(id);

  const el = {
    trainFile: $("trainFile"),
    testFile: $("testFile"),
    btnLoad: $("btnLoad"),

    toggleFamily: $("toggleFamily"),
    toggleAlone: $("toggleAlone"),
    btnPreprocess: $("btnPreprocess"),
    prepPill: $("prepPill"),
    prepLog: $("prepLog"),

    btnBuildModel: $("btnBuildModel"),
    btnSaveModel: $("btnSaveModel"),
    modelPill: $("modelPill"),
    modelSummary: $("modelSummary"),

    btnTrain: $("btnTrain"),
    btnEvaluate: $("btnEvaluate"),
    trainPill: $("trainPill"),

    thresholdSlider: $("thresholdSlider"),
    thresholdLabel: $("thresholdLabel"),
    aucPill: $("aucPill"),
    metricsTable: $("metricsTable"),

    btnPredict: $("btnPredict"),
    btnExportSubmission: $("btnExportSubmission"),
    btnExportProbs: $("btnExportProbs"),
    predPill: $("predPill"),

    btnExportDebug: $("btnExportDebug"),
    exportPill: $("exportPill"),

    statusPill: $("statusPill"),

    kpiTrain: $("kpiTrain"),
    kpiTest: $("kpiTest"),
    kpiMissing: $("kpiMissing"),

    previewTable: $("previewTable"),

    visEDA: $("visEDA"),
    visFit: $("visFit"),
    visHistory: $("visHistory"), // We'll reuse this area to render gate feature-importance chart.
    visROC: $("visROC"),
  };

  function setPill(pillEl, label, value) {
    pillEl.innerHTML = `<b>${escapeHtml(label)}:</b> ${escapeHtml(value)}`;
  }
  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }
  function alertUser(msg) {
    window.alert(msg);
  }

  // ----------------------------
  // State
  // ----------------------------
  const state = {
    raw: {
      trainRows: null,
      testRows: null,
    },
    inspect: {
      missingPct: null,
    },
    prep: {
      // Learned preprocessing parameters
      ageMedian: null,
      embarkedMode: null,
      ageMean: null,
      ageStd: null,
      fareMean: null,
      fareStd: null,
      // Categories for one-hot (fit on train, apply to both)
      cats: {
        Sex: [],
        Pclass: [],
        Embarked: [],
      },
      // Feature order after one-hot + engineered
      featureNames: [],
      // Debug summary object (for export)
      summary: null,
    },
    tensors: {
      xTrain: null,
      yTrain: null,
      xVal: null,
      yVal: null,
      xTest: null,
      testPassengerIds: null,
    },
    model: null,
    eval: {
      // Cached validation probabilities + labels for threshold slider updates
      valProbs: null, // Float32Array
      valLabels: null, // Int32Array
      auc: null,
      roc: null, // {fpr:[], tpr:[]}
    },
    pred: {
      testProbs: null, // Float32Array
      threshold: 0.5,
    },
  };

  // ----------------------------
  // (HW2 Task 4) Sigmoid Gate (Mask) Layer for Feature Importance
  // ----------------------------
  // Learns a per-feature gate g = sigmoid(w), then outputs x * g (Hadamard product).
  // Gate values are directly interpretable as feature importance (0..1).
  class SigmoidGate extends tf.layers.Layer {
    constructor(config) {
      super(config || {});
      this.supportsMasking = true;
    }

    build(inputShape) {
      const inputDim = inputShape[inputShape.length - 1];
      // Trainable logits; sigmoid(logit)=0.5 at init.
      this.w = this.addWeight(
        "gate_logits",
        [inputDim],
        "float32",
        tf.initializers.zeros(),
        // Optional L1 regularization encourages sparsity for clearer importance.
        tf.regularizers.l1({ l1: 1e-3 })
      );
      this.built = true;
    }

    call(inputs) {
      return tf.tidy(() => {
        const x = Array.isArray(inputs) ? inputs[0] : inputs; // [batch, d]
        const g = tf.sigmoid(this.w.read()); // [d], values in (0,1)
        return x.mul(g); // broadcast multiply -> masked input
      });
    }

    computeOutputShape(inputShape) {
      return inputShape;
    }

    getConfig() {
      const base = super.getConfig();
      return { ...base };
    }

    static get className() {
      return "SigmoidGate";
    }
  }
  tf.serialization.registerClass(SigmoidGate);

  function renderGateImportance() {
    if (!state.model || !state.prep.featureNames?.length) return;

    let gateLayer;
    try {
      gateLayer = state.model.getLayer("sigmoid_gate");
    } catch {
      return;
    }
    if (!gateLayer) return;

    // gate logits -> sigmoid -> importance
    const logits = gateLayer.getWeights()[0]; // tensor [d]
    const imp = tf.sigmoid(logits).dataSync(); // Float32Array length d

    const items = state.prep.featureNames
      .map((name, i) => ({ x: name, y: Number(imp[i]) }))
      .sort((a, b) => b.y - a.y);

    // Render into the right-side training panel (visHistory)
    el.visHistory.innerHTML = "";
    tfvis.render.barchart(
      el.visHistory,
      items,
      {
        title: "Sigmoid Gate Feature Importance (0..1)",
        xLabel: "Feature",
        yLabel: "Gate value",
        height: 360,
      }
    );
  }

  // ----------------------------
  // CSV Loading (safe CSV parsing without comma-escape bugs)
  // ----------------------------
  // IMPORTANT: We do not use naive split(',') parsing. That breaks on quoted commas (e.g. Name field).
  // We implement a small RFC4180-ish parser here to handle quotes correctly.
  // If you want, you can swap this out for PapaParse (not required in HW2 prompt).
  function parseCSVText(text) {
    const rows = [];
    let row = [];
    let field = "";
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      const next = text[i + 1];

      if (inQuotes) {
        if (c === '"' && next === '"') {
          field += '"'; // escaped quote
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
          // ignore possible \r already removed or handle it
          rows.push(row);
          row = [];
        } else if (c === "\r") {
          // ignore
        } else {
          field += c;
        }
      }
    }
    // last field
    row.push(field);
    rows.push(row);

    // Convert to objects using header row
    const header = rows[0];
    const out = [];
    for (let r = 1; r < rows.length; r++) {
      if (rows[r].length === 1 && rows[r][0] === "") continue; // skip empty last line
      const obj = {};
      for (let c = 0; c < header.length; c++) {
        obj[header[c]] = rows[r][c] ?? "";
      }
      out.push(obj);
    }
    return out;
  }

  async function readFileAsText(file) {
    return await file.text();
  }

  function dynamicType(v) {
    // Convert strings to numbers when appropriate; keep empty as ""
    if (v === null || v === undefined) return "";
    const s = String(v).trim();
    if (s === "") return "";
    // Kaggle has "NaN" sometimes; treat as empty
    if (s.toLowerCase() === "nan") return "";
    const n = Number(s);
    if (Number.isFinite(n) && s !== "") return n;
    return s;
  }

  function typeRows(rows) {
    // Convert each field with dynamicType
    return rows.map((r) => {
      const o = {};
      for (const k of Object.keys(r)) o[k] = dynamicType(r[k]);
      return o;
    });
  }

  async function loadFromFileInputs() {
    const trainFile = el.trainFile.files?.[0];
    const testFile = el.testFile.files?.[0];
    if (!trainFile || !testFile) {
      alertUser("Please upload both train.csv and test.csv.");
      return null;
    }

    try {
      setPill(el.statusPill, "Status", "loading CSV…");

      const [trainText, testText] = await Promise.all([
        readFileAsText(trainFile),
        readFileAsText(testFile),
      ]);

      // Robust parsing with quote handling fixes "comma escape problem"
      const trainRows = typeRows(parseCSVText(trainText));
      const testRows = typeRows(parseCSVText(testText));

      if (!trainRows.length || !testRows.length) throw new Error("CSV appears empty.");
      if (!(SCHEMA.target in trainRows[0])) {
        throw new Error(`train.csv must contain target column "${SCHEMA.target}".`);
      }

      state.raw.trainRows = trainRows;
      state.raw.testRows = testRows;

      setPill(el.statusPill, "Status", "loaded ✅");
      return { trainRows, testRows };
    } catch (e) {
      console.error(e);
      alertUser(`Load failed: ${e.message || e}`);
      setPill(el.statusPill, "Status", "load failed");
      return null;
    }
  }

  // ----------------------------
  // Inspection: preview, shape, missing %, survival charts
  // ----------------------------
  function getColumns(rows) {
    const set = new Set();
    for (const r of rows) Object.keys(r).forEach((k) => set.add(k));
    return [...set];
  }

  function isMissing(v) {
    return v === "" || v === null || v === undefined;
  }

  function missingPct(rows) {
    const cols = getColumns(rows);
    const total = rows.length;
    const miss = {};
    cols.forEach((c) => (miss[c] = 0));
    for (const r of rows) {
      for (const c of cols) if (isMissing(r[c])) miss[c] += 1;
    }
    const arr = cols.map((c) => ({ col: c, pct: total ? (miss[c] / total) * 100 : 0 }));
    arr.sort((a, b) => b.pct - a.pct);
    return arr;
  }

  function renderPreviewTable(rows, n = 10) {
    const sample = rows.slice(0, n);
    const cols = getColumns(sample);

    const thead = el.previewTable.querySelector("thead");
    const tbody = el.previewTable.querySelector("tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    const trh = document.createElement("tr");
    cols.forEach((c) => {
      const th = document.createElement("th");
      th.textContent = c;
      trh.appendChild(th);
    });
    thead.appendChild(trh);

    sample.forEach((r) => {
      const tr = document.createElement("tr");
      cols.forEach((c) => {
        const td = document.createElement("td");
        const v = r[c];
        td.textContent = isMissing(v) ? "—" : String(v);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  function survivalRateBy(trainRows, col) {
    // returns [{key, rate, n}]
    const map = new Map();
    for (const r of trainRows) {
      const k = r[col];
      const y = r[SCHEMA.target];
      if (isMissing(k) || (y !== 0 && y !== 1)) continue;
      const key = String(k);
      if (!map.has(key)) map.set(key, { n: 0, s: 0 });
      const obj = map.get(key);
      obj.n += 1;
      obj.s += (y === 1 ? 1 : 0);
    }
    const out = [...map.entries()].map(([key, v]) => ({
      key,
      n: v.n,
      rate: v.n ? (v.s / v.n) * 100 : 0,
    }));
    out.sort((a, b) => {
      const an = Number(a.key), bn = Number(b.key);
      if (Number.isFinite(an) && Number.isFinite(bn)) return an - bn;
      return a.key.localeCompare(b.key);
    });
    return out;
  }

  function renderQuickEDA(trainRows) {
    el.visEDA.innerHTML = ""; // clear
    const bySex = survivalRateBy(trainRows, "Sex");
    const byPclass = survivalRateBy(trainRows, "Pclass");

    // tfjs-vis bar charts
    tfvis.render.barchart(
      el.visEDA,
      bySex.map((d) => ({ x: d.key, y: d.rate })),
      { title: "Survival rate by Sex (train)", xLabel: "Sex", yLabel: "Survival rate (%)" }
    );
    tfvis.render.barchart(
      el.visEDA,
      byPclass.map((d) => ({ x: d.key, y: d.rate })),
      { title: "Survival rate by Pclass (train)", xLabel: "Pclass", yLabel: "Survival rate (%)" }
    );
  }

  function renderKPIs(trainRows, testRows) {
    el.kpiTrain.textContent = `${trainRows.length} / ${getColumns(trainRows).length}`;
    el.kpiTest.textContent = `${testRows.length} / ${getColumns(testRows).length}`;

    const miss = missingPct(trainRows);
    state.inspect.missingPct = miss;
    const top3 = miss
      .slice(0, 3)
      .map((d) => `${d.col} ${d.pct.toFixed(1)}%`)
      .join(" • ");
    el.kpiMissing.textContent = top3 || "—";
  }

  // ----------------------------
  // Preprocessing
  // ----------------------------
  function median(nums) {
    if (!nums.length) return null;
    const a = [...nums].sort((x, y) => x - y);
    const m = Math.floor(a.length / 2);
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
  }

  function mode(values) {
    const map = new Map();
    for (const v of values) {
      if (isMissing(v)) continue;
      const k = String(v);
      map.set(k, (map.get(k) || 0) + 1);
    }
    let best = null,
      bestN = -1;
    for (const [k, n] of map.entries()) {
      if (n > bestN) {
        bestN = n;
        best = k;
      }
    }
    return best;
  }

  function meanStd(nums) {
    if (!nums.length) return { mean: 0, std: 1 };
    const mean = nums.reduce((s, x) => s + x, 0) / nums.length;
    const v = nums.reduce((s, x) => s + (x - mean) ** 2, 0) / Math.max(1, nums.length - 1);
    const std = Math.sqrt(v) || 1;
    return { mean, std };
  }

  function addEngineeredFeatures(row, useFamily, useAlone) {
    const out = { ...row };
    if (useFamily || useAlone) {
      const sib = Number(out.SibSp);
      const par = Number(out.Parch);
      const familySize =
        (Number.isFinite(sib) ? sib : 0) + (Number.isFinite(par) ? par : 0) + 1;
      if (useFamily) out.FamilySize = familySize;
      if (useAlone) out.IsAlone = familySize === 1 ? 1 : 0;
    }
    return out;
  }

  function fitPreprocessParams(trainRows) {
    // Impute Age median
    const ages = trainRows
      .map((r) => r.Age)
      .filter((v) => typeof v === "number" && Number.isFinite(v));
    const ageMedian = median(ages);

    // Embarked mode (categorical)
    const embarkedMode = mode(trainRows.map((r) => r.Embarked));

    // For standardization: compute on train AFTER imputation
    const imputedAges = trainRows
      .map((r) => (typeof r.Age === "number" ? r.Age : ageMedian))
      .filter((v) => typeof v === "number");
    const fares = trainRows
      .map((r) => r.Fare)
      .filter((v) => typeof v === "number" && Number.isFinite(v));

    const ageMS = meanStd(imputedAges);
    const fareMS = meanStd(fares);

    // One-hot categories fit on train
    const sexCats = [...new Set(trainRows.map((r) => String(r.Sex)).filter((v) => v && v !== "undefined"))].sort();
    const pclassCats = [
      ...new Set(trainRows.map((r) => String(r.Pclass)).filter((v) => v && v !== "undefined")),
    ].sort((a, b) => Number(a) - Number(b));
    const embarkedCats = [
      ...new Set(
        trainRows
          .map((r) => String(r.Embarked || embarkedMode))
          .filter((v) => v && v !== "undefined")
      ),
    ].sort();

    return {
      ageMedian,
      embarkedMode,
      ageMean: ageMS.mean,
      ageStd: ageMS.std,
      fareMean: fareMS.mean,
      fareStd: fareMS.std,
      cats: { Sex: sexCats, Pclass: pclassCats, Embarked: embarkedCats },
    };
  }

  function zscore(x, mean, std) {
    const v = (x - mean) / (std || 1);
    return Number.isFinite(v) ? v : 0;
  }

  function oneHot(value, categories) {
    const v = String(value);
    return categories.map((c) => (v === String(c) ? 1 : 0));
  }

  function preprocessRows(rows, params, { useFamily, useAlone, includeTarget }) {
    // Build feature vector for each row, matching a fixed feature order.
    // One-hot order is stable based on params.cats.
    const featureNames = [];

    // numeric
    featureNames.push("Age_z");
    featureNames.push("Fare_z");
    featureNames.push("SibSp");
    featureNames.push("Parch");

    // optional engineered
    if (useFamily) featureNames.push("FamilySize");
    if (useAlone) featureNames.push("IsAlone");

    // one-hot Sex
    params.cats.Sex.forEach((c) => featureNames.push(`Sex_${c}`));
    // one-hot Pclass
    params.cats.Pclass.forEach((c) => featureNames.push(`Pclass_${c}`));
    // one-hot Embarked
    params.cats.Embarked.forEach((c) => featureNames.push(`Embarked_${c}`));

    const X = [];
    const y = [];
    const ids = [];

    for (const r0 of rows) {
      const r = addEngineeredFeatures(r0, useFamily, useAlone);

      // Identifier for exports
      const pid = r[SCHEMA.identifier];
      ids.push(pid);

      // Impute
      const age = typeof r.Age === "number" && Number.isFinite(r.Age) ? r.Age : params.ageMedian;
      const embarked = !isMissing(r.Embarked) ? r.Embarked : params.embarkedMode;

      // Standardize Age/Fare
      const ageZ = zscore(Number(age), params.ageMean, params.ageStd);
      const fareZ = zscore(Number(r.Fare), params.fareMean, params.fareStd);

      const sib = Number(r.SibSp);
      const par = Number(r.Parch);

      const vec = [];
      vec.push(ageZ);
      vec.push(fareZ);
      vec.push(Number.isFinite(sib) ? sib : 0);
      vec.push(Number.isFinite(par) ? par : 0);

      if (useFamily) vec.push(Number(r.FamilySize) || 1);
      if (useAlone) vec.push(Number(r.IsAlone) || 0);

      // one-hot categoricals
      vec.push(...oneHot(r.Sex, params.cats.Sex));
      vec.push(...oneHot(r.Pclass, params.cats.Pclass));
      vec.push(...oneHot(embarked, params.cats.Embarked));

      X.push(vec);

      if (includeTarget) {
        const t = r[SCHEMA.target];
        y.push(t === 1 ? 1 : 0);
      }
    }

    return { X, y, ids, featureNames };
  }

  function toTensor2D(X) {
    return tf.tensor2d(X, [X.length, X[0].length], "float32");
  }
  function toTensor1D(y) {
    return tf.tensor1d(y, "float32");
  }

  function stratifiedSplit(X, y, valFrac = 0.2, seed = 42) {
    // Simple stratified split by label
    const idx0 = [];
    const idx1 = [];
    for (let i = 0; i < y.length; i++) (y[i] === 1 ? idx1 : idx0).push(i);

    // deterministic shuffle
    function mulberry32(a) {
      return function () {
        let t = (a += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }
    const rand = mulberry32(seed);

    function shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
      }
    }
    shuffle(idx0);
    shuffle(idx1);

    const nVal0 = Math.floor(idx0.length * valFrac);
    const nVal1 = Math.floor(idx1.length * valFrac);

    const valIdx = idx0.slice(0, nVal0).concat(idx1.slice(0, nVal1));
    const trainIdx = idx0.slice(nVal0).concat(idx1.slice(nVal1));

    // shuffle combined (not necessary but nice)
    shuffle(valIdx);
    shuffle(trainIdx);

    const Xtrain = trainIdx.map((i) => X[i]);
    const ytrain = trainIdx.map((i) => y[i]);
    const Xval = valIdx.map((i) => X[i]);
    const yval = valIdx.map((i) => y[i]);

    return { Xtrain, ytrain, Xval, yval };
  }

  async function runPreprocessing() {
    if (!state.raw.trainRows || !state.raw.testRows) {
      alertUser("Load data first.");
      return;
    }

    setPill(el.prepPill, "Preprocess", "running…");

    const useFamily = el.toggleFamily.checked;
    const useAlone = el.toggleAlone.checked;

    const params = fitPreprocessParams(state.raw.trainRows);

    // Build matrices
    const train = preprocessRows(state.raw.trainRows, params, {
      useFamily,
      useAlone,
      includeTarget: true,
    });
    const test = preprocessRows(state.raw.testRows, params, {
      useFamily,
      useAlone,
      includeTarget: false,
    });

    // Stratified split
    const split = stratifiedSplit(train.X, train.y, 0.2, 42);

    // Dispose old tensors if any
    disposeTensors();

    state.prep = {
      ...state.prep,
      ...params,
      featureNames: train.featureNames,
      summary: {
        schema: SCHEMA,
        engineered: { FamilySize: useFamily, IsAlone: useAlone },
        imputation: { Age_median: params.ageMedian, Embarked_mode: params.embarkedMode },
        standardization: {
          Age: { mean: params.ageMean, std: params.ageStd },
          Fare: { mean: params.fareMean, std: params.fareStd },
        },
        onehot: params.cats,
        featureNames: train.featureNames,
      },
    };

    // Tensors
    state.tensors.xTrain = toTensor2D(split.Xtrain);
    state.tensors.yTrain = toTensor1D(split.ytrain);
    state.tensors.xVal = toTensor2D(split.Xval);
    state.tensors.yVal = toTensor1D(split.yval);

    state.tensors.xTest = toTensor2D(test.X);
    state.tensors.testPassengerIds = test.ids;

    // Log
    const logLines = [];
    logLines.push(`Features (after preprocessing): ${train.featureNames.length}`);
    logLines.push(train.featureNames.join(", "));
    logLines.push("");
    logLines.push(`Train X shape: [${split.Xtrain.length}, ${train.featureNames.length}]`);
    logLines.push(`Train y shape: [${split.ytrain.length}]`);
    logLines.push(`Val   X shape: [${split.Xval.length}, ${train.featureNames.length}]`);
    logLines.push(`Val   y shape: [${split.yval.length}]`);
    logLines.push(`Test  X shape: [${test.X.length}, ${train.featureNames.length}]`);
    el.prepLog.textContent = logLines.join("\n");

    setPill(el.prepPill, "Preprocess", "done ✅");
  }

  function disposeTensors() {
    for (const k of Object.keys(state.tensors)) {
      const t = state.tensors[k];
      if (t && typeof t.dispose === "function") t.dispose();
      state.tensors[k] = null;
    }
    state.tensors.testPassengerIds = null;
  }

  // ----------------------------
  // Model
  // ----------------------------
  function buildModel(inputDim) {
    // (HW2 Task 4) Add Sigmoid gate to interpret feature importance.
    // Functional API: Input -> SigmoidGate -> Dense(16,relu) -> Dense(1,sigmoid)
    const input = tf.input({ shape: [inputDim], name: "features" });
    const gated = new SigmoidGate({ name: "sigmoid_gate" }).apply(input);
    const hidden = tf.layers.dense({ units: 16, activation: "relu", name: "hidden" }).apply(gated);
    const output = tf.layers.dense({ units: 1, activation: "sigmoid", name: "output" }).apply(hidden);

    const model = tf.model({
      inputs: input,
      outputs: output,
      name: "titanic_shallow_with_gate",
    });

    model.compile({
      optimizer: tf.train.adam(),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"],
    });

    return model;
  }

  function captureModelSummary(model) {
    const lines = [];
    model.summary(120, undefined, (line) => lines.push(line));
    return lines.join("\n");
  }

  function onBuildModel() {
    if (!state.tensors.xTrain) {
      alertUser("Run preprocessing first.");
      return;
    }
    const inputDim = state.tensors.xTrain.shape[1];
    state.model = buildModel(inputDim);
    el.modelSummary.textContent = captureModelSummary(state.model);
    setPill(el.modelPill, "Model", `built ✅ (inputDim=${inputDim})`);
  }

  async function onSaveModel() {
    if (!state.model) {
      alertUser("Build/train a model first.");
      return;
    }
    try {
      await state.model.save("downloads://titanic-tfjs");
      setPill(el.modelPill, "Model", "saved ✅");
    } catch (e) {
      console.error(e);
      alertUser("Model save failed. Check console.");
    }
  }

  // ----------------------------
  // Training (with early stopping)
  // ----------------------------
  function earlyStopping(patience = 5) {
    let best = Infinity;
    let wait = 0;
    let bestWeights = null;

    return {
      onEpochEnd: async (epoch, logs) => {
        const valLoss = logs?.val_loss;
        if (typeof valLoss !== "number") return;

        if (valLoss < best - 1e-6) {
          best = valLoss;
          wait = 0;
          // clone weights
          bestWeights = state.model.getWeights().map((w) => w.clone());
        } else {
          wait++;
          if (wait >= patience) {
            // restore best weights and stop
            if (bestWeights) state.model.setWeights(bestWeights);
            state.model.stopTraining = true;
          }
        }
      },
      onTrainEnd: async () => {
        // dispose best weights clones
        if (bestWeights) bestWeights.forEach((w) => w.dispose());
      },
    };
  }

  async function onTrain() {
    if (!state.model) {
      alertUser("Build model first.");
      return;
    }
    if (!state.tensors.xTrain || !state.tensors.yTrain) {
      alertUser("Run preprocessing first.");
      return;
    }

    setPill(el.trainPill, "Train", "training…");

    // Clear vis containers
    el.visFit.innerHTML = "";
    el.visHistory.innerHTML = "";

    // tfjs-vis fit callbacks
    const fitCallbacks = tfvis.show.fitCallbacks(el.visFit, ["loss", "acc", "val_loss", "val_acc"], {
      callbacks: ["onEpochEnd"],
    });

    const es = earlyStopping(5);

    await state.model.fit(state.tensors.xTrain, state.tensors.yTrain, {
      epochs: 50,
      batchSize: 32,
      validationData: [state.tensors.xVal, state.tensors.yVal],
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          await fitCallbacks.onEpochEnd(epoch, logs);
          await es.onEpochEnd(epoch, logs);
          setPill(
            el.trainPill,
            "Train",
            `epoch ${epoch + 1}/50 loss=${logs.loss.toFixed(4)} val_loss=${logs.val_loss.toFixed(4)}`
          );
        },
        onTrainEnd: async (logs) => {
          await es.onTrainEnd(logs);
        },
      },
    });

    // After training, render feature importance from the sigmoid gate (HW2 Task 4).
    renderGateImportance();

    setPill(el.trainPill, "Train", "done ✅");
  }

  // ----------------------------
  // Metrics: ROC/AUC + threshold slider -> confusion matrix + PRF1
  // ----------------------------
  async function computeValProbsAndLabels() {
    const probsT = state.model.predict(state.tensors.xVal);
    const probs = await probsT.data();
    probsT.dispose();

    const labels = await state.tensors.yVal.data(); // float32 0/1
    const y = new Int32Array(labels.length);
    for (let i = 0; i < labels.length; i++) y[i] = labels[i] >= 0.5 ? 1 : 0;

    return { probs: Float32Array.from(probs), labels: y };
  }

  function rocCurve(labels, probs, steps = 101) {
    const tpr = [];
    const fpr = [];
    for (let i = 0; i < steps; i++) {
      const thr = i / (steps - 1);
      let tp = 0,
        fp = 0,
        tn = 0,
        fn = 0;
      for (let j = 0; j < labels.length; j++) {
        const pred = probs[j] >= thr ? 1 : 0;
        const y = labels[j];
        if (y === 1 && pred === 1) tp++;
        if (y === 0 && pred === 1) fp++;
        if (y === 0 && pred === 0) tn++;
        if (y === 1 && pred === 0) fn++;
      }
      const tprVal = tp + fn ? tp / (tp + fn) : 0;
      const fprVal = fp + tn ? fp / (fp + tn) : 0;
      tpr.push(tprVal);
      fpr.push(fprVal);
    }
    return { fpr, tpr };
  }

  function aucFromRoc(fpr, tpr) {
    // Trapezoidal rule, assuming fpr increasing with threshold direction (it will be)
    let auc = 0;
    for (let i = 1; i < fpr.length; i++) {
      const dx = fpr[i] - fpr[i - 1];
      const yAvg = (tpr[i] + tpr[i - 1]) / 2;
      auc += dx * yAvg;
    }
    return Math.abs(auc);
  }

  function confusionAndPRF1(labels, probs, threshold) {
    let tp = 0,
      fp = 0,
      tn = 0,
      fn = 0;
    for (let i = 0; i < labels.length; i++) {
      const pred = probs[i] >= threshold ? 1 : 0;
      const y = labels[i];
      if (y === 1 && pred === 1) tp++;
      if (y === 0 && pred === 1) fp++;
      if (y === 0 && pred === 0) tn++;
      if (y === 1 && pred === 0) fn++;
    }
    const acc = (tp + tn) / Math.max(1, tp + tn + fp + fn);
    const precision = tp / Math.max(1, tp + fp);
    const recall = tp / Math.max(1, tp + fn);
    const f1 = (2 * precision * recall) / Math.max(1e-12, precision + recall);
    return { tp, fp, tn, fn, acc, precision, recall, f1 };
  }

  function renderMetricsTable(m) {
    const thead = el.metricsTable.querySelector("thead");
    const tbody = el.metricsTable.querySelector("tbody");
    thead.innerHTML = "";
    tbody.innerHTML = "";

    // Confusion matrix section
    const trh = document.createElement("tr");
    ["", "Pred 1", "Pred 0"].forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h;
      trh.appendChild(th);
    });
    thead.appendChild(trh);

    const row1 = document.createElement("tr");
    row1.appendChild(td("Actual 1"));
    row1.appendChild(td(String(m.tp)));
    row1.appendChild(td(String(m.fn)));
    tbody.appendChild(row1);

    const row0 = document.createElement("tr");
    row0.appendChild(td("Actual 0"));
    row0.appendChild(td(String(m.fp)));
    row0.appendChild(td(String(m.tn)));
    tbody.appendChild(row0);

    // Spacer
    const spacer = document.createElement("tr");
    spacer.appendChild(td("—"));
    spacer.appendChild(td("—"));
    spacer.appendChild(td("—"));
    tbody.appendChild(spacer);

    // Metrics
    const metrics = [
      ["Accuracy", m.acc],
      ["Precision", m.precision],
      ["Recall", m.recall],
      ["F1", m.f1],
    ];
    for (const [name, val] of metrics) {
      const tr = document.createElement("tr");
      tr.appendChild(td(name));
      tr.appendChild(td(val.toFixed(4)));
      tr.appendChild(td(""));
      tbody.appendChild(tr);
    }

    function td(text) {
      const cell = document.createElement("td");
      cell.textContent = text;
      return cell;
    }
  }

  function renderROC(fpr, tpr) {
    el.visROC.innerHTML = "";
    const points = fpr.map((x, i) => ({ x, y: tpr[i] }));
    tfvis.render.linechart(
      el.visROC,
      { values: [points], series: ["ROC"] },
      { title: "ROC Curve (validation)", xLabel: "FPR", yLabel: "TPR", width: 520, height: 360 }
    );
  }

  async function onEvaluate() {
    if (!state.model || !state.tensors.xVal) {
      alertUser("Train a model first.");
      return;
    }

    setPill(el.trainPill, "Train", "evaluating…");

    const { probs, labels } = await computeValProbsAndLabels();
    state.eval.valProbs = probs;
    state.eval.valLabels = labels;

    const roc = rocCurve(labels, probs, 101);
    const auc = aucFromRoc(roc.fpr, roc.tpr);

    state.eval.roc = roc;
    state.eval.auc = auc;

    setPill(el.aucPill, "AUC", auc.toFixed(4));
    renderROC(roc.fpr, roc.tpr);

    // Render metrics at current threshold
    const thr = Number(el.thresholdSlider.value);
    state.pred.threshold = thr;
    const m = confusionAndPRF1(labels, probs, thr);
    renderMetricsTable(m);

    // Also show gate importance here (useful if user presses Evaluate after training).
    renderGateImportance();

    setPill(el.trainPill, "Train", "evaluation ready ✅");
  }

  function onThresholdChange() {
    const thr = Number(el.thresholdSlider.value);
    state.pred.threshold = thr;
    el.thresholdLabel.textContent = thr.toFixed(2);

    if (!state.eval.valProbs || !state.eval.valLabels) return;
    const m = confusionAndPRF1(state.eval.valLabels, state.eval.valProbs, thr);
    renderMetricsTable(m);
  }

  // ----------------------------
  // Prediction + Export
  // ----------------------------
  async function onPredictTest() {
    if (!state.model || !state.tensors.xTest) {
      alertUser("Need preprocessing + trained model before prediction.");
      return;
    }
    setPill(el.predPill, "Predict", "predicting…");
    try {
      const probsT = state.model.predict(state.tensors.xTest);
      const probs = await probsT.data();
      probsT.dispose();
      state.pred.testProbs = Float32Array.from(probs);
      setPill(el.predPill, "Predict", `done ✅ (n=${state.pred.testProbs.length})`);
    } catch (e) {
      console.error(e);
      alertUser("Prediction failed. Check console.");
      setPill(el.predPill, "Predict", "failed");
    }
  }

  function toCSV(rows, header) {
    const lines = [];
    lines.push(header.join(","));
    for (const r of rows) {
      lines.push(header.map((h) => r[h]).join(","));
    }
    return lines.join("\n");
  }

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

  function onExportSubmission() {
    if (!state.pred.testProbs || !state.tensors.testPassengerIds) {
      alertUser("Run Predict Test first.");
      return;
    }
    const thr = state.pred.threshold;
    const rows = [];
    for (let i = 0; i < state.pred.testProbs.length; i++) {
      rows.push({
        PassengerId: state.tensors.testPassengerIds[i],
        Survived: state.pred.testProbs[i] >= thr ? 1 : 0,
      });
    }
    const csv = toCSV(rows, ["PassengerId", "Survived"]);
    downloadBlob("submission.csv", new Blob([csv], { type: "text/csv;charset=utf-8" }));
    setPill(el.exportPill, "Export", "submission.csv downloaded ✅");
  }

  function onExportProbs() {
    if (!state.pred.testProbs || !state.tensors.testPassengerIds) {
      alertUser("Run Predict Test first.");
      return;
    }
    const rows = [];
    for (let i = 0; i < state.pred.testProbs.length; i++) {
      rows.push({
        PassengerId: state.tensors.testPassengerIds[i],
        Probability: Number(state.pred.testProbs[i]).toFixed(6),
      });
    }
    const csv = toCSV(rows, ["PassengerId", "Probability"]);
    downloadBlob("probabilities.csv", new Blob([csv], { type: "text/csv;charset=utf-8" }));
    setPill(el.exportPill, "Export", "probabilities.csv downloaded ✅");
  }

  function onExportDebug() {
    if (!state.prep.summary) {
      alertUser("Run preprocessing first.");
      return;
    }
    const json = JSON.stringify(state.prep.summary, null, 2);
    downloadBlob(
      "preprocessing_summary.json",
      new Blob([json], { type: "application/json;charset=utf-8" })
    );
    setPill(el.exportPill, "Export", "preprocessing_summary.json downloaded ✅");
  }

  // ----------------------------
  // Button wiring
  // ----------------------------
  async function onLoadInspect() {
    const loaded = await loadFromFileInputs();
    if (!loaded) return;

    const { trainRows, testRows } = loaded;

    // KPIs + preview + missing
    renderKPIs(trainRows, testRows);
    renderPreviewTable(trainRows, 10);

    // Quick EDA charts
    el.visEDA.innerHTML = "";
    renderQuickEDA(trainRows);

    setPill(el.statusPill, "Status", "inspect ready ✅");
  }

  function init() {
    // Default UI states
    setPill(el.statusPill, "Status", "waiting for files");
    setPill(el.prepPill, "Preprocess", "idle");
    setPill(el.modelPill, "Model", "not built");
    setPill(el.trainPill, "Train", "idle");
    setPill(el.aucPill, "AUC", "—");
    setPill(el.predPill, "Predict", "idle");
    setPill(el.exportPill, "Export", "idle");
    el.thresholdLabel.textContent = Number(el.thresholdSlider.value).toFixed(2);

    el.btnLoad.addEventListener("click", onLoadInspect);
    el.btnPreprocess.addEventListener("click", runPreprocessing);
    el.btnBuildModel.addEventListener("click", onBuildModel);
    el.btnSaveModel.addEventListener("click", onSaveModel);
    el.btnTrain.addEventListener("click", onTrain);
    el.btnEvaluate.addEventListener("click", onEvaluate);

    el.thresholdSlider.addEventListener("input", onThresholdChange);

    el.btnPredict.addEventListener("click", onPredictTest);
    el.btnExportSubmission.addEventListener("click", onExportSubmission);
    el.btnExportProbs.addEventListener("click", onExportProbs);
    el.btnExportDebug.addEventListener("click", onExportDebug);

    // Helpful status updates when files selected
    const maybe = () => {
      const ok = !!el.trainFile.files?.[0] && !!el.testFile.files?.[0];
      setPill(el.statusPill, "Status", ok ? "files selected — click Load & Inspect" : "waiting for files");
    };
    el.trainFile.addEventListener("change", maybe);
    el.testFile.addEventListener("change", maybe);
  }

  init();
})();
```
