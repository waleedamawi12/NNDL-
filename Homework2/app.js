// app.js
// Titanic TF.js — Browser-only shallow binary classifier (no server).
// IMPORTANT: CSV parsing here correctly handles quoted commas (e.g., "Braund, Mr. Owen Harris").
// Schema swap point: change FEATURES / TARGET / ID if you reuse for another dataset.

(() => {
  // -------------------------
  // Schema (swap here to reuse)
  // -------------------------
  const ID_COL = "PassengerId";
  const TARGET_COL = "Survived"; // 0/1
  const BASE_FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"];

  // -------------------------
  // UI elements
  // -------------------------
  const el = (id) => document.getElementById(id);

  const trainFileEl = el("https://waleedamawi12.github.io/NNDL-/Homework1/train.csv");
  const testFileEl = el("https://waleedamawi12.github.io/NNDL-/Homework1/test.csv");

  const btnAutoLoad = el("btnAutoLoad");
  const btnClear = el("btnClear");
  const btnPreprocess = el("btnPreprocess");
  const btnBuildModel = el("btnBuildModel");
  const btnTrain = el("btnTrain");
  const btnStop = el("btnStop");
  const btnEvaluate = el("btnEvaluate");
  const btnPredict = el("btnPredict");
  const btnDownloadSubmission = el("btnDownloadSubmission");
  const btnDownloadProbs = el("btnDownloadProbs");
  const btnDownloadModel = el("btnDownloadModel");

  const chkFamily = el("chkFamily");
  const chkAlone = el("chkAlone");

  const loadStatus = el("loadStatus");
  const prepStatus = el("prepStatus");
  const modelStatus = el("modelStatus");
  const trainStatus = el("trainStatus");
  const evalStatus = el("evalStatus");
  const predStatus = el("predStatus");
  const exportStatus = el("exportStatus");

  const trainShapeEl = el("trainShape");
  const testShapeEl = el("testShape");
  const missingTop3El = el("missingTop3");
  const featSizeEl = el("featSize");
  const previewTableEl = el("previewTable");

  const thresholdEl = el("threshold");
  const thrLabelEl = el("thrLabel");

  const visCharts = el("visCharts");
  const visFit = el("visFit");
  const metricsBox = el("metricsBox");
  const rocBox = el("rocBox");

  // -------------------------
  // App state
  // -------------------------
  const state = {
    rawTrain: null,
    rawTest: null,
    prep: null, // fitted preprocessing params
    Xtrain: null,
    ytrain: null,
    Xval: null,
    yval: null,
    valProbs: null,
    valPreds: null,
    model: null,
    testIds: null,
    testProbs: null,
    testPreds: null,
    stopRequested: false
  };

  // -------------------------
  // Helpers (status, alerts)
  // -------------------------
  function setStatus(node, text) {
    node.textContent = text;
  }
  function warn(msg) {
    alert(msg);
  }
  function clamp01(x) {
    return Math.max(0, Math.min(1, x));
  }

  // -------------------------
  // CSV parser (handles quoted commas + newlines)
  // -------------------------
  function parseCSV(text) {
    // Returns array of rows (array of fields). Handles:
    // - commas inside quotes
    // - escaped quotes ("")
    // - CRLF / LF line endings
    const rows = [];
    let row = [];
    let field = "";
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      const next = text[i + 1];

      if (inQuotes) {
        if (c === '"' && next === '"') {
          field += '"';
          i++; // skip second quote
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
          // push row if not empty (avoid trailing blank line noise)
          if (!(row.length === 1 && row[0] === "")) rows.push(row);
          row = [];
        } else if (c === "\r") {
          // ignore \r (CRLF)
        } else {
          field += c;
        }
      }
    }
    // flush last field
    row.push(field);
    if (!(row.length === 1 && row[0] === "")) rows.push(row);

    // Normalize: remove completely empty last row if any
    while (rows.length && rows[rows.length - 1].every((v) => (v ?? "").trim() === "")) rows.pop();

    return rows;
  }

  function csvToObjects(text) {
    const rows = parseCSV(text);
    if (!rows.length) throw new Error("CSV is empty.");
    const header = rows[0].map((h) => (h ?? "").trim());
    const data = [];
    for (let r = 1; r < rows.length; r++) {
      const obj = {};
      for (let c = 0; c < header.length; c++) {
        obj[header[c]] = rows[r][c] ?? "";
      }
      data.push(obj);
    }
    return data;
  }

  // -------------------------
  // Loaders
  // -------------------------
  async function readFileAsText(file) {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = () => reject(new Error("File read failed."));
      fr.readAsText(file);
    });
  }

  async function loadCSVFromFileInput(fileEl) {
    if (!fileEl.files || !fileEl.files[0]) return null;
    const text = await readFileAsText(fileEl.files[0]);
    return csvToObjects(text);
  }

  async function loadCSVFromFetch(path) {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) throw new Error(`Fetch failed: ${path} (${res.status})`);
    const text = await res.text();
    return csvToObjects(text);
  }

  async function loadData({ preferUploads = true } = {}) {
    setStatus(loadStatus, "Status: loading CSV files…");

    try {
      // Prefer file uploads if present (optional override)
      let train = null;
      let test = null;

      if (preferUploads) {
        train = await loadCSVFromFileInput(trainFileEl);
        test = await loadCSVFromFileInput(testFileEl);
      }

      // Auto-load from repo defaults if not provided
      if (!train) train = await loadCSVFromFetch("./train.csv");
      if (!test) test = await loadCSVFromFetch("./test.csv");

      // Basic schema sanity
      if (!train.length || !test.length) throw new Error("Train/Test data is empty.");
      if (!(TARGET_COL in train[0])) throw new Error(`Missing target column '${TARGET_COL}' in train.csv`);
      if (!(ID_COL in train[0]) || !(ID_COL in test[0])) throw new Error(`Missing id column '${ID_COL}'`);

      state.rawTrain = train;
      state.rawTest = test;

      renderInspection();
      renderCharts();

      setStatus(loadStatus, `Status: loaded ✓ (train=${train.length} rows, test=${test.length} rows)`);
      setStatus(prepStatus, "Not preprocessed yet.");
      setStatus(modelStatus, "Model: not built.");
      setStatus(trainStatus, "Train: idle.");
      setStatus(evalStatus, "Evaluate: idle.");
      setStatus(predStatus, "Predict: idle.");
      setStatus(exportStatus, "Export: idle.");

      // Enable next step
      btnPreprocess.disabled = false;
      btnBuildModel.disabled = true;
      btnTrain.disabled = true;
      btnEvaluate.disabled = true;
      btnPredict.disabled = true;
      btnDownloadSubmission.disabled = true;
      btnDownloadProbs.disabled = true;
      btnDownloadModel.disabled = true;

      // Reset state downstream
      resetPipelineFrom("prep");

    } catch (err) {
      console.error(err);
      setStatus(loadStatus, `Status: error — ${err.message}`);
      warn(`Data load error:\n${err.message}\n\nTip: Make sure train.csv/test.csv are in the same folder as index.html on GitHub Pages.`);
    }
  }

  function resetPipelineFrom(stage) {
    // stage: "prep" | "model" | "train" | "predict"
    if (stage === "prep") {
      state.prep = null;
      state.Xtrain = state.ytrain = state.Xval = state.yval = null;
      state.valProbs = state.valPreds = null;
      state.testIds = state.testProbs = state.testPreds = null;
      state.model = null;
      clearVis();
      featSizeEl.textContent = "—";
    }
  }

  function clearVis() {
    visFit.innerHTML = "";
    metricsBox.innerHTML = "";
    rocBox.innerHTML = "";
  }

  // -------------------------
  // Inspection UI
  // -------------------------
  function renderInspection() {
    const train = state.rawTrain;
    const test = state.rawTest;
    if (!train || !test) return;

    trainShapeEl.textContent = `${train.length} / ${Object.keys(train[0]).length}`;
    testShapeEl.textContent = `${test.length} / ${Object.keys(test[0]).length}`;

    // Missingness (overall on train for readability)
    const cols = Object.keys(train[0]);
    const missing = cols.map((c) => {
      let m = 0;
      for (const r of train) {
        const v = (r[c] ?? "").trim();
        if (v === "" || v.toLowerCase() === "null" || v.toLowerCase() === "nan") m++;
      }
      return { col: c, pct: (100 * m) / train.length };
    }).sort((a, b) => b.pct - a.pct);

    const top3 = missing.slice(0, 3).map((d) => `${d.col}: ${d.pct.toFixed(1)}%`).join(" • ");
    missingTop3El.textContent = top3 || "—";

    // Preview table (first 10 rows of train)
    const preview = train.slice(0, 10);
    previewTableEl.innerHTML = makeTable(preview, [
      "PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"
    ]);
  }

  function makeTable(rows, preferredCols = null) {
    if (!rows || !rows.length) return "<div class='muted'>No rows.</div>";
    const allCols = Object.keys(rows[0]);
    const cols = (preferredCols || allCols).filter((c) => allCols.includes(c));
    const head = `<tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr>`;
    const body = rows.map((r) => {
      return `<tr>${cols.map((c) => `<td>${escapeHtml(String(r[c] ?? ""))}</td>`).join("")}</tr>`;
    }).join("");
    return `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (m) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;" }[m]));
  }

  function renderCharts() {
    const train = state.rawTrain;
    if (!train) return;
    visCharts.innerHTML = "";

    // Survival by Sex
    const bySex = groupRate(train, "Sex", TARGET_COL);
    tfvis.render.barchart(
      visCharts,
      { values: bySex.map((d) => ({ x: d.key, y: d.rate })) },
      { title: "Survival rate by Sex", xLabel: "Sex", yLabel: "Rate", height: 180 }
    );

    // Survival by Pclass
    const byPclass = groupRate(train, "Pclass", TARGET_COL);
    tfvis.render.barchart(
      visCharts,
      { values: byPclass.map((d) => ({ x: d.key, y: d.rate })) },
      { title: "Survival rate by Pclass", xLabel: "Pclass", yLabel: "Rate", height: 180 }
    );
  }

  function groupRate(rows, keyCol, targetCol) {
    const map = new Map();
    for (const r of rows) {
      const k = (r[keyCol] ?? "").trim();
      const y = Number((r[targetCol] ?? "").trim());
      if (!map.has(k)) map.set(k, { key: k, n: 0, s: 0 });
      const obj = map.get(k);
      obj.n += 1;
      obj.s += (y === 1 ? 1 : 0);
    }
    const out = Array.from(map.values()).map((d) => ({ key: d.key, rate: d.n ? d.s / d.n : 0 }));
    // Nice ordering for Pclass
    out.sort((a, b) => {
      const na = Number(a.key), nb = Number(b.key);
      if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
      return String(a.key).localeCompare(String(b.key));
    });
    return out;
  }

  // -------------------------
  // Preprocessing
  // -------------------------
  function toNumberOrNull(v) {
    const s = (v ?? "").toString().trim();
    if (s === "" || s.toLowerCase() === "null" || s.toLowerCase() === "nan") return null;
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
  }

  function mode(values) {
    const freq = new Map();
    for (const v of values) {
      const key = v == null ? "" : String(v);
      if (!key) continue;
      freq.set(key, (freq.get(key) || 0) + 1);
    }
    let best = null, bestC = -1;
    for (const [k, c] of freq.entries()) {
      if (c > bestC) { best = k; bestC = c; }
    }
    return best;
  }

  function median(nums) {
    const arr = nums.filter((x) => Number.isFinite(x)).slice().sort((a, b) => a - b);
    if (!arr.length) return 0;
    const mid = Math.floor(arr.length / 2);
    return arr.length % 2 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
  }

  function meanStd(nums) {
    const arr = nums.filter((x) => Number.isFinite(x));
    if (!arr.length) return { mean: 0, std: 1 };
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const varr = arr.reduce((a, b) => a + (b - mean) * (b - mean), 0) / arr.length;
    const std = Math.sqrt(varr) || 1;
    return { mean, std };
  }

  function oneHot(categories, value) {
    const out = new Array(categories.length).fill(0);
    const idx = categories.indexOf(value);
    if (idx >= 0) out[idx] = 1;
    return out;
  }

  function fitPreprocessing(trainRows) {
    // Fit only on train. Save params to apply consistently to val/test.
    const ages = trainRows.map((r) => toNumberOrNull(r.Age)).filter((x) => x != null);
    const fares = trainRows.map((r) => toNumberOrNull(r.Fare)).filter((x) => x != null);

    const ageMedian = median(ages);
    const embarkedMode = mode(trainRows.map((r) => (r.Embarked ?? "").trim())) || "S";

    // Standardize only numeric we use (Age/Fare)
    const { mean: ageMean, std: ageStd } = meanStd(trainRows.map((r) => {
      const n = toNumberOrNull(r.Age);
      return n == null ? ageMedian : n;
    }));
    const { mean: fareMean, std: fareStd } = meanStd(trainRows.map((r) => {
      const n = toNumberOrNull(r.Fare);
      return n == null ? 0 : n;
    }));

    // One-hot categories: derive from train
    const sexCats = Array.from(new Set(trainRows.map((r) => (r.Sex ?? "").trim()).filter(Boolean))).sort();
    const pclassCats = Array.from(new Set(trainRows.map((r) => (r.Pclass ?? "").trim()).filter(Boolean))).sort((a, b) => Number(a) - Number(b));
    const embarkedCats = Array.from(new Set(trainRows.map((r) => {
      const e = (r.Embarked ?? "").trim();
      return e ? e : embarkedMode;
    }).filter(Boolean))).sort();

    return {
      ageMedian,
      embarkedMode,
      scaler: { ageMean, ageStd, fareMean, fareStd },
      cats: { sexCats, pclassCats, embarkedCats }
    };
  }

  function featurizeRows(rows, prep, { addFamilySize, addIsAlone, includeTarget } = {}) {
    const { ageMedian, embarkedMode, scaler, cats } = prep;
    const featureNames = [];

    // One-hot names
    for (const c of cats.pclassCats) featureNames.push(`Pclass_${c}`);
    for (const c of cats.sexCats) featureNames.push(`Sex_${c}`);
    featureNames.push("Age_z");
    featureNames.push("SibSp");
    featureNames.push("Parch");
    featureNames.push("Fare_z");
    for (const c of cats.embarkedCats) featureNames.push(`Embarked_${c}`);

    if (addFamilySize) featureNames.push("FamilySize");
    if (addIsAlone) featureNames.push("IsAlone");

    const X = [];
    const y = [];
    const ids = [];

    for (const r of rows) {
      const pclass = (r.Pclass ?? "").trim();
      const sex = (r.Sex ?? "").trim();
      const sibsp = toNumberOrNull(r.SibSp) ?? 0;
      const parch = toNumberOrNull(r.Parch) ?? 0;

      const rawAge = toNumberOrNull(r.Age);
      const age = rawAge == null ? ageMedian : rawAge;

      const rawFare = toNumberOrNull(r.Fare);
      const fare = rawFare == null ? 0 : rawFare;

      const embarkedRaw = (r.Embarked ?? "").trim();
      const embarked = embarkedRaw ? embarkedRaw : embarkedMode;

      const ageZ = (age - scaler.ageMean) / (scaler.ageStd || 1);
      const fareZ = (fare - scaler.fareMean) / (scaler.fareStd || 1);

      const vec = []
        .concat(oneHot(cats.pclassCats, pclass))
        .concat(oneHot(cats.sexCats, sex))
        .concat([ageZ, sibsp, parch, fareZ])
        .concat(oneHot(cats.embarkedCats, embarked));

      let familySize = null;
      if (addFamilySize || addIsAlone) {
        familySize = sibsp + parch + 1;
      }
      if (addFamilySize) vec.push(familySize);
      if (addIsAlone) vec.push(familySize === 1 ? 1 : 0);

      X.push(vec);

      if (includeTarget) {
        const yy = toNumberOrNull(r[TARGET_COL]);
        y.push(yy == null ? 0 : (yy >= 1 ? 1 : 0));
      }

      ids.push((r[ID_COL] ?? "").trim());
    }

    return { X, y, ids, featureNames };
  }

  function preprocess() {
    const train = state.rawTrain;
    const test = state.rawTest;
    if (!train || !test) {
      warn("Load train.csv and test.csv first.");
      return;
    }

    try {
      const addFamilySize = chkFamily.checked;
      const addIsAlone = chkAlone.checked;

      const prep = fitPreprocessing(train);
      const trainFeat = featurizeRows(train, prep, { addFamilySize, addIsAlone, includeTarget: true });
      const testFeat = featurizeRows(test, prep, { addFamilySize, addIsAlone, includeTarget: false });

      // Ensure same feature size
      if (trainFeat.X[0].length !== testFeat.X[0].length) {
        throw new Error("Feature size mismatch between train and test after preprocessing.");
      }

      state.prep = { ...prep, featureNames: trainFeat.featureNames, addFamilySize, addIsAlone };
      state.testIds = testFeat.ids;

      featSizeEl.textContent = String(trainFeat.X[0].length);

      // Stratified split (80/20)
      const { Xtrain, ytrain, Xval, yval } = stratifiedSplit(trainFeat.X, trainFeat.y, 0.2);

      // Store tensors later (we keep arrays until training to avoid memory pressure)
      state.Xtrain = Xtrain;
      state.ytrain = ytrain;
      state.Xval = Xval;
      state.yval = yval;

      setStatus(prepStatus,
        `Preprocess ✓
- Impute Age median=${prep.ageMedian.toFixed(3)}
- Impute Embarked mode=${prep.embarkedMode}
- Standardize Age/Fare
- One-hot: Pclass(${prep.cats.pclassCats.length}), Sex(${prep.cats.sexCats.length}), Embarked(${prep.cats.embarkedCats.length})
- Features=${trainFeat.X[0].length}
- Split: train=${Xtrain.length}, val=${Xval.length}`
      );

      // Enable next step
      btnBuildModel.disabled = false;
      btnTrain.disabled = true;
      btnEvaluate.disabled = true;
      btnPredict.disabled = true;
      btnDownloadSubmission.disabled = true;
      btnDownloadProbs.disabled = true;
      btnDownloadModel.disabled = true;

      // reset downstream
      state.model = null;
      state.valProbs = state.valPreds = null;
      state.testProbs = state.testPreds = null;
      clearVis();

      // Show feature names in console for debugging
      console.log("Feature order:", state.prep.featureNames);

    } catch (err) {
      console.error(err);
      setStatus(prepStatus, `Preprocess: error — ${err.message}`);
      warn(`Preprocessing error:\n${err.message}`);
    }
  }

  function stratifiedSplit(X, y, valFrac = 0.2) {
    const pos = [];
    const neg = [];
    for (let i = 0; i < y.length; i++) {
      (y[i] === 1 ? pos : neg).push(i);
    }

    shuffleInPlace(pos);
    shuffleInPlace(neg);

    const posValN = Math.max(1, Math.floor(pos.length * valFrac));
    const negValN = Math.max(1, Math.floor(neg.length * valFrac));

    const valIdx = pos.slice(0, posValN).concat(neg.slice(0, negValN));
    const trainIdx = pos.slice(posValN).concat(neg.slice(negValN));

    shuffleInPlace(valIdx);
    shuffleInPlace(trainIdx);

    const Xtrain = trainIdx.map((i) => X[i]);
    const ytrain = trainIdx.map((i) => y[i]);
    const Xval = valIdx.map((i) => X[i]);
    const yval = valIdx.map((i) => y[i]);

    return { Xtrain, ytrain, Xval, yval };
  }

  function shuffleInPlace(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  // -------------------------
  // Model
  // -------------------------
  function buildModel() {
    if (!state.prep || !state.Xtrain) {
      warn("Preprocess data first.");
      return;
    }

    const inputDim = state.Xtrain[0].length;

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [inputDim] })); // single hidden layer
    model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

    model.compile({
      optimizer: tf.train.adam(),
      loss: "binaryCrossentropy",
      metrics: ["accuracy"]
    });

    state.model = model;

    // Show summary
    const lines = [];
    model.summary(80, undefined, (line) => lines.push(line));
    setStatus(modelStatus, "Model built ✓\n" + lines.join("\n"));

    btnTrain.disabled = false;
    btnDownloadModel.disabled = false;
  }

  // -------------------------
  // Training (tfjs-vis + early stopping)
  // -------------------------
  async function train() {
    if (!state.model) {
      warn("Build the model first.");
      return;
    }

    state.stopRequested = false;
    btnTrain.disabled = true;
    btnStop.disabled = false;
    btnEvaluate.disabled = true;
    btnPredict.disabled = true;

    setStatus(trainStatus, "Training…");

    clearVis();
    metricsBox.innerHTML = "";
    rocBox.innerHTML = "";

    const XtrainT = tf.tensor2d(state.Xtrain);
    const ytrainT = tf.tensor2d(state.ytrain, [state.ytrain.length, 1]);
    const XvalT = tf.tensor2d(state.Xval);
    const yvalT = tf.tensor2d(state.yval, [state.yval.length, 1]);

    const surface = { name: "Training", tab: "Fit" };
    visFit.innerHTML = ""; // ensure empty
    tfvis.visor().open();

    // Early stopping on val_loss
    const patience = 5;
    let best = Number.POSITIVE_INFINITY;
    let wait = 0;

    const callbacks = [
      tfvis.show.fitCallbacks(
        { name: "Loss & Accuracy", tab: "Fit", styles: { height: "260px" } },
        ["loss", "val_loss", "acc", "val_acc"],
        { callbacks: ["onEpochEnd"] }
      ),
      {
        onEpochEnd: async (epoch, logs) => {
          const valLoss = logs.val_loss;
          if (valLoss + 1e-6 < best) {
            best = valLoss;
            wait = 0;
          } else {
            wait += 1;
          }

          if (wait >= patience) {
            state.model.stopTraining = true;
          }
          if (state.stopRequested) {
            state.model.stopTraining = true;
          }

          setStatus(trainStatus,
            `Epoch ${epoch + 1} — loss=${logs.loss.toFixed(4)}, acc=${(logs.acc ?? logs.accuracy).toFixed(4)}, ` +
            `val_loss=${logs.val_loss.toFixed(4)}, val_acc=${(logs.val_acc ?? logs.val_accuracy).toFixed(4)}`
          );
        }
      }
    ];

    try {
      await state.model.fit(XtrainT, ytrainT, {
        epochs: 50,
        batchSize: 32,
        shuffle: true,
        validationData: [XvalT, yvalT],
        callbacks
      });

      setStatus(trainStatus, `Training completed ✓ (best val_loss=${best.toFixed(4)})`);

      // After training, compute validation probabilities once (used for ROC + threshold slider)
      const probs = state.model.predict(XvalT);
      const probArr = Array.from((await probs.data()));
      state.valProbs = probArr;
      state.valPreds = null; // computed at threshold when needed

      btnEvaluate.disabled = false;
      btnPredict.disabled = false;

    } catch (err) {
      console.error(err);
      setStatus(trainStatus, `Training error — ${err.message}`);
      warn(`Training error:\n${err.message}`);
    } finally {
      btnTrain.disabled = false;
      btnStop.disabled = true;
      tf.dispose([XtrainT, ytrainT, XvalT, yvalT]);
    }
  }

  // -------------------------
  // Metrics: ROC/AUC + threshold slider + confusion matrix
  // -------------------------
  function computeRocAuc(yTrue, yProb) {
    // Build ROC points by sorting by probability desc.
    const pairs = yProb.map((p, i) => ({ p, y: yTrue[i] }));
    pairs.sort((a, b) => b.p - a.p);

    const P = yTrue.reduce((s, v) => s + (v === 1 ? 1 : 0), 0);
    const N = yTrue.length - P;

    let tp = 0, fp = 0;
    let prevP = Infinity;

    const points = [{ fpr: 0, tpr: 0 }];
    for (const item of pairs) {
      if (item.p !== prevP) {
        points.push({ fpr: N ? fp / N : 0, tpr: P ? tp / P : 0 });
        prevP = item.p;
      }
      if (item.y === 1) tp++; else fp++;
    }
    points.push({ fpr: 1, tpr: 1 });

    // AUC via trapezoidal rule over FPR
    let auc = 0;
    for (let i = 1; i < points.length; i++) {
      const x1 = points[i - 1].fpr, y1 = points[i - 1].tpr;
      const x2 = points[i].fpr, y2 = points[i].tpr;
      auc += (x2 - x1) * (y1 + y2) / 2;
    }

    return { points, auc };
  }

  function confusionAndMetrics(yTrue, yProb, thr) {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (let i = 0; i < yTrue.length; i++) {
      const pred = yProb[i] >= thr ? 1 : 0;
      const y = yTrue[i];
      if (pred === 1 && y === 1) tp++;
      else if (pred === 1 && y === 0) fp++;
      else if (pred === 0 && y === 0) tn++;
      else fn++;
    }
    const acc = (tp + tn) / (tp + tn + fp + fn);
    const prec = tp + fp === 0 ? 0 : tp / (tp + fp);
    const rec = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = (prec + rec) === 0 ? 0 : (2 * prec * rec) / (prec + rec);
    return { tp, tn, fp, fn, acc, prec, rec, f1 };
  }

  function renderConfusion(m) {
    // Simple HTML table + KPI
    metricsBox.innerHTML = `
      <div style="display:grid; gap:10px">
        <table>
          <thead>
            <tr><th></th><th>Predicted 1</th><th>Predicted 0</th></tr>
          </thead>
          <tbody>
            <tr><th>Actual 1</th><td>${m.tp}</td><td>${m.fn}</td></tr>
            <tr><th>Actual 0</th><td>${m.fp}</td><td>${m.tn}</td></tr>
          </tbody>
        </table>
        <div class="kpi">
          <div class="box"><div class="muted">Accuracy</div><div class="num">${(m.acc * 100).toFixed(2)}%</div></div>
          <div class="box"><div class="muted">Precision</div><div class="num">${m.prec.toFixed(3)}</div></div>
          <div class="box"><div class="muted">Recall</div><div class="num">${m.rec.toFixed(3)}</div></div>
          <div class="box"><div class="muted">F1</div><div class="num">${m.f1.toFixed(3)}</div></div>
        </div>
      </div>
    `;
  }

  function renderRoc(points, auc) {
    rocBox.innerHTML = "";
    const data = points.map((pt) => ({ x: pt.fpr, y: pt.tpr }));

    tfvis.render.linechart(
      rocBox,
      { values: [data], series: ["ROC"] },
      {
        title: `ROC Curve (AUC = ${auc.toFixed(4)})`,
        xLabel: "False Positive Rate",
        yLabel: "True Positive Rate",
        height: 260
      }
    );
  }

  function evaluate() {
    if (!state.valProbs || !state.yval) {
      warn("Train the model first to compute validation probabilities.");
      return;
    }
    const thr = clamp01(Number(thresholdEl.value));
    thrLabelEl.textContent = thr.toFixed(2);

    const { points, auc } = computeRocAuc(state.yval, state.valProbs);
    renderRoc(points, auc);

    const m = confusionAndMetrics(state.yval, state.valProbs, thr);
    renderConfusion(m);

    setStatus(evalStatus, `Evaluation ✓ (AUC=${auc.toFixed(4)}) at threshold=${thr.toFixed(2)}`);
  }

  // -------------------------
  // Prediction + Export
  // -------------------------
  async function predictOnTest() {
    if (!state.model || !state.prep || !state.rawTest) {
      warn("Load data, preprocess, and train (or at least build) the model first.");
      return;
    }

    try {
      setStatus(predStatus, "Predicting on test.csv…");

      const addFamilySize = state.prep.addFamilySize;
      const addIsAlone = state.prep.addIsAlone;

      const testFeat = featurizeRows(state.rawTest, state.prep, {
        addFamilySize,
        addIsAlone,
        includeTarget: false
      });

      const XtestT = tf.tensor2d(testFeat.X);
      const probsT = state.model.predict(XtestT);
      const probs = Array.from(await probsT.data());

      state.testProbs = probs;

      const thr = clamp01(Number(thresholdEl.value));
      state.testPreds = probs.map((p) => (p >= thr ? 1 : 0));

      tf.dispose([XtestT, probsT]);

      setStatus(predStatus, `Predict ✓ (n=${probs.length}) using threshold=${thr.toFixed(2)}`);

      btnDownloadSubmission.disabled = false;
      btnDownloadProbs.disabled = false;

    } catch (err) {
      console.error(err);
      setStatus(predStatus, `Predict: error — ${err.message}`);
      warn(`Prediction error:\n${err.message}`);
    }
  }

  function downloadCSV(filename, rows) {
    const header = Object.keys(rows[0] || {});
    const esc = (v) => {
      const s = String(v ?? "");
      // CSV escaping: wrap if contains comma/quote/newline
      if (/[",\n\r]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
      return s;
    };
    const lines = [
      header.join(","),
      ...rows.map((r) => header.map((h) => esc(r[h])).join(","))
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  function exportSubmission() {
    if (!state.testPreds || !state.testIds) {
      warn("Run prediction first.");
      return;
    }
    const rows = state.testIds.map((id, i) => ({
      PassengerId: id,
      Survived: state.testPreds[i]
    }));
    downloadCSV("submission.csv", rows);
    setStatus(exportStatus, "Export ✓ downloaded submission.csv");
  }

  function exportProbabilities() {
    if (!state.testProbs || !state.testIds) {
      warn("Run prediction first.");
      return;
    }
    const rows = state.testIds.map((id, i) => ({
      PassengerId: id,
      Survived_Prob: state.testProbs[i]
    }));
    downloadCSV("probabilities.csv", rows);
    setStatus(exportStatus, "Export ✓ downloaded probabilities.csv");
  }

  async function saveModel() {
    if (!state.model) {
      warn("Build a model first.");
      return;
    }
    try {
      setStatus(exportStatus, "Saving model…");
      await state.model.save("downloads://titanic-tfjs");
      setStatus(exportStatus, "Model saved ✓ (downloads://titanic-tfjs)");
    } catch (err) {
      console.error(err);
      setStatus(exportStatus, `Model save error — ${err.message}`);
      warn(`Model save error:\n${err.message}`);
    }
  }

  // -------------------------
  // Event wiring
  // -------------------------
  btnAutoLoad.addEventListener("click", () => loadData({ preferUploads: true }));
  btnClear.addEventListener("click", () => {
    // Reset everything
    trainFileEl.value = "";
    testFileEl.value = "";
    state.rawTrain = null;
    state.rawTest = null;
    resetPipelineFrom("prep");
    previewTableEl.innerHTML = "";
    visCharts.innerHTML = "";
    trainShapeEl.textContent = "—";
    testShapeEl.textContent = "—";
    missingTop3El.textContent = "—";
    featSizeEl.textContent = "—";
    clearVis();
    setStatus(loadStatus, "Status: cleared. Click Auto-load or upload files.");
    setStatus(prepStatus, "Not preprocessed yet.");
    setStatus(modelStatus, "Model: not built.");
    setStatus(trainStatus, "Train: idle.");
    setStatus(evalStatus, "Evaluate: idle.");
    setStatus(predStatus, "Predict: idle.");
    setStatus(exportStatus, "Export: idle.");
    btnPreprocess.disabled = true;
    btnBuildModel.disabled = true;
    btnTrain.disabled = true;
    btnStop.disabled = true;
    btnEvaluate.disabled = true;
    btnPredict.disabled = true;
    btnDownloadSubmission.disabled = true;
    btnDownloadProbs.disabled = true;
    btnDownloadModel.disabled = true;
  });

  btnPreprocess.addEventListener("click", preprocess);
  btnBuildModel.addEventListener("click", buildModel);
  btnTrain.addEventListener("click", train);
  btnStop.addEventListener("click", () => {
    state.stopRequested = true;
    setStatus(trainStatus, "Stop requested… finishing current epoch.");
  });

  btnEvaluate.addEventListener("click", evaluate);
  thresholdEl.addEventListener("input", () => {
    const thr = clamp01(Number(thresholdEl.value));
    thrLabelEl.textContent = thr.toFixed(2);
    // Live-update confusion matrix if we already have val probs
    if (state.valProbs && state.yval) {
      const m = confusionAndMetrics(state.yval, state.valProbs, thr);
      renderConfusion(m);
    }
  });

  btnPredict.addEventListener("click", predictOnTest);
  btnDownloadSubmission.addEventListener("click", exportSubmission);
  btnDownloadProbs.addEventListener("click", exportProbabilities);
  btnDownloadModel.addEventListener("click", saveModel);

  // Optional: auto-load on page open (repo defaults) without requiring user clicks.
  // If you DON'T want auto-load on page load, comment the next line.
  window.addEventListener("load", () => loadData({ preferUploads: true }));
})();
