// app.js
// “Looks-like-the-example” version (stable):
// - Baseline: MSE(x, y) -> copycat noise.
// - Student: learns an explicit left->right gradient + chess texture + mild smoothness.
// - Adds Row Consistency loss to straighten the gradient (big visual improvement).
//
// Includes:
// - Student architecture radios: compression/transformation/expansion
// - Radio buttons for λSmooth, λDir, λDist, λChess, λRow
//
// NOTE: This is optimized for the visual target. It is not the strict “rearrangement-only/sortedMSE” solution.

// -----------------------------------------------------------------------------
// DOM helpers
const $ = (sel) => document.querySelector(sel);
const logEl = $("#log");
const statusLine = $("#statusLine");
const memBadge = $("#memBadge");

const cvInput = $("#cvInput");
const cvBase = $("#cvBase");
const cvStudent = $("#cvStudent");

const btnStep = $("#btnStep");
const btnAuto = $("#btnAuto");
const btnReset = $("#btnReset");

// -----------------------------------------------------------------------------
// Constants
const H = 16, W = 16;
const SHAPE_4D = [1, H, W, 1];

const STEPS_PER_FRAME = 25;
const RENDER_EVERY = 1;
const LOG_EVERY = 50;

const BASE_LR = 1e-2;
const STUDENT_LR = 1e-2;

const AUTO_MAX_STEPS = 4500;
const PLATEAU_PATIENCE = 1000;
const MIN_DELTA = 1e-8;

// Loss weights (controlled by radios)
// Good defaults for the “close but can be better” phase:
let LAMBDA_SMOOTH = 0.015; // slightly less blur than 0.02
let LAMBDA_DIR = 9.0;      // stronger gradient pull
let LAMBDA_DIST = 2.0;     // mapped to texture strength (see studentLoss)
let LAMBDA_CHESS = 2.5;    // chess/texture strength
let LAMBDA_ROW = 0.5;      // straighten gradient across rows (0.2–0.8)

// -----------------------------------------------------------------------------
// State
let stepCount = 0;
let autoRunning = false;
let rafHandle = null;

let xInput = null;
let baselineModel = null;
let studentModel = null;

let baseOpt = null;
let studentOpt = null;

let studentArchType = "compression";

// Plateau
let bestComboLoss = Infinity;
let stepsSinceBest = 0;

// -----------------------------------------------------------------------------
// Logging / UI
function log(msg, kind = "info") {
  const prefix = kind === "error" ? "✖ " : kind === "ok" ? "✓ " : "• ";
  logEl.textContent = (prefix + msg + "\n" + logEl.textContent).slice(0, 8000);
}
function fmt(x) {
  if (x == null || Number.isNaN(x)) return "—";
  if (!Number.isFinite(x)) return String(x);
  if (Math.abs(x) < 1e-3) return x.toExponential(2);
  return x.toFixed(4);
}
function setStatus({ step, baseLoss, studentLoss }) {
  statusLine.textContent = `Step: ${step} | Baseline loss: ${fmt(baseLoss)} | Student loss: ${fmt(studentLoss)}`;
  memBadge.textContent = `tf.memory: ${tf.memory().numTensors} tensors`;
}

// -----------------------------------------------------------------------------
// Render tensor to 16×16 canvas
function drawTensorToCanvas(t4d, canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const img = ctx.createImageData(W, H);
  const data = img.data;

  const vals = t4d.dataSync();
  for (let i = 0; i < H * W; i++) {
    const v01 = Math.max(0, Math.min(1, vals[i]));
    const v = Math.round(v01 * 255);
    const j = i * 4;
    data[j + 0] = v;
    data[j + 1] = v;
    data[j + 2] = v;
    data[j + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// -----------------------------------------------------------------------------
// Loss pieces
function mse(yTrue, yPred) {
  return tf.mean(tf.losses.meanSquaredError(yTrue, yPred));
}

// Smoothness: penalize horizontal + vertical differences
function smoothness(yPred) {
  return tf.tidy(() => {
    const dx = yPred.slice([0, 0, 0, 0], [1, H, W - 1, 1])
      .sub(yPred.slice([0, 0, 1, 0], [1, H, W - 1, 1]));
    const dy = yPred.slice([0, 0, 0, 0], [1, H - 1, W, 1])
      .sub(yPred.slice([0, 1, 0, 0], [1, H - 1, W, 1]));
    return tf.add(tf.mean(tf.square(dx)), tf.mean(tf.square(dy)));
  });
}

// Target gradient loss: forces left->right gradient
function gradientLoss(yPred) {
  return tf.tidy(() => {
    const target = [];
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) target.push(j / (W - 1));
    }
    const targetTensor = tf.tensor(target, [1, H, W, 1], "float32");
    return tf.mean(tf.square(tf.sub(targetTensor, yPred)));
  });
}

// Chess texture loss (neighbor-based): encourages checker-ish texture
function chessNeighborLoss(yPred) {
  return tf.tidy(() => {
    const left = yPred.slice([0, 0, 0, 0], [1, H, W - 1, 1]);
    const right = yPred.slice([0, 0, 1, 0], [1, H, W - 1, 1]);
    const horizontalDiff = tf.abs(left.sub(right));
    const horizontalLoss = tf.mean(tf.square(horizontalDiff.sub(tf.scalar(0.3))));

    const top = yPred.slice([0, 0, 0, 0], [1, H - 1, W, 1]);
    const bottom = yPred.slice([0, 1, 0, 0], [1, H - 1, W, 1]);
    const verticalDiff = tf.abs(top.sub(bottom));
    const verticalLoss = tf.mean(tf.square(verticalDiff.sub(tf.scalar(0.3))));

    const diag1 = yPred.slice([0, 0, 0, 0], [1, H - 1, W - 1, 1]);
    const diag2 = yPred.slice([0, 1, 1, 0], [1, H - 1, W - 1, 1]);
    const diagonalDiff = tf.abs(diag1.sub(diag2));
    const diagonalLoss = tf.mean(tf.square(diagonalDiff)).mul(tf.scalar(2));

    return horizontalLoss.add(verticalLoss).add(diagonalLoss).div(tf.scalar(3));
  });
}

// Row consistency loss: makes each row follow the same column profile (straightens gradient)
function rowConsistencyLoss(yPred) {
  return tf.tidy(() => {
    // mean over rows (axis=1) -> shape [1, W, 1] then reshape to [1,1,W,1]
    const colMean = tf.mean(yPred, 1).reshape([1, 1, W, 1]);
    const colMeanTile = colMean.tile([1, H, 1, 1]);
    return tf.mean(tf.square(yPred.sub(colMeanTile)));
  });
}

// -----------------------------------------------------------------------------
// Models
function createBaselineModel() {
  const inp = tf.input({ shape: [H, W, 1] });
  const flat = tf.layers.flatten().apply(inp);
  const h1 = tf.layers.dense({ units: 64, activation: "relu" }).apply(flat);
  const outFlat = tf.layers.dense({ units: H * W, activation: "sigmoid" }).apply(h1);
  const out = tf.layers.reshape({ targetShape: [H, W, 1] }).apply(outFlat);
  return tf.model({ inputs: inp, outputs: out, name: "baseline" });
}

function createStudentModel(archType) {
  const inp = tf.input({ shape: [H, W, 1] });

  if (archType === "compression") {
    const flat = tf.layers.flatten().apply(inp);
    const z = tf.layers.dense({ units: 32, activation: "relu" }).apply(flat);
    const outFlat = tf.layers.dense({ units: H * W, activation: "sigmoid" }).apply(z);
    const out = tf.layers.reshape({ targetShape: [H, W, 1] }).apply(outFlat);
    return tf.model({ inputs: inp, outputs: out, name: "student_compression" });
  }

  if (archType === "transformation") {
    let x = inp;
    x = tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: "same", activation: "relu" }).apply(x);
    x = tf.layers.conv2d({ filters: 16, kernelSize: 1, padding: "same", activation: "relu" }).apply(x);
    const out = tf.layers.conv2d({ filters: 1, kernelSize: 1, padding: "same", activation: "sigmoid" }).apply(x);
    return tf.model({ inputs: inp, outputs: out, name: "student_transformation" });
  }

  // expansion
  let x = inp;
  x = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: "same", activation: "relu" }).apply(x);
  x = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: "same", activation: "relu" }).apply(x);
  x = tf.layers.conv2d({ filters: 32, kernelSize: 1, padding: "same", activation: "relu" }).apply(x);
  const out = tf.layers.conv2d({ filters: 1, kernelSize: 1, padding: "same", activation: "sigmoid" }).apply(x);
  return tf.model({ inputs: inp, outputs: out, name: "student_expansion" });
}

// -----------------------------------------------------------------------------
// Losses
function baselineLoss(yTrue, yPred) {
  return mse(yTrue, yPred);
}

function studentLoss(yPred) {
  return tf.tidy(() => {
    // Gradient term (main)
    const Lg = gradientLoss(yPred).mul(tf.scalar(LAMBDA_DIR));

    // Texture (chess) term(s)
    const chess = chessNeighborLoss(yPred);
    const Lc = chess.mul(tf.scalar(LAMBDA_CHESS));

    // “λDist” mapped to texture strength as requested (keeps the 3 knobs meaningful)
    const Ld = chess.mul(tf.scalar(LAMBDA_DIST));

    // Gentle smoothness
    const Ls = smoothness(yPred).mul(tf.scalar(LAMBDA_SMOOTH));

    // Row-consistency to straighten gradient
    const Lrow = rowConsistencyLoss(yPred).mul(tf.scalar(LAMBDA_ROW));

    return tf.addN([Lg, Lc, Ld, Ls, Lrow]);
  });
}

// -----------------------------------------------------------------------------
// Training step
async function trainOneStepReturnLosses() {
  let baseLossVal = NaN;
  let studentLossVal = NaN;

  try {
    // Baseline update
    const baseLossTensor = tf.tidy(() => {
      const vars = baselineModel.trainableWeights.map(w => w.val);
      const { value, grads } = tf.variableGrads(() => {
        const yPred = baselineModel.apply(xInput);
        return baselineLoss(xInput, yPred);
      }, vars);

      baseOpt.applyGradients(grads);
      Object.values(grads).forEach(g => g.dispose());
      return value;
    });
    baseLossVal = (await baseLossTensor.data())[0];
    baseLossTensor.dispose();

    // Student update
    const studentLossTensor = tf.tidy(() => {
      const vars = studentModel.trainableWeights.map(w => w.val);
      const { value, grads } = tf.variableGrads(() => {
        const yPred = studentModel.apply(xInput);
        return studentLoss(yPred);
      }, vars);

      studentOpt.applyGradients(grads);
      Object.values(grads).forEach(g => g.dispose());
      return value;
    });
    studentLossVal = (await studentLossTensor.data())[0];
    studentLossTensor.dispose();

    stepCount++;
    setStatus({ step: stepCount, baseLoss: baseLossVal, studentLoss: studentLossVal });

    if (stepCount % LOG_EVERY === 0 || stepCount === 1) {
      log(`step=${stepCount} | baseline=${fmt(baseLossVal)} | student=${fmt(studentLossVal)}`, "info");
    }

    if (stepCount % RENDER_EVERY === 0) {
      tf.tidy(() => {
        const b = baselineModel.predict(xInput);
        const s = studentModel.predict(xInput);
        drawTensorToCanvas(xInput, cvInput);
        drawTensorToCanvas(b, cvBase);
        drawTensorToCanvas(s, cvStudent);
      });
    }
  } catch (err) {
    log(String(err?.message || err), "error");
  }

  // plateau tracking
  const combo =
    (Number.isFinite(baseLossVal) ? baseLossVal : 0) +
    (Number.isFinite(studentLossVal) ? studentLossVal : 0);

  if (combo + MIN_DELTA < bestComboLoss) {
    bestComboLoss = combo;
    stepsSinceBest = 0;
  } else {
    stepsSinceBest++;
  }

  return { baseLossVal, studentLossVal };
}

// -----------------------------------------------------------------------------
// Auto train loop
async function autoLoop() {
  if (!autoRunning) return;

  for (let i = 0; i < STEPS_PER_FRAME; i++) {
    await trainOneStepReturnLosses();
    await tf.nextFrame();

    if (!autoRunning) return;

    if (stepCount >= AUTO_MAX_STEPS) {
      log(`Auto-stop: reached max steps (${AUTO_MAX_STEPS}).`, "ok");
      stopAuto();
      return;
    }
    if (stepsSinceBest >= PLATEAU_PATIENCE) {
      log(`Auto-stop: plateau (${PLATEAU_PATIENCE} steps without improvement).`, "ok");
      stopAuto();
      return;
    }
  }

  rafHandle = requestAnimationFrame(autoLoop);
}

function startAuto() {
  if (autoRunning) return;
  autoRunning = true;
  btnAuto.textContent = "Auto Train (Stop)";
  log("Auto train started.", "ok");
  rafHandle = requestAnimationFrame(autoLoop);
}

function stopAuto() {
  autoRunning = false;
  btnAuto.textContent = "Auto Train (Start)";
  if (rafHandle != null) cancelAnimationFrame(rafHandle);
  rafHandle = null;
  log("Auto train stopped.", "ok");
}

// -----------------------------------------------------------------------------
// Init/reset
function makeFixedNoiseInput() {
  return tf.tidy(() => tf.randomUniform(SHAPE_4D, 0, 1, "float32"));
}

function rebuildStudentModel(newArchType) {
  if (studentModel) studentModel.dispose();
  studentModel = createStudentModel(newArchType);
  studentOpt = tf.train.adam(STUDENT_LR);
  bestComboLoss = Infinity;
  stepsSinceBest = 0;
  log(`Student model rebuilt: arch='${newArchType}'.`, "ok");
}

function resetAllWeights() {
  stopAuto();

  stepCount = 0;
  bestComboLoss = Infinity;
  stepsSinceBest = 0;

  if (baselineModel) baselineModel.dispose();
  if (studentModel) studentModel.dispose();

  baselineModel = createBaselineModel();
  studentModel = createStudentModel(studentArchType);

  baseOpt = tf.train.adam(BASE_LR);
  studentOpt = tf.train.adam(STUDENT_LR);

  tf.tidy(() => {
    const b = baselineModel.predict(xInput);
    const s = studentModel.predict(xInput);
    drawTensorToCanvas(xInput, cvInput);
    drawTensorToCanvas(b, cvBase);
    drawTensorToCanvas(s, cvStudent);
  });

  setStatus({ step: stepCount, baseLoss: NaN, studentLoss: NaN });
  log("Weights reset.", "ok");
}

// -----------------------------------------------------------------------------
// Add lambda radio UI
function addLambdaControls() {
  const panelBody = document.querySelector(".panel .body");
  if (!panelBody) return;

  const wrap = document.createElement("div");
  wrap.className = "arch";
  wrap.style.marginTop = "10px";

  wrap.innerHTML = `
    <div class="title">
      <span>Student loss weights</span>
      <span class="badge">live</span>
    </div>

    <div class="tiny" style="margin: 0 0 6px;">λSmooth</div>
    <div id="lamSmooth" style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;"></div>

    <div class="tiny" style="margin: 0 0 6px;">λDir (gradient strength)</div>
    <div id="lamDir" style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;"></div>

    <div class="tiny" style="margin: 0 0 6px;">λDist (mapped to texture strength)</div>
    <div id="lamDist" style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;"></div>

    <div class="tiny" style="margin: 0 0 6px;">λChess (extra texture knob)</div>
    <div id="lamChess" style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;"></div>

    <div class="tiny" style="margin: 0 0 6px;">λRow (straighten gradient)</div>
    <div id="lamRow" style="display:flex; gap:10px; flex-wrap:wrap;"></div>
  `;

  panelBody.appendChild(wrap);

  function makeRadioRow(containerId, name, options, getVal, setVal) {
    const c = document.getElementById(containerId);
    options.forEach((v) => {
      const label = document.createElement("label");
      label.style.display = "flex";
      label.style.alignItems = "center";
      label.style.gap = "6px";
      label.style.cursor = "pointer";

      const input = document.createElement("input");
      input.type = "radio";
      input.name = name;
      input.value = String(v);
      input.checked = (v === getVal());
      input.addEventListener("change", () => {
        setVal(v);
        log(`Set ${name}=${v}`, "info");
      });

      const span = document.createElement("span");
      span.textContent = String(v);

      label.appendChild(input);
      label.appendChild(span);
      c.appendChild(label);
    });
  }

  makeRadioRow("lamSmooth", "lambdaSmooth", [0.0, 0.01, 0.015, 0.02, 0.05], () => LAMBDA_SMOOTH, (v) => { LAMBDA_SMOOTH = v; });
  makeRadioRow("lamDir", "lambdaDir", [6.0, 8.0, 9.0, 10.0, 12.0], () => LAMBDA_DIR, (v) => { LAMBDA_DIR = v; });
  makeRadioRow("lamDist", "lambdaDist", [0.0, 1.0, 2.0, 3.0, 5.0], () => LAMBDA_DIST, (v) => { LAMBDA_DIST = v; });
  makeRadioRow("lamChess", "lambdaChess", [0.0, 1.0, 2.5, 3.0, 5.0], () => LAMBDA_CHESS, (v) => { LAMBDA_CHESS = v; });
  makeRadioRow("lamRow", "lambdaRow", [0.0, 0.3, 0.5, 0.8, 1.2], () => LAMBDA_ROW, (v) => { LAMBDA_ROW = v; });
}

// -----------------------------------------------------------------------------
// UI wiring
btnStep.addEventListener("click", async () => {
  stopAuto();
  await trainOneStepReturnLosses();
});

btnAuto.addEventListener("click", () => {
  autoRunning ? stopAuto() : startAuto();
});

btnReset.addEventListener("click", () => {
  resetAllWeights();
});

document.querySelectorAll("input[name='arch']").forEach(radio => {
  radio.addEventListener("change", () => {
    const newType = document.querySelector("input[name='arch']:checked").value;
    studentArchType = newType;

    stopAuto();
    rebuildStudentModel(newType);

    tf.tidy(() => {
      const b = baselineModel.predict(xInput);
      const s = studentModel.predict(xInput);
      drawTensorToCanvas(xInput, cvInput);
      drawTensorToCanvas(b, cvBase);
      drawTensorToCanvas(s, cvStudent);
    });
  });
});

// -----------------------------------------------------------------------------
// Boot
async function main() {
  await tf.ready();

  addLambdaControls();

  xInput = makeFixedNoiseInput();

  baselineModel = createBaselineModel();
  studentModel = createStudentModel(studentArchType);

  baseOpt = tf.train.adam(BASE_LR);
  studentOpt = tf.train.adam(STUDENT_LR);

  tf.tidy(() => {
    const b = baselineModel.predict(xInput);
    const s = studentModel.predict(xInput);
    drawTensorToCanvas(xInput, cvInput);
    drawTensorToCanvas(b, cvBase);
    drawTensorToCanvas(s, cvStudent);
  });

  setStatus({ step: stepCount, baseLoss: NaN, studentLoss: NaN });
  log("Ready. Student optimizes gradient + chess + smooth + row-straightening.", "ok");
  log("Best recipe: Compression, λDir=9, λRow=0.5, λChess=2.5, λDist=2, λSmooth=0.015, then Auto Train.", "info");
}

main().catch((err) => log(String(err?.message || err), "error"));
