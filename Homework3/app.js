// app.js
// Neural Network Design: The Gradient Puzzle (2-file version)
//
// Fix: index.html loads THIS file as "app.js".
// Baseline: pixel-wise MSE (copycat / stuck).
// Student: sortedMSE (distribution constraint) + smoothness + direction (forms gradient).

// -----------------------------------------------------------------------------
// DOM
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
// Config
const H = 16, W = 16;
const SHAPE_4D = [1, H, W, 1];

// Training speed / UI smoothness
const STEPS_PER_FRAME = 10;
const RENDER_EVERY = 1;
const LOG_EVERY = 50;

// Optimizers
const BASE_LR = 1e-2;
const STUDENT_LR = 3e-2;

// Auto-stop
const AUTO_MAX_STEPS = 5000;
const PLATEAU_PATIENCE = 900;
const MIN_DELTA = 1e-8;

// Loss weights
const LAMBDA_SMOOTH = 1.2; // try 0.5–2.5
const LAMBDA_DIR = 2.2;    // try 1.0–4.0

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

// Direction mask [-1..+1] across x
let xCoordMask = null;

// Plateau tracking
let bestComboLoss = Infinity;
let stepsSinceBest = 0;

// -----------------------------------------------------------------------------
// Logging helpers
function log(msg, kind = "info") {
  const prefix = kind === "error" ? "✖ " : kind === "ok" ? "✓ " : "• ";
  logEl.textContent = (prefix + msg + "\n" + logEl.textContent).slice(0, 5000);
}
function fmt(x) {
  if (x == null || Number.isNaN(x)) return "—";
  if (!Number.isFinite(x)) return String(x);
  if (x < 1e-3) return x.toExponential(2);
  return x.toFixed(4);
}
function setStatus({ step, baseLoss, studentLoss }) {
  statusLine.textContent = `Step: ${step} | Baseline loss: ${fmt(baseLoss)} | Student loss: ${fmt(studentLoss)}`;
  memBadge.textContent = `tf.memory: ${tf.memory().numTensors} tensors`;
}

// -----------------------------------------------------------------------------
// Canvas renderer (16×16 grayscale)
function drawTensorToCanvas(t4d, canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const img = ctx.createImageData(W, H);
  const data = img.data;

  const vals = t4d.dataSync(); // length 256
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
// Loss helpers
function mse(yTrue, yPred) {
  return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

function sortedMSE(yTrue, yPred) {
  // MSE(sort(true), sort(pred)) using topk sort (descending).
  return tf.tidy(() => {
    const a = yTrue.flatten();
    const b = yPred.flatten();
    const N = a.size;

    // topk is a practical way to sort 1D tensors in tfjs
    const aSorted = tf.topk(a, N, true).values;
    const bSorted = tf.topk(b, N, true).values;
    return tf.mean(tf.square(tf.sub(aSorted, bSorted)));
  });
}

function smoothness(yPred) {
  // TV-like: squared neighbor diffs
  return tf.tidy(() => {
    const dx = tf.sub(
      yPred.slice([0, 0, 1, 0], [1, H, W - 1, 1]),
      yPred.slice([0, 0, 0, 0], [1, H, W - 1, 1])
    );
    const dy = tf.sub(
      yPred.slice([0, 1, 0, 0], [1, H - 1, W, 1]),
      yPred.slice([0, 0, 0, 0], [1, H - 1, W, 1])
    );
    return tf.add(tf.mean(tf.square(dx)), tf.mean(tf.square(dy)));
  });
}

function directionX(yPred) {
  // Minimize -mean(output * mask) => bright on right, dark on left
  return tf.tidy(() => tf.neg(tf.mean(tf.mul(yPred, xCoordMask))));
}

// -----------------------------------------------------------------------------
// Models
function createBaselineModel() {
  const inp = tf.input({ shape: [H, W, 1] });
  const flat = tf.layers.flatten().apply(inp);
  const h1 = tf.layers.dense({ units: 64, activation: "relu" }).apply(flat);
  const h2 = tf.layers.dense({ units: H * W, activation: "sigmoid" }).apply(h1);
  const out = tf.layers.reshape({ targetShape: [H, W, 1] }).apply(h2);
  return tf.model({ inputs: inp, outputs: out, name: "baselineModel" });
}

function createStudentModel(archType) {
  const inp = tf.input({ shape: [H, W, 1] });

  if (archType === "compression") {
    const flat = tf.layers.flatten().apply(inp);
    const z = tf.layers.dense({ units: 32, activation: "relu" }).apply(flat);
    const h = tf.layers.dense({ units: 128, activation: "relu" }).apply(z);
    const outFlat = tf.layers.dense({ units: H * W, activation: "sigmoid" }).apply(h);
    const out = tf.layers.reshape({ targetShape: [H, W, 1] }).apply(outFlat);
    return tf.model({ inputs: inp, outputs: out, name: "student_compression" });
  }

  if (archType === "transformation") {
    let x = inp;
    x = tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: "same", activation: "relu" }).apply(x);
    x = tf.layers.conv2d({ filters: 16, kernelSize: 1, padding: "same", activation: "relu" }).apply(x);
    const skip = tf.layers.conv2d({ filters: 16, kernelSize: 1, padding: "same", activation: "linear" }).apply(inp);
    x = tf.layers.add().apply([x, skip]);
    x = tf.layers.activation({ activation: "relu" }).apply(x);
    const out = tf.layers.conv2d({ filters: 1, kernelSize: 1, padding: "same", activation: "sigmoid" }).apply(x);
    return tf.model({ inputs: inp, outputs: out, name: "student_transformation" });
  }

  if (archType === "expansion") {
    let x = inp;
    x = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: "same", activation: "relu" }).apply(x);
    x = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: "same", activation: "relu" }).apply(x);
    x = tf.layers.conv2d({ filters: 32, kernelSize: 1, padding: "same", activation: "relu" }).apply(x);
    const out = tf.layers.conv2d({ filters: 1, kernelSize: 1, padding: "same", activation: "sigmoid" }).apply(x);
    return tf.model({ inputs: inp, outputs: out, name: "student_expansion" });
  }

  throw new Error(`Unknown student architecture: ${archType}`);
}

// -----------------------------------------------------------------------------
// Losses
function baselineLoss(yTrue, yPred) {
  // Baseline = Level 1 trap: pixel-wise MSE only
  return mse(yTrue, yPred);
}

function studentLoss(yTrue, yPred) {
  // Homework solution:
  // - sortedMSE keeps histogram
  // - smoothness makes it locally consistent
  // - direction makes it bright on the right
  const Lsorted = sortedMSE(yTrue, yPred);
  const Lsmooth = smoothness(yPred);
  const Ldir = directionX(yPred);
  return tf.addN([Lsorted, tf.mul(LAMBDA_SMOOTH, Lsmooth), tf.mul(LAMBDA_DIR, Ldir)]);
}

// -----------------------------------------------------------------------------
// Training (custom loop, separate optimizers, tidy)
async function trainOneStepReturnLosses() {
  const yTrue = xInput;

  let baseLossVal = NaN;
  let studentLossVal = NaN;

  try {
    const baseLossTensor = tf.tidy(() => {
      const vars = baselineModel.trainableWeights.map(w => w.val);
      const { value, grads } = tf.variableGrads(() => {
        const yPred = baselineModel.apply(yTrue);
        return baselineLoss(yTrue, yPred);
      }, vars);
      baseOpt.applyGradients(grads);
      Object.values(grads).forEach(g => g.dispose());
      return value;
    });
    baseLossVal = (await baseLossTensor.data())[0];
    baseLossTensor.dispose();

    const studentLossTensor = tf.tidy(() => {
      const vars = studentModel.trainableWeights.map(w => w.val);
      const { value, grads } = tf.variableGrads(() => {
        const yPred = studentModel.apply(yTrue);
        return studentLoss(yTrue, yPred);
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

  const combo = (Number.isFinite(baseLossVal) ? baseLossVal : 0) + (Number.isFinite(studentLossVal) ? studentLossVal : 0);
  if (combo + MIN_DELTA < bestComboLoss) {
    bestComboLoss = combo;
    stepsSinceBest = 0;
  } else {
    stepsSinceBest++;
  }

  return { baseLossVal, studentLossVal };
}

async function trainOneStep() {
  await trainOneStepReturnLosses();
}

// -----------------------------------------------------------------------------
// Auto loop
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
// Init / Reset / rebuild
function makeXCoordMask() {
  return tf.tidy(() => {
    const xs = tf.linspace(-1, 1, W);
    const row = xs.reshape([1, 1, W, 1]);
    return row.tile([1, H, 1, 1]);
  });
}

function makeFixedNoiseInput() {
  // fixed random seed isn’t exposed cleanly in TFJS without extra work,
  // but we keep the SAME tensor for the whole run, so it is “fixed”.
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
  log("Weights reset. (Baseline + Student reinitialized.)", "ok");
}

// -----------------------------------------------------------------------------
// UI wiring
btnStep.addEventListener("click", async () => {
  stopAuto();
  await trainOneStep();
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

  xInput = makeFixedNoiseInput();
  xCoordMask = makeXCoordMask();

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
  log("Ready. Baseline is MSE-copycat. Student is Level-3 (sortedMSE + smooth + direction).", "ok");
  log(`Loss weights: lambdaSmooth=${LAMBDA_SMOOTH}, lambdaDir=${LAMBDA_DIR}.`, "info");
}

main().catch((err) => log(String(err?.message || err), "error"));
