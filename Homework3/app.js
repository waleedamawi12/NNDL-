// app.js
// Neural Network Design: The Gradient Puzzle (TFJS, 2-file version)
//
// FINAL FIXED VERSION for GitHub Pages / WebGL:
//
// Problem you hit:
// - tf.topk() (used for "sorting") has NO gradient in TFJS on WebGL:
//   "Cannot compute gradient: gradient function not found for TopK."
//
// Fixes in this file:
// 1) Replace topk/sort-based loss with a differentiable distribution constraint:
//    soft-histogram CDF loss (approx 1D Wasserstein / quantile match).
// 2) Replace the "direction" loss that encourages saturation (step function) with
//    a bounded correlation-based direction loss (Pearson correlation), which
//    prevents the model from cheating by slamming pixels to 0/1.
// 3) Add an explicit weight for distribution loss, so histogram preservation
//    actually wins.
//
// Expected behavior:
// - Baseline output stays noise-like (pixel MSE copycat).
// - Student output becomes a smooth left→right gradient while roughly preserving
//   the input brightness distribution.

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
const AUTO_MAX_STEPS = 6000;
const PLATEAU_PATIENCE = 1200;
const MIN_DELTA = 1e-8;

// Loss weights (tuned to avoid "banded step function" solutions)
const LAMBDA_DIST = 12.0;  // strong “keep the same histogram”
const LAMBDA_SMOOTH = 2.5; // smoothness / TV-ish
const LAMBDA_DIR = 1.0;    // direction (keep modest)

// Differentiable distribution loss parameters
const DIST_BINS = 64;      // more bins = stricter match (32–128)
const DIST_SIGMA = 0.02;   // smaller = stricter, but can be harder (0.02–0.05)

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
  logEl.textContent = (prefix + msg + "\n" + logEl.textContent).slice(0, 7000);
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
// Canvas renderer (16×16 grayscale)
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
// Loss helpers
function mse(yTrue, yPred) {
  return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

// Differentiable distribution constraint (no topk/sort):
// soft-histogram CDF loss ≈ 1D Wasserstein-ish distribution match
function cdfLoss(yTrue, yPred, bins = 64, sigma = 0.02) {
  return tf.tidy(() => {
    const a = yTrue.flatten(); // [N]
    const b = yPred.flatten(); // [N]
    const centers = tf.linspace(0, 1, bins).reshape([1, bins]); // [1,B]

    function softHist(x) {
      const x2d = x.reshape([-1, 1]); // [N,1]
      const dist2 = tf.square(tf.sub(x2d, centers)); // [N,B]
      const w = tf.exp(tf.mul(dist2, -1 / (2 * sigma * sigma))); // [N,B]
      const hist = tf.sum(w, 0); // [B]
      return tf.div(hist, tf.sum(hist)); // normalize
    }

    const ha = softHist(a);     // [B]
    const hb = softHist(b);     // [B]
    const cdfa = tf.cumsum(ha); // [B]
    const cdfb = tf.cumsum(hb); // [B]

    return tf.mean(tf.square(tf.sub(cdfa, cdfb)));
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

// Correlation-based direction loss (bounded, avoids "step function" cheating)
function directionCorrX(yPred) {
  return tf.tidy(() => {
    const y = yPred.flatten();
    const x = xCoordMask.flatten();

    const y0 = tf.sub(y, tf.mean(y));
    const x0 = tf.sub(x, tf.mean(x));

    const cov = tf.mean(tf.mul(y0, x0));
    const yStd = tf.sqrt(tf.add(tf.mean(tf.square(y0)), 1e-8));
    const xStd = tf.sqrt(tf.add(tf.mean(tf.square(x0)), 1e-8));

    const corr = tf.div(cov, tf.mul(yStd, xStd)); // [-1, 1]
    return tf.neg(corr); // minimize => maximize corr
  });
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

// Projection types for student (all implemented)
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
  return mse(yTrue, yPred);
}

function studentLoss(yTrue, yPred) {
  const Ldist = cdfLoss(yTrue, yPred, DIST_BINS, DIST_SIGMA);
  const Lsmooth = smoothness(yPred);
  const Ldir = directionCorrX(yPred);

  return tf.addN([
    tf.mul(LAMBDA_DIST, Ldist),
    tf.mul(LAMBDA_SMOOTH, Lsmooth),
    tf.mul(LAMBDA_DIR, Ldir),
  ]);
}

// -----------------------------------------------------------------------------
// Training loop
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

  // Plateau tracking (combined score)
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
     
