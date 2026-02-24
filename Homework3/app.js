// app.js
// Neural Network Design: The Gradient Puzzle (TFJS / GitHub Pages)
//
// Matches the homework slide intent:
// Level 2: MSE(Sort(Input), Sort(Output))
// Level 3: + Smoothness (TV) + Direction: L_dir = -Mean(Output * Mask)
// :contentReference[oaicite:1]{index=1}
//
// IMPORTANT: tf.topk/sort has no gradient in TFJS WebGL.
// So we use a differentiable sorting approximation (SoftSort / NeuralSort-style).
//
// Expected outcome:
// - Baseline: pixel-wise MSE => copycat noise-like output.
// - Student: same "inventory of colors" (distribution preserved) rearranged into left→right gradient,
//   with a mild checker texture possible (like the demo/slide).

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
const STEPS_PER_FRAME = 5;   // softsort is heavier than simple losses
const RENDER_EVERY = 1;
const LOG_EVERY = 50;

// Optimizers
const BASE_LR = 1e-2;
const STUDENT_LR = 3e-2;

// Auto-stop
const AUTO_MAX_STEPS = 7000;
const PLATEAU_PATIENCE = 1400;
const MIN_DELTA = 1e-8;

// Loss weights (close to the slide/demo defaults)
const LAMBDA_SMOOTH = 1.2; // TV
const LAMBDA_DIR = 2.2;    // direction

// SoftSort temperature:
// smaller => closer to true sort, but can be harder to optimize.
// good starting range: 0.05–0.2
const SOFTSORT_TAU = 0.10;

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

/**
 * Differentiable SoftSort approximation (NeuralSort-style).
 * Returns approximately sorted values (descending).
 *
 * For vector s (length n):
 * A_ij = |s_i - s_j|
 * B_j  = sum_k A_jk
 * scaling_i = (n + 1 - 2i), i=1..n
 * logits_{i,j} = (scaling_i * s_j - B_j) / tau
 * P = softmax(logits, axis=1)  (row-wise)
 * sorted ≈ P @ s
 */
function softSort1D(s, tau = 0.1) {
  return tf.tidy(() => {
    const v = s.flatten();                // [n]
    const n = v.size;

    // [n,1] and [1,n]
    const vCol = v.reshape([n, 1]);
    const vRow = v.reshape([1, n]);

    // Pairwise abs diffs: [n,n]
    const A = tf.abs(tf.sub(vCol, vRow));

    // B_j = sum_k |v_j - v_k|  -> shape [1,n] for broadcasting
    const ones = tf.ones([n, 1]);
    const B = tf.matMul(A, ones).reshape([1, n]); // [1,n]

    // scaling_i = n + 1 - 2i, i=1..n  -> [n,1]
    const i = tf.range(1, n + 1, 1, "float32"); // [n]
    const scaling = tf.sub(tf.scalar(n + 1, "float32"), tf.mul(tf.scalar(2, "float32"), i)).reshape([n, 1]);

    // C_{i,j} = scaling_i * v_j -> [n,n]
    const C = tf.matMul(scaling, vRow);

    // logits: [n,n]
    const logits = tf.div(tf.sub(C, B), tf.scalar(tau, "float32"));

    // P: [n,n], each row sums to 1
    const P = tf.softmax(logits, 1);

    // sorted approx: [n,1] -> [n]
    return tf.matMul(P, vCol).reshape([n]);
  });
}

function sortedMSE_soft(yTrue, yPred) {
  // MSE( SoftSort(yTrue), SoftSort(yPred) )
  return tf.tidy(() => {
    const a = softSort1D(yTrue, SOFTSORT_TAU);
    const b = softSort1D(yPred, SOFTSORT_TAU);
    return tf.mean(tf.square(tf.sub(a, b)));
  });
}

function smoothnessTV(yPred) {
  // TV-like: squared neighbor diffs (matches the slide idea)
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

function directionLossSlide(yPred) {
  // Slide formula: L_dir = -Mean(Output * Mask)  :contentReference[oaicite:2]{index=2}
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
  // Level 1 trap: pixel-wise MSE only
  return mse(yTrue, yPred);
}

function studentLoss(yTrue, yPred) {
  // Level 2 + Level 3 (per slides) :contentReference[oaicite:3]{index=3}
  const Lsorted = sortedMSE_soft(yTrue, yPred);
  const Ltv = smoothnessTV(yPred);
  const Ldir = directionLossSlide(yPred);
  return tf.addN([Lsorted, tf.mul(LAMBDA_SMOOTH, Ltv), tf.mul(LAMBDA_DIR, Ldir)]);
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
  // Not seeded, but fixed for the whole run since we keep this tensor.
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
  log("Ready. Student uses SoftSortedMSE + TV + Direction (slide form).", "ok");
  log(`SoftSort tau=${SOFTSORT_TAU} | lambdaSmooth=${LAMBDA_SMOOTH} | lambdaDir=${LAMBDA_DIR}`, "info");
  log("Tip: If gradient is too blurry, lower tau to 0.07. If unstable, raise tau to 0.15.", "info");
}

main().catch((err) => log(String(err?.message || err), "error"));
