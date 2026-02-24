// app.js
// Neural Network Design: The Gradient Puzzle
// Browser-only TensorFlow.js demo (no build tools). Students edit TODO blocks below.
//
// Key lesson:
// - Networks are projection machines (compression / transformation / expansion).
// - The loss is the "rule of the game".
// - Pixel-wise MSE is a *trap* if you want pixel rearrangement: it freezes pixel positions.
//
// Intended progression:
//   Level 1: L = mse(input, output)               -> identity/copycat (noise stays noise).
//   Level 2: L = sortedMSE(input, output)         -> "no new colors, rearrange only" (histogram match).
//   Level 3: L = sortedMSE + λ_smooth*TV + λ_dir*direction
//                                                -> smooth left→right gradient while preserving histogram.
//
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

// Config
const H = 16, W = 16;
const SHAPE_4D = [1, H, W, 1];

// Speed / stability
const STEPS_PER_FRAME = 6;   // keep modest for smooth UI
const RENDER_EVERY = 1;      // render every N steps (keep 1 for immediate feedback)
const LOG_EVERY = 50;        // avoid spamming the log

// Auto-stop: prevents "it keeps running and never finishes"
const AUTO_STOP_LOSS = 1e-6;      // stop when BOTH losses below this (Level-1 MSE case)
const AUTO_STOP_MAX_STEPS = 8000; // safety cap
const AUTO_STOP_PATIENCE = 600;   // stop if no meaningful improvement for this many steps
const AUTO_STOP_MIN_DELTA = 1e-9; // "meaningful" improvement threshold

const BASE_LR = 1e-2;
const STUDENT_LR = 1e-2;

// Global state
let stepCount = 0;
let autoRunning = false;
let rafHandle = null;

let xInput = null;            // fixed per session
let baselineModel = null;
let studentModel = null;

let baseOpt = null;
let studentOpt = null;

let studentArchType = "compression";

// For directionX loss
let xCoordMask = null;

// Auto-stop tracking
let bestComboLoss = Infinity;
let stepsSinceBest = 0;

// Logging helpers
function log(msg, kind = "info") {
  const prefix = kind === "error" ? "✖ " : kind === "ok" ? "✓ " : "• ";
  const line = `${prefix}${msg}\n`;
  logEl.textContent = (line + logEl.textContent).slice(0, 5000);
}

function fmt(x) {
  if (x == null || Number.isNaN(x)) return "—";
  if (!Number.isFinite(x)) return String(x);
  if (x < 1e-3) return x.toExponential(2);
  return x.toFixed(4);
}

function setStatus({ step, baseLoss, studentLoss }) {
  statusLine.textContent =
    `Step: ${step} | Baseline loss: ${fmt(baseLoss)} | Student loss: ${fmt(studentLoss)}`;

  const m = tf.memory();
  memBadge.textContent = `tf.memory: ${m.numTensors} tensors`;
}

// Canvas renderer (16×16 tensor -> pixelated grayscale canvas)
function drawTensorToCanvas(t4d, canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const img = ctx.createImageData(W, H);
  const data = img.data;

  const vals = t4d.dataSync(); // size H*W
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

// Loss helpers (implemented, tested, and used)
function mse(yTrue, yPred) {
  // Pixel-wise MSE: forces each pixel to match its original *position*.
  // Great for copying. Terrible for rearranging.
  return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

function sortedMSE(yTrue, yPred) {
  // Level 2 distribution constraint:
  // MSE(sort(true_pixels), sort(pred_pixels))
  // Implementation requirement: use tf.topk-based sorting on flattened pixels.
  return tf.tidy(() => {
    const a = yTrue.flatten(); // [N]
    const b = yPred.flatten(); // [N]
    const N = a.size;

    const aSorted = tf.topk(a, N, true).values; // descending
    const bSorted = tf.topk(b, N, true).values;

    return tf.mean(tf.square(tf.sub(aSorted, bSorted)));
  });
}

function smoothness(yPred) {
  // Total-variation-ish penalty (squared neighbor diffs).
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
  // Encourage left-dark / right-bright:
  // maximize mean(yPred * mask) => minimize negative of it
  return tf.tidy(() => tf.neg(tf.mean(tf.mul(yPred, xCoordMask))));
}

// Models
function createBaselineModel() {
  // Fixed architecture, fixed MSE-only objective.
  const inp = tf.input({ shape: [H, W, 1] });

  const flat = tf.layers.flatten().apply(inp);
  const h1 = tf.layers.dense({ units: 64, activation: "relu" }).apply(flat);
  const h2 = tf.layers.dense({ units: H * W, activation: "sigmoid" }).apply(h1);
  const out = tf.layers.reshape({ targetShape: [H, W, 1] }).apply(h2);

  return tf.model({ inputs: inp, outputs: out, name: "baselineModel" });
}

// -----------------------------------------------------------------------------
// TODO-A (Architecture)
// -----------------------------------------------------------------------------
// Fully implemented now, but still intentionally "student editable":
// - Compression: flatten -> bottleneck -> expand -> reshape
// - Transformation: same spatial size, conv-residual mixing (basis change)
// - Expansion: expand channels a lot, then project back down (overcomplete projection)
//
// Students can modify widths, activations, add residuals, etc.
function createStudentModel(archType) {
  const inp = tf.input({ shape: [H, W, 1] });

  if (archType === "compression") {
    const flat = tf.layers.flatten().apply(inp);
    const z = tf.layers.dense({ units: 32, activation: "relu" }).apply(flat);
    const h = tf.layers.dense({ units: 128, activation: "relu" }).apply(z);
    const outFlat = tf.layers.dense({ units: H * W, activation: "sigmoid" }).apply(h);
    const out = tf.layers.reshape({ targetShape: [H, W, 1] }).apply(outFlat);
    return tf.model({ inputs: inp, outputs: out, name: "studentModel_compression" });
  }

  if (archType === "transformation") {
    // Transformation: keep dimension, change basis with conv mixing + residual.
    // This is a classic "projection as transformation" idea.
    let x = inp;
    x = tf.layers.conv2d({ filters: 16, kernelSize: 3, padding: "same", activation: "relu" }).apply(x);
    x = tf.layers.conv2d({ filters: 16, kernelSize: 1, padding: "same", activation: "relu" }).apply(x);

    // Residual-ish: map input to 16 channels then add
    const skip = tf.layers.conv2d({ filters: 16, kernelSize: 1, padding: "same", activation: "linear" }).apply(inp);
    x = tf.layers.add().apply([x, skip]);
    x = tf.layers.activation({ activation: "relu" }).apply(x);

    // Project back to 1 channel with sigmoid
    const out = tf.layers.conv2d({ filters: 1, kernelSize: 1, padding: "same", activation: "sigmoid" }).apply(x);
    return tf.model({ inputs: inp, outputs: out, name: "studentModel_transformation" });
  }

  if (archType === "expansion") {
    // Expansion: lift to high-dimensional representation, then compress back.
    // Overcomplete representation makes it easier to "rearrange" once loss changes (TODO-B).
    let x = inp;
    x = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: "same", activation: "relu" }).apply(x);
    x = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: "same", activation: "relu" }).apply(x);
    x = tf.layers.conv2d({ filters: 64, kernelSize: 1, padding: "same", activation: "relu" }).apply(x);

    // Optional mild regularization via a tiny bottleneck in channels (still "expansion" overall)
    x = tf.layers.conv2d({ filters: 32, kernelSize: 1, padding: "same", activation: "relu" }).apply(x);

    const out = tf.layers.conv2d({ filters: 1, kernelSize: 1, padding: "same", activation: "sigmoid" }).apply(x);
    return tf.model({ inputs: inp, outputs: out, name: "studentModel_expansion" });
  }

  throw new Error(`Unknown student architecture: ${archType}`);
}

// -----------------------------------------------------------------------------
// TODO-B (Custom Loss) — THE KEY PIVOT
// -----------------------------------------------------------------------------
// Critical conceptual requirement:
// Students MUST NOT build custom loss on top of pixel-wise MSE if they want rearrangement.
// Pixel-wise MSE freezes pixel positions.
// Escape hatch: sortedMSE(input, output) matches histogram ("no new colors") and allows movement.
//
// Stages:
//   1) Start: return mse(yTrue, yPred)
//   2) Pivot: return sortedMSE(yTrue, yPred)
//   3) Intent: sortedMSE + λ_smooth*smoothness + λ_dir*directionX
//
// Recommended ranges:
//   λ_smooth: 0.05 – 1.0
//   λ_dir:    0.05 – 1.0
function studentLoss(yTrue, yPred) {
  // LEVEL 1 (default): identical to baseline (MSE trap).
  // TODO-B: Replace this when teaching the "escape" moment.
  return mse(yTrue, yPred);

  // ---- LEVEL 2 (pivot): histogram match ("no new colors"), rearrange allowed ----
  // const Lsorted = sortedMSE(yTrue, yPred);
  // return Lsorted;

  // ---- LEVEL 3 (intent): add smoothness + direction ----
  // const Lsorted = sortedMSE(yTrue, yPred); // base term (do NOT re-add pixel-wise MSE)
  // const Lsmooth = smoothness(yPred);
  // const Ldir = directionX(yPred);
  // const lambdaSmooth = 0.25; // try 0.05–1.0
  // const lambdaDir = 0.25;    // try 0.05–1.0
  // return tf.addN([Lsorted, tf.mul(lambdaSmooth, Lsmooth), tf.mul(lambdaDir, Ldir)]);
}

function baselineLoss(yTrue, yPred) {
  return mse(yTrue, yPred); // always Level 1 trap
}

// -----------------------------------------------------------------------------
// Training loop (custom): separate optimizers; tidy; auto-stop; no log spam
// -----------------------------------------------------------------------------
async function trainOneStepReturnLosses() {
  const yTrue = xInput;

  let baseLossVal = NaN;
  let studentLossVal = NaN;

  try {
    // Baseline update
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

    // Student update
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

    // Status is cheap; update every step
    setStatus({ step: stepCount, baseLoss: baseLossVal, studentLoss: studentLossVal });

    // TODO-C (Comparison): log less frequently to avoid "keeps running" feeling
    if (stepCount % LOG_EVERY === 0 || stepCount === 1) {
      log(`step=${stepCount} | baseline=${fmt(baseLossVal)} | student=${fmt(studentLossVal)}`, "info");
    }

    // Render
    if (stepCount % RENDER_EVERY === 0) {
      tf.tidy(() => {
        const baseOut = baselineModel.predict(xInput);
        const studentOut = studentModel.predict(xInput);
        drawTensorToCanvas(xInput, cvInput);
        drawTensorToCanvas(baseOut, cvBase);
        drawTensorToCanvas(studentOut, cvStudent);
      });
    }
  } catch (err) {
    log(String(err?.message || err), "error");
  }

  return { baseLossVal, studentLossVal };
}

async function trainOneStep() {
  await trainOneStepReturnLosses();
}

// Auto loop with auto-stop
async function autoLoop() {
  if (!autoRunning) return;

  for (let i = 0; i < STEPS_PER_FRAME; i++) {
    const { baseLossVal, studentLossVal } = await trainOneStepReturnLosses();
    await tf.nextFrame();
    if (!autoRunning) return;

    // Auto-stop logic:
    // 1) Stop if converged (Level-1 MSE case)
    if (Number.isFinite(baseLossVal) && Number.isFinite(studentLossVal)) {
      if (baseLossVal < AUTO_STOP_LOSS && studentLossVal < AUTO_STOP_LOSS) {
        log(`Auto-stop: converged (loss < ${AUTO_STOP_LOSS}).`, "ok");
        stopAuto();
        return;
      }
    }

    // 2) Stop if no improvement for patience steps (works even after TODO-B changes)
    const combo = (Number.isFinite(baseLossVal) ? baseLossVal : 0) + (Number.isFinite(studentLossVal) ? studentLossVal : 0);
    if (combo + AUTO_STOP_MIN_DELTA < bestComboLoss) {
      bestComboLoss = combo;
      stepsSinceBest = 0;
    } else {
      stepsSinceBest++;
      if (stepsSinceBest >= AUTO_STOP_PATIENCE) {
        log(`Auto-stop: plateau (no improvement for ${AUTO_STOP_PATIENCE} steps).`, "ok");
        stopAuto();
        return;
      }
    }

    // 3) Safety cap
    if (stepCount >= AUTO_STOP_MAX_STEPS) {
      log(`Auto-stop: reached max steps (${AUTO_STOP_MAX_STEPS}).`, "ok");
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

// Init / Reset
function makeXCoordMask() {
  return tf.tidy(() => {
    const xs = tf.linspace(-1, 1, W);       // [W]
    const row = xs.reshape([1, 1, W, 1]);   // [1,1,W,1]
    return row.tile([1, H, 1, 1]);          // [1,H,W,1]
  });
}

function makeFixedNoiseInput() {
  return tf.tidy(() => tf.randomUniform(SHAPE_4D, 0, 1, "float32"));
}

function rebuildStudentModel(newArchType) {
  if (studentModel) studentModel.dispose();
  studentModel = createStudentModel(newArchType);

  // Optimizer has state; re-init when model changes (keeps behavior clean for students)
  studentOpt = tf.train.adam(STUDENT_LR);

  // Reset plateau tracking when architecture changes
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

// Wire UI
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

    // Re-render
    tf.tidy(() => {
      const b = baselineModel.predict(xInput);
      const s = studentModel.predict(xInput);
      drawTensorToCanvas(xInput, cvInput);
      drawTensorToCanvas(b, cvBase);
      drawTensorToCanvas(s, cvStudent);
    });
  });
});

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
  log("Ready. Baseline+Student start in the MSE trap (identity/copycat).", "ok");
  log("TODO-B is the escape hatch: switch to sortedMSE + smoothness + direction.", "info");
  log("Auto-train now stops automatically (converged/plateau/max steps).", "ok");
}

main().catch((err) => log(String(err?.message || err), "error"));
