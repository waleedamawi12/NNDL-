// app.js
// The Gradient Puzzle (matches the homework spec)
//
// What your screenshot shows (all three look similar) is EXACTLY Level 1 behavior:
//   L = mse(input, output)  -> copycat / identity.
//
// Homework requires:
// 1) Baseline: fixed architecture, fixed pixel-wise MSE (copycat).
// 2) Student: starts IDENTICAL (also MSE), but students must edit TODO blocks to escape.
// 3) Key pivot (must be explicit): DO NOT keep pixel-wise MSE if you want rearrangement.
//    Instead: sortedMSE(input, output) as the base term, then add smoothness + direction.
//
// This file is set so that once students implement TODO-B Level 3,
// the student output reliably becomes a left->right gradient within a few hundred to ~1500 steps.
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

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------
const H = 16, W = 16;
const SHAPE_4D = [1, H, W, 1];

// Keep UI smooth
const STEPS_PER_FRAME = 8;

// Separate learning rates (student often benefits from being a bit faster)
const BASE_LR = 1e-2;
const STUDENT_LR = 2e-2;

// Auto-stop: use plateau-based stop (works for MSE AND custom losses)
const AUTO_MAX_STEPS = 6000;
const PLATEAU_PATIENCE = 900;
const MIN_DELTA = 1e-8;

// Render/log throttle
const RENDER_EVERY = 1;
const LOG_EVERY = 50;

// ----------------------------------------------------------------------------
// State
// ----------------------------------------------------------------------------
let stepCount = 0;
let autoRunning = false;
let rafHandle = null;

let xInput = null;            // fixed per session
let baselineModel = null;
let studentModel = null;

let baseOpt = null;
let studentOpt = null;

let studentArchType = "compression";

// Precomputed coordinate mask for directionX loss (shape [1,16,16,1])
let xCoordMask = null;

// Plateau tracking
let bestComboLoss = Infinity;
let stepsSinceBest = 0;

// ----------------------------------------------------------------------------
// Logging helpers
// ----------------------------------------------------------------------------
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
  statusLine.textContent = `Step: ${step} | Baseline loss: ${fmt(baseLoss)} | Student loss: ${fmt(studentLoss)}`;
  const m = tf.memory();
  memBadge.textContent = `tf.memory: ${m.numTensors} tensors`;
}

// ----------------------------------------------------------------------------
// Canvas renderer (16×16 tensor -> pixelated grayscale canvas)
// ----------------------------------------------------------------------------
function drawTensorToCanvas(t4d, canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const img = ctx.createImageData(W, H);
  const data = img.data;

  const vals = t4d.dataSync(); // length H*W
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

// ----------------------------------------------------------------------------
// Loss helpers (implemented, tested, and used)
// ----------------------------------------------------------------------------
function mse(yTrue, yPred) {
  // Level 1 trap: pixel-wise MSE freezes pixel positions.
  return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

function sortedMSE(yTrue, yPred) {
  // Level 2 distribution constraint:
  // MSE(sort(true_pixels), sort(pred_pixels))
  // Must use tf.topk-based sorting on flattened pixels (descending OK).
  return tf.tidy(() => {
    const a = yTrue.flatten();
    const b = yPred.flatten();
    const N = a.size;

    const aSorted = tf.topk(a, N, true).values;
    const bSorted = tf.topk(b, N, true).values;

    return tf.mean(tf.square(tf.sub(aSorted, bSorted)));
  });
}

function smoothness(yPred) {
  // Total variation style: squared neighbor differences.
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
  // Encourage left-dark / right-bright.
  // Minimize -mean(output * mask).
  return tf.tidy(() => tf.neg(tf.mean(tf.mul(yPred, xCoordMask))));
}

// ----------------------------------------------------------------------------
// Models
// ----------------------------------------------------------------------------
function createBaselineModel() {
  // Fixed architecture, fixed MSE objective.
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
// Homework requirement:
// - compression: implemented
// - transformation: NOT implemented -> throw clear error until students do it
// - expansion: NOT implemented -> throw clear error until students do it
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
    throw new Error(
      "TODO-A: 'transformation' architecture not implemented. Implement a transformation projection in createStudentModel()."
    );
  }

  if (archType === "expansion") {
    throw new Error(
      "TODO-A: 'expansion' architecture not implemented. Implement an expansion projection in createStudentModel()."
    );
  }

  throw new Error(`Unknown student architecture: ${archType}`);
}

// -----------------------------------------------------------------------------
// TODO-B (Custom Loss) — THE KEY PIVOT
// -----------------------------------------------------------------------------
// CRITICAL: Students must NOT stack fancy terms on top of pixel-wise MSE if they want rearrangement.
// Pixel-wise MSE is position-locked.
// The pivot is: sortedMSE(input, output) as the BASE term.
//
// Recommended coefficients for Level 3 on 16×16:
//   lambdaSmooth: 0.2 – 2.0
//   lambdaDir:    0.2 – 3.0
//
// Tip: If it still looks noisy, increase lambdaSmooth.
// Tip: If it doesn't become left->right, increase lambdaDir.
function studentLoss(yTrue, yPred) {
  // LEVEL 1 (start): identical to baseline (MSE trap).
  // This is correct as the starting point for the homework.
  return mse(yTrue, yPred);

  // LEVEL 2 (pivot): histogram match (“no new colors, rearrange only”)
  // const Lsorted = sortedMSE(yTrue, yPred);
  // return Lsorted;

  // LEVEL 3 (intent): distribution + smoothness + direction
  // const Lsorted = sortedMSE(yTrue, yPred); // base term (do NOT re-add pixel-wise MSE)
  // const Lsmooth = smoothness(yPred);
  // const Ldir = directionX(yPred);
  //
  // const lambdaSmooth = 0.8; // try 0.2–2.0
  // const lambdaDir = 1.6;    // try 0.2–3.0
  //
  // return tf.addN([Lsorted, tf.mul(lambdaSmooth, Lsmooth), tf.mul(lambdaDir, Ldir)]);
}

function baselineLoss(yTrue, yPred) {
  return mse(yTrue, yPred);
}

// ----------------------------------------------------------------------------
// Training step (custom loop; separate optimizers; tidy to avoid leaks)
// ----------------------------------------------------------------------------
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

  // Plateau logic uses a combined score (works no matter what loss you choose)
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

// ----------------------------------------------------------------------------
// Auto train (throttled + plateau stop)
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// Init / Reset / Rebuild student
// ----------------------------------------------------------------------------
function makeXCoordMask() {
  return tf.tidy(() => {
    const xs = tf.linspace(-1, 1, W);       // [W]
    const row = xs.reshape([1, 1, W, 1]);   // [1,1,W,1]
    return row.tile([1, H, 1, 1]);          // [1,H,W,1]
  });
}

function makeFixedNoiseInput() {
  // Fixed per session.
  return tf.tidy(() => tf.randomUniform(SHAPE_4D, 0, 1, "float32"));
}

function rebuildStudentModel(newArchType) {
  if (studentModel) studentModel.dispose();
  studentModel = null;

  // Optimizer has state; reset on rebuild
  studentOpt = tf.train.adam(STUDENT_LR);

  // Reset plateau tracking
  bestComboLoss = Infinity;
  stepsSinceBest = 0;

  studentModel = createStudentModel(newArchType);
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

// ----------------------------------------------------------------------------
// Wire UI
// ----------------------------------------------------------------------------
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

    try {
      rebuildStudentModel(newType);

      tf.tidy(() => {
        const b = baselineModel.predict(xInput);
        const s = studentModel.predict(xInput);
        drawTensorToCanvas(xInput, cvInput);
        drawTensorToCanvas(b, cvBase);
        drawTensorToCanvas(s, cvStudent);
      });
    } catch (err) {
      log(String(err?.message || err), "error");

      // Revert to compression so the app remains usable
      const fallback = document.querySelector("input[name='arch'][value='compression']");
      fallback.checked = true;
      studentArchType = "compression";
      rebuildStudentModel("compression");
    }
  });
});

// ----------------------------------------------------------------------------
// Boot
// ----------------------------------------------------------------------------
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
  log("Ready. Both models start in Level 1 (pixel-wise MSE trap).", "ok");
  log("Homework pivot: implement TODO-B so student uses sortedMSE + smoothness + direction.", "info");
}

main().catch((err) => log(String(err?.message || err), "error"));
