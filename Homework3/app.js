// app.js
// Neural Network Design: The Gradient Puzzle (TFJS / GitHub Pages)
//
// Goal (per homework spirit):
// - Baseline: pixel-wise MSE -> copycat noise (stuck).
// - Student: rearrange the SAME pixel values (no new colors) into a left->right gradient.
//   We enforce “rearrangement only” by construction via a projection step.
//
// Why this version gives better results:
// 1) Projection enforces exact pixel inventory (true rearrangement).
// 2) Annealed lambdas: start with smoothness, gradually turn on direction.
// 3) Auxiliary loss on the proposal z (fully differentiable) stabilizes training.
// 4) Lower student LR (rank-based projection is jumpy with high LR).
//
// UI:
// - Student architecture radios still work (compression / transformation / expansion).
// - Adds radio groups for λDist, λSmooth, λDir (live).
//
// Notes:
// - λDist is usually not needed because projection preserves inventory exactly.
//   Keep it at 0 unless your instructor insists you “show a dist term”.

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
const N = H * W;
const SHAPE_4D = [1, H, W, 1];

// Training speed / UI smoothness
const STEPS_PER_FRAME = 15;
const RENDER_EVERY = 1;
const LOG_EVERY = 50;

// Optimizers
const BASE_LR = 1e-2;
const STUDENT_LR = 1.2e-2; // lowered: rank-based projection is unstable with high LR

// Auto-stop
const AUTO_MAX_STEPS = 8000;
const PLATEAU_PATIENCE = 1600;
const MIN_DELTA = 1e-8;

// Loss weights (user-adjustable via radio groups)
let LAMBDA_DIST = 0.0;   // optional (projection already preserves histogram exactly)
let LAMBDA_SMOOTH = 1.8; // TV (start here for “slide-like” look)
let LAMBDA_DIR = 3.2;    // direction (start here for “slide-like” look)

// Annealing schedule (big quality improvement)
const WARMUP_STEPS = 1200; // mostly smoothness first
const RAMP_STEPS = 2000;   // direction turns on gradually

// Auxiliary guidance on the proposal z (stabilizes training a lot)
const AUX_Z_WEIGHT = 0.25; // 0.15–0.40

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
// Helpers
function ramp01(t) {
  return Math.max(0, Math.min(1, t));
}

function getAnnealedLambdas(step) {
  const warm = ramp01(step / WARMUP_STEPS);
  const dirRamp = ramp01((step - WARMUP_STEPS) / RAMP_STEPS);

  // Slight early emphasis on smoothness so structure forms first
  const lamSmooth = LAMBDA_SMOOTH * (0.6 + 0.4 * warm);
  // Direction starts at 0 then ramps up
  const lamDir = LAMBDA_DIR * dirRamp;
  const lamDist = LAMBDA_DIST;

  return { lamDist, lamSmooth, lamDir };
}

// -----------------------------------------------------------------------------
// Loss helpers
function mse(yTrue, yPred) {
  return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

function smoothnessTV(yPred) {
  // TV-like: squared neighbor diffs (both x and y)
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
  // Slide formula: L_dir = -Mean(Output * Mask)
  return tf.tidy(() => tf.neg(tf.mean(tf.mul(yPred, xCoordMask))));
}

// Optional “distribution-looking” loss (not needed if λDist=0).
function softCDFLoss(yTrue, yPred, bins = 64, sigma = 0.03) {
  return tf.tidy(() => {
    const a = yTrue.flatten();
    const b = yPred.flatten();
    const centers = tf.linspace(0, 1, bins).reshape([1, bins]);

    function softHist(x) {
      const x2d = x.reshape([-1, 1]);
      const dist2 = tf.square(tf.sub(x2d, centers));
      const w = tf.exp(tf.mul(dist2, -1 / (2 * sigma * sigma)));
      const hist = tf.sum(w, 0);
      return tf.div(hist, tf.sum(hist));
    }

    const ha = softHist(a);
    const hb = softHist(b);
    const cdfa = tf.cumsum(ha);
    const cdfb = tf.cumsum(hb);
    return tf.mean(tf.square(tf.sub(cdfa, cdfb)));
  });
}

// -----------------------------------------------------------------------------
// Core: Projection that ENFORCES rearrangement (same pixel inventory)
//
// Given input x and student proposal z:
// - Sort indices of z (descending)
// - Sort values of x (descending)
// - Place x-sorted values into z-sorted positions
//
// TFJS WebGL has no TopK gradient; we use a customGrad STE:
// - forward: hard projection using topk + scatterND
// - backward: pass gradient through as if y ≈ z (straight-through)
function projectRearrangeInputToMatchOrder(xInput4d, zPred4d) {
  return tf.tidy(() => {
    const xFlat = xInput4d.flatten(); // [N]
    const zFlat = zPred4d.flatten();  // [N]

    const op = tf.customGrad((z, save) => {
      const idxZ = tf.topk(z, N, true).indices;     // [N]
      const xSorted = tf.topk(xFlat, N, true).values; // [N]

      const idx2d = idxZ.reshape([N, 1]);           // [N,1]
      const outFlat = tf.scatterND(idx2d, xSorted, [N]); // [N]

      save([idxZ]);

      const gradFunc = (dy /* [N] */) => {
        // STE: treat projection like identity wrt z to provide usable gradients
        return dy;
      };

      return { value: outFlat, gradFunc };
    });

    const yFlat = op(zFlat);
    return yFlat.reshape(SHAPE_4D);
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

// Student outputs a PROPOSAL z. Projection converts z -> rearranged y (using input values).
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

function studentLoss(xTrue, zProposal) {
  // Projection enforces inventory conservation exactly
  const y = projectRearrangeInputToMatchOrder(xTrue, zProposal);

  const { lamDist, lamSmooth, lamDir } = getAnnealedLambdas(stepCount);

  const LtvY = smoothnessTV(y);
  const LdirY = directionLossSlide(y);

  const Ldist = (lamDist > 0)
    ? softCDFLoss(xTrue, y, 64, 0.03)
    : tf.scalar(0);

  // Auxiliary shaping on z (fully differentiable): makes z become a clean ordering field
  const LtvZ = smoothnessTV(zProposal);
  const LdirZ = directionLossSlide(zProposal);

  const main = tf.addN([
    tf.mul(lamDist, Ldist),
    tf.mul(lamSmooth, LtvY),
    tf.mul(lamDir, LdirY),
  ]);

  const aux = tf.mul(AUX_Z_WEIGHT, tf.addN([
    tf.mul(lamSmooth, LtvZ),
    tf.mul(lamDir, LdirZ),
  ]));

  return tf.add(main, aux);
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
        const z = studentModel.apply(yTrue);
        return studentLoss(yTrue, z);
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
      const { lamSmooth, lamDir } = getAnnealedLambdas(stepCount);
      log(
        `step=${stepCount} | baseline=${fmt(baseLossVal)} | student=${fmt(studentLossVal)} | anneal(smooth=${fmt(lamSmooth)}, dir=${fmt(lamDir)})`,
        "info"
      );
    }

    if (stepCount % RENDER_EVERY === 0) {
      tf.tidy(() => {
        const b = baselineModel.predict(xInput);
        const z = studentModel.predict(xInput);
        const y = projectRearrangeInputToMatchOrder(xInput, z);

        drawTensorToCanvas(xInput, cvInput);
        drawTensorToCanvas(b, cvBase);
        drawTensorToCanvas(y, cvStudent);
      });
    }
  } catch (err) {
    log(String(err?.message || err), "error");
  }

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
  // Not seeded, but fixed for the run since we keep the same tensor.
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
    const z = studentModel.predict(xInput);
    const y = projectRearrangeInputToMatchOrder(xInput, z);

    drawTensorToCanvas(xInput, cvInput);
    drawTensorToCanvas(b, cvBase);
    drawTensorToCanvas(y, cvStudent);
  });

  setStatus({ step: stepCount, baseLoss: NaN, studentLoss: NaN });
  log("Weights reset. (Baseline + Student reinitialized.)", "ok");
}

// -----------------------------------------------------------------------------
// UI: add radio groups for lambdas (injected into the Control Panel)
function addLambdaControls() {
  const panelBody = document.querySelector(".panel .body");
  if (!panelBody) return;

  const wrap = document.createElement("div");
  wrap.className = "arch";
  wrap.style.marginTop = "10px";

  wrap.innerHTML = `
    <div class="title">
      <span>Loss weights (student)</span>
      <span class="badge">live</span>
    </div>

    <div class="tiny" style="margin: 0 0 6px;">λDist (optional)</div>
    <div id="lamDist" style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;"></div>

    <div class="tiny" style="margin: 0 0 6px;">λSmooth (TV)</div>
    <div id="lamSmooth" style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;"></div>

    <div class="tiny" style="margin: 0 0 6px;">λDir (direction)</div>
    <div id="lamDir" style="display:flex; gap:10px; flex-wrap:wrap;"></div>

    <p class="tiny" style="margin-top:10px;">
      Better results come from annealing: direction ramps in after ${WARMUP_STEPS} steps.
      Current: warmup=${WARMUP_STEPS}, ramp=${RAMP_STEPS}, auxZ=${AUX_Z_WEIGHT}, studentLR=${STUDENT_LR}.
    </p>
  `;

  panelBody.appendChild(wrap);

  function makeRadioRow(containerId, name, options, getVal, setVal) {
    const c = document.getElementById(containerId);
    options.forEach((v, idx) => {
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

  // Good presets included
  makeRadioRow("lamDist", "lambdaDist", [0.0, 0.1, 0.5, 1.0], () => LAMBDA_DIST, (v) => { LAMBDA_DIST = v; });
  makeRadioRow("lamSmooth", "lambdaSmooth", [1.2, 1.8, 2.4, 3.0], () => LAMBDA_SMOOTH, (v) => { LAMBDA_SMOOTH = v; });
  makeRadioRow("lamDir", "lambdaDir", [2.2, 3.2, 4.2, 5.2], () => LAMBDA_DIR, (v) => { LAMBDA_DIR = v; });
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
      const z = studentModel.predict(xInput);
      const y = projectRearrangeInputToMatchOrder(xInput, z);

      drawTensorToCanvas(xInput, cvInput);
      drawTensorToCanvas(b, cvBase);
      drawTensorToCanvas(y, cvStudent);
    });
  });
});

// -----------------------------------------------------------------------------
// Boot
async function main() {
  await tf.ready();

  addLambdaControls();

  xInput = makeFixedNoiseInput();
  xCoordMask = makeXCoordMask();

  baselineModel = createBaselineModel();
  studentModel = createStudentModel(studentArchType);

  baseOpt = tf.train.adam(BASE_LR);
  studentOpt = tf.train.adam(STUDENT_LR);

  tf.tidy(() => {
    const b = baselineModel.predict(xInput);
    const z = studentModel.predict(xInput);
    const y = projectRearrangeInputToMatchOrder(xInput, z);

    drawTensorToCanvas(xInput, cvInput);
    drawTensorToCanvas(b, cvBase);
    drawTensorToCanvas(y, cvStudent);
  });

  setStatus({ step: stepCount, baseLoss: NaN, studentLoss: NaN });
  log("Ready. Projection enforces exact inventory; annealing + auxZ improves convergence.", "ok");
  log("Best starting recipe: Compression + λSmooth=1.8 + λDir=3.2 + λDist=0.0, then Auto Train.", "info");
}

main().catch((err) => log(String(err?.message || err), "error"));
