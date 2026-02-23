// app.js
// Autoencoder Inpainting — Your Photo (browser-only TF.js)
//
// What this demo does:
// - You upload a photo.
// - We resize to 128×128 and train an inpainting model by randomly masking squares.
// - Loss is (by default) computed only over the masked region (so the model learns filling).
//
// Educational hooks:
// - Reconstruction objective is a "game" (loss defines the game).
// - Students can modify the loss (TODO) to explore:
//   - masked-only vs whole-image loss
//   - smoothness / TV penalty
//   - perceptual-ish losses (edge consistency)
//
// -----------------------------------------------------------------------------
// DOM
const $ = (s) => document.querySelector(s);
const logEl = $("#log");
const statusEl = $("#status");
const memEl = $("#mem");

const filePhoto = $("#filePhoto");
const maskSizeEl = $("#maskSize");
const spfEl = $("#spf");
const lrEl = $("#lr");
const maskedOnlyEl = $("#maskedOnly");

const btnBuild = $("#btnBuild");
const btnStep = $("#btnStep");
const btnAuto = $("#btnAuto");
const btnReset = $("#btnReset");
const btnPredict = $("#btnPredict");

const cvOrig = $("#cvOrig");
const cvMasked = $("#cvMasked");
const cvRecon = $("#cvRecon");

// -----------------------------------------------------------------------------
// Config
const H = 128, W = 128, C = 3;
const SHAPE = [1, H, W, C];

// Auto-stop (prevents endless running)
const MAX_STEPS = 5000;
const PLATEAU_PATIENCE = 500;     // stop if no improvement for N steps
const MIN_DELTA = 1e-6;           // improvement threshold
const RENDER_EVERY = 5;           // render every N steps
const LOG_EVERY = 50;

// -----------------------------------------------------------------------------
// State
let step = 0;
let auto = false;
let raf = null;

let model = null;
let opt = null;

let imgTarget = null;   // [1,128,128,3] in [0,1]
let maskXY = { x: 48, y: 48 };  // top-left for the "interactive" mask
let bestLoss = Infinity;
let stepsSinceBest = 0;

// -----------------------------------------------------------------------------
// Logging / status
function log(msg, kind="info"){
  const p = kind==="error" ? "✖ " : kind==="ok" ? "✓ " : "• ";
  logEl.textContent = (p + msg + "\n" + logEl.textContent).slice(0, 6000);
}
function fmt(x){
  if (x == null || Number.isNaN(x)) return "—";
  if (!Number.isFinite(x)) return String(x);
  if (x < 1e-3) return x.toExponential(2);
  return x.toFixed(4);
}
function setStatus(lossVal){
  statusEl.textContent = `Step: ${step} | Loss: ${fmt(lossVal)}`;
  const m = tf.memory();
  memEl.textContent = `tf.memory: ${m.numTensors} tensors`;
}

// -----------------------------------------------------------------------------
// Canvas helpers
function drawRGBTensorToCanvas(t4d, canvas){
  // t4d: [1,H,W,3] values in [0,1]
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const img = ctx.createImageData(W, H);
  const data = img.data;

  const vals = t4d.dataSync(); // length H*W*3
  let k = 0;
  for (let i=0; i<H*W; i++){
    const r = Math.max(0, Math.min(1, vals[k++])) * 255;
    const g = Math.max(0, Math.min(1, vals[k++])) * 255;
    const b = Math.max(0, Math.min(1, vals[k++])) * 255;
    const j = i*4;
    data[j+0] = r|0;
    data[j+1] = g|0;
    data[j+2] = b|0;
    data[j+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function drawMaskOverlay(canvas, x, y, size){
  const ctx = canvas.getContext("2d");
  ctx.save();
  ctx.strokeStyle = "rgba(122,162,255,0.95)";
  ctx.lineWidth = 2;
  ctx.strokeRect(x + 0.5, y + 0.5, size, size);
  ctx.restore();
}

// -----------------------------------------------------------------------------
// Image load + resize (no extra libs)
async function loadImageToTensor(file){
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.crossOrigin = "anonymous";

  await new Promise((res, rej) => {
    img.onload = () => res();
    img.onerror = (e) => rej(e);
    img.src = url;
  });

  // Draw to an offscreen canvas at 128x128
  const off = document.createElement("canvas");
  off.width = W;
  off.height = H;
  const ctx = off.getContext("2d");
  ctx.drawImage(img, 0, 0, W, H);
  URL.revokeObjectURL(url);

  // Read pixels -> tensor [1,H,W,3] in [0,1]
  const im = ctx.getImageData(0, 0, W, H);
  return tf.tidy(() => {
    const t = tf.browser.fromPixels(im).toFloat().div(255);
    return t.expandDims(0); // [1,H,W,3]
  });
}

// -----------------------------------------------------------------------------
// Mask generation
function makeMask(x, y, size){
  // Returns mask tensor [1,H,W,1] with 1 inside the square, else 0
  return tf.tidy(() => {
    const xs = tf.range(0, W, 1, "int32");
    const ys = tf.range(0, H, 1, "int32");
    const X = xs.reshape([1, 1, W, 1]).tile([1, H, 1, 1]); // [1,H,W,1]
    const Y = ys.reshape([1, H, 1, 1]).tile([1, 1, W, 1]); // [1,H,W,1]

    const x0 = tf.scalar(x, "int32");
    const y0 = tf.scalar(y, "int32");
    const s = tf.scalar(size, "int32");

    const inX = tf.logicalAnd(tf.greaterEqual(X, x0), tf.less(X, tf.add(x0, s)));
    const inY = tf.logicalAnd(tf.greaterEqual(Y, y0), tf.less(Y, tf.add(y0, s)));
    const inside = tf.logicalAnd(inX, inY);
    return inside.toFloat(); // [1,H,W,1]
  });
}

function applyMaskToImage(img, mask){
  // img: [1,H,W,3], mask: [1,H,W,1]
  // Masked region replaced by 0.5 gray (constant) to create an obvious hole.
  return tf.tidy(() => {
    const hole = tf.mul(mask, tf.scalar(0.5));
    const holeRGB = hole.tile([1, 1, 1, 3]); // [1,H,W,3]
    const keep = tf.sub(tf.onesLike(mask), mask).tile([1, 1, 1, 3]);
    return tf.add(tf.mul(img, keep), holeRGB);
  });
}

function randomMaskXY(size){
  const x = Math.floor(Math.random() * (W - size));
  const y = Math.floor(Math.random() * (H - size));
  return { x, y };
}

// -----------------------------------------------------------------------------
// Model: small conv autoencoder for inpainting
function buildModel(){
  // Encoder-decoder with skip connections (tiny U-Net-ish without heavy complexity)
  const inp = tf.input({ shape: [H, W, C] });

  // Encoder
  const c1 = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: "same", activation: "relu" }).apply(inp);
  const p1 = tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }).apply(c1);

  const c2 = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: "same", activation: "relu" }).apply(p1);
  const p2 = tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }).apply(c2);

  const c3 = tf.layers.conv2d({ filters: 128, kernelSize: 3, padding: "same", activation: "relu" }).apply(p2);

  // Decoder
  const u2 = tf.layers.upSampling2d({ size: [2, 2] }).apply(c3);
  const m2 = tf.layers.concatenate().apply([u2, c2]);
  const d2 = tf.layers.conv2d({ filters: 64, kernelSize: 3, padding: "same", activation: "relu" }).apply(m2);

  const u1 = tf.layers.upSampling2d({ size: [2, 2] }).apply(d2);
  const m1 = tf.layers.concatenate().apply([u1, c1]);
  const d1 = tf.layers.conv2d({ filters: 32, kernelSize: 3, padding: "same", activation: "relu" }).apply(m1);

  // Output in [0,1]
  const out = tf.layers.conv2d({ filters: 3, kernelSize: 1, padding: "same", activation: "sigmoid" }).apply(d1);

  return tf.model({ inputs: inp, outputs: out, name: "inpaintAE" });
}

// -----------------------------------------------------------------------------
// Loss
function mse(a, b){ return tf.mean(tf.square(tf.sub(a, b))); }

function maskedMSE(yTrue, yPred, mask){
  // mask: [1,H,W,1] where 1 indicates "missing region"
  return tf.tidy(() => {
    const m = mask.tile([1,1,1,3]); // RGB
    const diff2 = tf.square(tf.sub(yTrue, yPred));
    const num = tf.sum(tf.mul(diff2, m));
    const den = tf.add(tf.sum(m), tf.scalar(1e-6)); // avoid divide-by-zero
    return tf.div(num, den);
  });
}

// TODO (Students): add smoothness / TV to reduce block artifacts
function smoothnessTV(yPred){
  return tf.tidy(() => {
    const dx = tf.sub(
      yPred.slice([0,0,1,0],[1,H,W-1,3]),
      yPred.slice([0,0,0,0],[1,H,W-1,3])
    );
    const dy = tf.sub(
      yPred.slice([0,1,0,0],[1,H-1,W,3]),
      yPred.slice([0,0,0,0],[1,H-1,W,3])
    );
    return tf.add(tf.mean(tf.square(dx)), tf.mean(tf.square(dy)));
  });
}

function lossFn(yTrue, yPred, mask){
  const maskedOnly = maskedOnlyEl.value === "1";
  if (maskedOnly) {
    // Recommended: learn to fill the hole, not re-learn the whole image.
    const Lfill = maskedMSE(yTrue, yPred, mask);

    // Optional TV regularization (small). Keeps recon smoother.
    // Students can tune this or remove it.
    const lambdaTV = 0.02;
    const Ltv = smoothnessTV(yPred);

    return tf.add(Lfill, tf.mul(lambdaTV, Ltv));
  } else {
    // Whole-image reconstruction (classic AE). Can over-focus on copying everything.
    return mse(yTrue, yPred);
  }
}

// -----------------------------------------------------------------------------
// One training step (custom loop, tidy, no leaks)
async function trainStep(){
  if (!imgTarget || !model) {
    log("Upload a photo and build the model first.", "error");
    return NaN;
  }

  const size = clampInt(+maskSizeEl.value, 8, 64);
  const { x, y } = randomMaskXY(size);

  const lossTensor = tf.tidy(() => {
    const mask = makeMask(x, y, size);
    const xMasked = applyMaskToImage(imgTarget, mask);

    const vars = model.trainableWeights.map(w => w.val);
    const { value, grads } = tf.variableGrads(() => {
      const yPred = model.apply(xMasked);
      return lossFn(imgTarget, yPred, mask);
    }, vars);

    opt.applyGradients(grads);
    Object.values(grads).forEach(g => g.dispose());
    return value;
  });

  const lossVal = (await lossTensor.data())[0];
  lossTensor.dispose();

  step++;
  setStatus(lossVal);

  // Auto-stop tracking
  if (lossVal + MIN_DELTA < bestLoss) {
    bestLoss = lossVal;
    stepsSinceBest = 0;
  } else {
    stepsSinceBest++;
  }

  if (step % LOG_EVERY === 0 || step === 1) {
    log(`step=${step} loss=${fmt(lossVal)}`, "info");
  }

  if (step % RENDER_EVERY === 0) {
    renderInteractive();
  }

  return lossVal;
}

// -----------------------------------------------------------------------------
// Interactive mask + prediction
function clampInt(v, lo, hi){
  v = Math.floor(Number(v));
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, v));
}

function renderInteractive(){
  if (!imgTarget) return;

  const size = clampInt(+maskSizeEl.value, 8, 64);
  const x = clampInt(maskXY.x, 0, W - size);
  const y = clampInt(maskXY.y, 0, H - size);
  maskXY = { x, y };

  tf.tidy(() => {
    drawRGBTensorToCanvas(imgTarget, cvOrig);

    const mask = makeMask(x, y, size);
    const xMasked = applyMaskToImage(imgTarget, mask);
    drawRGBTensorToCanvas(xMasked, cvMasked);

    if (model) {
      const yPred = model.predict(xMasked);
      drawRGBTensorToCanvas(yPred, cvRecon);
    } else {
      // If no model, show masked as "recon" placeholder
      drawRGBTensorToCanvas(xMasked, cvRecon);
    }
  });

  // Overlay mask rectangle on masked canvas
  drawMaskOverlay(cvMasked, x, y, size);
}

async function predictOnce(){
  if (!imgTarget || !model) {
    log("Upload a photo and build/train the model first.", "error");
    return;
  }
  renderInteractive();
  log("Predicted inpaint for current mask location.", "ok");
}

// -----------------------------------------------------------------------------
// Auto loop (throttled + auto-stop)
async function autoLoop(){
  if (!auto) return;

  const spf = clampInt(+spfEl.value, 1, 30);

  for (let i=0; i<spf; i++){
    const lossVal = await trainStep();
    await tf.nextFrame();

    if (!auto) return;

    // Stop conditions
    if (step >= MAX_STEPS) {
      log(`Auto-stop: reached max steps (${MAX_STEPS}).`, "ok");
      stopAuto();
      return;
    }
    if (stepsSinceBest >= PLATEAU_PATIENCE) {
      log(`Auto-stop: plateau (${PLATEAU_PATIENCE} steps without improvement).`, "ok");
      stopAuto();
      return;
    }
    // If loss goes weird, stop
    if (!Number.isFinite(lossVal)) {
      log("Auto-stop: loss is not finite.", "error");
      stopAuto();
      return;
    }
  }

  raf = requestAnimationFrame(autoLoop);
}

function startAuto(){
  if (auto) return;
  auto = true;
  btnAuto.textContent = "Auto Train (Stop)";
  log("Auto train started.", "ok");
  raf = requestAnimationFrame(autoLoop);
}
function stopAuto(){
  auto = false;
  btnAuto.textContent = "Auto Train (Start)";
  if (raf) cancelAnimationFrame(raf);
  raf = null;
  log("Auto train stopped.", "ok");
}

// -----------------------------------------------------------------------------
// Build / reset
function rebuild(){
  stopAuto();

  if (model) model.dispose();
  model = buildModel();

  const lr = Number(lrEl.value);
  opt = tf.train.adam(lr);

  step = 0;
  bestLoss = Infinity;
  stepsSinceBest = 0;

  setStatus(NaN);
  log(`Model built. Adam lr=${lr}.`, "ok");
  renderInteractive();
}

function resetWeights(){
  rebuild();
  log("Weights reset (rebuilt model).", "ok");
}

// -----------------------------------------------------------------------------
// Events
filePhoto.addEventListener("change", async (e) => {
  const f = e.target.files?.[0];
  if (!f) return;

  try{
    if (imgTarget) imgTarget.dispose();
    imgTarget = await loadImageToTensor(f);

    // Center the mask initially
    const size = clampInt(+maskSizeEl.value, 8, 64);
    maskXY = { x: Math.floor((W - size)/2), y: Math.floor((H - size)/2) };

    log("Photo loaded and resized to 128×128.", "ok");

    // Build a model automatically the first time
    if (!model) rebuild();
    else renderInteractive();
  } catch (err){
    log(String(err?.message || err), "error");
  }
});

btnBuild.addEventListener("click", rebuild);
btnReset.addEventListener("click", resetWeights);

btnStep.addEventListener("click", async () => {
  stopAuto();
  await trainStep();
});

btnAuto.addEventListener("click", () => {
  auto ? stopAuto() : startAuto();
});

btnPredict.addEventListener("click", async () => {
  stopAuto();
  await predictOnce();
});

// Click to move the mask
cvMasked.addEventListener("click", (ev) => {
  const rect = cvMasked.getBoundingClientRect();
  const px = (ev.clientX - rect.left) / rect.width;   // 0..1
  const py = (ev.clientY - rect.top) / rect.height;  // 0..1
  const size = clampInt(+maskSizeEl.value, 8, 64);

  const x = clampInt(Math.floor(px * W - size/2), 0, W - size);
  const y = clampInt(Math.floor(py * H - size/2), 0, H - size);

  maskXY = { x, y };
  renderInteractive();
  log(`Mask moved to x=${x}, y=${y}, size=${size}.`, "ok");
});

// If mask size changes, just re-render (and recommend rebuild for best training)
maskSizeEl.addEventListener("change", () => {
  renderInteractive();
});

// -----------------------------------------------------------------------------
// Boot
(async function main(){
  await tf.ready();
  setStatus(NaN);
  log("Ready. Upload a photo to begin.", "ok");
})();
