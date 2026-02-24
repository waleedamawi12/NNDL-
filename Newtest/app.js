// app.js
// Autoencoder Inpainting — Your Photo (higher quality version)
//
// Why your previous result looked blurry:
// - Training on ONE image with whole-image MSE encourages “average-looking” solutions.
// - MSE punishes sharp edges heavily → models often learn blur to reduce penalty.
// - Also, training only on a single fixed mask location is not enough data.
//
// Fixes in this version:
// - Train on RANDOM CROPS of your photo (creates lots of training examples).
// - Default loss = masked-only L1 + edge loss + small TV (less blur, more structure).
// - Slightly stronger U-Net-ish model.
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
const H = 256, W = 256, C = 3;
const SHAPE = [1, H, W, C];

// Auto-stop
const MAX_STEPS = 7000;
const PLATEAU_PATIENCE = 700;
const MIN_DELTA = 1e-6;
const RENDER_EVERY = 6;
const LOG_EVERY = 60;

// Training crop settings (creates many examples from one photo)
const CROP = 160;                // random crop size
const CROP_SHAPE = [1, CROP, CROP, C];
const USE_CROP = true;           // keep true for better quality

// Mask fill strategy:
// "noise" prevents the network from keying on a constant flat gray block.
const MASK_FILL = "noise"; // "noise" or "gray"

// -----------------------------------------------------------------------------
// State
let step = 0;
let auto = false;
let raf = null;

let model = null;
let opt = null;

let imgTargetFull = null;  // [1,H,W,3] in [0,1]
let maskXY = { x: 60, y: 60 };   // mask pos for interactive preview (on full image)

let bestLoss = Infinity;
let stepsSinceBest = 0;

// -----------------------------------------------------------------------------
// Logging / status
function log(msg, kind="info"){
  const p = kind==="error" ? "✖ " : kind==="ok" ? "✓ " : "• ";
  logEl.textContent = (p + msg + "\n" + logEl.textContent).slice(0, 7000);
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
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const img = ctx.createImageData(canvas.width, canvas.height);
  const data = img.data;

  // t4d expected [1,h,w,3] matching canvas size
  const vals = t4d.dataSync();
  let k = 0;
  for (let i=0; i<canvas.width*canvas.height; i++){
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

function clampInt(v, lo, hi){
  v = Math.floor(Number(v));
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, v));
}

// -----------------------------------------------------------------------------
// Image load + resize
async function loadImageToTensor(file){
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.crossOrigin = "anonymous";

  await new Promise((res, rej) => {
    img.onload = () => res();
    img.onerror = (e) => rej(e);
    img.src = url;
  });

  const off = document.createElement("canvas");
  off.width = W;
  off.height = H;
  const ctx = off.getContext("2d");
  ctx.drawImage(img, 0, 0, W, H);
  URL.revokeObjectURL(url);

  const im = ctx.getImageData(0, 0, W, H);
  return tf.tidy(() => tf.browser.fromPixels(im).toFloat().div(255).expandDims(0));
}

// -----------------------------------------------------------------------------
// Random crop (creates many training examples from one photo)
function randomCrop(t4d){
  // t4d: [1,H,W,3] -> [1,CROP,CROP,3]
  return tf.tidy(() => {
    const x0 = Math.floor(Math.random() * (W - CROP));
    const y0 = Math.floor(Math.random() * (H - CROP));
    return t4d.slice([0, y0, x0, 0], [1, CROP, CROP, 3]);
  });
}

// -----------------------------------------------------------------------------
// Mask generation
function makeMask(h, w, x, y, size){
  // Returns mask tensor [1,h,w,1] with 1 inside the square, else 0
  return tf.tidy(() => {
    const xs = tf.range(0, w, 1, "int32");
    const ys = tf.range(0, h, 1, "int32");
    const X = xs.reshape([1, 1, w, 1]).tile([1, h, 1, 1]);
    const Y = ys.reshape([1, h, 1, 1]).tile([1, 1, w, 1]);

    const x0 = tf.scalar(x, "int32");
    const y0 = tf.scalar(y, "int32");
    const s = tf.scalar(size, "int32");

    const inX = tf.logicalAnd(tf.greaterEqual(X, x0), tf.less(X, tf.add(x0, s)));
    const inY = tf.logicalAnd(tf.greaterEqual(Y, y0), tf.less(Y, tf.add(y0, s)));
    return tf.logicalAnd(inX, inY).toFloat();
  });
}

function applyMaskToImage(img, mask){
  // img: [1,h,w,3], mask: [1,h,w,1]
  return tf.tidy(() => {
    const mRGB = mask.tile([1, 1, 1, 3]);

    let fill;
    if (MASK_FILL === "noise") {
      fill = tf.randomUniform(img.shape, 0, 1, "float32");
      // keep fill “roughly plausible” by mixing with blurred-ish base (simple average)
      fill = tf.add(tf.mul(0.35, fill), tf.mul(0.65, tf.mean(img, 3, true).tile([1,1,1,3])));
    } else {
      fill = tf.fill(img.shape, 0.5);
    }

    const keep = tf.sub(tf.onesLike(mRGB), mRGB);
    return tf.add(tf.mul(img, keep), tf.mul(fill, mRGB));
  });
}

function randomMaskXY(h, w, size){
  const x = Math.floor(Math.random() * (w - size));
  const y = Math.floor(Math.random() * (h - size));
  return { x, y };
}

// -----------------------------------------------------------------------------
// Model: stronger U-Net-ish (still lightweight)
function convBlock(x, filters){
  x = tf.layers.conv2d({ filters, kernelSize: 3, padding: "same", useBias: false }).apply(x);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.activation({ activation: "relu" }).apply(x);

  x = tf.layers.conv2d({ filters, kernelSize: 3, padding: "same", useBias: false }).apply(x);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.activation({ activation: "relu" }).apply(x);
  return x;
}

function buildModel(){
  // NOTE: model input size depends on whether we train on crops.
  const inH = USE_CROP ? CROP : H;
  const inW = USE_CROP ? CROP : W;
  const inp = tf.input({ shape: [inH, inW, C] });

  // Encoder
  const c1 = convBlock(inp, 32);
  const p1 = tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }).apply(c1);

  const c2 = convBlock(p1, 64);
  const p2 = tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }).apply(c2);

  const c3 = convBlock(p2, 128);
  const p3 = tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }).apply(c3);

  // Bottleneck
  const bn = convBlock(p3, 192);

  // Decoder
  const u3 = tf.layers.upSampling2d({ size: [2,2] }).apply(bn);
  const m3 = tf.layers.concatenate().apply([u3, c3]);
  const d3 = convBlock(m3, 128);

  const u2 = tf.layers.upSampling2d({ size: [2,2] }).apply(d3);
  const m2 = tf.layers.concatenate().apply([u2, c2]);
  const d2 = convBlock(m2, 64);

  const u1 = tf.layers.upSampling2d({ size: [2,2] }).apply(d2);
  const m1 = tf.layers.concatenate().apply([u1, c1]);
  const d1 = convBlock(m1, 32);

  // Output in [0,1]
  const out = tf.layers.conv2d({ filters: 3, kernelSize: 1, padding: "same", activation: "sigmoid" }).apply(d1);

  return tf.model({ inputs: inp, outputs: out, name: "inpaint_unet" });
}

// -----------------------------------------------------------------------------
// Losses (better than plain MSE)
function l1(a, b){ return tf.mean(tf.abs(tf.sub(a, b))); }

function maskedL1(yTrue, yPred, mask){
  return tf.tidy(() => {
    const m = mask.tile([1,1,1,3]);
    const diff = tf.abs(tf.sub(yTrue, yPred));
    const num = tf.sum(tf.mul(diff, m));
    const den = tf.add(tf.sum(m), tf.scalar(1e-6));
    return tf.div(num, den);
  });
}

function tv(yPred){
  return tf.tidy(() => {
    const [_, h, w, __] = yPred.shape;
    const dx = tf.sub(
      yPred.slice([0,0,1,0],[1,h,w-1,3]),
      yPred.slice([0,0,0,0],[1,h,w-1,3])
    );
    const dy = tf.sub(
      yPred.slice([0,1,0,0],[1,h-1,w,3]),
      yPred.slice([0,0,0,0],[1,h-1,w,3])
    );
    return tf.add(tf.mean(tf.square(dx)), tf.mean(tf.square(dy)));
  });
}

function edgeLoss(yTrue, yPred, mask){
  // Sobel-based gradient matching reduces blur and preserves edges.
  // We compute edge difference (L1) mostly on masked area + a small ring around it.
  return tf.tidy(() => {
    const eT = tf.image.sobelEdges(yTrue); // [1,h,w,3,2]
    const eP = tf.image.sobelEdges(yPred);

    // magnitude approx = |gx|+|gy|
    const magT = tf.sum(tf.abs(eT), -1); // [1,h,w,3]
    const magP = tf.sum(tf.abs(eP), -1);

    // Expand mask slightly to include border context (cheap “ring”)
    // Using maxPool as dilation (kernel 7).
    const m = mask;
    const dil = tf.maxPool(m, 7, 1, "same"); // [1,h,w,1]
    const mRGB = dil.tile([1,1,1,3]);

    const diff = tf.abs(tf.sub(magT, magP));
    const num = tf.sum(tf.mul(diff, mRGB));
    const den = tf.add(tf.sum(mRGB), tf.scalar(1e-6));
    return tf.div(num, den);
  });
}

// TODO (students): try replacing edgeLoss with SSIM or adding a color-histogram constraint
function lossFn(yTrue, yPred, mask){
  const maskedOnly = maskedOnlyEl.value === "1";

  if (maskedOnly) {
    const Lfill = maskedL1(yTrue, yPred, mask);
    const Ledge = edgeLoss(yTrue, yPred, mask);
    const Ltv = tv(yPred);

    // Coeffs tuned for “looks better” on single-photo training
    const lambdaEdge = 0.35;  // 0.15–0.6
    const lambdaTV = 0.02;    // 0.00–0.05

    return tf.addN([Lfill, tf.mul(lambdaEdge, Ledge), tf.mul(lambdaTV, Ltv)]);
  }

  // Whole image loss (still better than MSE):
  return l1(yTrue, yPred);
}

// -----------------------------------------------------------------------------
// Rendering (interactive preview uses FULL image and a fixed mask)
function renderInteractive(){
  if (!imgTargetFull) return;

  const size = clampInt(+maskSizeEl.value, 12, 256);
  const x = clampInt(maskXY.x, 0, W - size);
  const y = clampInt(maskXY.y, 0, H - size);
  maskXY = { x, y };

  tf.tidy(() => {
    drawRGBTensorToCanvas(imgTargetFull, cvOrig);

    const mask = makeMask(H, W, x, y, size);
    const xMasked = applyMaskToImage(imgTargetFull, mask);
    drawRGBTensorToCanvas(xMasked, cvMasked);

    if (model) {
      // If model was built for crops, we still can predict on full image by running on full size:
      // easiest: rebuild expects crop input, so here we do a center crop for preview prediction.
      // But users expect full prediction; we handle both cases:
      let pred;
      if (USE_CROP) {
        // Predict on a full image by tiling crop inference is heavy.
        // Instead, we do preview by resizing to crop size and back (still looks decent).
        const resized = tf.image.resizeBilinear(xMasked, [CROP, CROP], false);
        const outSmall = model.predict(resized);
        pred = tf.image.resizeBilinear(outSmall, [H, W], false);
      } else {
        pred = model.predict(xMasked);
      }
      drawRGBTensorToCanvas(pred, cvRecon);
    } else {
      drawRGBTensorToCanvas(xMasked, cvRecon);
    }
  });

  drawMaskOverlay(cvMasked, x, y, size);
}

// -----------------------------------------------------------------------------
// Training step (custom loop)
async function trainStep(){
  if (!imgTargetFull || !model) {
    log("Upload a photo and build the model first.", "error");
    return NaN;
  }

  const size = clampInt(+maskSizeEl.value, 12, 256);

  const lossTensor = tf.tidy(() => {
    // Training example = random crop (more data), random mask inside that crop.
    const yTrue = USE_CROP ? randomCrop(imgTargetFull) : imgTargetFull;
    const [_, h, w, __] = yTrue.shape;

    const { x, y } = randomMaskXY(h, w, size);
    const mask = makeMask(h, w, x, y, size);
    const xMasked = applyMaskToImage(yTrue, mask);

    const vars = model.trainableWeights.map(wt => wt.val);
    const { value, grads } = tf.variableGrads(() => {
      const yPred = model.apply(xMasked);
      return lossFn(yTrue, yPred, mask);
    }, vars);

    opt.applyGradients(grads);
    Object.values(grads).forEach(g => g.dispose());

    // Dispose training tensors
    yTrue.dispose();
    mask.dispose();
    xMasked.dispose();

    return value;
  });

  const lossVal = (await lossTensor.data())[0];
  lossTensor.dispose();

  step++;
  setStatus(lossVal);

  // Plateau tracking
  if (lossVal + MIN_DELTA < bestLoss) {
    bestLoss = lossVal;
    stepsSinceBest = 0;
  } else {
    stepsSinceBest++;
  }

  if (step % LOG_EVERY === 0 || step === 1) log(`step=${step} loss=${fmt(lossVal)}`, "info");
  if (step % RENDER_EVERY === 0) renderInteractive();

  return lossVal;
}

// -----------------------------------------------------------------------------
// Auto loop
async function autoLoop(){
  if (!auto) return;

  const spf = clampInt(+spfEl.value, 1, 30);

  for (let i=0; i<spf; i++){
    const lossVal = await trainStep();
    await tf.nextFrame();
    if (!auto) return;

    if (!Number.isFinite(lossVal)) {
      log("Auto-stop: loss not finite.", "error");
      stopAuto();
      return;
    }
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
  log(`Model built. Adam lr=${lr}. Train mode: ${USE_CROP ? "random crops" : "full image"}.`, "ok");
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
    if (imgTargetFull) imgTargetFull.dispose();
    imgTargetFull = await loadImageToTensor(f);

    const size = clampInt(+maskSizeEl.value, 12, 256);
    maskXY = { x: Math.floor((W - size)/2), y: Math.floor((H - size)/2) };

    log(`Photo loaded and resized to ${W}×${H}.`, "ok");

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

btnPredict.addEventListener("click", () => {
  stopAuto();
  renderInteractive();
  log("Predicted inpaint for current mask location (preview).", "ok");
});

// Click to move the mask (interactive preview)
cvMasked.addEventListener("click", (ev) => {
  const rect = cvMasked.getBoundingClientRect();
  const px = (ev.clientX - rect.left) / rect.width;
  const py = (ev.clientY - rect.top) / rect.height;

  const size = clampInt(+maskSizeEl.value, 12, 256);
  const x = clampInt(Math.floor(px * W - size/2), 0, W - size);
  const y = clampInt(Math.floor(py * H - size/2), 0, H - size);

  maskXY = { x, y };
  renderInteractive();
  log(`Mask moved to x=${x}, y=${y}, size=${size}.`, "ok");
});

maskSizeEl.addEventListener("change", () => renderInteractive());

// -----------------------------------------------------------------------------
// Boot
(async function main(){
  await tf.ready();
  setStatus(NaN);
  log("Ready. Upload a photo to begin.", "ok");
  log("Quality tips: use Masked-only loss, mask 32–64, train 2–5k steps.", "info");
})();



