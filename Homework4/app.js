```js
// app-2.js
// ChineseMNIST TF.js — DENOISING CNN AUTOENCODER (Browser-only, CSV Upload + Downloads Save/Load)
// ---------------------------------------------------------------------------------------------
// This version is a FULL FIX for 64×64 datasets like ChineseMNIST.
//
// What it fixes vs your current (28×28 MNIST) assumptions:
// 1) Works with 64×64 (=4096 pixels) images.
// 2) Loads CSV where pixels come first and label/character are at the END (common ChineseMNIST CSV layout).
// 3) Does NOT require labels for autoencoding (we only need xs). We still tolerate labels if present.
// 4) Updates model inputShape, slicing, preview drawing, evaluation to 64×64.
// 5) Keeps your UI the same (index.html unchanged).
//
// Expected CSV row formats supported (NO external libs):
// A) ChineseMNIST typical: pixel_0..pixel_4095,label,character  (4098 columns)
//    In our split files written without header, each row is:
///     p0,p1,...,p4095,label,character
// B) MNIST style: label,p0..p4095  (4097 columns)  (rare, but we support it)
//
// IMPORTANT:
// - This is a denoising autoencoder (no classification). We display MSE/PSNR as "Overall Test Accuracy" label.

(() => {
  // ---------------------------
  // Constants (ChineseMNIST)
  // ---------------------------
  const IMG_H = 64;
  const IMG_W = 64;
  const CHANNELS = 1;
  const PIXELS = IMG_H * IMG_W; // 4096

  // Noise factor: tune if you see too much corruption
  const NOISE_FACTOR = 0.25;

  // Preview scale (canvas pixel upscaling)
  const PREVIEW_SCALE = 3;

  // ---------------------------
  // DOM helpers / UI elements
  // ---------------------------
  const el = (id) => document.getElementById(id);

  const trainCsvInput = el("trainCsv");
  const testCsvInput = el("testCsv");
  const trainName = el("trainName");
  const testName = el("testName");

  const btnLoadData = el("btnLoadData");
  const btnTrain = el("btnTrain");
  const btnEval = el("btnEval");
  const btnTest5 = el("btnTest5");
  const btnSave = el("btnSave");
  const btnToggleVisor = el("btnToggleVisor");
  const btnReset = el("btnReset");
  const btnLoadModel = el("btnLoadModel");

  const modelJsonInput = el("modelJson");
  const modelBinInput = el("modelBin");
  const jsonName = el("jsonName");
  const binName = el("binName");

  const dataStatus = el("dataStatus");
  const trainLogs = el("trainLogs");
  const overallAcc = el("overallAcc");
  const previewStrip = el("previewStrip");
  const modelInfo = el("modelInfo");

  // ---------------------------
  // App state (tensors + models)
  // ---------------------------
  let trainXsAll = null; // tf.Tensor4D [N,64,64,1]
  let testXsAll = null;  // tf.Tensor4D [N,64,64,1]

  let split = null;      // {trainXs, valXs}

  let modelMax = null;
  let modelAvg = null;

  let activeModelKey = "max"; // "max" | "avg"
  let busy = false;

  // ---------------------------
  // Utility: logging + status
  // ---------------------------
  function setStatus(text) {
    dataStatus.textContent = text;
  }

  function log(text) {
    const ts = new Date().toLocaleTimeString();
    trainLogs.textContent += `[${ts}] ${text}\n`;
    trainLogs.scrollTop = trainLogs.scrollHeight;
  }

  function clearLogs() {
    trainLogs.textContent = "";
  }

  function clearPreview() {
    previewStrip.innerHTML = "";
  }

  function getActiveModel() {
    return activeModelKey === "avg" ? modelAvg : modelMax;
  }

  function setButtonsEnabled() {
    const hasData = !!(trainXsAll && testXsAll && split);
    const hasAnyModel = !!(modelMax || modelAvg);
    const active = getActiveModel();

    btnTrain.disabled = !hasData || busy;
    btnEval.disabled = !hasData || !hasAnyModel || busy;
    btnTest5.disabled = !hasData || !hasAnyModel || busy;
    btnSave.disabled = !active || busy;

    btnLoadData.disabled = busy;
    btnLoadModel.disabled = busy;
    btnReset.disabled = busy;
  }

  function friendlyError(err) {
    const msg = (err && err.message) ? err.message : String(err);
    return msg.replace(/\s+/g, " ").trim();
  }

  function bytesToMB(x) {
    return (x / (1024 * 1024)).toFixed(1);
  }

  // ---------------------------
  // CSV loading (NO external libs)
  // ---------------------------
  // Reads a CSV file into xs tensor [N,64,64,1] normalized to [0,1].
  //
  // Supports:
  // - 4098 columns: p0..p4095,label,character  (we ignore label/character)
  // - 4097 columns: label,p0..p4095           (we ignore label)
  //
  // Ignores empty lines.
  async function loadXsFromCsvFile(file) {
    if (!file) throw new Error("No CSV file provided.");

    // Read as text (OK for 15k rows; still manageable). If your file is huge, we can chunk,
    // but this keeps the code simple and reliable across browsers.
    const text = await file.text();

    // Split into lines (handle Windows newlines)
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length === 0) throw new Error("CSV appears empty.");

    // Detect whether the first line is a header (contains non-numeric tokens).
    // Our split files were written without headers, but we guard anyway.
    const firstParts = lines[0].split(",");
    const headerLike = firstParts.some((p) => p.trim() === "" || Number.isNaN(+p.trim()));
    const startLine = headerLike ? 1 : 0;

    // We'll build a Float32Array for all pixels: N * 4096
    const rows = [];
    for (let i = startLine; i < lines.length; i++) {
      const parts = lines[i].split(",");
      // Quick skip for obviously bad rows
      if (parts.length < PIXELS) continue;
      rows.push(parts);
    }

    if (rows.length === 0) throw new Error("No valid data rows found.");

    // Determine row format by column count (from first valid row)
    const colCount = rows[0].length;

    const isPixelsFirst_4098 = (colCount === (PIXELS + 2)); // p0..p4095,label,character
    const isLabelFirst_4097 = (colCount === (PIXELS + 1));  // label,p0..p4095
    const isPixelsOnly_4096 = (colCount === PIXELS);        // p0..p4095 (no label)
    if (!isPixelsFirst_4098 && !isLabelFirst_4097 && !isPixelsOnly_4096) {
      throw new Error(
        `Unexpected column count: ${colCount}. Expected 4096, 4097, or 4098 columns for 64×64 data.`
      );
    }

    const n = rows.length;
    const all = new Float32Array(n * PIXELS);

    // Parse rows → pixels
    // We normalize /255.
    for (let r = 0; r < n; r++) {
      const parts = rows[r];

      let pixelStart = 0;
      if (isLabelFirst_4097) {
        pixelStart = 1; // skip label
      } else {
        pixelStart = 0; // pixels first
      }

      // For pixels-first 4098, the last 2 columns are label+character. We stop at pixelStart+4096 anyway.
      const base = r * PIXELS;
      for (let j = 0; j < PIXELS; j++) {
        const v = +parts[pixelStart + j];
        all[base + j] = (v / 255.0);
      }

      // Keep UI responsive for big files
      if (r % 2000 === 0) await tf.nextFrame();
    }

    // Create tensor4d [N,64,64,1]
    const xs = tf.tensor4d(all, [n, IMG_H, IMG_W, CHANNELS], "float32");
    return xs;
  }

  // Split train/val tensors (no labels needed)
  function splitTrainVal(xs, valRatio = 0.1) {
    const n = xs.shape[0];
    const valN = Math.max(1, Math.floor(n * valRatio));
    const trainN = n - valN;

    const trainXs = xs.slice([0, 0, 0, 0], [trainN, IMG_H, IMG_W, CHANNELS]);
    const valXs = xs.slice([trainN, 0, 0, 0], [valN, IMG_H, IMG_W, CHANNELS]);

    return { trainXs, valXs };
  }

  // Random batch from xs only (autoencoder doesn't require ys)
  function getRandomBatch(xs, k = 5) {
    const n = xs.shape[0];
    const kk = Math.min(k, n);

    const idx = new Int32Array(kk);
    for (let i = 0; i < kk; i++) idx[i] = (Math.random() * n) | 0;

    const idxTensor = tf.tensor1d(idx, "int32");
    const batchXs = tf.gather(xs, idxTensor);
    idxTensor.dispose();
    return batchXs;
  }

  // Generic draw (works for any H×W)
  function drawToCanvas(imageTensor, canvas, h, w, scale = 3) {
    const s = scale | 0;
    canvas.width = w * s;
    canvas.height = h * s;

    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(w * s, h * s);

    const data = tf.tidy(() => {
      let t = imageTensor;
      if (t.rank === 4) t = t.squeeze([0, 3]);     // [1,H,W,1] -> [H,W]
      else if (t.rank === 3) t = t.squeeze([2]);   // [H,W,1]   -> [H,W]
      return t.dataSync(); // length H*W
    });

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = data[y * w + x];
        const c = Math.max(0, Math.min(255, (v * 255) | 0));

        for (let dy = 0; dy < s; dy++) {
          for (let dx = 0; dx < s; dx++) {
            const px = (y * s + dy) * (w * s) + (x * s + dx);
            const o = px * 4;
            imgData.data[o + 0] = c;
            imgData.data[o + 1] = c;
            imgData.data[o + 2] = c;
            imgData.data[o + 3] = 255;
          }
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }

  // ---------------------------
  // Step 1: Noise injection
  // ---------------------------
  function addRandomNoise(xs, noiseFactor = NOISE_FACTOR) {
    // xs shape [N,64,64,1], values in [0,1]
    return tf.tidy(() => {
      const noise = tf.randomNormal(xs.shape, 0, 1, "float32");
      return xs.add(noise.mul(noiseFactor)).clipByValue(0, 1);
    });
  }

  // ---------------------------
  // Step 2: Autoencoder builder (64×64)
  // ---------------------------
  // We keep the same idea:
  // Encoder: Conv -> Pool -> Conv -> Pool
  // Decoder: UpSample -> Conv -> UpSample -> Conv -> Output
  //
  // Output uses sigmoid, loss uses BCE to avoid the "all-black" shortcut.
  function buildAutoencoder(poolType = "max") {
    const isMax = poolType === "max";
    const m = tf.sequential();

    // Encoder
    m.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      inputShape: [IMG_H, IMG_W, CHANNELS]
    }));

    m.add(isMax
      ? tf.layers.maxPooling2d({ poolSize: 2, strides: 2 })
      : tf.layers.averagePooling2d({ poolSize: 2, strides: 2 })
    );

    m.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
      padding: "same"
    }));

    m.add(isMax
      ? tf.layers.maxPooling2d({ poolSize: 2, strides: 2 })
      : tf.layers.averagePooling2d({ poolSize: 2, strides: 2 })
    );

    // Decoder (UpSampling + Conv)
    m.add(tf.layers.upSampling2d({ size: [2, 2] }));
    m.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
      padding: "same"
    }));

    m.add(tf.layers.upSampling2d({ size: [2, 2] }));
    m.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      padding: "same"
    }));

    // Output layer: sigmoid → [0,1]
    m.add(tf.layers.conv2d({
      filters: 1,
      kernelSize: 3,
      activation: "sigmoid",
      padding: "same"
    }));

    m.compile({
      optimizer: tf.train.adam(1e-3),
      loss: "binaryCrossentropy"
    });

    return m;
  }

  // ---------------------------
  // Model info rendering + selector (no index.html edits)
  // ---------------------------
  function renderModelInfo() {
    const active = getActiveModel();

    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.alignItems = "center";
    container.style.justifyContent = "space-between";
    container.style.gap = "10px";
    container.style.marginBottom = "10px";

    const title = document.createElement("div");
    title.textContent = "Active model for Save/Eval:";
    title.style.fontSize = "11px";
    title.style.color = "#93a4c7";

    const select = document.createElement("select");
    select.style.background = "rgba(255,255,255,.06)";
    select.style.border = "1px solid rgba(255,255,255,.14)";
    select.style.borderRadius = "10px";
    select.style.color = "#e8eeff";
    select.style.padding = "8px 10px";
    select.style.fontSize = "12px";
    select.innerHTML = `
      <option value="max">Autoencoder (Max Pooling)</option>
      <option value="avg">Autoencoder (Avg Pooling)</option>
    `;
    select.value = activeModelKey;
    select.addEventListener("change", () => {
      activeModelKey = select.value;
      renderModelInfo();
      setButtonsEnabled();
      log(`Active model switched to: ${activeModelKey.toUpperCase()} pooling`);
    });

    container.appendChild(title);
    container.appendChild(select);

    const summaryLines = [];
    if (active) active.summary(200, undefined, (line) => summaryLines.push(line));
    else summaryLines.push("No model built yet.");

    modelInfo.innerHTML = "";
    modelInfo.appendChild(container);

    const pre = document.createElement("pre");
    pre.style.margin = "0";
    pre.style.whiteSpace = "pre-wrap";
    pre.textContent = summaryLines.join("\n");
    modelInfo.appendChild(pre);
  }

  // ---------------------------
  // tfjs-vis visor toggle
  // ---------------------------
  function showOrToggleVisor() {
    const visor = tfvis.visor();
    visor.isOpen() ? visor.close() : visor.open();
  }

  // ---------------------------
  // Data loading
  // ---------------------------
  async function onLoadData() {
    if (busy) return;
    busy = true;
    setButtonsEnabled();

    try {
      clearLogs();
      clearPreview();
      overallAcc.textContent = "—";

      const trainFile = trainCsvInput.files && trainCsvInput.files[0];
      const testFile = testCsvInput.files && testCsvInput.files[0];
      if (!trainFile || !testFile) {
        throw new Error("Please select BOTH train.csv and test.csv.");
      }

      disposeAllTensors();
      disposeModels();

      setStatus(
        `Loading...\n` +
        `Train file: ${trainFile.name} (${bytesToMB(trainFile.size)} MB)\n` +
        `Test file:  ${testFile.name} (${bytesToMB(testFile.size)} MB)\n\n` +
        `Parsing CSV → tensors (64×64)...`
      );

      log("Parsing training CSV (64×64)...");
      trainXsAll = await loadXsFromCsvFile(trainFile);
      await tf.nextFrame();

      log("Parsing test CSV (64×64)...");
      testXsAll = await loadXsFromCsvFile(testFile);
      await tf.nextFrame();

      split = splitTrainVal(trainXsAll, 0.1);

      modelMax = buildAutoencoder("max");
      modelAvg = buildAutoencoder("avg");
      activeModelKey = "max";
      renderModelInfo();

      const trainN = trainXsAll.shape[0];
      const valN = split.valXs.shape[0];
      const testN = testXsAll.shape[0];

      setStatus(
        `Loaded (64×64).\n` +
        `Train: ${trainN} samples → xs ${trainXsAll.shape}\n` +
        `Val:   ${valN} samples → xs ${split.valXs.shape}\n` +
        `Test:  ${testN} samples → xs ${testXsAll.shape}\n\n` +
        `Noise: Gaussian factor=${NOISE_FACTOR}\n` +
        `Loss: binaryCrossentropy (prevents all-black collapse)\n` +
        `Decoder: UpSampling2D + Conv2D (stable in browsers)`
      );

      log(`Data ready. Train=${trainN}, Val=${valN}, Test=${testN}`);
      log("Built two autoencoders: MAX pooling and AVG pooling.");
    } catch (err) {
      setStatus(`Error loading data:\n${friendlyError(err)}`);
      log(`ERROR: ${friendlyError(err)}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // ---------------------------
  // Training
  // ---------------------------
  async function trainOneModel(modelToTrain, label, trainXsClean, valXsClean) {
    const epochs = 15;
    const batchSize = 64; // 64×64 uses more memory; 128 can be heavy on Safari.

    log(`Training ${label}: epochs=${epochs}, batchSize=${batchSize}, noise=${NOISE_FACTOR}`);

    const t0 = performance.now();

    // Create noisy inputs; targets remain clean.
    const noisyTrainXs = addRandomNoise(trainXsClean, NOISE_FACTOR);
    const noisyValXs = addRandomNoise(valXsClean, NOISE_FACTOR);

    const fitSurface = { name: `AE ${label} (Loss)`, tab: "Training" };
    const callbacks = tfvis.show.fitCallbacks(
      fitSurface,
      ["loss", "val_loss"],
      {
        callbacks: [{
          onEpochEnd: async (epoch, logs) => {
            log(`${label} epoch ${epoch + 1}: loss=${logs.loss?.toFixed(5)} val_loss=${logs.val_loss?.toFixed(5)}`);
            await tf.nextFrame();
          }
        }]
      }
    );

    const history = await modelToTrain.fit(noisyTrainXs, trainXsClean, {
      epochs,
      batchSize,
      shuffle: true,
      validationData: [noisyValXs, valXsClean],
      callbacks
    });

    noisyTrainXs.dispose();
    noisyValXs.dispose();

    const t1 = performance.now();
    const durSec = (t1 - t0) / 1000;

    const valLossHist = history.history.val_loss || [];
    const bestValLoss = valLossHist.length ? Math.min(...valLossHist) : NaN;

    log(`Training ${label} done in ${durSec.toFixed(2)}s. Best val_loss=${bestValLoss.toFixed(5)}`);
    return { durSec, bestValLoss };
  }

  async function onTrain() {
    if (busy) return;
    if (!split || !modelMax || !modelAvg) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();
      overallAcc.textContent = "—";

      log("=== TRAINING START (Autoencoders) ===");
      const resMax = await trainOneModel(modelMax, "MAX", split.trainXs, split.valXs);
      await tf.nextFrame();
      const resAvg = await trainOneModel(modelAvg, "AVG", split.trainXs, split.valXs);

      log("=== TRAINING COMPLETE ===");
      log(`MAX: ${resMax.durSec.toFixed(2)}s, best val_loss=${resMax.bestValLoss.toFixed(5)}`);
      log(`AVG: ${resAvg.durSec.toFixed(2)}s, best val_loss=${resAvg.bestValLoss.toFixed(5)}`);

      setStatus(
        dataStatus.textContent +
        `\n\nTraining finished.\n` +
        `MAX best val_loss: ${resMax.bestValLoss.toFixed(5)}\n` +
        `AVG best val_loss: ${resAvg.bestValLoss.toFixed(5)}\n` +
        `Now click "Test 5 Random" to see denoising outputs.`
      );

      renderModelInfo();
    } catch (err) {
      log(`ERROR (training): ${friendlyError(err)}`);
      setStatus(`Training error:\n${friendlyError(err)}\n\n${dataStatus.textContent}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // ---------------------------
  // Evaluation (MSE/PSNR)
  // ---------------------------
  async function evaluateMSE(modelToEval, label, testXsClean) {
    const batchSize = 128;
    const n = testXsClean.shape[0];

    const testXsNoisy = addRandomNoise(testXsClean, NOISE_FACTOR);

    let sum = 0;
    let count = 0;

    for (let start = 0; start < n; start += batchSize) {
      const size = Math.min(batchSize, n - start);

      const mseVal = tf.tidy(() => {
        const xClean = testXsClean.slice([start, 0, 0, 0], [size, IMG_H, IMG_W, CHANNELS]);
        const xNoisy = testXsNoisy.slice([start, 0, 0, 0], [size, IMG_H, IMG_W, CHANNELS]);
        const recon = modelToEval.predict(xNoisy);
        const mse = recon.sub(xClean).square().mean();
        return mse.dataSync()[0];
      });

      sum += mseVal * size;
      count += size;
      await tf.nextFrame();
    }

    testXsNoisy.dispose();

    const mse = sum / count;
    const psnr = 20 * Math.log10(1.0) - 10 * Math.log10(mse);
    log(`${label} test MSE=${mse.toFixed(6)} PSNR=${psnr.toFixed(2)} dB`);
    return { mse, psnr };
  }

  async function onEvaluate() {
    if (busy) return;
    if (!testXsAll || !modelMax || !modelAvg) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();
      log("Evaluating denoising performance (MSE/PSNR) on test set...");
      await tf.nextFrame();

      const rMax = await evaluateMSE(modelMax, "MAX", testXsAll);
      const rAvg = await evaluateMSE(modelAvg, "AVG", testXsAll);

      overallAcc.textContent =
        `MAX MSE ${rMax.mse.toFixed(5)} | AVG MSE ${rAvg.mse.toFixed(5)} (noise=${NOISE_FACTOR})`;

      tfvis.render.barchart(
        { name: "Denoising MSE (Lower is Better)", tab: "Evaluation" },
        [
          { index: "MAX", value: rMax.mse },
          { index: "AVG", value: rAvg.mse }
        ],
        { xLabel: "Model", yLabel: "MSE", height: 300 }
      );

      tfvis.render.barchart(
        { name: "Denoising PSNR (Higher is Better)", tab: "Evaluation" },
        [
          { index: "MAX", value: rMax.psnr },
          { index: "AVG", value: rAvg.psnr }
        ],
        { xLabel: "Model", yLabel: "PSNR (dB)", height: 300 }
      );

      log("Evaluation charts rendered in Visor (Evaluation tab).");
    } catch (err) {
      log(`ERROR (evaluation): ${friendlyError(err)}`);
      setStatus(`Evaluation error:\n${friendlyError(err)}\n\n${dataStatus.textContent}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // ---------------------------
  // Step 3: Test 5 Random preview
  // ---------------------------
  async function onTestFive() {
    if (busy) return;
    if (!testXsAll || !modelMax || !modelAvg) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();

      const batchXs = getRandomBatch(testXsAll, 5);
      const noisyBatch = addRandomNoise(batchXs, NOISE_FACTOR);

      const reconMax = modelMax.predict(noisyBatch);
      const reconAvg = modelAvg.predict(noisyBatch);

      // Range debug: if outputs are near 0, previews look black.
      const maxMin = reconMax.min().dataSync()[0];
      const maxMax = reconMax.max().dataSync()[0];
      const avgMin = reconAvg.min().dataSync()[0];
      const avgMax = reconAvg.max().dataSync()[0];
      log(`Recon ranges -> MAX: [${maxMin.toFixed(4)}, ${maxMax.toFixed(4)}], AVG: [${avgMin.toFixed(4)}, ${avgMax.toFixed(4)}]`);

      for (let i = 0; i < 5; i++) {
        const item = document.createElement("div");
        item.className = "previewItem";
        item.style.minWidth = "320px";

        const title = document.createElement("div");
        title.style.fontSize = "11px";
        title.style.color = "#93a4c7";
        title.style.marginBottom = "6px";
        title.textContent = `Sample #${i + 1}`;
        item.appendChild(title);

        const grid = document.createElement("div");
        grid.style.display = "grid";
        grid.style.gridTemplateColumns = "repeat(4, 1fr)";
        grid.style.gap = "6px";

        const makeCell = (labelText) => {
          const wrap = document.createElement("div");
          wrap.style.textAlign = "center";

          const c = document.createElement("canvas");
          c.width = IMG_W * PREVIEW_SCALE;
          c.height = IMG_H * PREVIEW_SCALE;

          const lbl = document.createElement("div");
          lbl.style.fontSize = "10px";
          lbl.style.color = "#93a4c7";
          lbl.style.marginTop = "4px";
          lbl.textContent = labelText;

          wrap.appendChild(c);
          wrap.appendChild(lbl);
          return { wrap, canvas: c };
        };

        const cleanCell = makeCell("Clean");
        const noisyCell = makeCell("Noisy");
        const maxCell = makeCell("Denoised (Max)");
        const avgCell = makeCell("Denoised (Avg)");

        grid.appendChild(cleanCell.wrap);
        grid.appendChild(noisyCell.wrap);
        grid.appendChild(maxCell.wrap);
        grid.appendChild(avgCell.wrap);

        item.appendChild(grid);
        previewStrip.appendChild(item);

        const clean = batchXs.slice([i, 0, 0, 0], [1, IMG_H, IMG_W, CHANNELS]);
        const noisy = noisyBatch.slice([i, 0, 0, 0], [1, IMG_H, IMG_W, CHANNELS]);
        const dMax = reconMax.slice([i, 0, 0, 0], [1, IMG_H, IMG_W, CHANNELS]);
        const dAvg = reconAvg.slice([i, 0, 0, 0], [1, IMG_H, IMG_W, CHANNELS]);

        drawToCanvas(clean, cleanCell.canvas, IMG_H, IMG_W, PREVIEW_SCALE);
        drawToCanvas(noisy, noisyCell.canvas, IMG_H, IMG_W, PREVIEW_SCALE);
        drawToCanvas(dMax, maxCell.canvas, IMG_H, IMG_W, PREVIEW_SCALE);
        drawToCanvas(dAvg, avgCell.canvas, IMG_H, IMG_W, PREVIEW_SCALE);

        clean.dispose(); noisy.dispose(); dMax.dispose(); dAvg.dispose();
        await tf.nextFrame();
      }

      log(`Previewed 5 random test samples with noise=${NOISE_FACTOR} (Clean|Noisy|Max|Avg).`);

      reconMax.dispose();
      reconAvg.dispose();
      noisyBatch.dispose();
      batchXs.dispose();
    } catch (err) {
      log(`ERROR (test 5): ${friendlyError(err)}`);
      setStatus(`Preview error:\n${friendlyError(err)}\n\n${dataStatus.textContent}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // ---------------------------
  // Step 4: Save / Load model files (ACTIVE slot)
  // ---------------------------
  async function onSaveDownload() {
    if (busy) return;
    const active = getActiveModel();
    if (!active) return;

    busy = true;
    setButtonsEnabled();

    try {
      const name = activeModelKey === "avg" ? "chinesemnist-ae-avgpool-64" : "chinesemnist-ae-maxpool-64";
      log(`Saving ACTIVE model (${activeModelKey.toUpperCase()}) to downloads as '${name}'...`);
      await active.save(`downloads://${name}`);
      log(`Model download triggered: ${name}.json + ${name}.weights.bin`);
    } catch (err) {
      log(`ERROR (save): ${friendlyError(err)}`);
      setStatus(`Save error:\n${friendlyError(err)}\n\n${dataStatus.textContent}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  async function onLoadFromFiles() {
    if (busy) return;
    busy = true;
    setButtonsEnabled();

    try {
      const jsonFile = modelJsonInput.files && modelJsonInput.files[0];
      const binFile = modelBinInput.files && modelBinInput.files[0];
      if (!jsonFile || !binFile) {
        throw new Error("Please choose BOTH model.json and weights.bin to load the model.");
      }

      log(`Loading model from files into ACTIVE slot (${activeModelKey.toUpperCase()})...`);
      await tf.nextFrame();

      const loaded = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));

      // Compile for eval/further training
      loaded.compile({
        optimizer: tf.train.adam(1e-3),
        loss: "binaryCrossentropy"
      });

      if (activeModelKey === "avg") {
        if (modelAvg) modelAvg.dispose();
        modelAvg = loaded;
      } else {
        if (modelMax) modelMax.dispose();
        modelMax = loaded;
      }

      renderModelInfo();
      log("Model loaded successfully from files.");
    } catch (err) {
      log(`ERROR (load model): ${friendlyError(err)}`);
      setStatus(`Load model error:\n${friendlyError(err)}\n\n${dataStatus.textContent}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // ---------------------------
  // Reset + cleanup
  // ---------------------------
  function disposeAllTensors() {
    if (split) {
      split.trainXs?.dispose?.();
      split.valXs?.dispose?.();
      split = null;
    }
    if (trainXsAll) {
      trainXsAll.dispose();
      trainXsAll = null;
    }
    if (testXsAll) {
      testXsAll.dispose();
      testXsAll = null;
    }
  }

  function disposeModels() {
    if (modelMax) { modelMax.dispose(); modelMax = null; }
    if (modelAvg) { modelAvg.dispose(); modelAvg = null; }
  }

  function onReset() {
    if (busy) return;

    try {
      clearPreview();
      clearLogs();
      overallAcc.textContent = "—";

      disposeAllTensors();
      disposeModels();

      activeModelKey = "max";
      renderModelInfo();

      setStatus(`Reset.\nUpload train/test CSV files, then click Load Data.\nImage size=${IMG_H}×${IMG_W}\nNoise factor=${NOISE_FACTOR}`);

      trainCsvInput.value = "";
      testCsvInput.value = "";
      modelJsonInput.value = "";
      modelBinInput.value = "";

      trainName.textContent = "No file selected";
      testName.textContent = "No file selected";
      jsonName.textContent = "No file";
      binName.textContent = "No file";
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // ---------------------------
  // Wire UI
  // ---------------------------
  function bindUI() {
    trainCsvInput.addEventListener("change", () => {
      const f = trainCsvInput.files && trainCsvInput.files[0];
      trainName.textContent = f ? f.name : "No file selected";
    });

    testCsvInput.addEventListener("change", () => {
      const f = testCsvInput.files && testCsvInput.files[0];
      testName.textContent = f ? f.name : "No file selected";
    });

    modelJsonInput.addEventListener("change", () => {
      const f = modelJsonInput.files && modelJsonInput.files[0];
      jsonName.textContent = f ? f.name : "No file";
    });

    modelBinInput.addEventListener("change", () => {
      const f = modelBinInput.files && modelBinInput.files[0];
      binName.textContent = f ? f.name : "No file";
    });

    btnLoadData.addEventListener("click", () => onLoadData());
    btnTrain.addEventListener("click", () => onTrain());
    btnEval.addEventListener("click", () => onEvaluate());
    btnTest5.addEventListener("click", () => onTestFive());
    btnSave.addEventListener("click", () => onSaveDownload());
    btnLoadModel.addEventListener("click", () => onLoadFromFiles());
    btnReset.addEventListener("click", () => onReset());
    btnToggleVisor.addEventListener("click", () => showOrToggleVisor());
  }

  // ---------------------------
  // Boot
  // ---------------------------
  function boot() {
    try { tfvis.visor().close(); } catch (_) {}

    setStatus(
      `Ready.\nUpload train/test CSV files, then click Load Data.\n` +
      `Image size=${IMG_H}×${IMG_W} (${PIXELS} pixels)\n` +
      `Noise factor=${NOISE_FACTOR}`
    );

    renderModelInfo();
    bindUI();
    setButtonsEnabled();

    log("Autoencoder mode (64×64): Train noisy→clean. Preview shows Clean|Noisy|Denoised(Max)|Denoised(Avg).");
    log("Loss: BCE + sigmoid output (prevents all-black collapse).");
    log("CSV loader supports pixels-first (4098 cols) and label-first (4097 cols).");
  }

  boot();
})();
```
