// app.js
// MNIST TF.js — DENOISING CNN AUTOENCODER (Browser-only, CSV Upload + Downloads Save/Load)
// -------------------------------------------------------------------------------------
// This replaces the classifier logic with a denoising autoencoder homework flow:
//
// Step 1) Add random noise to test data (no network fetch).
// Step 2) Train CNN autoencoder to map noisy -> clean.
// Step 3) "Test 5 Random" shows Clean | Noisy | Denoised for BOTH max-pooling and avg-pooling
//         (two reconstructions side-by-side, same input batch).
// Step 4) Save model and reload to reproduce results.
//
// IMPORTANT:
// - This file assumes data-loader.js provides:
//    loadTrainFromFiles, loadTestFromFiles, splitTrainVal, getRandomTestBatch, draw28x28ToCanvas
// - We DO NOT use labels for training (autoencoder target is the clean image itself).
// - We still keep ys tensors around because your loader makes them; we dispose properly on reset.
//
// UI NOTE:
// - We keep the existing index.html controls.
// - We support saving/downloading the currently "active" model (max or avg), selected via a dropdown
//   we create dynamically in Model Info area (so no index.html edits required).
//
// SAFETY/PERFORMANCE:
// - Uses tf.tidy in hot paths.
// - Disposes intermediate tensors.
// - Uses batching for evaluation metrics.
// - Yields to UI via tf.nextFrame()/requestAnimationFrame.

(() => {
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
  let trainAll = null; // {xs, ys}
  let testAll = null;  // {xs, ys}
  let split = null;    // {trainXs, trainYs, valXs, valYs} -- we will use xs only

  // We train TWO models: max-pooling AE and avg-pooling AE
  let modelMax = null;
  let modelAvg = null;

  // Which model is currently "active" for Save/Load/Eval buttons
  let activeModelKey = "max"; // "max" | "avg"
  let busy = false;

  // Noise settings (homework step 1)
  // You can expose this to the UI later; for now, keep it here.
  const NOISE_FACTOR = 0.4;

  // For preview drawing
  const PREVIEW_SCALE = 4;

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
    const hasData = !!(trainAll && testAll && split);
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
  // Noise injection (Homework Step 1)
  // ---------------------------
  // Adds Gaussian noise to xs and clips to [0,1].
  // xs expected shape: [N,28,28,1]
  function addRandomNoise(xs, noiseFactor = NOISE_FACTOR) {
    return tf.tidy(() => {
      const noise = tf.randomNormal(xs.shape, 0, 1, "float32");
      return xs.add(noise.mul(noiseFactor)).clipByValue(0, 1);
    });
  }

  // ---------------------------
  // Autoencoder model builders (Homework Step 2)
  // ---------------------------
  // We build SAME architecture except pooling type:
  // - Encoder: Conv -> Pool -> Conv -> Pool
  // - Decoder: Conv2DTranspose upsample twice -> final Conv sigmoid
  function buildAutoencoder(poolType = "max") {
    // poolType: "max" or "avg"
    const isMax = poolType === "max";

    const m = tf.sequential();

    // Encoder
    m.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      inputShape: [28, 28, 1]
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

    // Decoder
    // Conv2DTranspose with stride 2 upsamples spatially
    m.add(tf.layers.conv2dTranspose({
      filters: 64,
      kernelSize: 3,
      strides: 2,
      activation: "relu",
      padding: "same"
    }));

    m.add(tf.layers.conv2dTranspose({
      filters: 32,
      kernelSize: 3,
      strides: 2,
      activation: "relu",
      padding: "same"
    }));

    // Output layer: 1 channel image in [0,1]
    m.add(tf.layers.conv2d({
      filters: 1,
      kernelSize: 3,
      activation: "sigmoid",
      padding: "same"
    }));

    // Denoising objective: pixel reconstruction
    m.compile({
      optimizer: "adam",
      loss: "meanSquaredError"
    });

    return m;
  }

  // Renders model summary + adds a tiny model selector UI (no index.html edits needed)
  function renderModelInfo() {
    const active = getActiveModel();

    // Build a little selector above summary (inside modelInfo)
    // Using plain DOM to stay dependency-free.
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
      renderModelInfo(); // re-render summary for the new active model
      setButtonsEnabled();
      log(`Active model switched to: ${activeModelKey.toUpperCase()} pooling`);
    });

    container.appendChild(title);
    container.appendChild(select);

    // Capture model.summary() output
    const summaryLines = [];
    if (active) {
      active.summary(100, undefined, (line) => summaryLines.push(line));
    } else {
      summaryLines.push("No model built yet.");
    }

    // Replace modelInfo content
    modelInfo.innerHTML = "";
    modelInfo.appendChild(container);

    const pre = document.createElement("pre");
    pre.style.margin = "0";
    pre.style.whiteSpace = "pre-wrap";
    pre.textContent = summaryLines.join("\n");
    modelInfo.appendChild(pre);
  }

  // ---------------------------
  // tfjs-vis utilities
  // ---------------------------
  function showOrToggleVisor() {
    const visor = tfvis.visor();
    visor.isOpen() ? visor.close() : visor.open();
  }

  // ---------------------------
  // Data loading (same as before, but now used for denoising)
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
        throw new Error("Please select BOTH mnist_train.csv and mnist_test.csv.");
      }

      disposeAllTensors();

      setStatus(
        `Loading...\n` +
        `Train file: ${trainFile.name} (${bytesToMB(trainFile.size)} MB)\n` +
        `Test file:  ${testFile.name} (${bytesToMB(testFile.size)} MB)\n\n` +
        `Parsing CSV → tensors (this can take a bit).`
      );

      log("Parsing training CSV...");
      trainAll = await window.loadTrainFromFiles(trainFile);
      await tf.nextFrame();

      log("Parsing test CSV...");
      testAll = await window.loadTestFromFiles(testFile);
      await tf.nextFrame();

      split = window.splitTrainVal(trainAll.xs, trainAll.ys, 0.1);

      // Build both models fresh (max + avg)
      if (modelMax) modelMax.dispose();
      if (modelAvg) modelAvg.dispose();
      modelMax = buildAutoencoder("max");
      modelAvg = buildAutoencoder("avg");

      // Default active: max
      activeModelKey = "max";
      renderModelInfo();

      const trainN = trainAll.xs.shape[0];
      const valN = split.valXs.shape[0];
      const testN = testAll.xs.shape[0];

      setStatus(
        `Loaded.\n` +
        `Train: ${trainN} samples → xs ${trainAll.xs.shape}\n` +
        `Val:   ${valN} samples → xs ${split.valXs.shape}\n` +
        `Test:  ${testN} samples → xs ${testAll.xs.shape}\n\n` +
        `Noise: Gaussian, factor=${NOISE_FACTOR}\n` +
        `Training target: clean images (denoising autoencoder)`
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
  // Training (Homework Step 2)
  // ---------------------------
  async function trainOneModel(modelToTrain, label, trainXsClean, valXsClean) {
    // We train: noisy -> clean
    // Inputs: noisy version of train/val
    // Targets: clean images
    //
    // We generate noise per-epoch implicitly by regenerating noisy tensors before fit.
    // For simplicity & speed, we generate once per training call (good enough for homework).

    const epochs = 10;
    const batchSize = 128;

    log(`Training ${label}: epochs=${epochs}, batchSize=${batchSize}, noise=${NOISE_FACTOR}`);

    const t0 = performance.now();

    // Create noisy train/val tensors
    const noisyTrainXs = addRandomNoise(trainXsClean, NOISE_FACTOR);
    const noisyValXs = addRandomNoise(valXsClean, NOISE_FACTOR);

    // tfjs-vis live curves
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

    // Train
    const history = await modelToTrain.fit(noisyTrainXs, trainXsClean, {
      epochs,
      batchSize,
      shuffle: true,
      validationData: [noisyValXs, valXsClean],
      callbacks
    });

    // Cleanup noisy tensors
    noisyTrainXs.dispose();
    noisyValXs.dispose();

    const t1 = performance.now();
    const durSec = (t1 - t0) / 1000;

    // Best val loss (lower is better)
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

      // Train both models sequentially (more stable for browser memory)
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
        `Try "Test 5 Random" to compare denoising side-by-side.`
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
  // Evaluation for autoencoder (Homework Step 2/3)
  // ---------------------------
  // For denoising, "accuracy" isn't directly meaningful.
  // Instead we compute:
  // - MSE between reconstruction and clean images (lower is better)
  // - PSNR (optional, derived from MSE): 20*log10(MAX_I) - 10*log10(MSE), MAX_I=1.0
  //
  // We still display "Overall Test Accuracy" field, but now it's "Overall Test MSE / PSNR".
  async function evaluateDenoising(modelToEval, label, testXsClean) {
    const batchSize = 512;
    const n = testXsClean.shape[0];

    // We'll generate one noisy version of entire test set (consistent evaluation)
    // NOTE: This can be memory-heavy for big test sets. For MNIST (10k), it's fine.
    const testXsNoisy = addRandomNoise(testXsClean, NOISE_FACTOR);

    let sumMSE = 0;
    let count = 0;

    for (let start = 0; start < n; start += batchSize) {
      const size = Math.min(batchSize, n - start);

      const mseVal = await tf.tidy(async () => {
        const xClean = testXsClean.slice([start, 0, 0, 0], [size, 28, 28, 1]);
        const xNoisy = testXsNoisy.slice([start, 0, 0, 0], [size, 28, 28, 1]);

        const recon = modelToEval.predict(xNoisy);

        // MSE per batch: mean((recon-clean)^2)
        const mse = recon.sub(xClean).square().mean();
        const v = (await mse.data())[0];

        // Dispose explicitly for clarity (tidy will do it too, but this is educational)
        xClean.dispose();
        xNoisy.dispose();
        recon.dispose();
        mse.dispose();

        return v;
      });

      sumMSE += mseVal * size;
      count += size;

      await tf.nextFrame();
    }

    testXsNoisy.dispose();

    const mse = sumMSE / count;
    const psnr = 20 * Math.log10(1.0) - 10 * Math.log10(mse); // MAX_I=1
    log(`${label} test MSE=${mse.toFixed(6)} PSNR=${psnr.toFixed(2)} dB`);

    return { mse, psnr };
  }

  async function onEvaluate() {
    if (busy) return;
    if (!testAll || !modelMax || !modelAvg) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();
      log("Evaluating denoising performance (MSE/PSNR) on test set...");
      await tf.nextFrame();

      // Evaluate both models
      const rMax = await evaluateDenoising(modelMax, "MAX", testAll.xs);
      const rAvg = await evaluateDenoising(modelAvg, "AVG", testAll.xs);

      // Show summary in the "Overall Test Accuracy" field (reused UI slot)
      overallAcc.textContent =
        `MAX MSE ${rMax.mse.toFixed(5)} | AVG MSE ${rAvg.mse.toFixed(5)} (noise=${NOISE_FACTOR})`;

      // Render a comparison bar chart in tfjs-vis
      tfvis.render.barchart(
        { name: "Denoising MSE (Lower is Better)", tab: "Evaluation" },
        [
          { index: "MAX", value: rMax.mse },
          { index: "AVG", value: rAvg.mse },
        ],
        { xLabel: "Model", yLabel: "MSE", height: 300 }
      );

      tfvis.render.barchart(
        { name: "Denoising PSNR (Higher is Better)", tab: "Evaluation" },
        [
          { index: "MAX", value: rMax.psnr },
          { index: "AVG", value: rAvg.psnr },
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
  // Test 5 Random (Homework Step 3)
  // ---------------------------
  // Display for each sample:
  //   Clean | Noisy | Denoised(MAX) | Denoised(AVG)
  //
  // So each preview item becomes a mini grid of 4 images.
  async function onTestFive() {
    if (busy) return;
    if (!testAll || !modelMax || !modelAvg) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();

      // Get 5 random test images (we only need xs; ys unused here)
      const { batchXs, batchYs } = window.getRandomTestBatch(testAll.xs, testAll.ys, 5);

      // Create noisy inputs
      const noisyBatch = addRandomNoise(batchXs, NOISE_FACTOR);

      // Predict reconstructions from both models
      const reconMax = modelMax.predict(noisyBatch);
      const reconAvg = modelAvg.predict(noisyBatch);

      // Build UI rows
      for (let i = 0; i < 5; i++) {
        const item = document.createElement("div");
        item.className = "previewItem";
        item.style.minWidth = "280px"; // wider because we show 4 images

        // A small title line
        const title = document.createElement("div");
        title.style.fontSize = "11px";
        title.style.color = "#93a4c7";
        title.style.marginBottom = "6px";
        title.textContent = `Sample #${i + 1}`;
        item.appendChild(title);

        // Grid container for 4 canvases
        const grid = document.createElement("div");
        grid.style.display = "grid";
        grid.style.gridTemplateColumns = "repeat(4, 1fr)";
        grid.style.gap = "6px";

        // Helper to create one labeled canvas cell
        const cell = (labelText) => {
          const wrap = document.createElement("div");
          wrap.style.textAlign = "center";

          const c = document.createElement("canvas");
          c.style.width = `${28 * PREVIEW_SCALE}px`;
          c.style.height = `${28 * PREVIEW_SCALE}px`;
          c.width = 28 * PREVIEW_SCALE;
          c.height = 28 * PREVIEW_SCALE;

          const lbl = document.createElement("div");
          lbl.style.fontSize = "10px";
          lbl.style.color = "#93a4c7";
          lbl.style.marginTop = "4px";
          lbl.textContent = labelText;

          wrap.appendChild(c);
          wrap.appendChild(lbl);
          return { wrap, canvas: c };
        };

        const cleanCell = cell("Clean");
        const noisyCell2 = cell("Noisy");
        const maxCell = cell("Denoised (Max)");
        const avgCell = cell("Denoised (Avg)");

        grid.appendChild(cleanCell.wrap);
        grid.appendChild(noisyCell2.wrap);
        grid.appendChild(maxCell.wrap);
        grid.appendChild(avgCell.wrap);

        item.appendChild(grid);
        previewStrip.appendChild(item);

        // Slice tensors for this index and draw
        const clean = batchXs.slice([i, 0, 0, 0], [1, 28, 28, 1]);
        const noisy = noisyBatch.slice([i, 0, 0, 0], [1, 28, 28, 1]);
        const dMax = reconMax.slice([i, 0, 0, 0], [1, 28, 28, 1]);
        const dAvg = reconAvg.slice([i, 0, 0, 0], [1, 28, 28, 1]);

        window.draw28x28ToCanvas(clean, cleanCell.canvas, PREVIEW_SCALE);
        window.draw28x28ToCanvas(noisy, noisyCell2.canvas, PREVIEW_SCALE);
        window.draw28x28ToCanvas(dMax, maxCell.canvas, PREVIEW_SCALE);
        window.draw28x28ToCanvas(dAvg, avgCell.canvas, PREVIEW_SCALE);

        clean.dispose();
        noisy.dispose();
        dMax.dispose();
        dAvg.dispose();

        // Yield so Safari doesn't freeze while drawing
        await tf.nextFrame();
      }

      log(`Previewed 5 random test samples with noise=${NOISE_FACTOR} (Clean|Noisy|Max|Avg).`);

      // Cleanup tensors
      reconMax.dispose();
      reconAvg.dispose();
      noisyBatch.dispose();
      batchXs.dispose();
      batchYs.dispose();
    } catch (err) {
      log(`ERROR (test 5): ${friendlyError(err)}`);
      setStatus(`Preview error:\n${friendlyError(err)}\n\n${dataStatus.textContent}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // ---------------------------
  // Save / Load model (Homework Step 4)
  // ---------------------------
  // Save the ACTIVE model to downloads://
  // NOTE: downloads:// will name the files:
  //   <name>.json and <name>.weights.bin
  async function onSaveDownload() {
    if (busy) return;

    const active = getActiveModel();
    if (!active) return;

    busy = true;
    setButtonsEnabled();

    try {
      const name = activeModelKey === "avg" ? "mnist-ae-avgpool" : "mnist-ae-maxpool";
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

  // Load model from user selected files and assign it to the ACTIVE slot (max or avg).
  // That way students can:
  // - Train both models
  // - Save max model
  // - Reset
  // - Load max model back into max slot
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

      // Ensure it's compiled for evaluation and potential further training
      loaded.compile({
        optimizer: "adam",
        loss: "meanSquaredError"
      });

      // Replace the correct slot
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
  // Reset: dispose everything and clear UI
  // ---------------------------
  function disposeAllTensors() {
    if (split) {
      split.trainXs?.dispose?.();
      split.trainYs?.dispose?.();
      split.valXs?.dispose?.();
      split.valYs?.dispose?.();
      split = null;
    }
    if (trainAll) {
      trainAll.xs?.dispose?.();
      trainAll.ys?.dispose?.();
      trainAll = null;
    }
    if (testAll) {
      testAll.xs?.dispose?.();
      testAll.ys?.dispose?.();
      testAll = null;
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

      setStatus("Reset.\nUpload train/test CSV files, then click Load Data.");

      // Clear file inputs (allow reselect same file)
      trainCsvInput.value = "";
      testCsvInput.value = "";
      modelJsonInput.value = "";
      modelBinInput.value = "";

      trainName.textContent = "No file selected";
      testName.textContent = "No file selected";
      jsonName.textContent = "No file";
      binName.textContent = "No file";

      // Optional: keep visor state as user preference
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // ---------------------------
  // Wire DOM events
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
    // Close visor initially; user can open it.
    try { tfvis.visor().close(); } catch (_) {}

    setStatus(`Ready.\nUpload train/test CSV files, then click Load Data.\nNoise factor=${NOISE_FACTOR}`);
    renderModelInfo();
    bindUI();
    setButtonsEnabled();

    log("Autoencoder mode: Train noisy→clean. Use Test 5 Random to see Clean|Noisy|Denoised(Max)|Denoised(Avg).");
  }

  boot();
})();
