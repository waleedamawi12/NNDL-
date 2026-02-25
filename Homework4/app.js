// app.js
// MNIST TF.js — DENOISING CNN AUTOENCODER (Browser-only, CSV Upload + Downloads Save/Load)
// -------------------------------------------------------------------------------------
// Fixes applied vs. previous version:
// 1) Prevent "all black" trivial solution by using loss='binaryCrossentropy' (sigmoid output).
// 2) Use stable decoder: UpSampling2D + Conv2D (instead of Conv2DTranspose).
// 3) Reduce noise for smaller dataset subsets (6000 train): NOISE_FACTOR=0.25
// 4) Add reconstruction range logging (min/max) to confirm outputs are not near-zero.
//
// Homework mapping:
// Step 1: addRandomNoise() applied to test + training inputs.
// Step 2: train autoencoders (max pooling and avg pooling) noisy->clean.
// Step 3: Test 5 Random shows Clean | Noisy | Denoised(Max) | Denoised(Avg)
// Step 4: Save/Load active model using downloads:// and browserFiles().

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
  let split = null;    // {trainXs, trainYs, valXs, valYs} -- we use xs only

  let modelMax = null;
  let modelAvg = null;

  let activeModelKey = "max"; // "max" | "avg"
  let busy = false;

  // IMPORTANT FIX:
  // With only 6000 training samples, noise=0.4 often encourages the trivial solution (all zeros).
  // Use a slightly gentler noise for stable learning.
  const NOISE_FACTOR = 0.25;

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
  // Step 1: Noise injection
  // ---------------------------
  function addRandomNoise(xs, noiseFactor = NOISE_FACTOR) {
    // xs shape [N,28,28,1], values in [0,1]
    return tf.tidy(() => {
      const noise = tf.randomNormal(xs.shape, 0, 1, "float32");
      return xs.add(noise.mul(noiseFactor)).clipByValue(0, 1);
    });
  }

  // ---------------------------
  // Step 2: Autoencoder builder
  // ---------------------------
  // Key Fix: UpSampling2D + Conv2D decoder is more stable than Conv2DTranspose in browsers.
  // Key Fix: loss='binaryCrossentropy' with sigmoid output prevents "all black" MSE shortcut.
  function buildAutoencoder(poolType = "max") {
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
      // IMPORTANT FIX:
      // BCE punishes missing bright pixels strongly and avoids trivial all-zero outputs.
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
    if (active) active.summary(100, undefined, (line) => summaryLines.push(line));
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
        throw new Error("Please select BOTH mnist_train.csv and mnist_test.csv.");
      }

      disposeAllTensors();

      setStatus(
        `Loading...\n` +
        `Train file: ${trainFile.name} (${bytesToMB(trainFile.size)} MB)\n` +
        `Test file:  ${testFile.name} (${bytesToMB(testFile.size)} MB)\n\n` +
        `Parsing CSV → tensors...`
      );

      log("Parsing training CSV...");
      trainAll = await window.loadTrainFromFiles(trainFile);
      await tf.nextFrame();

      log("Parsing test CSV...");
      testAll = await window.loadTestFromFiles(testFile);
      await tf.nextFrame();

      split = window.splitTrainVal(trainAll.xs, trainAll.ys, 0.1);

      if (modelMax) modelMax.dispose();
      if (modelAvg) modelAvg.dispose();

      modelMax = buildAutoencoder("max");
      modelAvg = buildAutoencoder("avg");

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
    const batchSize = 128;

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
  // Evaluation (simple metric for homework)
  // ---------------------------
  // We re-use the "Overall Test Accuracy" label to show reconstruction quality.
  // We'll compute MSE just as a numeric baseline, even though training uses BCE.
  async function evaluateMSE(modelToEval, label, testXsClean) {
    const batchSize = 512;
    const n = testXsClean.shape[0];

    const testXsNoisy = addRandomNoise(testXsClean, NOISE_FACTOR);

    let sum = 0;
    let count = 0;

    for (let start = 0; start < n; start += batchSize) {
      const size = Math.min(batchSize, n - start);

      const mseVal = tf.tidy(() => {
        const xClean = testXsClean.slice([start, 0, 0, 0], [size, 28, 28, 1]);
        const xNoisy = testXsNoisy.slice([start, 0, 0, 0], [size, 28, 28, 1]);
        const recon = modelToEval.predict(xNoisy);
        const mse = recon.sub(xClean).square().mean();
        const v = mse.dataSync()[0];
        return v;
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
    if (!testAll || !modelMax || !modelAvg) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();
      log("Evaluating denoising performance (MSE/PSNR) on test set...");
      await tf.nextFrame();

      const rMax = await evaluateMSE(modelMax, "MAX", testAll.xs);
      const rAvg = await evaluateMSE(modelAvg, "AVG", testAll.xs);

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
    if (!testAll || !modelMax || !modelAvg) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();

      const { batchXs, batchYs } = window.getRandomTestBatch(testAll.xs, testAll.ys, 5);
      const noisyBatch = addRandomNoise(batchXs, NOISE_FACTOR);

      const reconMax = modelMax.predict(noisyBatch);
      const reconAvg = modelAvg.predict(noisyBatch);

      // DEBUG: If these ranges are near [0,0], it will look black.
      const maxMin = reconMax.min().dataSync()[0];
      const maxMax = reconMax.max().dataSync()[0];
      const avgMin = reconAvg.min().dataSync()[0];
      const avgMax = reconAvg.max().dataSync()[0];
      log(`Recon ranges -> MAX: [${maxMin.toFixed(4)}, ${maxMax.toFixed(4)}], AVG: [${avgMin.toFixed(4)}, ${avgMax.toFixed(4)}]`);

      for (let i = 0; i < 5; i++) {
        const item = document.createElement("div");
        item.className = "previewItem";
        item.style.minWidth = "280px";

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

        const clean = batchXs.slice([i, 0, 0, 0], [1, 28, 28, 1]);
        const noisy = noisyBatch.slice([i, 0, 0, 0], [1, 28, 28, 1]);
        const dMax = reconMax.slice([i, 0, 0, 0], [1, 28, 28, 1]);
        const dAvg = reconAvg.slice([i, 0, 0, 0], [1, 28, 28, 1]);

        window.draw28x28ToCanvas(clean, cleanCell.canvas, PREVIEW_SCALE);
        window.draw28x28ToCanvas(noisy, noisyCell.canvas, PREVIEW_SCALE);
        window.draw28x28ToCanvas(dMax, maxCell.canvas, PREVIEW_SCALE);
        window.draw28x28ToCanvas(dAvg, avgCell.canvas, PREVIEW_SCALE);

        clean.dispose(); noisy.dispose(); dMax.dispose(); dAvg.dispose();
        await tf.nextFrame();
      }

      log(`Previewed 5 random test samples with noise=${NOISE_FACTOR} (Clean|Noisy|Max|Avg).`);

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
  // Step 4: Save / Load model files (ACTIVE slot)
  // ---------------------------
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

      setStatus(`Reset.\nUpload train/test CSV files, then click Load Data.\nNoise factor=${NOISE_FACTOR}`);

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

    setStatus(`Ready.\nUpload train/test CSV files, then click Load Data.\nNoise factor=${NOISE_FACTOR}`);
    renderModelInfo();
    bindUI();
    setButtonsEnabled();

    log("Autoencoder mode: Train noisy→clean. Preview shows Clean|Noisy|Denoised(Max)|Denoised(Avg).");
    log("Fixes: BCE loss + UpSampling decoder (prevents all-black collapse).");
  }

  boot();
})();
