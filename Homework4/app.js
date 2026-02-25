// app.js
// Browser-only TensorFlow.js Autoencoder demo that uses the shared data-loader.js
// -----------------------------------------------------------------------------
// This file is intentionally "dumb about CSV". All CSV parsing + shape inference
// (28×28 MNIST or 64×64 ChineseMNIST) is done in data-loader.js.
//
// Required functions provided by data-loader.js (make sure index.html loads it first):
//   - loadTrainFromFiles(file) -> { xs, ys, meta:{imgSize,pixelCount,format} }
//   - loadTestFromFiles(file)  -> { xs, ys, meta:{imgSize,pixelCount,format} }
//   - splitTrainVal(xs, ys, valRatio) -> { trainXs, trainYs, valXs, valYs }
//   - getRandomTestBatch(xs, ys, k) -> { batchXs, batchYs }
//   - drawToCanvas(tensor, canvas, scale) OR drawImageToCanvas(tensor, canvas, scale)
//
// IMPORTANT:
// - This app is an AUTOENCODER: targets are images themselves (ys is ignored).
// - We show reconstruction loss (MSE) during evaluation and preview reconstructions.
//
// Notes:
// - If you want classification later, you can reuse ys and switch model/loss.

(() => {
  // -----------------------------
  // DOM helpers
  // -----------------------------
  const el = (id) => document.getElementById(id);

  // File inputs (must match your index.html ids)
  const trainCsvInput = el("trainCsv");
  const testCsvInput  = el("testCsv");
  const trainName = el("trainName");
  const testName  = el("testName");

  // Buttons
  const btnLoadData = el("btnLoadData");
  const btnTrain    = el("btnTrain");
  const btnEval     = el("btnEval");
  const btnTest5    = el("btnTest5");
  const btnSave     = el("btnSave");
  const btnLoadModel= el("btnLoadModel");
  const btnReset    = el("btnReset");
  const btnToggleVisor = el("btnToggleVisor");

  // Model load inputs
  const modelJsonInput = el("modelJson");
  const modelBinInput  = el("modelBin");
  const jsonName = el("jsonName");
  const binName  = el("binName");

  // Output areas
  const dataStatus  = el("dataStatus");
  const trainLogs   = el("trainLogs");
  const overallAcc  = el("overallAcc");   // we will display "Test MSE" here
  const previewStrip= el("previewStrip");
  const modelInfo   = el("modelInfo");

  // -----------------------------
  // App state
  // -----------------------------
  let trainXsAll = null;
  let testXsAll  = null;
  let split = null; // {trainXs,valXs}
  let model = null;

  // Dynamic image size (28 or 64) determined from data-loader output
  let IMG_SIZE = 28;

  // Anti-double-click guard
  let busy = false;

  // -----------------------------
  // UI helpers
  // -----------------------------
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

  function setButtonsEnabled() {
    const hasData = !!(trainXsAll && testXsAll && split);
    const hasModel = !!model;

    btnLoadData.disabled = busy;
    btnTrain.disabled = !hasData || !hasModel || busy;
    btnEval.disabled  = !hasData || !hasModel || busy;
    btnTest5.disabled = !hasData || !hasModel || busy;
    btnSave.disabled  = !hasModel || busy;
    btnLoadModel.disabled = busy;
    btnReset.disabled = busy;
  }

  function renderModelSummary() {
    if (!model) {
      modelInfo.textContent = "—";
      return;
    }
    const lines = [];
    model.summary(200, undefined, (line) => lines.push(line));
    modelInfo.textContent = lines.join("\n");
  }

  function showOrToggleVisor() {
    const visor = tfvis.visor();
    visor.isOpen() ? visor.close() : visor.open();
  }

  // -----------------------------
  // Memory cleanup
  // -----------------------------
  function disposeTensors() {
    // Split tensors
    if (split) {
      split.trainXs?.dispose?.();
      split.trainYs?.dispose?.(); // may exist but unused
      split.valXs?.dispose?.();
      split.valYs?.dispose?.();   // may exist but unused
      split = null;
    }
    // Full datasets
    trainXsAll?.dispose?.(); trainXsAll = null;
    testXsAll?.dispose?.();  testXsAll = null;
  }

  function disposeModel() {
    if (model) {
      model.dispose();
      model = null;
    }
    renderModelSummary();
  }

  function resetUI() {
    clearLogs();
    clearPreview();
    overallAcc.textContent = "—";
  }

  // -----------------------------
  // Model (Autoencoder)
  // -----------------------------
  function buildAutoencoder(imgSize) {
    // Simple, stable autoencoder:
    // Encoder: Conv -> Pool -> Conv -> Pool
    // Decoder: UpSample -> Conv -> UpSample -> Conv -> Output
    //
    // Output uses sigmoid so pixels are predicted in [0,1].
    // Loss uses binaryCrossentropy; for grayscale 0..1 this works well and avoids "all black" collapse.
    const m = tf.sequential();

    // Encoder
    m.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      inputShape: [imgSize, imgSize, 1]
    }));
    m.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    m.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
      padding: "same"
    }));
    m.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    // Decoder
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

    // Output: 1 channel grayscale
    m.add(tf.layers.conv2d({
      filters: 1,
      kernelSize: 3,
      activation: "sigmoid",
      padding: "same"
    }));

    m.compile({
      optimizer: "adam",
      loss: "binaryCrossentropy"
    });

    return m;
  }

  // -----------------------------
  // Data loading (USING data-loader.js)
  // -----------------------------
  async function onLoadData() {
    if (busy) return;
    busy = true;
    setButtonsEnabled();

    try {
      resetUI();
      disposeTensors();
      disposeModel();

      const trainFile = trainCsvInput?.files?.[0];
      const testFile  = testCsvInput?.files?.[0];

      if (!trainFile || !testFile) {
        throw new Error("Please select BOTH Train CSV and Test CSV.");
      }

      setStatus(
        `Loading...\n` +
        `Train: ${trainFile.name}\n` +
        `Test:  ${testFile.name}\n\n` +
        `Parsing CSV via data-loader.js...`
      );
      log("Loading train CSV via data-loader.js...");
      const trainData = await window.loadTrainFromFiles(trainFile);

      log("Loading test CSV via data-loader.js...");
      const testData  = await window.loadTestFromFiles(testFile);

      // We only need xs for autoencoder (ys can exist, but unused)
      trainXsAll = trainData.xs;
      testXsAll  = testData.xs;

      // Detect image size dynamically (28 or 64)
      IMG_SIZE = trainData?.meta?.imgSize || trainXsAll.shape[1];

      // We split using helper. It expects xs,ys but ys may be irrelevant.
      // Pass trainData.ys so the function works unchanged.
      split = window.splitTrainVal(trainXsAll, trainData.ys, 0.1);

      // Build model for detected size
      model = buildAutoencoder(IMG_SIZE);
      renderModelSummary();

      setStatus(
        `Loaded.\n` +
        `Train xs: ${trainXsAll.shape}\n` +
        `Val xs:   ${split.valXs.shape}\n` +
        `Test xs:  ${testXsAll.shape}\n` +
        `Image size detected: ${IMG_SIZE}×${IMG_SIZE}\n\n` +
        `Autoencoder mode: target = input image`
      );
      log(`Data loaded. IMG_SIZE=${IMG_SIZE}. Model built.`);
    } catch (err) {
      const msg = (err && err.message) ? err.message : String(err);
      setStatus(`Error loading data:\n${msg}`);
      log(`ERROR: ${msg}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // -----------------------------
  // Training
  // -----------------------------
  async function onTrain() {
    if (busy) return;
    if (!split || !model) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();
      overallAcc.textContent = "—";

      const epochs = 10;
      const batchSize = (IMG_SIZE >= 64) ? 64 : 128;

      log(`Training autoencoder: epochs=${epochs}, batchSize=${batchSize}`);

      const fitSurface = { name: "Training (Loss)", tab: "Training" };
      const callbacks = tfvis.show.fitCallbacks(
        fitSurface,
        ["loss", "val_loss"],
        {
          callbacks: [{
            onEpochEnd: async (epoch, logs) => {
              log(`Epoch ${epoch + 1}: loss=${logs.loss?.toFixed(5)} val_loss=${logs.val_loss?.toFixed(5)}`);
              await tf.nextFrame();
            }
          }]
        }
      );

      const t0 = performance.now();

      // Autoencoder target is the clean input itself
      await model.fit(split.trainXs, split.trainXs, {
        epochs,
        batchSize,
        shuffle: true,
        validationData: [split.valXs, split.valXs],
        callbacks
      });

      const t1 = performance.now();
      log(`Training done in ${((t1 - t0) / 1000).toFixed(2)}s`);
    } catch (err) {
      const msg = (err && err.message) ? err.message : String(err);
      log(`ERROR (train): ${msg}`);
      setStatus(`Training error:\n${msg}\n\n${dataStatus.textContent}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // -----------------------------
  // Evaluation (report MSE)
  // -----------------------------
  async function onEvaluate() {
    if (busy) return;
    if (!testXsAll || !model) return;

    busy = true;
    setButtonsEnabled();

    try {
      log("Evaluating reconstruction (MSE) on test set...");

      const n = testXsAll.shape[0];
      const batchSize = 128;
      let sum = 0;
      let count = 0;

      for (let start = 0; start < n; start += batchSize) {
        const size = Math.min(batchSize, n - start);

        const mse = tf.tidy(() => {
          const x = testXsAll.slice([start, 0, 0, 0], [size, IMG_SIZE, IMG_SIZE, 1]);
          const y = model.predict(x);
          // Mean squared error over batch
          return y.sub(x).square().mean().dataSync()[0];
        });

        sum += mse * size;
        count += size;

        if (start % (batchSize * 10) === 0) await tf.nextFrame();
      }

      const mseAll = sum / count;
      overallAcc.textContent = `Test MSE: ${mseAll.toFixed(6)}`;
      log(`Test MSE = ${mseAll.toFixed(6)}`);

      // Show a simple bar chart in visor
      tfvis.render.barchart(
        { name: "Test MSE (Lower is Better)", tab: "Evaluation" },
        [{ index: "Autoencoder", value: mseAll }],
        { xLabel: "Model", yLabel: "MSE", height: 260 }
      );
    } catch (err) {
      const msg = (err && err.message) ? err.message : String(err);
      log(`ERROR (eval): ${msg}`);
      setStatus(`Evaluation error:\n${msg}\n\n${dataStatus.textContent}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // -----------------------------
  // Preview: 5 random test images + reconstructions
  // -----------------------------
  async function onTestFive() {
    if (busy) return;
    if (!testXsAll || !model) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();

      const { batchXs } = window.getRandomTestBatch(testXsAll, testXsAll, 5);

      const recon = tf.tidy(() => model.predict(batchXs));

      for (let i = 0; i < 5; i++) {
        const item = document.createElement("div");
        item.className = "previewItem";
        item.style.minWidth = "220px";

        const row = document.createElement("div");
        row.style.display = "grid";
        row.style.gridTemplateColumns = "repeat(2, 1fr)";
        row.style.gap = "8px";

        const makeCell = (label) => {
          const wrap = document.createElement("div");
          wrap.style.textAlign = "center";

          const canvas = document.createElement("canvas");
          canvas.style.borderRadius = "10px";
          canvas.style.imageRendering = "pixelated";

          const txt = document.createElement("div");
          txt.style.fontSize = "10px";
          txt.style.color = "#93a4c7";
          txt.style.marginTop = "4px";
          txt.textContent = label;

          wrap.appendChild(canvas);
          wrap.appendChild(txt);
          return { wrap, canvas };
        };

        const a = makeCell("Input");
        const b = makeCell("Reconstruction");

        row.appendChild(a.wrap);
        row.appendChild(b.wrap);
        item.appendChild(row);
        previewStrip.appendChild(item);

        const x = batchXs.slice([i, 0, 0, 0], [1, IMG_SIZE, IMG_SIZE, 1]);
        const y = recon.slice([i, 0, 0, 0], [1, IMG_SIZE, IMG_SIZE, 1]);

        // data-loader.js may expose drawToCanvas or drawImageToCanvas, use whichever exists.
        const draw = window.drawToCanvas || window.drawImageToCanvas || window.draw28x28ToCanvas;
        draw(x, a.canvas, IMG_SIZE >= 64 ? 2 : 4);
        draw(y, b.canvas, IMG_SIZE >= 64 ? 2 : 4);

        x.dispose();
        y.dispose();

        await tf.nextFrame();
      }

      batchXs.dispose();
      recon.dispose();

      log("Previewed 5 random reconstructions.");
    } catch (err) {
      const msg = (err && err.message) ? err.message : String(err);
      log(`ERROR (preview): ${msg}`);
      setStatus(`Preview error:\n${msg}\n\n${dataStatus.textContent}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // -----------------------------
  // Save/Load model (downloads + browserFiles)
  // -----------------------------
  async function onSaveDownload() {
    if (busy) return;
    if (!model) return;

    busy = true;
    setButtonsEnabled();

    try {
      log("Saving model to downloads...");
      await model.save("downloads://autoencoder");
      log("Download triggered: autoencoder.json + autoencoder.weights.bin");
    } catch (err) {
      const msg = (err && err.message) ? err.message : String(err);
      log(`ERROR (save): ${msg}`);
      setStatus(`Save error:\n${msg}\n\n${dataStatus.textContent}`);
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
      const jsonFile = modelJsonInput?.files?.[0];
      const binFile  = modelBinInput?.files?.[0];
      if (!jsonFile || !binFile) throw new Error("Choose BOTH model.json and weights.bin.");

      log("Loading model from files...");
      const loaded = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));

      // Compile for evaluate/fit
      loaded.compile({ optimizer: "adam", loss: "binaryCrossentropy" });

      // Replace old model safely
      disposeModel();
      model = loaded;
      renderModelSummary();

      log("Model loaded successfully.");
    } catch (err) {
      const msg = (err && err.message) ? err.message : String(err);
      log(`ERROR (load model): ${msg}`);
      setStatus(`Load model error:\n${msg}\n\n${dataStatus.textContent}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // -----------------------------
  // Reset
  // -----------------------------
  function onReset() {
    if (busy) return;

    disposeTensors();
    disposeModel();
    resetUI();

    // Clear file inputs so same file can be reselected
    if (trainCsvInput) trainCsvInput.value = "";
    if (testCsvInput) testCsvInput.value = "";
    if (modelJsonInput) modelJsonInput.value = "";
    if (modelBinInput) modelBinInput.value = "";

    if (trainName) trainName.textContent = "No file selected";
    if (testName)  testName.textContent  = "No file selected";
    if (jsonName)  jsonName.textContent  = "No file";
    if (binName)   binName.textContent   = "No file";

    IMG_SIZE = 28;
    setStatus("Reset.\nUpload train/test CSV files, then click Load Data.");
    setButtonsEnabled();
  }

  // -----------------------------
  // Bind UI
  // -----------------------------
  function bindUI() {
    // Defensive: if IDs mismatch, fail loudly in console
    if (!trainCsvInput || !testCsvInput || !btnLoadData) {
      console.error("Missing required DOM elements. Check index.html IDs:", {
        trainCsv: !!trainCsvInput,
        testCsv: !!testCsvInput,
        btnLoadData: !!btnLoadData
      });
    }

    trainCsvInput?.addEventListener("change", () => {
      const f = trainCsvInput.files?.[0];
      if (trainName) trainName.textContent = f ? f.name : "No file selected";
    });

    testCsvInput?.addEventListener("change", () => {
      const f = testCsvInput.files?.[0];
      if (testName) testName.textContent = f ? f.name : "No file selected";
    });

    modelJsonInput?.addEventListener("change", () => {
      const f = modelJsonInput.files?.[0];
      if (jsonName) jsonName.textContent = f ? f.name : "No file";
    });

    modelBinInput?.addEventListener("change", () => {
      const f = modelBinInput.files?.[0];
      if (binName) binName.textContent = f ? f.name : "No file";
    });

    btnLoadData?.addEventListener("click", onLoadData);
    btnTrain?.addEventListener("click", onTrain);
    btnEval?.addEventListener("click", onEvaluate);
    btnTest5?.addEventListener("click", onTestFive);
    btnSave?.addEventListener("click", onSaveDownload);
    btnLoadModel?.addEventListener("click", onLoadFromFiles);
    btnReset?.addEventListener("click", onReset);
    btnToggleVisor?.addEventListener("click", showOrToggleVisor);
  }

  // -----------------------------
  // Boot
  // -----------------------------
  function boot() {
    // Close visor by default (optional)
    try { tfvis.visor().close(); } catch (_) {}

    setStatus("Ready.\nUpload train/test CSV files, then click Load Data.");
    renderModelSummary();
    bindUI();
    setButtonsEnabled();

    log("Autoencoder app loaded. Using data-loader.js for CSV parsing + shape inference.");
  }

  boot();

})();
