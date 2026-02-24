/* app.js
   UI wiring + model training/evaluation for the MNIST CSV demo.

   Key design principles for students:
   - Everything runs client-side (no server).
   - Data tensors are created from uploaded CSV files.
   - Model persistence is FILE-BASED only:
       Save: model.save('downloads://mnist-cnn')
       Load: tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]))
   - We aggressively dispose intermediate tensors to avoid GPU/CPU memory leaks.
   - We keep the UI responsive by awaiting animation frames during long operations.

   IMPORTANT:
   - data-loader.js must be loaded BEFORE this file.
*/

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
  // App state (tensors + model)
  // ---------------------------
  let trainAll = null;   // {xs, ys} full training tensors
  let testAll = null;    // {xs, ys} full test tensors
  let split = null;      // {trainXs, trainYs, valXs, valYs}
  let model = null;

  // A small flag to prevent concurrent training/eval clicks.
  let busy = false;

  // ---------------------------
  // Utility: logging + status
  // ---------------------------
  function setStatus(text) {
    dataStatus.textContent = text;
  }

  function log(text) {
    // Append with timestamp
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
    const hasData = !!(trainAll && testAll);
    const hasModel = !!model;

    btnTrain.disabled = !hasData || busy;
    btnEval.disabled = !hasData || !hasModel || busy;
    btnTest5.disabled = !hasData || !hasModel || busy;
    btnSave.disabled = !hasModel || busy;
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
  // Model definition
  // ---------------------------
  function buildModel() {
    // Dispose old model if present (avoid leaks when rebuilding).
    if (model) {
      model.dispose();
      model = null;
    }

    // The exact architecture requested.
    model = tf.sequential();

    model.add(tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      padding: "same",
      inputShape: [28, 28, 1]
    }));
    model.add(tf.layers.conv2d({
      filters: 64,
      kernelSize: 3,
      activation: "relu",
      padding: "same"
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    model.add(tf.layers.dropout({ rate: 0.5 }));
    model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

    model.compile({
      optimizer: "adam",
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"]
    });

    renderModelSummary();
    setButtonsEnabled();
    return model;
  }

  function renderModelSummary() {
    if (!model) {
      modelInfo.textContent = "—";
      return;
    }

    // Capture model.summary() output into a string.
    // tfjs lets you pass a custom print function.
    const lines = [];
    model.summary(100, undefined, (line) => lines.push(line));
    modelInfo.textContent = lines.join("\n");
  }

  // ---------------------------
  // tfjs-vis utilities
  // ---------------------------
  function showOrToggleVisor() {
    // tfjs-vis uses a "visor" panel. This toggles its visibility.
    const visor = tfvis.visor();
    visor.isOpen() ? visor.close() : visor.open();
  }

  function clearVisorTabs() {
    // There is no single "clear everything" API, but we can overwrite surfaces as we render.
    // So this function is mostly conceptual; we keep it for teaching clarity.
  }

  // ---------------------------
  // Data loading pipeline
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

      // Clean up old tensors before reloading (critical to avoid memory leaks).
      disposeAllTensors();

      setStatus(
        `Loading...\n` +
        `Train file: ${trainFile.name} (${bytesToMB(trainFile.size)} MB)\n` +
        `Test file:  ${testFile.name} (${bytesToMB(testFile.size)} MB)\n\n` +
        `Parsing CSV → tensors (this can take a bit).`
      );

      log("Parsing training CSV...");
      trainAll = await window.loadTrainFromFiles(trainFile);
      await tf.nextFrame(); // yield

      log("Parsing test CSV...");
      testAll = await window.loadTestFromFiles(testFile);
      await tf.nextFrame(); // yield

      // Create train/val split tensors.
      split = window.splitTrainVal(trainAll.xs, trainAll.ys, 0.1);

      // Build model fresh after data is ready (so Train button becomes meaningful).
      buildModel();

      // Report status.
      const trainN = trainAll.xs.shape[0];
      const testN = testAll.xs.shape[0];
      setStatus(
        `Loaded.\n` +
        `Train: ${trainN} samples → xs ${trainAll.xs.shape} ys ${trainAll.ys.shape}\n` +
        `Val:   ${split.valXs.shape[0]} samples\n` +
        `Test:  ${testN} samples → xs ${testAll.xs.shape} ys ${testAll.ys.shape}\n\n` +
        `Tip: Open the tfjs-vis visor for live charts during training.`
      );

      log(`Data ready. Train=${trainN}, Val=${split.valXs.shape[0]}, Test=${testN}`);
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
  async function onTrain() {
    if (busy) return;
    if (!split || !model) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();
      overallAcc.textContent = "—";

      // Training defaults (as requested: 5–10 epochs, batchSize 64–128)
      const epochs = 8;
      const batchSize = 128;

      log(`Training start: epochs=${epochs}, batchSize=${batchSize}`);
      const t0 = performance.now();

      // tfjs-vis surfaces for live curves
      const fitSurface = { name: "Training (Loss / Accuracy)", tab: "Training" };

      // This callback bundle auto-renders live curves:
      // - loss + val_loss
      // - acc + val_acc
      const callbacks = tfvis.show.fitCallbacks(
        fitSurface,
        ["loss", "val_loss", "acc", "val_acc"],
        {
          callbacks: [
            // Keep UI responsive, and also log each epoch.
            {
              onEpochEnd: async (epoch, logs) => {
                const line =
                  `Epoch ${epoch + 1}: ` +
                  `loss=${logs.loss?.toFixed(4)} ` +
                  `acc=${(logs.acc * 100)?.toFixed(2)}% ` +
                  `val_loss=${logs.val_loss?.toFixed(4)} ` +
                  `val_acc=${(logs.val_acc * 100)?.toFixed(2)}%`;
                log(line);
                await tf.nextFrame();
              }
            }
          ]
        }
      );

      // Train!
      const history = await model.fit(split.trainXs, split.trainYs, {
        epochs,
        batchSize,
        validationData: [split.valXs, split.valYs],
        shuffle: true,
        callbacks
      });

      const t1 = performance.now();
      const durSec = (t1 - t0) / 1000;

      // Best val accuracy (nice teaching moment: generalization is the point)
      const valAccHist = history.history.val_acc || history.history.val_accuracy || [];
      const bestValAcc = valAccHist.length ? Math.max(...valAccHist) : NaN;

      log(`Training done in ${durSec.toFixed(2)}s. Best val acc: ${(bestValAcc * 100).toFixed(2)}%`);
      setStatus(
        dataStatus.textContent +
        `\n\nLast train: ${durSec.toFixed(2)}s, best val acc: ${(bestValAcc * 100).toFixed(2)}%`
      );
    } catch (err) {
      log(`ERROR (training): ${friendlyError(err)}`);
      setStatus(`Training error:\n${friendlyError(err)}\n\n${dataStatus.textContent}`);
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // ---------------------------
  // Evaluation: accuracy + confusion matrix + per-class accuracy
  // ---------------------------
  async function onEvaluate() {
    if (busy) return;
    if (!testAll || !model) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();

      log("Evaluating on test set...");
      await tf.nextFrame();

      // Compute test accuracy using model.evaluate.
      // model.evaluate returns tensors; we must dispose them after reading values.
      const evalOut = await model.evaluate(testAll.xs, testAll.ys, { batchSize: 256 });
      const lossT = Array.isArray(evalOut) ? evalOut[0] : evalOut;
      const accT = Array.isArray(evalOut) ? evalOut[1] : null;

      const loss = (await lossT.data())[0];
      const acc = accT ? (await accT.data())[0] : NaN;

      lossT.dispose();
      if (accT) accT.dispose();

      overallAcc.textContent = `${(acc * 100).toFixed(2)}% (loss ${loss.toFixed(4)})`;
      log(`Test accuracy: ${(acc * 100).toFixed(2)}% | loss: ${loss.toFixed(4)}`);

      // Confusion matrix + per-class accuracy requires predicted + true class labels.
      // IMPORTANT: We must avoid keeping huge tensors around. We'll:
      // - compute argMax for preds and labels
      // - bring results to CPU arrays
      // - dispose intermediate tensors
      const { labelsArr, predsArr } = await computePredsAndLabels(testAll.xs, testAll.ys);

      // Build confusion matrix: 10x10
      const cm = buildConfusionMatrix(labelsArr, predsArr, 10);

      // Per-class accuracy: diagonal / row sum (true class count)
      const perClass = [];
      for (let c = 0; c < 10; c++) {
        const rowSum = cm[c].reduce((a, b) => a + b, 0);
        const correct = cm[c][c];
        perClass.push(rowSum ? correct / rowSum : 0);
      }

      // Render confusion matrix as heatmap in tfjs-vis
      tfvis.render.confusionMatrix(
        { name: "Confusion Matrix (Test)", tab: "Evaluation" },
        { values: cm, tickLabels: Array.from({ length: 10 }, (_, i) => String(i)) }
      );

      // Render per-class accuracy as a bar chart
      // tfvis.render.barchart expects array of {index, value} or {x, y}
      tfvis.render.barchart(
        { name: "Per-class Accuracy (Test)", tab: "Evaluation" },
        perClass.map((v, i) => ({ index: i, value: v })),
        { xLabel: "Digit", yLabel: "Accuracy", height: 320 }
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

  async function computePredsAndLabels(xs, ys) {
    // Use batches to avoid a single huge forward pass that can spike memory.
    // For MNIST test (10k), either is fine, but batching teaches good habits.
    const batchSize = 512;
    const n = xs.shape[0];
    const labelsArr = new Int32Array(n);
    const predsArr = new Int32Array(n);

    for (let start = 0; start < n; start += batchSize) {
      const size = Math.min(batchSize, n - start);

      // Use tidy so intermediate tensors inside the block are auto-disposed.
      const { labelBatch, predBatch } = tf.tidy(() => {
        const xB = xs.slice([start, 0, 0, 0], [size, 28, 28, 1]);
        const yB = ys.slice([start, 0], [size, 10]);

        const logits = model.predict(xB);
        const pred = logits.argMax(-1);  // [size]
        const lab = yB.argMax(-1);       // [size]

        // Return tensors; tidy will NOT dispose returned values.
        return { labelBatch: lab, predBatch: pred };
      });

      const [labData, predData] = await Promise.all([labelBatch.data(), predBatch.data()]);
      labelsArr.set(labData, start);
      predsArr.set(predData, start);

      labelBatch.dispose();
      predBatch.dispose();

      // Yield for responsiveness.
      await tf.nextFrame();
    }

    return { labelsArr, predsArr };
  }

  function buildConfusionMatrix(labels, preds, numClasses) {
    const cm = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));
    for (let i = 0; i < labels.length; i++) {
      const t = labels[i];
      const p = preds[i];
      cm[t][p] += 1;
    }
    return cm;
  }

  // ---------------------------
  // Random 5 preview
  // ---------------------------
  async function onTestFive() {
    if (busy) return;
    if (!testAll || !model) return;

    busy = true;
    setButtonsEnabled();

    try {
      clearPreview();

      // Get a tiny random batch (k=5) from test tensors.
      const { batchXs, batchYs } = window.getRandomTestBatch(testAll.xs, testAll.ys, 5);

      // Predict and compare.
      const { preds, truths } = await tf.tidy(async () => {
        const logits = model.predict(batchXs);
        const pred = logits.argMax(-1);   // [5]
        const truth = batchYs.argMax(-1); // [5]
        const [p, t] = await Promise.all([pred.data(), truth.data()]);
        // pred/truth tensors are intermediates; dispose after data extraction.
        pred.dispose();
        truth.dispose();
        return { preds: Array.from(p), truths: Array.from(t) };
      });

      // Render each image to its own canvas in a horizontal strip.
      // We draw from batchXs slice-by-slice to keep memory tidy.
      for (let i = 0; i < preds.length; i++) {
        const item = document.createElement("div");
        item.className = "previewItem";

        const canvas = document.createElement("canvas");
        canvas.width = 28 * 4;
        canvas.height = 28 * 4;

        // Extract i-th image tensor [1,28,28,1] -> draw
        const img = batchXs.slice([i, 0, 0, 0], [1, 28, 28, 1]);
        window.draw28x28ToCanvas(img, canvas, 4);
        img.dispose();

        const predEl = document.createElement("div");
        const ok = preds[i] === truths[i];
        predEl.className = `pred ${ok ? "ok" : "bad"}`;
        predEl.textContent = `Pred: ${preds[i]}`;

        const truthEl = document.createElement("div");
        truthEl.className = "truth";
        truthEl.textContent = `True: ${truths[i]}`;

        item.appendChild(canvas);
        item.appendChild(predEl);
        item.appendChild(truthEl);
        previewStrip.appendChild(item);
      }

      log("Previewed 5 random test samples.");
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
  // Save / Load model (FILE-BASED only)
  // ---------------------------
  async function onSaveDownload() {
    if (busy) return;
    if (!model) return;

    busy = true;
    setButtonsEnabled();

    try {
      log("Saving model to downloads...");
      // This triggers browser downloads for model.json + weights.bin
      await model.save("downloads://mnist-cnn");
      log("Model download triggered (mnist-cnn.json + mnist-cnn.weights.bin).");
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

      log("Loading model from files...");
      await tf.nextFrame();

      // Load new model
      const newModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, binFile]));

      // Dispose old model before replacing to avoid leaks.
      if (model) model.dispose();
      model = newModel;

      // IMPORTANT: The loaded model may or may not include optimizer state.
      // We compile again to ensure evaluate/fit metrics behave as expected.
      model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
      });

      renderModelSummary();
      log("Model loaded successfully from files.");

      // Now that model exists, enable Save/Eval/Test buttons (if data present).
      setButtonsEnabled();
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
    // Dispose split tensors
    if (split) {
      split.trainXs?.dispose?.();
      split.trainYs?.dispose?.();
      split.valXs?.dispose?.();
      split.valYs?.dispose?.();
      split = null;
    }

    // Dispose full datasets
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

  function onReset() {
    if (busy) return;

    try {
      clearPreview();
      clearLogs();
      overallAcc.textContent = "—";
      setStatus("Reset.\nUpload train/test CSV files, then click Load Data.");

      disposeAllTensors();

      if (model) {
        model.dispose();
        model = null;
      }
      renderModelSummary();

      // Clear file inputs (must set value="" to allow re-selecting same file)
      trainCsvInput.value = "";
      testCsvInput.value = "";
      modelJsonInput.value = "";
      modelBinInput.value = "";
      trainName.textContent = "No file selected";
      testName.textContent = "No file selected";
      jsonName.textContent = "No file";
      binName.textContent = "No file";

      // Also clear the visor panels (close it if open)
      // (Not required, but keeps reset feeling "clean")
      const visor = tfvis.visor();
      // We won't force close; just leave user preference.
    } finally {
      busy = false;
      setButtonsEnabled();
    }
  }

  // ---------------------------
  // Wire DOM events
  // ---------------------------
  function bindUI() {
    // File name previews
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

    // Buttons
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
    // Safer defaults: hide visor initially (student can open)
    try { tfvis.visor().close(); } catch (_) {}

    setStatus("Ready.\nUpload train/test CSV files, then click Load Data.");
    renderModelSummary();
    bindUI();
    setButtonsEnabled();

    // Friendly hint for memory usage
    log("Tip: If your browser gets slow, click Reset to dispose tensors/models.");
  }

  boot();
})();
