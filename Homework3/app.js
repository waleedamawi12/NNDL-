// app.js
// Neural Network Design: The Gradient Puzzle (browser-only TF.js)
//
// Baseline idea (Level 1): pixel-wise MSE(Input, Output) => identity / copycat.
// Level 2 idea: distribution constraint via sorted pixels => MSE(sort(Input), sort(Output))
// Level 3 idea: add shape constraints (smoothness + direction) => emergent gradient.
//
// IMPORTANT: Students should edit this file directly (no in-browser editor).
// Look for TODO-A / TODO-B / TODO-C.

(() => {
  // ---------- DOM ----------
  const el = {
    btnStep: document.getElementById("btnStep"),
    btnAuto: document.getElementById("btnAuto"),
    btnReset: document.getElementById("btnReset"),
    log: document.getElementById("log"),
    badgeStep: document.getElementById("badgeStep"),
    metaBase: document.getElementById("metaBase"),
    metaStudent: document.getElementById("metaStudent"),
    canvasInput: document.getElementById("canvasInput"),
    canvasBaseline: document.getElementById("canvasBaseline"),
    canvasStudent: document.getElementById("canvasStudent"),
    archRadios: Array.from(document.querySelectorAll('input[name="arch"]')),
  };

  // ---------- Globals / State ----------
  const H = 16, W = 16, C = 1;
  const BATCH = 1;

  // Throttle auto-train
  const STEPS_PER_FRAME = 2;

  let stepCount = 0;
  let isAuto = false;

  // Fixed input noise (the "puzzle tiles")
  // Shape: [1, 16, 16, 1]
  let xInput = null;

  // Models + optimizers
  let baselineModel = null;
  let studentModel = null;
  let baselineOpt = null;
  let studentOpt = null;

  // Student architecture selection (applies to studentModel only)
  let studentArchType = "compression";

  // ---------- Logging ----------
  function logLine(msg) {
    const t = new Date().toLocaleTimeString();
    el.log.textContent = `[${t}] ${msg}\n` + el.log.textContent;
  }

  function setAutoButtonLabel() {
    el.btnAuto.textContent = isAuto ? "Auto Train (Stop)" : "Auto Train (Start)";
  }

  // ---------- Rendering (pixelated 16x16) ----------
  function drawTensorToCanvas01(t4d, canvas) {
    // Expects shape [1,16,16,1], values in any range; we normalize to [0,255].
    tf.tidy(() => {
      const t = t4d.squeeze(); // [16,16]
      const min = t.min();
      const max = t.max();
      const denom = tf.maximum(max.sub(min), tf.scalar(1e-8));
      const norm = t.sub(min).div(denom).mul(255).clipByValue(0, 255).toInt(); // [16,16]
      const data = norm.dataSync(); // 256 ints

      const ctx = canvas.getContext("2d", { willReadFrequently: false });
      const img = ctx.createImageData(W, H);
      for (let i = 0; i < W * H; i++) {
        const v = data[i];
        const j = i * 4;
        img.data[j + 0] = v; // R
        img.data[j + 1] = v; // G
        img.data[j + 2] = v; // B
        img.data[j + 3] = 255; // A
      }
      ctx.putImageData(img, 0, 0);
    });
  }

  // ---------- Loss helpers (provided for students) ----------
  function mse(yTrue, yPred) {
    // Pixel-wise MSE (Level 1 trap)
    return tf.mean(tf.square(yTrue.sub(yPred)));
  }

  function smoothness(yPred) {
    // Total-variation-like smoothness:
    // sum of squared differences between neighbors (encourages local consistency)
    // yPred shape: [1,H,W,1]
    const dy = yPred.slice([0, 1, 0, 0], [BATCH, H - 1, W, C])
      .sub(yPred.slice([0, 0, 0, 0], [BATCH, H - 1, W, C]));
    const dx = yPred.slice([0, 0, 1, 0], [BATCH, H, W - 1, C])
      .sub(yPred.slice([0, 0, 0, 0], [BATCH, H, W - 1, C]));
    return tf.mean(tf.square(dx)).add(tf.mean(tf.square(dy)));
  }

  function directionX(yPred) {
    // Encourage left-dark / right-bright by aligning intensity with an x-coordinate mask.
    // We create a mask from -1 (left) to +1 (right). Higher output on right => higher correlation.
    // Loss is NEG correlation (so minimizing increases correlation).
    //
    // Intuition: If output is brighter on the right, output * mask is larger on average.
    const x = tf.linspace(-1, 1, W).reshape([1, 1, W, 1]); // [1,1,W,1]
    const mask = x.tile([1, H, 1, 1]); // [1,H,W,1]
    return tf.neg(tf.mean(yPred.mul(mask)));
  }

  function sortedMSE(yTrue, yPred) {
    // Level 2: distribution constraint (aka "quantile loss" / 1D Wasserstein-ish flavor)
    // We compare sorted pixel values, which preserves the histogram ("no new colors"),
    // while allowing pixels to move anywhere.
    //
    // NOTE: This uses topk-sort (hard sort). It’s simple for teaching but not perfectly smooth.
    // For 256 pixels it’s fine and runs fast in the browser.
    const a = yTrue.reshape([W * H]); // [256]
    const b = yPred.reshape([W * H]); // [256]

    // tf.topk sorts descending. We'll use descending for both so it's consistent.
    const aSorted = tf.topk(a, W * H, true).values;
    const bSorted = tf.topk(b, W * H, true).values;
    return tf.mean(tf.square(aSorted.sub(bSorted)));
  }

  // ---------- Models ----------
  function createCompressionModel() {
    // "Compression" projection: bottleneck latent
    // Minimal autoencoder-ish model
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [H, W, C] })); // 256
    model.add(tf.layers.dense({ units: 32, activation: "relu" })); // bottleneck
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" })); // output in [0,1]
    model.add(tf.layers.reshape({ targetShape: [H, W, C] }));
    return model;
  }

  // TODO-A (Architecture):
  // Implement createStudentModel(archType) with three projection types:
  // - compression (already implemented)
  // - transformation (TODO; same-size latent, think “rearrange without compressing much”)
  // - expansion (TODO; bigger latent / more channels, then decode)
  //
  // Baseline requirement for this assignment demo:
  // Only compression is implemented by default. Others must throw a clear error.
  function createStudentModel(archType) {
    if (archType === "compression") return createCompressionModel();

    // Students: replace these errors with real implementations.
    if (archType === "transformation") {
      throw new Error("Student architecture 'transformation' is not implemented yet. (TODO-A)");
    }
    if (archType === "expansion") {
      throw new Error("Student architecture 'expansion' is not implemented yet. (TODO-A)");
    }
    throw new Error(`Unknown student architecture: ${archType}`);
  }

  function createBaselineModel() {
    // Baseline is intentionally fixed to keep comparisons honest.
    return createCompressionModel();
  }

  // ---------- Student loss ----------
  // TODO-B (Custom loss):
  // Starting point is the Level 1 trap:
  //   L = MSE(Input, Output)
  //
  // The homework wants you to escape that trap:
  // Level 2: enforce same pixel distribution ("no new colors"):
  //   L_sorted = MSE(sort(Input), sort(Output))
  //
  // Level 3: add shape constraints:
  //   L_total = L_sorted + λ_smooth * smoothness(Output) + λ_dir * directionX(Output)
  //
  // Default below keeps student model identical to baseline (MSE only).
  function studentLoss(yTrue, yPred) {
    // Baseline / starter: pixel-wise MSE (Level 1)
    return mse(yTrue, yPred);

    // Example (students can uncomment and tune):
    // const Lsorted = sortedMSE(yTrue, yPred);
    // const Lsmooth = smoothness(yPred);
    // const Ldir = directionX(yPred);
    // const lambdaSmooth = 0.20; // try 0.05 to 1.0
    // const lambdaDir = 0.30;    // try 0.05 to 1.0
    // return Lsorted.add(Lsmooth.mul(lambdaSmooth)).add(Ldir.mul(lambdaDir));
  }

  // ---------- Training step (custom loop; no model.fit) ----------
  function trainOneStep() {
    // Returns { baseLoss: number, studentLoss: number } or throws.
    const out = tf.tidy(() => {
      // Baseline grads
      const baseRes = tf.variableGrads(() => {
        const yBase = baselineModel.apply(xInput);
        return mse(xInput, yBase);
      });
      baselineOpt.applyGradients(baseRes.grads);

      // Student grads
      const studentRes = tf.variableGrads(() => {
        const yStu = studentModel.apply(xInput);
        return studentLoss(xInput, yStu);
      });
      studentOpt.applyGradients(studentRes.grads);

      const baseLossVal = baseRes.value.dataSync()[0];
      const studentLossVal = studentRes.value.dataSync()[0];

      // Clean up (variableGrads returns tensors for grads map)
      Object.values(baseRes.grads).forEach(t => t.dispose());
      Object.values(studentRes.grads).forEach(t => t.dispose());
      baseRes.value.dispose();
      studentRes.value.dispose();

      return { baseLossVal, studentLossVal };
    });

    stepCount += 1;
    el.badgeStep.textContent = `step: ${stepCount}`;
    el.metaBase.textContent = `loss: ${out.baseLossVal.toFixed(5)}`;
    el.metaStudent.textContent = `loss: ${out.studentLossVal.toFixed(5)}`;
    return out;
  }

  function renderAll() {
    tf.tidy(() => {
      const yBase = baselineModel.apply(xInput);
      const yStu = studentModel.apply(xInput);

      drawTensorToCanvas01(xInput, el.canvasInput);
      drawTensorToCanvas01(yBase, el.canvasBaseline);
      drawTensorToCanvas01(yStu, el.canvasStudent);
    });
  }

  // TODO-C (Comparison):
  // - Print and compare baseline vs student losses
  // - Visually inspect differences between baseline/student outputs.
  // Starter implementation below logs losses every step.

  // ---------- Lifecycle / reset ----------
  function disposeModel(m) {
    if (!m) return;
    m.layers?.forEach(layer => layer.dispose?.());
    m.dispose?.();
  }

  function initInput() {
    if (xInput) xInput.dispose();
    // Fixed seed behavior in browser JS is not guaranteed; we keep it fixed per session.
    // Use uniform noise in [0,1] so "colors" are bounded.
    xInput = tf.randomUniform([BATCH, H, W, C], 0, 1, "float32");
  }

  function buildAllModels() {
    // Dispose old
    disposeModel(baselineModel);
    disposeModel(studentModel);

    // Build new
    baselineModel = createBaselineModel();
    studentModel = createStudentModel(studentArchType);

    // Fresh optimizers
    baselineOpt = tf.train.adam(0.03);
    studentOpt = tf.train.adam(0.03);
  }

  function resetAll() {
    stepCount = 0;
    el.badgeStep.textContent = `step: ${stepCount}`;
    el.metaBase.textContent = `loss: —`;
    el.metaStudent.textContent = `loss: —`;
    el.log.textContent = "";

    initInput();

    try {
      buildAllModels();
      renderAll();
      logLine("Reset complete. Baseline: MSE. Student: starter (MSE).");
    } catch (err) {
      logLine(`ERROR during reset: ${err.message}`);
      console.error(err);
    }
  }

  // ---------- Auto train loop ----------
  function autoLoop() {
    if (!isAuto) return;

    try {
      let last = null;
      for (let i = 0; i < STEPS_PER_FRAME; i++) {
        last = trainOneStep();
      }
      renderAll();
      if (last) {
        logLine(`step=${stepCount} base=${last.baseLossVal.toFixed(5)} student=${last.studentLossVal.toFixed(5)}`);
      }
    } catch (err) {
      isAuto = false;
      setAutoButtonLabel();
      logLine(`AUTO STOP (error): ${err.message}`);
      console.error(err);
    }

    requestAnimationFrame(autoLoop);
  }

  // ---------- Events ----------
  el.btnStep.addEventListener("click", () => {
    try {
      const r = trainOneStep();
      renderAll();
      logLine(`step=${stepCount} base=${r.baseLossVal.toFixed(5)} student=${r.studentLossVal.toFixed(5)}`);
    } catch (err) {
      logLine(`ERROR: ${err.message}`);
      console.error(err);
    }
  });

  el.btnAuto.addEventListener("click", () => {
    isAuto = !isAuto;
    setAutoButtonLabel();
    if (isAuto) {
      logLine("Auto-train started.");
      requestAnimationFrame(autoLoop);
    } else {
      logLine("Auto-train stopped.");
    }
  });

  el.btnReset.addEventListener("click", () => {
    isAuto = false;
    setAutoButtonLabel();
    resetAll();
  });

  el.archRadios.forEach(r => {
    r.addEventListener("change", () => {
      if (!r.checked) return;
      studentArchType = r.value;

      // Changing architecture should rebuild student model so weights match chosen structure.
      isAuto = false;
      setAutoButtonLabel();
      try {
        // Keep the same input noise; just rebuild models.
        buildAllModels();
        stepCount = 0;
        el.badgeStep.textContent = `step: ${stepCount}`;
        el.metaBase.textContent = `loss: —`;
        el.metaStudent.textContent = `loss: —`;
        renderAll();
        logLine(`Student architecture set to "${studentArchType}". Models rebuilt.`);
      } catch (err) {
        logLine(`ERROR selecting architecture "${studentArchType}": ${err.message}`);
        console.error(err);
      }
    });
  });

  // ---------- Boot ----------
  (async function main() {
    try {
      // Warm up TF.js backend
      await tf.ready();
      logLine(`TF.js ready. Backend: ${tf.getBackend()}`);
      resetAll();
    } catch (err) {
      logLine(`FATAL: ${err.message}`);
      console.error(err);
    }
  })();
})();
