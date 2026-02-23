// app.js
// Neural Network Design: The Gradient Puzzle (browser-only TF.js)
//
// Goal: Transform a fixed 16x16 noise image into a structured left→right gradient
// WITHOUT target labels. The loss function is the "rule of the game".
//
// Homework progression (matches slides):
//   Level 1: Pixel-wise MSE(Input, Output) -> identity mapping (copycat trap)
//   Level 2: Distribution constraint via sorted pixels (preserve histogram / "no new colors"):
//            L_sorted = MSE(sort(Input), sort(Output))
//   Level 3: Shape constraints:
//            L_total = L_sorted + λ_smooth * Smoothness(Output) + λ_dir * DirectionX(Output)
//
// Students edit this file directly (no in-browser editor).
// Look for TODO-A / TODO-B / TODO-C.

(() => {
  // -------------------- DOM --------------------
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

  // -------------------- Globals / State --------------------
  const H = 16, W = 16, C = 1, B = 1;
  const STEPS_PER_FRAME = 2; // throttle auto-train for stability

  let stepCount = 0;
  let isAuto = false;

  // Fixed input noise per session: [1,16,16,1]
  let xInput = null;

  // Two separate models and optimizers
  let baselineModel = null;
  let studentModel = null;
  let baselineOpt = null;
  let studentOpt = null;

  // Student-only architecture selector
  let studentArchType = "compression";

  // -------------------- Logging --------------------
  function logLine(msg) {
    const t = new Date().toLocaleTimeString();
    el.log.textContent = `[${t}] ${msg}\n` + el.log.textContent;
  }

  function setAutoButtonLabel() {
    el.btnAuto.textContent = isAuto ? "Auto Train (Stop)" : "Auto Train (Start)";
  }

  // -------------------- Rendering (pixelated 16x16) --------------------
  function drawTensorToCanvas01(t4d, canvas) {
    // Expects shape [1,16,16,1]. Values can be any range; we normalize to [0..255].
    tf.tidy(() => {
      const t = t4d.squeeze(); // [16,16]
      const min = t.min();
      const max = t.max();
      const denom = tf.maximum(max.sub(min), tf.scalar(1e-8));
      const norm = t.sub(min).div(denom).mul(255).clipByValue(0, 255).toInt(); // [16,16]
      const data = norm.dataSync();

      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(W, H);
      for (let i = 0; i < W * H; i++) {
        const v = data[i];
        const j = i * 4;
        img.data[j + 0] = v;
        img.data[j + 1] = v;
        img.data[j + 2] = v;
        img.data[j + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    });
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

  // -------------------- Helper losses (provided, implemented, used) --------------------
  function mse(yTrue, yPred) {
    // Level 1 trap: pins every pixel to its original position.
    return tf.mean(tf.square(yTrue.sub(yPred)));
  }

  function sortedMSE(yTrue, yPred) {
    // Level 2 distribution constraint:
    // Compare sorted pixel values -> preserves histogram ("no new colors") but frees positions.
    //
    // Implementation: flatten to [256], use tf.topk to sort (descending).
    const a = yTrue.reshape([H * W]);
    const b = yPred.reshape([H * W]);
    const aSorted = tf.topk(a, H * W, true).values;
    const bSorted = tf.topk(b, H * W, true).values;
    return tf.mean(tf.square(aSorted.sub(bSorted)));
  }

  function smoothness(yPred) {
    // Total-variation-like smoothness:
    // Squared neighbor differences (encourages local consistency).
    const dy = yPred.slice([0, 1, 0, 0], [B, H - 1, W, C])
      .sub(yPred.slice([0, 0, 0, 0], [B, H - 1, W, C]));
    const dx = yPred.slice([0, 0, 1, 0], [B, H, W - 1, C])
      .sub(yPred.slice([0, 0, 0, 0], [B, H, W - 1, C]));
    return tf.mean(tf.square(dx)).add(tf.mean(tf.square(dy)));
  }

  function directionX(yPred) {
    // Encourage left-dark / right-bright via correlation with an x mask from -1..+1.
    // Minimizing -mean(output * mask) increases brightness on the right.
    const x = tf.linspace(-1, 1, W).reshape([1, 1, W, 1]); // [1,1,W,1]
    const mask = x.tile([1, H, 1, 1]); // [1,H,W,1]
    return tf.neg(tf.mean(yPred.mul(mask)));
  }

  // -------------------- Models --------------------
  function createCompressionModel() {
    // Simple bottleneck autoencoder (minimal for browser demo).
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [H, W, C] }));          // 256
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));   // bottleneck
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" })); // keep outputs in [0,1]
    model.add(tf.layers.reshape({ targetShape: [H, W, C] }));
    return model;
  }

  function createBaselineModel() {
    // Baseline is fixed: compression + pixel-wise MSE.
    return createCompressionModel();
  }

  // TODO-A (Architecture):
  // Implement createStudentModel(archType) with three projection types:
  //  - "compression"     (implemented)
  //  - "transformation"  (TODO) -> must throw until implemented
  //  - "expansion"       (TODO) -> must throw until implemented
  //
  // NOTE: The UI selector must rebuild ONLY the student model when changed.
  function createStudentModel(archType) {
    if (archType === "compression") return createCompressionModel();

    // Students: replace these errors with real implementations.
    if (archType === "transformation") {
      throw new Error("Student architecture 'transformation' not implemented yet. (TODO-A)");
    }
    if (archType === "expansion") {
      throw new Error("Student architecture 'expansion' not implemented yet. (TODO-A)");
    }
    throw new Error(`Unknown student architecture: ${archType}`);
  }

  // -------------------- Student loss --------------------
  // TODO-B (Custom Loss) — THE KEY FIX / PIVOT:
  //
  // Students MUST NOT build custom loss on top of pixel-wise MSE if they want rearrangement.
  // Why? Pixel-wise MSE locks pixels to their positions, fighting any attempt to move them.
  //
  // Intended progression:
  //   1) Level 1 (start): return mse(input, output)  -> copycat identity
  //   2) Level 2 (pivot): REPLACE pixel-wise MSE with sortedMSE:
  //         Lsorted = sortedMSE(input, output)
  //      This enforces "same colors" / histogram match while allowing movement.
  //   3) Level 3 (intent): add geometry constraints:
  //         Ltotal = Lsorted + λ_smooth*smoothness(output) + λ_dir*directionX(output)
  //
  // Recommended starting ranges:
  //   λ_smooth in [0.05 .. 1.0]
  //   λ_dir    in [0.05 .. 1.0]
  //
  // IMPORTANT: Once you move to Level 2/3, sortedMSE MUST be the base term.
  function studentLoss(yTrue, yPred) {
    // Level 1 starter (identical to baseline):
    return mse(yTrue, yPred);

    // ---- Level 2 / Level 3 template (students: implement by editing) ----
    // const Lsorted = sortedMSE(yTrue, yPred);    // <- BASE TERM (the pivot!)
    // const Lsmooth = smoothness(yPred);
    // const Ldir = directionX(yPred);
    // const lambdaSmooth = 0.20;
    // const lambdaDir = 0.30;
    // return Lsorted.add(Lsmooth.mul(lambdaSmooth)).add(Ldir.mul(lambdaDir));
  }

  // -------------------- Training loop (custom, no model.fit) --------------------
  // Each step:
  //   - compute baseline pred + baseline loss
  //   - compute student pred  + student loss
  //   - apply gradients separately
  function trainOneStep() {
    const result = tf.tidy(() => {
      // Baseline grads
      const baseRes = tf.variableGrads(() => {
        const yBase = baselineModel.apply(xInput);
        return mse(xInput, yBase);
      });
      baselineOpt.applyGradients(baseRes.grads);

      // Student grads
      const stuRes = tf.variableGrads(() => {
        const yStu = studentModel.apply(xInput);
        return studentLoss(xInput, yStu);
      });
      studentOpt.applyGradients(stuRes.grads);

      const baseLossVal = baseRes.value.dataSync()[0];
      const stuLossVal = stuRes.value.dataSync()[0];

      // Dispose grads/value tensors created by variableGrads
      Object.values(baseRes.grads).forEach(t => t.dispose());
      Object.values(stuRes.grads).forEach(t => t.dispose());
      baseRes.value.dispose();
      stuRes.value.dispose();

      return { baseLossVal, stuLossVal };
    });

    stepCount += 1;
    el.badgeStep.textContent = `step: ${stepCount}`;
    el.metaBase.textContent = `loss: ${result.baseLossVal.toFixed(5)}`;
    el.metaStudent.textContent = `loss: ${result.stuLossVal.toFixed(5)}`;

    // TODO-C (Comparison):
    // This starter log already compares losses each step.
    // Students can enhance it (e.g., show deltas, highlight when student diverges).
    logLine(`step=${stepCount}  baseline=${result.baseLossVal.toFixed(5)}  student=${result.stuLossVal.toFixed(5)}`);

    return result;
  }

  // -------------------- Setup / Reset --------------------
  function disposeModel(m) {
    if (!m) return;
    try { m.dispose(); } catch (_) {}
  }

  function initInput() {
    if (xInput) xInput.dispose();
    // Fixed per session (not re-randomized unless Reset is pressed).
    // Values in [0,1] so histogram is bounded and "no new colors" is meaningful.
    xInput = tf.randomUniform([B, H, W, C], 0, 1, "float32");
  }

  function buildBaseline() {
    disposeModel(baselineModel);
    baselineModel = createBaselineModel();
    baselineOpt = tf.train.adam(0.03);
  }

  function buildStudent() {
    disposeModel(studentModel);
    studentModel = createStudentModel(studentArchType);
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
      buildBaseline();
      buildStudent();
      renderAll();
      logLine("Reset complete. Baseline: pixel-wise MSE. Student: starter (pixel-wise MSE).");
    } catch (err) {
      logLine(`ERROR during reset: ${err.message}`);
      console.error(err);
    }
  }

  // -------------------- Auto-train loop --------------------
  function autoLoop() {
    if (!isAuto) return;

    try {
      for (let i = 0; i < STEPS_PER_FRAME; i++) trainOneStep();
      renderAll();
    } catch (err) {
      isAuto = false;
      setAutoButtonLabel();
      logLine(`AUTO STOP (error): ${err.message}`);
      console.error(err);
    }

    requestAnimationFrame(autoLoop);
  }

  // -------------------- Events --------------------
  el.btnStep.addEventListener("click", () => {
    try {
      trainOneStep();
      renderAll();
    } catch (err) {
      logLine(`ERROR: ${err.message}`);
      console.error(err);
    }
  });

  el.btnAuto.addEventListener("click", () => {
    isAuto = !isAuto;
    setAutoButtonLabel();
    logLine(isAuto ? "Auto-train started." : "Auto-train stopped.");
    if (isAuto) requestAnimationFrame(autoLoop);
  });

  el.btnReset.addEventListener("click", () => {
    isAuto = false;
    setAutoButtonLabel();
    resetAll();
  });

  // Architecture selector applies ONLY to student model
  el.archRadios.forEach(r => {
    r.addEventListener("change", () => {
      if (!r.checked) return;
      studentArchType = r.value;

      isAuto = false;
      setAutoButtonLabel();

      try {
        buildStudent();
        stepCount = 0;
        el.badgeStep.textContent = `step: ${stepCount}`;
        el.metaStudent.textContent = `loss: —`;
        renderAll();
        logLine(`Student architecture set to "${studentArchType}". Student model rebuilt.`);
      } catch (err) {
        logLine(`ERROR selecting architecture "${studentArchType}": ${err.message}`);
        console.error(err);
      }
    });
  });

  // -------------------- Boot --------------------
  (async function main() {
    try {
      await tf.ready();
      logLine(`TF.js ready. Backend: ${tf.getBackend()}`);
      resetAll();
    } catch (err) {
      logLine(`FATAL: ${err.message}`);
      console.error(err);
    }
  })();
})();
