/* data-loader.js
   Dynamic CSV loader for MNIST (28×28) and ChineseMNIST (64×64)
   Fix: handles ChineseMNIST rows where the last column (character) is NON-numeric.
*/

(() => {
  window.loadTrainFromFiles = (file) => loadCsvToTensors(file, { kind: "train" });
  window.loadTestFromFiles  = (file) => loadCsvToTensors(file, { kind: "test" });

  window.splitTrainVal = splitTrainVal;
  window.getRandomTestBatch = getRandomTestBatch;

  // Backwards compat + generic name
  window.draw28x28ToCanvas = (t, c, s = 4) => drawToCanvas(t, c, s);
  window.drawToCanvas = drawToCanvas;

  const NUM_CLASSES = 10;

  function inferShapeFromColumns(colCount) {
    const candidates = [
      { imgSize: 28, pixelCount: 28 * 28 },
      { imgSize: 64, pixelCount: 64 * 64 },
    ];

    for (const c of candidates) {
      const p = c.pixelCount;
      if (colCount === p)      return { ...c, format: "pixels_only" };
      if (colCount === p + 1)  return { ...c, format: "plus_one" }; // label-first OR pixels-first+label
      if (colCount === p + 2)  return { ...c, format: "plus_two" }; // pixels + label + character (ChineseMNIST)
    }
    return null;
  }

  // Header detection that works for ChineseMNIST:
  // - If column count matches a known layout (784/785/4096/4097/4098),
  //   we DO NOT require every token to be numeric.
  // - We only require that the pixel region is numeric enough to parse.
  function looksLikeDataRow(parts) {
    const inferred = inferShapeFromColumns(parts.length);
    if (!inferred) return false; // unknown layout, probably header/bad row

    const p = inferred.pixelCount;

    // Check that at least the first few pixel tokens are numeric (fast and robust)
    // For label-first, pixels start at 1; for pixels-first, pixels start at 0.
    // For plus_two, pixels definitely start at 0.
    const checkCount = 12; // quick sanity, not full validation
    if (parts.length === p) {
      // pixels only
      for (let i = 0; i < Math.min(checkCount, p); i++) if (!Number.isFinite(+parts[i])) return false;
      return true;
    }

    if (parts.length === p + 2) {
      // pixels..., label, character (character may be non-numeric)
      for (let i = 0; i < Math.min(checkCount, p); i++) if (!Number.isFinite(+parts[i])) return false;
      return true;
    }

    // parts.length === p + 1 (ambiguous)
    // Accept if either:
    // - label-first: parts[0] numeric 0..9 and next pixels numeric
    // - pixels-first: first pixels numeric and last token numeric 0..9
    const a = +parts[0];
    const b = +parts[parts.length - 1];

    const labelFirstOk =
      Number.isFinite(a) &&
      Number.isInteger(a) &&
      a >= 0 && a <= 9 &&
      Number.isFinite(+parts[1]);

    const pixelsFirstOk =
      Number.isFinite(+parts[0]) &&
      Number.isFinite(+parts[1]) &&
      Number.isFinite(b) &&
      Number.isInteger(b) &&
      b >= 0 && b <= 9;

    return labelFirstOk || pixelsFirstOk;
  }

  async function loadCsvToTensors(file, { kind = "data" } = {}) {
    if (!file) throw new Error(`No ${kind} CSV file provided.`);

    const decoder = new TextDecoder("utf-8");
    const CHUNK_BYTES = 8 * 1024 * 1024;
    let offset = 0;
    let carry = "";

    let imageChunks = [];
    let labelChunks = [];
    let rowCount = 0;

    let imgSize = null;
    let pixelCount = null;

    const BATCH_ROWS = 1024;
    let stagingPixels = null;
    let stagingLabels = null;
    let stagingCount = 0;

    function ensureStaging() {
      if (stagingPixels) return;
      stagingPixels = new Float32Array(BATCH_ROWS * pixelCount);
      stagingLabels = new Uint8Array(BATCH_ROWS);
    }

    function flushStaging() {
      if (stagingCount === 0) return;
      imageChunks.push(stagingPixels.slice(0, stagingCount * pixelCount));
      labelChunks.push(stagingLabels.slice(0, stagingCount));
      stagingPixels = new Float32Array(BATCH_ROWS * pixelCount);
      stagingLabels = new Uint8Array(BATCH_ROWS);
      stagingCount = 0;
    }

    function parseRow(parts, lineNo) {
      // infer shape once
      if (pixelCount == null) {
        const inferred = inferShapeFromColumns(parts.length);
        if (!inferred) throw new Error(`Row ${lineNo}: unsupported column count ${parts.length}.`);
        imgSize = inferred.imgSize;
        pixelCount = inferred.pixelCount;
        ensureStaging();
      }

      if (![pixelCount, pixelCount + 1, pixelCount + 2].includes(parts.length)) {
        throw new Error(
          `Row ${lineNo}: column count ${parts.length} does not match expected ${pixelCount}/${pixelCount + 1}/${pixelCount + 2}.`
        );
      }

      let label = 0;
      let pixelStart = 0;

      if (parts.length === pixelCount) {
        // pixels only (autoencoder-style)
        label = 0;
        pixelStart = 0;
      } else if (parts.length === pixelCount + 2) {
        // pixels...,label,character
        label = (+parts[pixelCount]) | 0;
        pixelStart = 0;
      } else {
        // pixelCount + 1 ambiguous
        const a = +parts[0];
        const b = +parts[parts.length - 1];

        const firstLooksLikeLabel = Number.isInteger(a) && a >= 0 && a <= 9;
        const lastLooksLikeLabel  = Number.isInteger(b) && b >= 0 && b <= 9;

        if (firstLooksLikeLabel) {
          label = a | 0;
          pixelStart = 1;
        } else if (lastLooksLikeLabel) {
          label = b | 0;
          pixelStart = 0;
        } else {
          // default to label-first
          label = (Number.isFinite(a) ? a : 0) | 0;
          pixelStart = 1;
        }
      }

      label = Math.max(0, Math.min(9, label | 0));
      stagingLabels[stagingCount] = label;

      const base = stagingCount * pixelCount;
      for (let i = 0; i < pixelCount; i++) {
        const v = +parts[pixelStart + i];
        stagingPixels[base + i] = (Number.isFinite(v) ? v : 0) / 255.0;
      }

      stagingCount++;
      rowCount++;
      if (stagingCount >= BATCH_ROWS) flushStaging();
    }

    let lineNo = 0;

    while (offset < file.size) {
      const slice = file.slice(offset, offset + CHUNK_BYTES);
      const buf = await slice.arrayBuffer();
      await new Promise(requestAnimationFrame);

      carry += decoder.decode(buf, { stream: true });

      const lines = carry.split(/\n/);
      carry = lines.pop() ?? "";

      for (const raw of lines) {
        lineNo++;
        const line = raw.trim();
        if (!line) continue;

        const parts = line.split(",").map((s) => s.trim());

        // Skip header ONLY if it doesn't look like a valid data row.
        if (pixelCount == null && !looksLikeDataRow(parts)) continue;

        parseRow(parts, lineNo);
      }

      offset += CHUNK_BYTES;
    }

    carry += decoder.decode();
    if (carry.trim()) {
      for (const raw of carry.split(/\n/)) {
        lineNo++;
        const line = raw.trim();
        if (!line) continue;
        const parts = line.split(",").map((s) => s.trim());
        if (pixelCount == null && !looksLikeDataRow(parts)) continue;
        parseRow(parts, lineNo);
      }
    }

    flushStaging();

    if (!rowCount) throw new Error(`No rows parsed from ${kind} file`);

    const images = concatFloat32(imageChunks, rowCount * pixelCount);
    const labels = concatUint8(labelChunks, rowCount);

    const xs = tf.tensor4d(images, [rowCount, imgSize, imgSize, 1], "float32");
    const labelTensor = tf.tensor1d(labels, "int32");
    const ys = tf.oneHot(labelTensor, NUM_CLASSES).toFloat();
    labelTensor.dispose();

    return { xs, ys, meta: { imgSize, pixelCount } };
  }

  function concatFloat32(chunks, totalLength) {
    const out = new Float32Array(totalLength);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out;
  }

  function concatUint8(chunks, totalLength) {
    const out = new Uint8Array(totalLength);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out;
  }

  function splitTrainVal(xs, ys, valRatio = 0.1) {
    const n = xs.shape[0];
    const valN = Math.max(1, Math.floor(n * valRatio));
    const trainN = n - valN;

    const h = xs.shape[1];
    const w = xs.shape[2];

    const trainXs = xs.slice([0, 0, 0, 0], [trainN, h, w, 1]);
    const valXs   = xs.slice([trainN, 0, 0, 0], [valN, h, w, 1]);

    const trainYs = ys.slice([0, 0], [trainN, NUM_CLASSES]);
    const valYs   = ys.slice([trainN, 0], [valN, NUM_CLASSES]);

    return { trainXs, trainYs, valXs, valYs };
  }

  function getRandomTestBatch(xs, ys, k = 5) {
    const n = xs.shape[0];
    const kk = Math.min(k, n);
    const idx = new Int32Array(kk);
    for (let i = 0; i < kk; i++) idx[i] = (Math.random() * n) | 0;

    const idxTensor = tf.tensor1d(idx, "int32");
    const batchXs = tf.gather(xs, idxTensor);
    const batchYs = tf.gather(ys, idxTensor);
    idxTensor.dispose();
    return { batchXs, batchYs };
  }

  function drawToCanvas(imageTensor, canvas, scale = 3) {
    const s = scale | 0;

    const { h, w, data } = tf.tidy(() => {
      let t = imageTensor;
      if (t.rank === 4) t = t.squeeze([0, 3]);
      else if (t.rank === 3) t = t.squeeze([2]);
      return { h: t.shape[0], w: t.shape[1], data: t.dataSync() };
    });

    canvas.width = w * s;
    canvas.height = h * s;

    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(w * s, h * s);

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
})();
