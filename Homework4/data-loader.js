/* data-loader.js
   Dynamic CSV loader for MNIST (28×28) and ChineseMNIST (64×64)
   ------------------------------------------------------------
   Supports these row formats (no header):
   1) label,p0,p1,...            (MNIST / label-first)
   2) p0,p1,...,label            (pixels-first)
   3) p0,p1,...,label,character  (ChineseMNIST common)

   Output:
     xs: tf.Tensor4D [N,H,W,1] float32 in [0,1]
     ys: tf.Tensor2D [N,10] one-hot float32  (always depth 10 for compatibility)
     meta: { imgSize, pixelCount, format }
*/

(() => {
  window.loadTrainFromFiles = (file) => loadCsvToTensors(file, { kind: "train" });
  window.loadTestFromFiles  = (file) => loadCsvToTensors(file, { kind: "test" });

  window.splitTrainVal = splitTrainVal;
  window.getRandomTestBatch = getRandomTestBatch;

  // Backwards-compatible name:
  window.draw28x28ToCanvas = (t, c, s = 4) => drawToCanvas(t, c, s);
  // New generic name:
  window.drawToCanvas = drawToCanvas;

  const NUM_CLASSES = 10;

  function inferShapeFromColumns(colCount) {
    // Allow: pixels only, label-first, pixels-first, pixels-first+character
    const candidates = [
      { imgSize: 28, pixelCount: 28 * 28 },
      { imgSize: 64, pixelCount: 64 * 64 },
    ];

    for (const c of candidates) {
      const p = c.pixelCount;
      if (colCount === p) return { ...c, format: "pixels_only" };
      if (colCount === p + 1) return { ...c, format: "label_first_or_pixels_first" };
      if (colCount === p + 2) return { ...c, format: "pixels_first_plus_char" };
    }
    throw new Error(
      `Unsupported column count: ${colCount}. Expected 784/785 or 4096/4097/4098 (no header).`
    );
  }

  function isNumericRow(parts) {
    // Quick header detection: if any token is non-numeric (like "label"), treat as header.
    for (let i = 0; i < parts.length; i++) {
      const t = parts[i].trim();
      if (t === "") return false;
      // Allow things like "12" "12.0"
      if (!Number.isFinite(+t)) return false;
    }
    return true;
  }

  async function loadCsvToTensors(file, { kind = "data" } = {}) {
    if (!file) throw new Error(`No ${kind} CSV file provided.`);

    // Read file in chunks to avoid Safari choking on huge text()
    const decoder = new TextDecoder("utf-8");
    const CHUNK_BYTES = 8 * 1024 * 1024;
    let offset = 0;
    let carry = "";

    // We don't know row count in advance → store chunks
    let imageChunks = [];
    let labelChunks = [];
    let rowCount = 0;

    // Determined after first valid row
    let imgSize = null;
    let pixelCount = null;
    let rowFormat = null;

    // Staging buffers (reused)
    const BATCH_ROWS = 1024;
    let stagingPixels = null; // Float32Array
    let stagingLabels = null; // Uint8Array
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
      // Initialize shape/format from first valid row
      if (pixelCount == null) {
        const inferred = inferShapeFromColumns(parts.length);
        imgSize = inferred.imgSize;
        pixelCount = inferred.pixelCount;
        rowFormat = inferred.format;
        ensureStaging();
      }

      // If shapes don’t match after inference, error out clearly.
      if (![pixelCount, pixelCount + 1, pixelCount + 2].includes(parts.length)) {
        throw new Error(
          `Row ${lineNo}: column count ${parts.length} does not match expected ${pixelCount}/${pixelCount + 1}/${pixelCount + 2}.`
        );
      }

      // Decide where pixels start + where label is.
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
        // pixelCount + 1: ambiguous (could be label-first OR pixels-first-with-label)
        // Heuristic: if first token is integer 0..9 and last token looks like pixel (0..255),
        // assume label-first. Otherwise if last token is 0..9 and first looks like pixel, assume pixels-first.
        const a = +parts[0];
        const b = +parts[parts.length - 1];
        const firstLooksLikeLabel = Number.isInteger(a) && a >= 0 && a <= 9;
        const lastLooksLikeLabel = Number.isInteger(b) && b >= 0 && b <= 9;

        const firstLooksLikePixel = a >= 0 && a <= 255;
        const lastLooksLikePixel = b >= 0 && b <= 255;

        if (firstLooksLikeLabel && lastLooksLikePixel) {
          label = a | 0;
          pixelStart = 1;
        } else if (lastLooksLikeLabel && firstLooksLikePixel) {
          label = b | 0;
          pixelStart = 0;
        } else {
          // default to label-first (MNIST convention)
          label = a | 0;
          pixelStart = 1;
        }
      }

      // Clamp label into 0..9 to satisfy oneHot depth=10
      if (!Number.isFinite(label)) label = 0;
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

        // Skip header row if detected
        if (pixelCount == null) {
          if (!isNumericRow(parts)) continue;
        }

        parseRow(parts, lineNo);
      }

      offset += CHUNK_BYTES;
    }

    carry += decoder.decode();
    if (carry.trim()) {
      const tailLines = carry.split(/\n/);
      for (const raw of tailLines) {
        lineNo++;
        const line = raw.trim();
        if (!line) continue;
        const parts = line.split(",").map((s) => s.trim());
        if (pixelCount == null) {
          if (!isNumericRow(parts)) continue;
        }
        parseRow(parts, lineNo);
      }
    }

    flushStaging();

    if (!rowCount) throw new Error(`No rows parsed from ${kind} file.`);

    // Flatten chunks
    const images = concatFloat32(imageChunks, rowCount * pixelCount);
    const labels = concatUint8(labelChunks, rowCount);

    // Build tensors
    const xs = tf.tensor4d(images, [rowCount, imgSize, imgSize, 1], "float32");

    const labelTensor = tf.tensor1d(labels, "int32");
    const ys = tf.oneHot(labelTensor, NUM_CLASSES).toFloat();
    labelTensor.dispose();

    return { xs, ys, meta: { imgSize, pixelCount, format: rowFormat } };
  }

  function concatFloat32(chunks, totalLength) {
    const out = new Float32Array(totalLength);
    let off = 0;
    for (const c of chunks) {
      out.set(c, off);
      off += c.length;
    }
    return out;
  }

  function concatUint8(chunks, totalLength) {
    const out = new Uint8Array(totalLength);
    let off = 0;
    for (const c of chunks) {
      out.set(c, off);
      off += c.length;
    }
    return out;
  }

  function splitTrainVal(xs, ys, valRatio = 0.1) {
    const n = xs.shape[0];
    const valN = Math.max(1, Math.floor(n * valRatio));
    const trainN = n - valN;

    const h = xs.shape[1];
    const w = xs.shape[2];

    const trainXs = xs.slice([0, 0, 0, 0], [trainN, h, w, 1]);
    const valXs = xs.slice([trainN, 0, 0, 0], [valN, h, w, 1]);

    const trainYs = ys.slice([0, 0], [trainN, NUM_CLASSES]);
    const valYs = ys.slice([trainN, 0], [valN, NUM_CLASSES]);

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
    // imageTensor: [H,W] or [H,W,1] or [1,H,W,1]
    const s = scale | 0;

    const { h, w, data } = tf.tidy(() => {
      let t = imageTensor;
      if (t.rank === 4) t = t.squeeze([0, 3]);
      else if (t.rank === 3) t = t.squeeze([2]);
      const hh = t.shape[0];
      const ww = t.shape[1];
      return { h: hh, w: ww, data: t.dataSync() };
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
