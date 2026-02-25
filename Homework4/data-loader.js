/* data-loader.js
   File-based MNIST CSV parsing and tensor utilities (NO external libraries).

   CSV format (no header):
     label,p0,p1,...,p783
   where label в€€ {0..9} and pixels в€€ {0..255}

   Goals:
   - Parse user-uploaded file
   - Normalize pixels to [0,1] by /255
   - Create tensors:
       xs: [N, 28, 28, 1] float32
       ys: [N, 10] one-hot float32
   - Provide helpers:
       loadTrainFromFiles(file), loadTestFromFiles(file)
       splitTrainVal(xs, ys, valRatio)
       getRandomTestBatch(xs, ys, k)
       draw28x28ToCanvas(tensor, canvas, scale)
   - Avoid leaks: dispose intermediate tensors/arrays when appropriate.

   NOTE ABOUT PERFORMANCE:
   - MNIST train has 60k lines. Reading full text is usually OK in modern browsers,
     but we still implement a chunked reader using Blob.slice + TextDecoder so
     Safari/Chrome won't choke as easily on large files.
   - We parse lines incrementally; we DO NOT store the entire file string at once.
*/

(() => {
  // Expose functions on window so app.js can use them without modules/bundlers.
  window.loadTrainFromFiles = loadTrainFromFiles;
  window.loadTestFromFiles = loadTestFromFiles;
  window.splitTrainVal = splitTrainVal;
  window.getRandomTestBatch = getRandomTestBatch;
  window.draw28x28ToCanvas = draw28x28ToCanvas;

  const PIXELS = 28 * 28;
  const NUM_CLASSES = 10;

  async function loadTrainFromFiles(file) {
    return loadMnistCsv(file, { kind: "train" });
  }

  async function loadTestFromFiles(file) {
    return loadMnistCsv(file, { kind: "test" });
  }

  async function loadMnistCsv(file, { kind = "data" } = {}) {
    if (!file) throw new Error(`No ${kind} CSV file provided.`);

    // We'll parse into JS typed arrays first (fast + compact), then create tensors once.
    // Memory plan:
    // - imagesFloat: Float32Array length N*784
    // - labelsInt: Uint8Array length N
    //
    // But we don't know N upfront. We store chunks in arrays then flatten once.
    const imageChunks = []; // each chunk is Float32Array
    const labelChunks = []; // each chunk is Uint8Array
    let rowCount = 0;

    // Robust incremental line parsing:
    // - read bytes in slices
    // - decode to text chunk
    // - append to carry string
    // - split by newline, parse complete lines, keep leftover in carry
    const decoder = new TextDecoder("utf-8");
    const CHUNK_BYTES = 8 * 1024 * 1024; // 8MB per read (tune if needed)
    let offset = 0;
    let carry = "";

    // We batch allocations to reduce overhead: gather parsed rows in JS arrays then flush to typed arrays.
    // "BATCH_ROWS" controls how often we convert the staging arrays into typed arrays.
    const BATCH_ROWS = 2048;
    let stagingPixels = new Float32Array(BATCH_ROWS * PIXELS);
    let stagingLabels = new Uint8Array(BATCH_ROWS);
    let stagingCount = 0;

    // Helper: flush staging arrays into chunks and reset staging.
    function flushStaging() {
      if (stagingCount === 0) return;
      // Slice the used portion only (avoid extra unused capacity).
      imageChunks.push(stagingPixels.slice(0, stagingCount * PIXELS));
      labelChunks.push(stagingLabels.slice(0, stagingCount));
      stagingPixels = new Float32Array(BATCH_ROWS * PIXELS);
      stagingLabels = new Uint8Array(BATCH_ROWS);
      stagingCount = 0;
    }

    // Parsing a CSV line fast-ish without external libs:
    // - MNIST CSV has no quoted commas, so split(',') is safe.
    // - We still guard against empty lines / whitespace.
    function parseLine(line) {
      const t = line.trim();
      if (!t) return; // ignore empty lines

      const parts = t.split(",");
      if (parts.length !== (1 + PIXELS)) {
        // Some files might have \r at end; trim handles it, but still guard.
        throw new Error(`Bad row with ${parts.length} columns; expected ${1 + PIXELS}.`);
      }

      const label = parts[0] | 0;
      if (label < 0 || label >= NUM_CLASSES) {
        throw new Error(`Invalid label '${parts[0]}' (expected 0-9).`);
      }

      // Write into staging buffers
      stagingLabels[stagingCount] = label;

      const base = stagingCount * PIXELS;
      for (let i = 0; i < PIXELS; i++) {
        // Convert pixel string to number and normalize to [0,1]
        // Using +parts[...] is fast and yields number (NaN check optional).
        const v = +parts[i + 1];
        stagingPixels[base + i] = v / 255.0;
      }

      stagingCount++;
      rowCount++;

      // When staging is full, flush into chunk arrays.
      if (stagingCount >= BATCH_ROWS) flushStaging();
    }

    // Read and parse file incrementally.
    while (offset < file.size) {
      const slice = file.slice(offset, offset + CHUNK_BYTES);
      const buf = await slice.arrayBuffer();

      // Yield to UI thread between chunks (keeps page responsive).
      await new Promise(requestAnimationFrame);

      // Decode this chunk and merge with leftover carry from previous chunk.
      carry += decoder.decode(buf, { stream: true });

      // Split into lines; keep last partial line in carry.
      const lines = carry.split(/\n/);
      carry = lines.pop() ?? "";

      for (const line of lines) parseLine(line);

      offset += CHUNK_BYTES;
    }

    // Finish decoder stream and parse any remaining text.
    carry += decoder.decode();
    if (carry.trim()) {
      // carry may contain multiple lines if file doesn't end with newline
      const tailLines = carry.split(/\n/);
      for (const line of tailLines) parseLine(line);
    }

    // Flush whatever remains staged
    flushStaging();

    if (rowCount === 0) throw new Error(`No rows parsed from ${kind} file. Is it empty?`);

    // Flatten chunks into contiguous typed arrays.
    const images = concatFloat32(imageChunks, rowCount * PIXELS);
    const labels = concatUint8(labelChunks, rowCount);

    // Create tensors:
    // xs: [N, 28, 28, 1]
    // ys: one-hot [N, 10]
    //
    // IMPORTANT: tensor creation copies from typed array into TF-managed memory.
    // After tensors are created, we can let JS typed arrays be garbage collected.
    const xs = tf.tensor4d(images, [rowCount, 28, 28, 1], "float32");
    const labelTensor = tf.tensor1d(labels, "int32");
    const ys = tf.oneHot(labelTensor, NUM_CLASSES).toFloat();

    // labelTensor is intermediate; dispose now to avoid leaks.
    labelTensor.dispose();

    return { xs, ys };
  }

  function concatFloat32(chunks, totalLength) {
    const out = new Float32Array(totalLength);
    let offset = 0;
    for (const c of chunks) {
      out.set(c, offset);
      offset += c.length;
    }
    return out;
  }

  function concatUint8(chunks, totalLength) {
    const out = new Uint8Array(totalLength);
    let offset = 0;
    for (const c of chunks) {
      out.set(c, offset);
      offset += c.length;
    }
    return out;
  }

  function splitTrainVal(xs, ys, valRatio = 0.1) {
    // We assume xs, ys are aligned on first dimension N.
    // We split WITHOUT shuffling here (training will shuffle=true).
    // If you want a randomized split, shuffle indices first.
    const n = xs.shape[0];
    const valN = Math.max(1, Math.floor(n * valRatio));
    const trainN = n - valN;

    // Slicing creates new tensors; caller should dispose when done.
    const trainXs = xs.slice([0, 0, 0, 0], [trainN, 28, 28, 1]);
    const trainYs = ys.slice([0, 0], [trainN, NUM_CLASSES]);
    const valXs = xs.slice([trainN, 0, 0, 0], [valN, 28, 28, 1]);
    const valYs = ys.slice([trainN, 0], [valN, NUM_CLASSES]);

    return { trainXs, trainYs, valXs, valYs };
  }

  function getRandomTestBatch(xs, ys, k = 5) {
    const n = xs.shape[0];
    const kk = Math.min(k, n);

    // Random integer indices in [0, n)
    const idx = new Int32Array(kk);
    for (let i = 0; i < kk; i++) idx[i] = (Math.random() * n) | 0;

    // Gather is convenient, but for 4D itвЂ™s a bit heavier; still fine for k=5.
    const idxTensor = tf.tensor1d(idx, "int32");
    const batchXs = tf.gather(xs, idxTensor);
    const batchYs = tf.gather(ys, idxTensor);

    // idxTensor is intermediate.
    idxTensor.dispose();

    return { batchXs, batchYs };
  }

  function draw28x28ToCanvas(imageTensor, canvas, scale = 4) {
    // imageTensor: either [28,28] or [28,28,1] or [1,28,28,1]
    // We pull it to CPU once, then paint pixels.
    const s = scale | 0;
    const w = 28 * s, h = 28 * s;
    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d");
    const imgData = ctx.createImageData(w, h);

    // Use tidy to avoid keeping reshaped tensors around.
    const data = tf.tidy(() => {
      let t = imageTensor;
      if (t.rank === 4) t = t.squeeze([0, 3]);     // [1,28,28,1] -> [28,28]
      else if (t.rank === 3) t = t.squeeze([2]);   // [28,28,1] -> [28,28]
      // Ensure float32 in [0,1]
      return t.dataSync(); // TypedArray length 784
    });

    // Paint scaled pixels (nearest-neighbor upscale).
    // MNIST is grayscale; map v in [0,1] to 0..255.
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = data[y * 28 + x];
        const c = Math.max(0, Math.min(255, (v * 255) | 0));

        for (let dy = 0; dy < s; dy++) {
          for (let dx = 0; dx < s; dx++) {
            const px = (y * s + dy) * w + (x * s + dx);
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
