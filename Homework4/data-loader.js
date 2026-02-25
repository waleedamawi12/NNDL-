/* data-loader.js
   Dynamic MNIST/ChineseMNIST CSV loader
   Supports 28x28 (784 px) and 64x64 (4096 px)
*/

(() => {

  window.loadTrainFromFiles = loadFromFiles;
  window.loadTestFromFiles = loadFromFiles;
  window.splitTrainVal = splitTrainVal;
  window.getRandomTestBatch = getRandomTestBatch;
  window.drawImageToCanvas = drawImageToCanvas;

  async function loadFromFiles(file) {
    if (!file) throw new Error("No CSV file provided.");

    const text = await file.text();
    const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
    if (lines.length === 0) throw new Error("CSV empty.");

    const first = lines[0].split(",");
    const colCount = first.length;

    let imgSize;
    let pixelCount;

    if (colCount === 785 || colCount === 784) {
      imgSize = 28;
      pixelCount = 28 * 28;
    } else if (colCount === 4097 || colCount === 4098 || colCount === 4096) {
      imgSize = 64;
      pixelCount = 64 * 64;
    } else {
      throw new Error(`Unsupported column count: ${colCount}`);
    }

    const images = new Float32Array(lines.length * pixelCount);
    const labels = new Int32Array(lines.length);

    for (let r = 0; r < lines.length; r++) {
      const parts = lines[r].split(",");

      let pixelStart = 0;

      // detect format
      if (parts.length === pixelCount + 1) {
        // label first
        labels[r] = +parts[0];
        pixelStart = 1;
      } else if (parts.length === pixelCount + 2) {
        // pixels first + label + character
        labels[r] = +parts[pixelCount];
        pixelStart = 0;
      } else if (parts.length === pixelCount) {
        // no label
        labels[r] = 0;
        pixelStart = 0;
      } else {
        throw new Error(`Row ${r} has unexpected column count.`);
      }

      const base = r * pixelCount;
      for (let i = 0; i < pixelCount; i++) {
        images[base + i] = (+parts[pixelStart + i]) / 255.0;
      }

      if (r % 2000 === 0) await tf.nextFrame();
    }

    const xs = tf.tensor4d(images, [lines.length, imgSize, imgSize, 1], "float32");
    const ys = tf.oneHot(tf.tensor1d(labels, "int32"), 10).toFloat();

    return { xs, ys, imgSize };
  }

  function splitTrainVal(xs, ys, valRatio = 0.1) {
    const n = xs.shape[0];
    const valN = Math.floor(n * valRatio);
    const trainN = n - valN;

    const imgSize = xs.shape[1];

    const trainXs = xs.slice([0, 0, 0, 0], [trainN, imgSize, imgSize, 1]);
    const valXs = xs.slice([trainN, 0, 0, 0], [valN, imgSize, imgSize, 1]);

    const trainYs = ys.slice([0, 0], [trainN, 10]);
    const valYs = ys.slice([trainN, 0], [valN, 10]);

    return { trainXs, trainYs, valXs, valYs };
  }

  function getRandomTestBatch(xs, ys, k = 5) {
    const n = xs.shape[0];
    const idx = new Int32Array(k);
    for (let i = 0; i < k; i++) idx[i] = (Math.random() * n) | 0;

    const idxTensor = tf.tensor1d(idx, "int32");
    const batchXs = tf.gather(xs, idxTensor);
    const batchYs = tf.gather(ys, idxTensor);
    idxTensor.dispose();

    return { batchXs, batchYs };
  }

  function drawImageToCanvas(tensor, canvas, scale = 3) {
    const size = tensor.shape[1];
    const ctx = canvas.getContext("2d");
    canvas.width = size * scale;
    canvas.height = size * scale;

    const data = tensor.squeeze().dataSync();
    const imgData = ctx.createImageData(size * scale, size * scale);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const v = Math.floor(data[y * size + x] * 255);

        for (let dy = 0; dy < scale; dy++) {
          for (let dx = 0; dx < scale; dx++) {
            const px = (y * scale + dy) * (size * scale) + (x * scale + dx);
            const o = px * 4;
            imgData.data[o] = v;
            imgData.data[o + 1] = v;
            imgData.data[o + 2] = v;
            imgData.data[o + 3] = 255;
          }
        }
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }

})();
