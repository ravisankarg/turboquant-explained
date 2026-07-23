const original = [0.12, -0.08, 0.15, 9.8, -0.11, 0.06, -10.2, 0.18];
const rotated = [4.71, -5.02, 3.88, -4.54, 5.31, -4.26, 4.63, -4.78];
const turboDequant = [0.121, -0.079, 0.148, 9.807, -0.108, 0.061, -10.191, 0.177];
const pairs = [
  { cart: [0.12, -0.08], radius: 0.144, angle: -33.7 },
  { cart: [0.15, 9.8], radius: 9.801, angle: 89.1 },
  { cart: [-0.11, 0.06], radius: 0.125, angle: 151.4 },
  { cart: [-10.2, 0.18], radius: 10.202, angle: 179.0 }
];
const polarPairDequant = [
  [0.119, -0.079],
  [0.148, 9.807],
  [-0.108, 0.061],
  [-10.191, 0.177]
];
function benchRow(dataset, bits, selfR1, selfR10, randomR10, msQuery, rom, vectorStaging, ram) {
  return { dataset, vectors: dataset.replace("Cohere ", ""), bits, selfR1, selfR10, randomR10, msQuery, rom, romMb: Number.parseFloat(rom), vectorStaging, vectorRam: ram };
}

const androidBenchTables = [
  {
    title: "FlatIndex — A) real traffic / query-major",
    note: "1 timed query worker; 8 Rayon workers for recall/truth and preparation. One-time build: 50K FP32 0.0, FP16 78.0, 8-bit 1,237.2, 4-bit 1,154.4 ms; 100K FP32 0.0, FP16 170.9, 8-bit 2,461.6, 4-bit 3,112.6 ms.",
    rows: [
      benchRow("Cohere 50K", 32, "100.00%", "100.00%", "100.00%", 33.604, "146.5 MB", "29.0 MB raw FP32 chunk", "30.0 MB"),
      benchRow("Cohere 50K", 16, "100.00%", "99.88%", "99.64%", 21.381, "73.2 MB", "28.9 MB raw FP16 + fused dot", "30.0 MB"),
      benchRow("Cohere 50K", 8, "100.00%", "99.10%", "98.54%", 30.123, "36.8 MB", "28.9 MB blocked 8-bit range", "30.0 MB"),
      benchRow("Cohere 50K", 4, "100.00%", "93.04%", "90.00%", 16.210, "18.5 MB", "28.9 MB blocked 4-bit range", "30.0 MB"),
      benchRow("Cohere 100K", 32, "100.00%", "100.00%", "100.00%", 68.712, "293.0 MB", "49.0 MB raw FP32 chunk", "50.0 MB"),
      benchRow("Cohere 100K", 16, "100.00%", "99.74%", "99.73%", 45.118, "146.5 MB", "48.9 MB raw FP16 + fused dot", "50.0 MB"),
      benchRow("Cohere 100K", 8, "100.00%", "99.31%", "98.77%", 85.440, "73.6 MB", "48.9 MB blocked 8-bit range", "50.0 MB"),
      benchRow("Cohere 100K", 4, "100.00%", "93.46%", "90.32%", 70.775, "37.0 MB", "48.9 MB blocked 4-bit range", "50.0 MB")
    ]
  },
  {
    title: "FlatIndex — B) batched throughput",
    note: "8 Rayon workers in the timed range-major batch. Each bounded chunk/range is reused across 2,000 queries. One-time build: 50K FP32 0.0, FP16 87.1, 8-bit 1,300.7, 4-bit 1,183.4 ms; 100K FP32 0.0, FP16 191.8, 8-bit 2,857.8, 4-bit 4,512.1 ms. This is the last B report pulled before final B-only source edits; rerun B on a connected S25 for updated latency.",
    rows: [
      benchRow("Cohere 50K", 32, "100.00%", "100.00%", "100.00%", 0.587, "146.5 MB", "23.1 MB raw FP32 chunk", "30.0 MB"),
      benchRow("Cohere 50K", 16, "100.00%", "99.88%", "99.64%", 1.330, "73.2 MB", "23.1 MB raw FP16 chunk", "30.0 MB"),
      benchRow("Cohere 50K", 8, "100.00%", "99.10%", "98.54%", 4.727, "36.8 MB", "22.7 MB blocked 8-bit range", "30.0 MB"),
      benchRow("Cohere 50K", 4, "100.00%", "93.04%", "90.00%", 0.381, "18.5 MB", "21.9 MB blocked 4-bit range", "30.0 MB"),
      benchRow("Cohere 100K", 32, "100.00%", "100.00%", "100.00%", 1.294, "293.0 MB", "43.1 MB raw FP32 chunk", "50.0 MB"),
      benchRow("Cohere 100K", 16, "100.00%", "99.74%", "99.73%", 3.363, "146.5 MB", "43.1 MB raw FP16 chunk", "50.0 MB"),
      benchRow("Cohere 100K", 8, "100.00%", "99.31%", "98.77%", 11.400, "73.6 MB", "42.7 MB blocked 8-bit range", "50.0 MB"),
      benchRow("Cohere 100K", 4, "100.00%", "93.46%", "90.32%", 1.383, "37.0 MB", "41.9 MB blocked 4-bit range", "50.0 MB")
    ]
  },
  {
    title: "HNSW — A) real traffic / query-major",
    note: "1 timed query worker; 8 Rayon workers for recall/truth and preparation. M=16, efConstruction=128, efSearch=1024, max layers=8, base degree ≤32. Compact graph resident; payload vectors disk-backed. Graph build was a persistent-cache hit. Quantized rows use same-bit finalization only; no raw-FP32 cross-bit rerank.",
    rows: [
      benchRow("Cohere 50K", 32, "99.90%", "99.75%", "98.99%", 41.544, "227.4 MB", "6.0 MB FP16 navigation cache + FP32 candidate scratch", "≤30.0 MB"),
      benchRow("Cohere 50K", 16, "99.90%", "99.65%", "98.65%", 39.506, "80.9 MB", "6.0 MB FP16 navigation cache", "≤30.0 MB"),
      benchRow("Cohere 50K", 8, "99.90%", "98.89%", "97.67%", 56.208, "154.6 MB", "6.0 MB cache + compressed block", "≤30.0 MB"),
      benchRow("Cohere 50K", 4, "99.90%", "92.94%", "89.50%", 53.324, "117.9 MB", "6.0 MB cache + compressed block", "≤30.0 MB"),
      benchRow("Cohere 100K", 32, "99.30%", "99.50%", "98.42%", 49.873, "454.8 MB", "6.0 MB FP16 navigation cache + FP32 candidate scratch", "≤50.0 MB"),
      benchRow("Cohere 100K", 16, "99.30%", "99.25%", "98.22%", 48.026, "161.9 MB", "6.0 MB FP16 navigation cache", "≤50.0 MB"),
      benchRow("Cohere 100K", 8, "99.30%", "98.85%", "97.39%", 67.199, "309.1 MB", "6.0 MB cache + compressed block", "≤50.0 MB"),
      benchRow("Cohere 100K", 4, "99.30%", "93.21%", "89.51%", 63.597, "235.9 MB", "6.0 MB cache + compressed block", "≤50.0 MB")
    ]
  },
  {
    title: "HNSW — B) batched throughput",
    note: "8 Rayon workers in the timed graph-search batch; each worker has a bounded 256-vector FP16 candidate cache. M=16, efConstruction=128, efSearch=1024, max layers=8, base degree ≤32. Graph build is one-time and excluded. Quantized rows use same-bit finalization only; no raw-FP32 cross-bit rerank.",
    rows: [
      benchRow("Cohere 50K", 32, "99.90%", "99.75%", "98.99%", 6.054, "227.4 MB", "384 KB FP16 candidates/worker + FP32 scratch", "≤30.0 MB"),
      benchRow("Cohere 50K", 16, "99.90%", "99.65%", "98.65%", 5.411, "80.9 MB", "384 KB FP16 candidates/worker", "≤30.0 MB"),
      benchRow("Cohere 50K", 8, "99.90%", "98.89%", "97.67%", 8.849, "154.6 MB", "384 KB cache + compressed block", "≤30.0 MB"),
      benchRow("Cohere 50K", 4, "99.90%", "92.94%", "89.50%", 12.448, "117.9 MB", "384 KB cache + compressed block", "≤30.0 MB"),
      benchRow("Cohere 100K", 32, "99.30%", "99.50%", "98.42%", 11.969, "454.8 MB", "384 KB FP16 candidates/worker + FP32 scratch", "≤50.0 MB"),
      benchRow("Cohere 100K", 16, "99.30%", "99.25%", "98.22%", 8.446, "161.9 MB", "384 KB FP16 candidates/worker", "≤50.0 MB"),
      benchRow("Cohere 100K", 8, "99.30%", "98.85%", "97.39%", 16.042, "309.1 MB", "384 KB cache + compressed block", "≤50.0 MB"),
      benchRow("Cohere 100K", 4, "99.30%", "93.21%", "89.51%", 15.200, "235.9 MB", "384 KB cache + compressed block", "≤50.0 MB")
    ]
  }
];

const androidBenchRows = androidBenchTables[0].rows;

function renderBars(targetId, values, maxAbs = Math.max(...values.map(Math.abs))) {
  const target = document.getElementById(targetId);
  if (!target) return;

  target.innerHTML = values.map((value, index) => {
    const width = Math.max(1, Math.abs(value) / maxAbs * 50);
    const isNegative = value < 0;
    const isOutlier = Math.abs(value) > maxAbs * 0.75;
    const left = isNegative ? `${50 - width}%` : "50%";
    const cls = ["bar-fill", isNegative ? "negative" : "", isOutlier ? "outlier" : ""].join(" ");
    return `
      <div class="bar-row">
        <span class="dim">d${index + 1}</span>
        <span class="val">${value.toFixed(value >= 1 || value <= -1 ? 2 : 3)}</span>
        <span class="bar-track">
          <span class="${cls}" style="left:${left};width:${width}%"></span>
        </span>
      </div>
    `;
  }).join("");
}

function renderPolar() {
  const target = document.getElementById("polarGrid");
  if (!target) return;

  const maxRadius = Math.max(...pairs.map(pair => pair.radius));
  target.innerHTML = pairs.map((pair, index) => {
    const len = 24 + pair.radius / maxRadius * 58;
    return `
      <article class="polar-item">
        <div class="polar-viz">
          <div class="axis">
            <span class="needle" style="--angle:${90 - pair.angle}deg;--len:${len}px"></span>
          </div>
        </div>
        <code>p${index + 1} = (${pair.cart[0]}, ${pair.cart[1]})</code>
        <strong>r = ${pair.radius.toFixed(3)}, theta = ${pair.angle.toFixed(1)} deg</strong>
      </article>
    `;
  }).join("");
}

function renderCacheStack() {
  const target = document.getElementById("cacheStack");
  if (!target) return;

  target.innerHTML = Array.from({ length: 9 }, (_, index) => {
    const width = 46 + index * 5;
    return `
      <div class="cache-token">
        <span>token ${index + 1}</span>
        <div class="cache-block key" style="width:${width}%"></div>
        <div class="cache-block value" style="width:${Math.min(96, width + 8)}%"></div>
      </div>
    `;
  }).join("");
}

function formatSigned(value, digits = 3) {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

function quantizeCode(value, min, max) {
  const step = (max - min) / 15;
  return {
    step,
    code: Math.max(0, Math.min(15, Math.round((value - min) / step))),
  };
}

function buildBinMeta(min, max, unit = "") {
  const step = (max - min) / 15;
  return Array.from({ length: 16 }, (_, index) => {
    const center = min + index * step;
    const halfStep = step / 2;
    const lo = index === 0 ? min : center - halfStep;
    const hi = index === 15 ? max : center + halfStep;
    return {
      center,
      range: `${lo.toFixed(2)}..${hi.toFixed(2)}${unit}`,
      centroid: `${center.toFixed(2)}${unit}`,
    };
  });
}

function renderBinStrip(targetId, codes, binMeta) {
  const target = document.getElementById(targetId);
  if (!target) return;

  const counts = codes.reduce((map, code) => {
    map[code] = (map[code] || 0) + 1;
    return map;
  }, {});

  target.innerHTML = Array.from({ length: 16 }, (_, index) => {
    const count = counts[index] || 0;
    const cls = ["bin", count ? "active" : "", count > 1 ? "hot" : ""].join(" ");
    const meta = binMeta[index];
    return `
      <span class="${cls}" title="bin ${index}: ${meta.range}; centroid ${meta.centroid}">
        <strong>${index}</strong>
        <em>${meta.centroid}</em>
        <small>${meta.range}</small>
        ${count ? `<b>${count}</b>` : ""}
      </span>
    `;
  }).join("");
}

function renderErrorList(targetId, rows) {
  const target = document.getElementById(targetId);
  if (!target) return;

  target.innerHTML = rows.map(row => `
    <div>
      <strong>${row.title}</strong>
      <span>${row.text}</span>
    </div>
  `).join("");
}

function renderIndexQuantDetails() {
  const min = Math.min(...original);
  const max = Math.max(...original);
  const standardRows = original.map((value, index) => {
    const { step, code } = quantizeCode(value, min, max);
    const dequant = min + code * step;
    return { index, value, code, dequant, error: dequant - value };
  });

  renderBinStrip("standardBins", standardRows.map(row => row.code), buildBinMeta(min, max));
  renderErrorList("standardErrorList", [
    {
      title: "Six small coordinates collapse into bin 8",
      text: "Values near zero dequantize to 0.467 because the outliers set a wide 1.333 step.",
    },
    {
      title: "Example error",
      text: `d2 original -0.080 -> dequant ${standardRows[1].dequant.toFixed(3)}, error ${formatSigned(standardRows[1].error)}.`,
    },
  ]);

  const rotMin = Math.min(...rotated);
  const rotMax = Math.max(...rotated);
  const rotatedRows = rotated.map((value, index) => {
    const { step, code } = quantizeCode(value, rotMin, rotMax);
    return { index, value, code, step, final: turboDequant[index], error: turboDequant[index] - original[index] };
  });

  renderBinStrip("rotatedBins", rotatedRows.map(row => row.code), buildBinMeta(rotMin, rotMax));
  renderErrorList("rotatedErrorList", [
    {
      title: `Rotated min/max step is ${rotatedRows[0].step.toFixed(3)}`,
      text: "The bins are based on the rotated values, so bucket placement is spread across the code range instead of all small dimensions sharing one bucket.",
    },
    {
      title: "Example after inverse rotation",
      text: `d2 original -0.080 -> TurboQuant dequant ${rotatedRows[1].final.toFixed(3)}, error ${formatSigned(rotatedRows[1].error)}.`,
    },
  ]);

  const radiusMin = Math.min(...pairs.map(pair => pair.radius));
  const radiusMax = Math.max(...pairs.map(pair => pair.radius));
  const angleMin = Math.min(...pairs.map(pair => pair.angle));
  const angleMax = Math.max(...pairs.map(pair => pair.angle));
  const radiusCodes = pairs.map(pair => quantizeCode(pair.radius, radiusMin, radiusMax).code);
  const angleCodes = pairs.map(pair => quantizeCode(pair.angle, angleMin, angleMax).code);

  renderBinStrip("polarRadiusBins", radiusCodes, buildBinMeta(radiusMin, radiusMax));
  renderBinStrip("polarAngleBins", angleCodes, buildBinMeta(angleMin, angleMax, "deg"));
  renderErrorList("polarErrorList", [
    {
      title: "Radius and angle get separate bins",
      text: `Radius min/max step is ${((radiusMax - radiusMin) / 15).toFixed(3)}; angle min/max step is ${((angleMax - angleMin) / 15).toFixed(2)} deg.`,
    },
    {
      title: "Pair example",
      text: `p2 original (0.150, 9.800) -> dequant (${polarPairDequant[1][0].toFixed(3)}, ${polarPairDequant[1][1].toFixed(3)}), errors ${formatSigned(polarPairDequant[1][0] - pairs[1].cart[0])}, ${formatSigned(polarPairDequant[1][1] - pairs[1].cart[1])}.`,
    },
  ]);
}

function renderQuantTable() {
  const target = document.getElementById("quantTable");
  if (!target) return;

  const min = Math.min(...original);
  const max = Math.max(...original);
  const standardStep = (max - min) / 15;
  const turboMin = Math.min(...rotated);
  const turboMax = Math.max(...rotated);
  const turboStep = (turboMax - turboMin) / 15;

  target.innerHTML = original.map((value, index) => {
    const standardCode = Math.round((value - min) / standardStep);
    const standardDequant = min + standardCode * standardStep;
    const standardError = standardDequant - value;
    const turboCode = Math.round((rotated[index] - turboMin) / turboStep);
    const turboError = turboDequant[index] - value;

    return `
      <tr>
        <td><code>d${index + 1}</code><small>same vector slot</small></td>
        <td><code>${value.toFixed(3)}</code><small>FP32 original</small></td>
        <td><code>${standardStep.toFixed(3)}</code><small>standard min-max grid</small></td>
        <td><code>${standardCode.toString(2).padStart(4, "0")}</code><small>standard 4-bit bucket ${standardCode}</small></td>
        <td>
          <code>${standardDequant.toFixed(3)}</code>
          <small>standard dequant; original ${value.toFixed(3)}, <span class="error">error ${formatSigned(standardError)}</span></small>
        </td>
        <td><code>R(x)=${rotated[index].toFixed(2)}</code><small>Google TurboQuant rotates before coding</small></td>
        <td><code>${turboCode.toString(2).padStart(4, "0")}</code><small>TurboQuant 4-bit polar bucket + QJL sign help</small></td>
        <td>
          <code>${turboDequant[index].toFixed(3)}</code>
          <small>TurboQuant dequant; original ${value.toFixed(3)}, <span class="error turbo-error">error ${formatSigned(turboError)}</span></small>
        </td>
      </tr>
    `;
  }).join("");
}

function renderAndroidBench() {
  const target = document.getElementById("androidBenchTables");
  if (target) {
    target.innerHTML = androidBenchTables.map(table => `
      <article class="bench-table-wrap">
        <div class="panel-head">
          <h3>${table.title}</h3>
          <span>same-bit contract; 98% target</span>
        </div>
        <p class="section-note">${table.note}</p>
        <div class="table-scroll">
          <table class="bench-table">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Bits</th>
                <th>Self R@1</th>
                <th>Self R@10</th>
                <th>Random R@10</th>
                <th>ms/query</th>
                <th>ROM</th>
                <th>Vector staging (RAM only)</th>
                <th>Search/graph RAM</th>
              </tr>
            </thead>
            <tbody>${table.rows.map(row => `
              <tr>
                <td>${row.dataset}</td>
                <td>${row.bits}</td>
                <td>${row.selfR1}</td>
                <td>${row.selfR10}</td>
                <td>${row.randomR10}</td>
                <td>${row.msQuery.toFixed(3)}</td>
                <td>${row.rom}</td>
                <td>${row.vectorStaging}</td>
                <td>${row.vectorRam}</td>
              </tr>
            `).join("")}</tbody>
          </table>
        </div>
      </article>
    `).join("");
  }

  renderBenchBars("latencyBars", androidBenchRows, "msQuery", value => `${value.toFixed(3)} ms`, false);
  renderBenchBars("sizeBars", androidBenchRows, "romMb", value => `${value.toFixed(1)} MB`, true);
}

function renderBenchBars(targetId, rows, key, formatter, sizeMode) {
  const target = document.getElementById(targetId);
  if (!target) return;

  const max = Math.max(...rows.map(row => row[key]));
  target.innerHTML = rows.map(row => {
    const width = Math.max(2, row[key] / max * 100);
    const label = row.bits === 32 ? "FP32" : `${row.bits}-bit`;
    const cls = row.bits === 32 ? "fp32" : `b${row.bits}`;
    return `
      <div class="bench-bar ${cls}">
        <span>${label}</span>
        <div class="bench-bar-track">
          <i class="bench-bar-fill" style="--w:${width}%"></i>
        </div>
        <strong>${formatter(row[key])}</strong>
      </div>
    `;
  }).join("");
}

renderBars("heroBars", original);
renderBars("originalBars", original);
renderBars("beforeRotate", original);
renderBars("rotatedBars", rotated, 5.4);
renderPolar();
renderCacheStack();
renderIndexQuantDetails();
renderQuantTable();
renderAndroidBench();
