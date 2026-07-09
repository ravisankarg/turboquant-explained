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
const androidBenchRows = [
  {
    dataset: "Cohere 50K",
    vectors: "50K",
    method: "exact fp32",
    bits: 32,
    selfR1: "100.00%",
    selfR10: "100.00%",
    randomR1: "100.00%",
    randomR10: "100.00%",
    indexMs: 0.0,
    prepMs: 0.0,
    writeMs: 0.0,
    selfMs: 4155.7,
    randomMs: 4155.7,
    usQuery: 4155.7,
    rom: "146.5 MB",
    romMb: 146.5,
    ramDelta: "146.5 MB"
  },
  {
    dataset: "Cohere 50K",
    vectors: "50K",
    method: "turbovec",
    bits: 8,
    selfR1: "100.00%",
    selfR10: "100.00%",
    randomR1: "100.00%",
    randomR10: "100.00%",
    indexMs: 1293.7,
    prepMs: 259.4,
    writeMs: 17.2,
    selfMs: 4886.6,
    randomMs: 4890.9,
    usQuery: 4888.8,
    rom: "36.8 MB",
    romMb: 36.8,
    ramDelta: "124.4 MB"
  },
  {
    dataset: "Cohere 50K",
    vectors: "50K",
    method: "turbovec",
    bits: 4,
    selfR1: "100.00%",
    selfR10: "100.00%",
    randomR1: "99.20%",
    randomR10: "100.00%",
    indexMs: 994.9,
    prepMs: 147.4,
    writeMs: 8.1,
    selfMs: 143.6,
    randomMs: 149.8,
    usQuery: 146.7,
    rom: "18.5 MB",
    romMb: 18.5,
    ramDelta: "78.5 MB"
  },
  {
    dataset: "Cohere 50K",
    vectors: "50K",
    method: "turbovec",
    bits: 3,
    selfR1: "100.00%",
    selfR10: "100.00%",
    randomR1: "99.20%",
    randomR10: "100.00%",
    indexMs: 977.0,
    prepMs: 74.0,
    writeMs: 6.0,
    selfMs: 148.0,
    randomMs: 153.0,
    usQuery: 150.5,
    rom: "13.9 MB",
    romMb: 13.9,
    ramDelta: "34.4 MB"
  },
  {
    dataset: "Cohere 50K",
    vectors: "50K",
    method: "turbovec",
    bits: 2,
    selfR1: "99.60%",
    selfR10: "100.00%",
    randomR1: "89.70%",
    randomR10: "99.80%",
    indexMs: 927.1,
    prepMs: 79.5,
    writeMs: 4.3,
    selfMs: 83.2,
    randomMs: 80.4,
    usQuery: 81.8,
    rom: "9.4 MB",
    romMb: 9.4,
    ramDelta: "37.6 MB"
  }
];

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
  const table = document.getElementById("androidKpiTable");
  if (table) {
    table.innerHTML = androidBenchRows.map(row => `
      <tr>
        <td class="method">${row.method}</td>
        <td>${row.dataset}</td>
        <td>${row.vectors}</td>
        <td>${row.bits}</td>
        <td>${row.selfR1}</td>
        <td>${row.selfR10}</td>
        <td>${row.randomR1}</td>
        <td>${row.randomR10}</td>
        <td>${row.indexMs.toFixed(1)}</td>
        <td>${row.prepMs.toFixed(1)}</td>
        <td>${row.writeMs.toFixed(1)}</td>
        <td>${row.selfMs.toFixed(1)}</td>
        <td>${row.randomMs.toFixed(1)}</td>
        <td>${row.usQuery.toFixed(1)}</td>
        <td>${row.rom}</td>
        <td>${row.ramDelta}</td>
      </tr>
    `).join("");
  }

  renderBenchBars("latencyBars", androidBenchRows, "usQuery", value => `${value.toFixed(1)} us`, false);
  renderBenchBars("sizeBars", androidBenchRows, "romMb", value => `${value.toFixed(1)} MB`, true);
}

function renderBenchBars(targetId, rows, key, formatter, sizeMode) {
  const target = document.getElementById(targetId);
  if (!target) return;

  const max = Math.max(...rows.map(row => row[key]));
  target.innerHTML = rows.map(row => {
    const width = Math.max(2, row[key] / max * 100);
    const label = row.method === "exact fp32" ? "FP32" : `${row.bits}-bit`;
    const cls = row.method === "exact fp32" ? "fp32" : `b${row.bits}`;
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
