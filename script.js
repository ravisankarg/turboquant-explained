const original = [0.12, -0.08, 0.15, 9.8, -0.11, 0.06, -10.2, 0.18];
const rotated = [4.71, -5.02, 3.88, -4.54, 5.31, -4.26, 4.63, -4.78];
const pairs = [
  { cart: [0.12, -0.08], radius: 0.144, angle: -33.7 },
  { cart: [0.15, 9.8], radius: 9.801, angle: 89.1 },
  { cart: [-0.11, 0.06], radius: 0.125, angle: 151.4 },
  { cart: [-10.2, 0.18], radius: 10.202, angle: 179.0 }
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

renderBars("heroBars", original);
renderBars("originalBars", original);
renderBars("beforeRotate", original);
renderBars("rotatedBars", rotated, 5.4);
renderPolar();
renderCacheStack();
