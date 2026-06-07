// MobileHumanPose in-browser demo (ONNX Runtime Web).
// Pipeline: center-crop -> 256x256 -> normalize -> LpNet ONNX -> soft-argmax
// coords (already inside the graph) -> draw COCO-17 skeleton.

const INPUT = 256, GRID = 32;                 // model input / output-grid size
const NJOINTS = 19;                           // 17 COCO + Thorax + Pelvis
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// 0-indexed skeleton edges (repo MSCOCO 19-joint convention).
const SKELETON = [
  [1,2],[0,1],[0,2],[2,4],[1,3],[6,8],[8,10],[5,7],[7,9],
  [12,14],[14,16],[11,13],[13,15],[5,6],[11,12],[17,5],[17,6],[18,11],[18,12]
];

const statusEl = document.getElementById("status");
const runBtn = document.getElementById("run");
const fileEl = document.getElementById("file");
const inCanvas = document.getElementById("in");
const outCanvas = document.getElementById("out");
let session = null;
let haveImage = false;

function setStatus(msg) { statusEl.textContent = msg; }

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

(async () => {
  try {
    session = await ort.InferenceSession.create("pose_model.onnx",
      { executionProviders: ["wasm"] });
    setStatus("Model loaded. Choose an image.");
  } catch (e) {
    setStatus("Failed to load model: " + e);
  }
})();

// Draw an uploaded image center-cropped into the 256x256 input canvas.
fileEl.addEventListener("change", (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const img = new Image();
  img.onload = () => {
    const side = Math.min(img.width, img.height);
    const sx = (img.width - side) / 2, sy = (img.height - side) / 2;
    const ctx = inCanvas.getContext("2d");
    ctx.clearRect(0, 0, INPUT, INPUT);
    ctx.drawImage(img, sx, sy, side, side, 0, 0, INPUT, INPUT);
    outCanvas.getContext("2d").drawImage(inCanvas, 0, 0);
    haveImage = true;
    runBtn.disabled = session === null;
    setStatus("Image ready. Click “Estimate pose”.");
  };
  img.src = URL.createObjectURL(f);
});

function preprocess() {
  const { data } = inCanvas.getContext("2d").getImageData(0, 0, INPUT, INPUT);
  const f = new Float32Array(3 * INPUT * INPUT);
  const plane = INPUT * INPUT;
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      f[c * plane + i] = ((data[i * 4 + c] / 255) - MEAN[c]) / STD[c];
    }
  }
  return new ort.Tensor("float32", f, [1, 3, INPUT, INPUT]);
}

function drawPose(coords) {
  const ctx = outCanvas.getContext("2d");
  ctx.drawImage(inCanvas, 0, 0);
  const s = INPUT / GRID;                       // output grid -> input pixels
  const pts = [];
  for (let j = 0; j < NJOINTS; j++) pts.push([coords[j * 3] * s, coords[j * 3 + 1] * s]);
  ctx.lineWidth = 3; ctx.strokeStyle = "#00ff66";
  for (const [a, b] of SKELETON) {
    ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke();
  }
  ctx.fillStyle = "#ff3355";
  for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fill(); }
}

runBtn.addEventListener("click", async () => {
  if (!session || !haveImage) return;
  setStatus("Running inference…");
  try {
    const t0 = performance.now();
    const out = await session.run({ input: preprocess() });
    const coords = out.coords.data;             // length 17*3, output-grid units
    drawPose(coords);
    setStatus("Done in " + (performance.now() - t0).toFixed(0) + " ms.");
  } catch (e) {
    setStatus("Inference error: " + e);
  }
});
