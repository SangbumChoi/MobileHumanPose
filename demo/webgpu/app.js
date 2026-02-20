/**
 * MobileHumanPose WebGPU demo: webcam -> person detection (ONNX) -> pose (ONNX) -> 2D overlay + Three.js 3D.
 * Serve this folder over HTTPS (or localhost). Models in ./models/ (pose.onnx, person_detector.onnx).
 */

const SKELETON = [
  [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13],
  [8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
];
const JOINT_NUM = 18;
const OUTPUT_H = 32;
const OUTPUT_W = 32;
const DEPTH_DIM = 32;
const BBOX_3D_HALF = 1000;
const PIXEL_MEAN = [0.485, 0.456, 0.406];
const PIXEL_STD = [0.229, 0.224, 0.225];
const DET_INPUT = 640;
const POSE_INPUT = 256;

let video = document.getElementById('video');
let canvas2d = document.getElementById('canvas2d');
let canvas3d = document.getElementById('canvas3d');
let ctx2d = canvas2d.getContext('2d');
let startBtn = document.getElementById('startBtn');
let stopBtn = document.getElementById('stopBtn');
let statusEl = document.getElementById('status');

let poseSession = null;
let detSession = null;
let threeScene = null;
let threeRenderer = null;
let threeCamera = null;
let skeletonLines = [];
let skeletonPoints = [];
let animationId = null;
let stream = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function getBaseUrl() {
  const path = new URL(import.meta.url).pathname;
  return path.replace(/\/[^/]+$/, '/');
}

async function loadModels() {
  const base = getBaseUrl();
  setStatus('Loading ONNX models…');
  const opts = { executionProviders: ['webgpu', 'webgl'] };
  try {
    poseSession = await ort.InferenceSession.create(base + 'models/pose.onnx', opts);
    detSession = await ort.InferenceSession.create(base + 'models/person_detector.onnx', opts);
  } catch (e) {
    setStatus('Model load failed (serve over HTTP and ensure models/ exists): ' + e.message);
    throw e;
  }
  setStatus('Models loaded. Start camera.');
}

function processBbox(x, y, w, h, imgW, imgH) {
  const pad = Math.max(w, h) * 0.25;
  const cx = x + w / 2;
  const cy = y + h / 2;
  const s = Math.max(w, h) + 2 * pad;
  let x1 = cx - s / 2;
  let y1 = cy - s / 2;
  x1 = Math.max(0, Math.min(imgW - s, x1));
  y1 = Math.max(0, Math.min(imgH - s, y1));
  return [x1, y1, s, s];
}

function cropAndResize(imageData, bbox, size) {
  const [bx, by, bw, bh] = bbox;
  const sw = imageData.width;
  const sh = imageData.height;
  const scaleX = bw / size;
  const scaleY = bh / size;
  const data = new Float32Array(3 * size * size);
  const r = new Uint8Array(size * size);
  const g = new Uint8Array(size * size);
  const b = new Uint8Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const sx = Math.min(sw - 1, Math.floor(bx + j * scaleX));
      const sy = Math.min(sh - 1, Math.floor(by + i * scaleY));
      const idx = (sy * sw + sx) * 4;
      r[i * size + j] = imageData.data[idx];
      g[i * size + j] = imageData.data[idx + 1];
      b[i * size + j] = imageData.data[idx + 2];
    }
  }
  for (let i = 0; i < size * size; i++) {
    data[i] = (r[i] / 255 - PIXEL_MEAN[0]) / PIXEL_STD[0];
    data[size * size + i] = (g[i] / 255 - PIXEL_MEAN[1]) / PIXEL_STD[1];
    data[2 * size * size + i] = (b[i] / 255 - PIXEL_MEAN[2]) / PIXEL_STD[2];
  }
  return data;
}

function runPose(imageData, bbox) {
  const [bx, by, bw, bh] = processBbox(bbox[0], bbox[1], bbox[2], bbox[3], imageData.width, imageData.height);
  const patch = cropAndResize(imageData, [bx, by, bw, bh], POSE_INPUT);
  const tensor = new ort.Tensor('float32', patch, [1, 3, POSE_INPUT, POSE_INPUT]);
  const out = poseSession.run({ input: tensor });
  const coord = out[poseSession.outputNames[0]];
  const data = coord.data;
  const focal = [1500, 1500];
  const princpt = [imageData.width / 2, imageData.height / 2];
  const rootDepth = 1500;
  const pose2d = [];
  const pose3d = [];
  for (let j = 0; j < JOINT_NUM; j++) {
    const ox = data[j * 3];
    const oy = data[j * 3 + 1];
    const oz = data[j * 3 + 2];
    const px = (ox / OUTPUT_W) * bw + bx;
    const py = (oy / OUTPUT_H) * bh + by;
    const depth = ((oz / DEPTH_DIM) * 2 - 1) * BBOX_3D_HALF + rootDepth;
    pose2d.push([px, py, depth]);
    const camX = ((px - princpt[0]) / focal[0]) * depth;
    const camY = ((py - princpt[1]) / focal[1]) * depth;
    pose3d.push([camX, camY, depth]);
  }
  return { pose2d, pose3d };
}

function runDetection(imageData) {
  const w = imageData.width;
  const h = imageData.height;
  const scale = Math.min(DET_INPUT / w, DET_INPUT / h);
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);
  const padW = (DET_INPUT - nw) / 2;
  const padH = (DET_INPUT - nh) / 2;
  const inputData = new Float32Array(3 * DET_INPUT * DET_INPUT);
  for (let i = 0; i < DET_INPUT; i++) {
    for (let j = 0; j < DET_INPUT; j++) {
      const sy = Math.min(h - 1, Math.floor((i - padH) / scale));
      const sx = Math.min(w - 1, Math.floor((j - padW) / scale));
      const idx = (sy * w + sx) * 4;
      const r = imageData.data[idx] / 255;
      const g = imageData.data[idx + 1] / 255;
      const b = imageData.data[idx + 2] / 255;
      inputData[i * DET_INPUT + j] = r;
      inputData[DET_INPUT * DET_INPUT + i * DET_INPUT + j] = g;
      inputData[2 * DET_INPUT * DET_INPUT + i * DET_INPUT + j] = b;
    }
  }
  const inpName = detSession.inputNames[0];
  const tensor = new ort.Tensor('float32', inputData, [1, 3, DET_INPUT, DET_INPUT]);
  const out = detSession.run({ [inpName]: tensor });
  const outKey = detSession.outputNames[0];
  const raw = out[outKey];
  return postprocessYolo(raw, scale, padW, padH, w, h);
}

function postprocessYolo(raw, scale, padW, padH, imgW, imgH) {
  const shape = raw.dims || [];
  let data = raw.data;
  if (shape.length === 3 && shape[0] === 1) {
    const [, C, N] = shape;
    const transposed = new Float32Array(N * C);
    for (let i = 0; i < N; i++)
      for (let c = 0; c < C; c++) transposed[i * C + c] = data[c * N + i];
    data = transposed;
  }
  const numProposals = data.length / 84;
  const boxes = [];
  const confThres = 0.5;
  for (let i = 0; i < numProposals; i++) {
    const base = i * 84;
    const cx = data[base];
    const cy = data[base + 1];
    const bw = data[base + 2];
    const bh = data[base + 3];
    let maxScore = 0;
    let maxCls = 0;
    for (let c = 0; c < 80; c++) {
      const s = data[base + 4 + c];
      if (s > maxScore) { maxScore = s; maxCls = c; }
    }
    if (maxCls !== 0 || maxScore < confThres) continue;
    const x1 = (cx - bw / 2 - padW) / scale;
    const y1 = (cy - bh / 2 - padH) / scale;
    boxes.push([Math.max(0, x1), Math.max(0, y1), bw / scale, bh / scale]);
  }
  return boxes.slice(0, 4);
}

function draw2DSkeleton(pose2d) {
  ctx2d.strokeStyle = '#00ff00';
  ctx2d.lineWidth = 2;
  for (const [i1, i2] of SKELETON) {
    const a = pose2d[i1];
    const b = pose2d[i2];
    if (!a || !b) continue;
    ctx2d.beginPath();
    ctx2d.moveTo(a[0], a[1]);
    ctx2d.lineTo(b[0], b[1]);
    ctx2d.stroke();
  }
  ctx2d.fillStyle = '#00ff00';
  for (const p of pose2d) {
    ctx2d.beginPath();
    ctx2d.arc(p[0], p[1], 3, 0, Math.PI * 2);
    ctx2d.fill();
  }
}

function initThree() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);
  const camera = new THREE.PerspectiveCamera(50, 1, 100, 10000);
  camera.position.set(800, -600, 1200);
  camera.lookAt(0, 0, 0);
  const renderer = new THREE.WebGLRenderer({ canvas: canvas3d, alpha: true });
  renderer.setSize(400, 400);
  renderer.setPixelRatio(window.devicePixelRatio);

  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x00ccff });
  const pointMaterial = new THREE.PointsMaterial({ color: 0x00ccff, size: 8 });
  const lines = [];
  const points = [];
  for (let i = 0; i < 4; i++) {
    const lineGeo = new THREE.BufferGeometry();
    const line = new THREE.LineSegments(lineGeo, lineMaterial);
    scene.add(line);
    lines.push(line);
    const ptGeo = new THREE.BufferGeometry();
    const pt = new THREE.Points(ptGeo, pointMaterial);
    scene.add(pt);
    points.push(pt);
  }

  threeScene = scene;
  threeRenderer = renderer;
  threeCamera = camera;
  skeletonLines = lines;
  skeletonPoints = points;
}

function updateThree(allPose3d) {
  const colors = [0x00ccff, 0xffcc00, 0xff00cc, 0x00ffcc];
  for (let p = 0; p < skeletonLines.length; p++) {
    const pose3d = allPose3d[p] || [];
    const line = skeletonLines[p];
    const pt = skeletonPoints[p];
    line.visible = pose3d.length === JOINT_NUM;
    pt.visible = pose3d.length === JOINT_NUM;
    if (pose3d.length !== JOINT_NUM) continue;

    const vertices = [];
    for (const [i1, i2] of SKELETON) {
      const a = pose3d[i1];
      const b = pose3d[i2];
      vertices.push(a[0], a[2], -a[1], b[0], b[2], -b[1]);
    }
    line.geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    line.geometry.attributes.position.needsUpdate = true;

    const pos = [];
    for (const a of pose3d) pos.push(a[0], a[2], -a[1]);
    pt.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    pt.geometry.attributes.position.needsUpdate = true;
  }
  threeRenderer.render(threeScene, threeCamera);
}

async function tick() {
  if (!video.videoWidth || !poseSession || !detSession) return;
  canvas2d.width = video.videoWidth;
  canvas2d.height = video.videoHeight;
  ctx2d.drawImage(video, 0, 0);
  const imageData = ctx2d.getImageData(0, 0, canvas2d.width, canvas2d.height);

  let bboxes = [];
  try {
    bboxes = runDetection(imageData);
  } catch (e) {
    console.warn('Detection:', e);
  }
  if (bboxes.length === 0) bboxes = [[0, 0, canvas2d.width, canvas2d.height]];

  const allPose3d = [];
  for (const bbox of bboxes.slice(0, 4)) {
    try {
      const { pose2d, pose3d } = runPose(imageData, bbox);
      draw2DSkeleton(pose2d);
      allPose3d.push(pose3d);
    } catch (e) {
      console.warn('Pose:', e);
      allPose3d.push([]);
    }
  }
  if (threeScene) updateThree(allPose3d);
  animationId = requestAnimationFrame(tick);
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus('Running…');
    if (!threeScene) initThree();
    tick();
  } catch (e) {
    setStatus('Camera error: ' + e.message);
  }
}

function stopCamera() {
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus('Stopped.');
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);

(async function () {
  await loadModels();
})();
