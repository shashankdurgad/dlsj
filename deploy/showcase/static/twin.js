// Live 3D digital twin of the SO-101 (three.js + urdf-loader).
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import URDFLoader from './vendor/URDFLoader.js';

// ---- calibration between our measured degrees and the URDF zero --------------------------
// rad = SIGN * (deg + OFFSET_DEG) * pi/180. Tweak here if the twin's pose disagrees with the arm.
export const ARM_JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'];
export const JOINT_SIGN = { shoulder_pan: 1, shoulder_lift: 1, elbow_flex: 1, wrist_flex: 1, wrist_roll: 1 };
export const JOINT_OFFSET_DEG = { shoulder_pan: 0, shoulder_lift: 0, elbow_flex: 0, wrist_flex: 0, wrist_roll: 0 };
// gripper: measured 0..100 maps linearly onto the URDF limit [-0.1745, 1.745] rad
export const GRIPPER_RANGE = [-0.174533, 1.74533];

// Scene layout (metres, three.js Y-up, robot base at the origin, robot forward = +X, robot's right = +Z).
// Top-camera pixels map onto the table with a rough planar fit: the base sits near px (365, 40),
// image-down is robot-forward, image-left is robot-right, ~1 mm per px.
const TOPCAM = { baseX: 365, baseY: 40, mPerPx: 0.001 };
const pxToTable = (px, py) => [(py - TOPCAM.baseY) * TOPCAM.mPerPx, (TOPCAM.baseX - px) * TOPCAM.mPerPx]; // [x, z]
const PLATE_POS = { yellow: pxToTable(86, 159), purple: pxToTable(110, 301) }; // fallback when perception has no plates
const COLORS = {
  printed: 0xe9eef0, motor: 0x2b3238, table: 0x22394a,
  yellow: 0xe9b923, purple: 0x7646c6, unknownBrick: 0xcfd8dc,
};

const reduced = matchMedia('(prefers-reduced-motion: reduce)');

export function createTwin(canvas, { onReady, onError } = {}) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'low-power' });
  } catch (e) { onError && onError(e); return null; }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.75));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.01, 20);
  camera.position.set(0.3, 0.52, -0.74);

  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0.2, 0.1, 0.06);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.3;
  controls.maxDistance = 2.0;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.enablePan = false;
  controls.enableZoom = false; // keep the mouse wheel for scrolling the page
  controls.update();

  // soft studio lighting
  scene.add(new THREE.HemisphereLight(0xdde9f2, 0x1b2a35, 1.25));
  const key = new THREE.DirectionalLight(0xffffff, 2.1);
  key.position.set(0.5, 1.1, 0.6);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.left = -0.5; key.shadow.camera.right = 0.5;
  key.shadow.camera.top = 0.5; key.shadow.camera.bottom = -0.5;
  key.shadow.camera.near = 0.1; key.shadow.camera.far = 3;
  key.shadow.radius = 6;
  key.shadow.bias = -0.0005;
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x9cc4ff, 0.9);
  rim.position.set(-0.7, 0.5, -0.6);
  scene.add(rim);

  // table: a soft disc with a faint ring
  const table = new THREE.Mesh(
    new THREE.CircleGeometry(0.62, 96),
    new THREE.MeshStandardMaterial({ color: COLORS.table, roughness: 0.92, metalness: 0 }),
  );
  table.rotation.x = -Math.PI / 2;
  table.receiveShadow = true;
  scene.add(table);
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.42, 0.423, 128),
    new THREE.MeshBasicMaterial({ color: 0x4d6a7d, transparent: true, opacity: 0.6 }),
  );
  ring.rotation.x = -Math.PI / 2; ring.position.y = 0.0005;
  scene.add(ring);

  // plates
  const plates = {};
  for (const [name, [x, z]] of Object.entries(PLATE_POS)) {
    const g = new THREE.Group();
    const disc = new THREE.Mesh(
      new THREE.CylinderGeometry(1, 1.03, 0.008, 64),
      new THREE.MeshStandardMaterial({ color: COLORS[name], roughness: 0.55 }),
    );
    disc.position.y = 0.004; disc.castShadow = true; disc.receiveShadow = true;
    g.add(disc);
    const glow = new THREE.Mesh(
      new THREE.RingGeometry(1.1, 1.25, 64),
      new THREE.MeshBasicMaterial({ color: COLORS[name], transparent: true, opacity: 0 }),
    );
    glow.rotation.x = -Math.PI / 2; glow.position.y = 0.001;
    g.add(glow);
    g.position.set(x, 0, z);
    disc.scale.set(0.07, 1, 0.07); glow.scale.set(0.07, 0.07, 1);
    g.userData = { glow, disc };
    scene.add(g);
    plates[name] = g;
  }

  // the fixed inspection camera on its post, looking back at the inspect pose
  const camRig = new THREE.Group();
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x0f171d, roughness: 0.4, metalness: 0.3 });
  const post = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.008, 0.08, 16), darkMat);
  post.position.y = 0.04; post.castShadow = true; camRig.add(post);
  const camBody = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.022, 0.05), darkMat);
  camBody.position.y = 0.09; camBody.castShadow = true; camRig.add(camBody);
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.006, 20), new THREE.MeshStandardMaterial({ color: 0x0e9a80, emissive: 0x0e9a80, emissiveIntensity: 0.6 }));
  lens.rotation.z = Math.PI / 2; lens.position.set(-0.017, 0.09, 0); camRig.add(lens);
  camRig.position.set(0.46, 0, -0.03);
  scene.add(camRig);

  // the held brick: a 2x2 Duplo-ish block with studs
  const brickMat = new THREE.MeshStandardMaterial({ color: COLORS.unknownBrick, roughness: 0.45 });
  const brick = new THREE.Group();
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.032, 0.024, 0.032), brickMat);
  body.castShadow = true;
  brick.add(body);
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const stud = new THREE.Mesh(new THREE.CylinderGeometry(0.0048, 0.0048, 0.005, 20), brickMat);
    stud.position.set(sx * 0.008, 0.0145, sz * 0.008);
    brick.add(stud);
  }
  brick.visible = false;

  // bricks the top camera sees on the table (and already sorted onto plates)
  const tableBricks = new THREE.Group();
  scene.add(tableBricks);
  const brickGeo = new THREE.BoxGeometry(0.03, 0.022, 0.03);
  const BRICK_HEX = { blue: 0x3d7be0, brown: 0x8a5a3c, red: 0xd0453a, white: 0xf2f2f2, yellow: 0xe9b923, green: 0x3aa25a, orange: 0xe8862a };
  const brickMats = {};
  const matFor = (c) => (brickMats[c] ||= new THREE.MeshStandardMaterial({ color: BRICK_HEX[c] ?? 0xb8c4ca, roughness: 0.45 }));
  function setPerception(p) {
    if (!p) return;
    for (const pl of p.plates || []) {
      const g = plates[pl.colour]; if (!g) continue;
      const [x, z] = pxToTable(pl.cx, pl.cy);
      g.position.set(x, 0, z);
      const r = Math.max(0.04, Math.min(0.1, pl.r * TOPCAM.mPerPx * 0.85));
      g.userData.disc.scale.set(r, 1, r); g.userData.glow.scale.set(r, r, 1);
    }
    const list = [...(p.table || []), ...(p.sorted || [])];
    while (tableBricks.children.length > list.length) tableBricks.remove(tableBricks.children.at(-1));
    list.forEach((b, i) => {
      let m = tableBricks.children[i];
      if (!m) { m = new THREE.Mesh(brickGeo, matFor(b.colour)); m.castShadow = true; tableBricks.add(m); }
      m.material = matFor(b.colour);
      const [x0, y0, x1, y1] = b.box;
      const [x, z] = pxToTable((x0 + x1) / 2, (y0 + y1) / 2);
      m.position.set(x, b.plate ? 0.019 : 0.011, z);
    });
  }

  // ---- robot ----
  const root = new THREE.Group();
  root.rotation.x = -Math.PI / 2; // URDF is Z-up
  scene.add(root);
  let robot = null;
  const cur = {}; const tgt = {};
  const HOME_DEG = [15, -102, 97, 8, 4, 1]; // rest pose shown until the arm reports angles
  const manager = new THREE.LoadingManager();
  const loader = new URDFLoader(manager);
  // meshes arrive asynchronously: restyle once every STL has loaded
  manager.onLoad = () => { if (robot) restyle(robot); };
  function restyle(r) {
    r.traverse((o) => {
      if (!o.isMesh) return;
      o.castShadow = true; o.receiveShadow = true;
      const src = o.material && o.material.color;
      const dark = src && (src.r + src.g + src.b) < 0.9;
      o.material = new THREE.MeshStandardMaterial({
        color: dark ? COLORS.motor : COLORS.printed,
        roughness: dark ? 0.5 : 0.62, metalness: dark ? 0.15 : 0.0,
      });
    });
  }
  loader.load('./robot/so101_new_calib.urdf', (r) => {
    robot = r;
    root.add(r);
    const gf = r.links['gripper_frame_link'] || r.links['gripper_link'];
    if (gf) { gf.add(brick); brick.position.set(0, 0, -0.006); }
    if (!Object.keys(tgt).length) setMeasured(HOME_DEG);
    for (const n of [...ARM_JOINTS, 'gripper']) { cur[n] = tgt[n] ?? 0; if (r.joints[n]) r.joints[n].setJointValue(cur[n]); }
    onReady && onReady();
  }, undefined, (err) => onError && onError(err));

  function setMeasured(m) {
    if (!Array.isArray(m) || m.length < 6) return;
    ARM_JOINTS.forEach((n, i) => {
      if (typeof m[i] === 'number') tgt[n] = JOINT_SIGN[n] * (m[i] + JOINT_OFFSET_DEG[n]) * Math.PI / 180;
    });
    if (typeof m[5] === 'number') {
      const t = Math.max(0, Math.min(100, m[5])) / 100;
      tgt.gripper = GRIPPER_RANGE[0] + t * (GRIPPER_RANGE[1] - GRIPPER_RANGE[0]);
    }
  }

  let activePlate = null, inspecting = false;
  function setScene({ holding, verdict, targetPlate, spinning, brickColour }) {
    inspecting = !!spinning;
    brick.visible = !!holding;
    brickMat.color.setHex(verdict === 'plain' ? COLORS.yellow : (verdict === 'marked' || verdict === 'unknown') ? COLORS.purple : (BRICK_HEX[brickColour] ?? 0xff8a3d));
    activePlate = targetPlate || null;
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // pull back a little on narrow stages so the whole arm stays in frame
    camera.fov = w / h < 1.1 ? 42 : 32;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  let visible = true;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(canvas);

  let last = performance.now();
  function frame(now) {
    requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (!visible || document.hidden) return;
    if (robot) {
      const k = reduced.matches ? 1 : 1 - Math.exp(-dt * 9);
      for (const n of Object.keys(tgt)) {
        cur[n] = (cur[n] ?? tgt[n]) + (tgt[n] - (cur[n] ?? tgt[n])) * k;
        if (robot.joints[n]) robot.joints[n].setJointValue(cur[n]);
      }
    }
    lens.material.emissiveIntensity = inspecting ? (reduced.matches ? 1.4 : 1 + 0.6 * Math.sin(now / 120)) : 0.4;
    for (const [name, g] of Object.entries(plates)) {
      const on = activePlate === name;
      const glow = g.userData.glow.material;
      const want = on ? (reduced.matches ? 0.8 : 0.55 + 0.35 * Math.sin(now / 300)) : 0;
      glow.opacity += (want - glow.opacity) * 0.15;
    }
    controls.update();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);

  return { setMeasured, setScene, setPerception, get robot() { return robot; }, camera, controls, plates };
}
