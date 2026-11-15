/* One SO-101 in cool white. The median gesture is echoed as a few ghosts,
 * the way a chronophotograph stacks one motion on a dark plate. */
import * as THREE from "three";

const canvas = document.getElementById("arm");
if (canvas) {

  const DEG = Math.PI / 180;
  const QINDEX = {shoulder_pan: 0, shoulder_lift: 1, elbow_flex: 2, wrist_flex: 3, wrist_roll: 4, gripper: 5};
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(28, 1, 0.02, 8);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({canvas: canvas, antialias: true});
  renderer.setClearColor(0x1b1e25, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

  scene.add(new THREE.HemisphereLight(0xf4f1ea, 0x1a1c22, 0.85));
  const key = new THREE.DirectionalLight(0xffffff, 1.15);
  key.position.set(0.35, -0.4, 0.95);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  key.shadow.camera.near = 0.05;
  key.shadow.camera.far = 2.4;
  key.shadow.camera.left = key.shadow.camera.bottom = -0.55;
  key.shadow.camera.right = key.shadow.camera.top = 0.55;
  key.shadow.bias = -0.0004;
  scene.add(key);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(4, 4),
    new THREE.ShadowMaterial({opacity: 0.22})
  );
  ground.position.z = -0.004;
  ground.receiveShadow = true;
  scene.add(ground);

  const scratchA = new THREE.Matrix4();
  const scratchB = new THREE.Matrix4();
  const scratchC = new THREE.Matrix4();

  function setIdent(e) {
    e[0] = 1; e[1] = 0; e[2] = 0; e[3] = 0;
    e[4] = 0; e[5] = 1; e[6] = 0; e[7] = 0;
    e[8] = 0; e[9] = 0; e[10] = 1; e[11] = 0;
    e[12] = 0; e[13] = 0; e[14] = 0; e[15] = 1;
  }

  function setMul(out, a, b) {
    for (let col = 0; col < 4; col++) {
      for (let row = 0; row < 4; row++) {
        out[col * 4 + row] =
          a[row] * b[col * 4] +
          a[4 + row] * b[col * 4 + 1] +
          a[8 + row] * b[col * 4 + 2] +
          a[12 + row] * b[col * 4 + 3];
      }
    }
  }

  function setOrigin(m, xyz, rpy) {
    const cx = Math.cos(rpy[0]), sx = Math.sin(rpy[0]);
    const cy = Math.cos(rpy[1]), sy = Math.sin(rpy[1]);
    const cz = Math.cos(rpy[2]), sz = Math.sin(rpy[2]);
    const e = m.elements;
    e[0] = cy * cz; e[1] = cy * sz; e[2] = -sy; e[3] = 0;
    e[4] = sx * sy * cz - cx * sz; e[5] = sx * sy * sz + cx * cz; e[6] = sx * cy; e[7] = 0;
    e[8] = cx * sy * cz + sx * sz; e[9] = cx * sy * sz - sx * cz; e[10] = cx * cy; e[11] = 0;
    e[12] = xyz[0]; e[13] = xyz[1]; e[14] = xyz[2]; e[15] = 1;
  }

  function setAxis(m, axis, angle) {
    const x = axis[0], y = axis[1], z = axis[2];
    const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
    const e = m.elements;
    e[0] = c + x * x * t; e[1] = y * x * t + z * s; e[2] = z * x * t - y * s; e[3] = 0;
    e[4] = x * y * t - z * s; e[5] = c + y * y * t; e[6] = z * y * t + x * s; e[7] = 0;
    e[8] = x * z * t + y * s; e[9] = y * z * t - x * s; e[10] = c + z * z * t; e[11] = 0;
    e[12] = 0; e[13] = 0; e[14] = 0; e[15] = 1;
  }

  function fillLinks(q, dest, joints) {
    setIdent(dest.base_link.elements);
    for (let i = 0; i < joints.length; i++) {
      const j = joints[i];
      setOrigin(scratchA, j.xyz, j.rpy);
      setAxis(scratchB, j.axis, (q[QINDEX[j.name]] || 0) * DEG);
      setMul(scratchC.elements, scratchA.elements, scratchB.elements);
      setMul(dest[j.child].elements, dest[j.parent].elements, scratchC.elements);
    }
  }

  function point(m, p) {
    const e = m.elements;
    return [
      e[0] * p[0] + e[4] * p[1] + e[8] * p[2] + e[12],
      e[1] * p[0] + e[5] * p[1] + e[9] * p[2] + e[13],
      e[2] * p[0] + e[6] * p[1] + e[10] * p[2] + e[14],
    ];
  }

  function sampleInto(frames, u, out) {
    const n = frames.length;
    const x = Math.min(0.9999, Math.max(0, u)) * (n - 1);
    const i = Math.min(n - 2, Math.floor(x));
    const f = x - i;
    const p0 = frames[Math.max(0, i - 1)];
    const p1 = frames[i];
    const p2 = frames[i + 1];
    const p3 = frames[Math.min(n - 1, i + 2)];
    for (let k = 0; k < out.length; k++) {
      const a = 2 * p1[k];
      const b = p2[k] - p0[k];
      const c = 2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k];
      const d = -p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k];
      out[k] = 0.5 * (((d * f + c) * f + b) * f + a);
    }
    return out;
  }

  function gesture(now) {
    const trip = 12000;
    const x = ((now / trip) % 2 + 2) % 2;
    const fold = x < 1 ? x : 2 - x;
    return fold * fold * (3 - 2 * fold);
  }

  function medianOf(traces) {
    const steps = traces[0].q.length;
    const joints = traces[0].q[0].length;
    const out = [];
    for (let t = 0; t < steps; t++) {
      const frame = [];
      for (let j = 0; j < joints; j++) {
        const col = traces.map(function (tr) { return tr.q[t][j]; });
        col.sort(function (a, b) { return a - b; });
        frame.push(col[col.length >> 1]);
      }
      out.push(frame);
    }
    return out;
  }

  function parseSTL(buf) {
    const view = new DataView(buf);
    const n = view.getUint32(80, true);
    const pos = new Float32Array(n * 9);
    let offset = 84;
    for (let i = 0; i < n; i++) {
      offset += 12;
      for (let k = 0; k < 9; k++) {
        pos[i * 9 + k] = view.getFloat32(offset, true);
        offset += 4;
      }
      offset += 2;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.computeVertexNormals();
    return geo;
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (w < 2 || h < 2) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  const ECHO = [0.22, 0.14, 0.09, 0.05];

  function bodyMaterial(kind, opacity) {
    const plastic = kind !== "motor";
    const ghost = opacity < 1;
    return new THREE.MeshStandardMaterial({
      color: plastic ? 0xe7e9ee : 0x2a2d33,
      roughness: plastic ? 0.42 : 0.5,
      metalness: plastic ? 0.04 : 0.25,
      transparent: ghost,
      opacity: opacity,
      depthWrite: !ghost,
    });
  }

  let kin = null;
  let median = null;
  let layers = [];
  let linkNames = [];
  const qnow = [0, 0, 0, 0, 0, 0];
  const ready = Promise.all([
    fetch("/so101/kin.json").then(function (r) { return r.json(); }),
    fetch("/api/arm").then(function (r) { return r.json(); }),
  ]).then(function (pair) {
    kin = pair[0];
    const traces = (pair[1].traces || []).filter(function (tr) { return tr.q && tr.q.length > 1; });
    if (!traces.length || !kin.visuals) return;
    median = medianOf(traces);
    linkNames = ["base_link"].concat(kin.joints.map(function (j) { return j.child; }));
    const note = document.getElementById("why");
    if (note) note.textContent = "One gesture, echoed.";
    const opacities = [1].concat(ECHO);
    return Promise.all(kin.visuals.map(function (vis) {
      return fetch("/so101/" + vis.file).then(function (r) { return r.arrayBuffer(); }).then(function (buf) {
        const geo = parseSTL(buf);
        opacities.forEach(function (opacity, layer) {
          const mesh = new THREE.Mesh(geo, bodyMaterial(vis.material, opacity));
          mesh.matrixAutoUpdate = false;
          mesh.frustumCulled = false;
          mesh.castShadow = opacity === 1;
          mesh.renderOrder = opacity === 1 ? 2 : 0;
          scene.add(mesh);
          (layers[layer] = layers[layer] || []).push({link: vis.link, mesh: mesh});
        });
      });
    }));
  });

  const pose = {};
  let posed = false;

  function applyLayer(layer) {
    const list = layers[layer];
    for (let i = 0; i < list.length; i++) {
      list[i].mesh.matrix.copy(pose[list[i].link]);
      list[i].mesh.matrixWorldNeedsUpdate = true;
    }
  }

  function frame(now) {
    requestAnimationFrame(frame);
    resize();
    camera.position.set(0.96, -0.96, 0.54);
    camera.lookAt(0.08, 0, 0.12);
    if (kin && median && layers.length) {
      if (!posed) {
        linkNames.forEach(function (name) { pose[name] = new THREE.Matrix4(); });
        posed = true;
      }
      fillLinks(sampleInto(median, gesture(now), qnow), pose, kin.joints);
      applyLayer(0);
      for (let echo = 0; echo < ECHO.length; echo++) {
        fillLinks(sampleInto(median, gesture(now - (echo + 1) * 280), qnow), pose, kin.joints);
        applyLayer(echo + 1);
      }
    }
    renderer.render(scene, camera);
  }

  ready.catch(function () {});
  requestAnimationFrame(frame);
  addEventListener("resize", resize);
}
