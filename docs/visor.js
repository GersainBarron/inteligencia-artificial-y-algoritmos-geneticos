// evocarros · visor 3D de repeticiones (esquema evocarros.repeticion/1).
// Simulación en el plano (x, y) → escena Three.js (x, altura, -y): igual que el exportador glTF de Blender.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const aEscena = (x, y, h = 0) => new THREE.Vector3(x, h, -y);
const MODELO_PISTA = { mexico: "hermanos_rodriguez.glb", spa: "spa.glb", monza: "monza.glb", hungaroring: "hungaroring.glb" };
const ETIQ_ENT = ["rayo 1", "rayo 2", "rayo 3", "rayo 4", "rayo 5", "velocidad", "áng. CP"];
const ETIQ_SAL = ["volante", "pedal"];

// ------------------------------------------------------------------ escena
const lienzo = $("lienzo");
const render = new THREE.WebGLRenderer({ antialias: true });
render.setPixelRatio(window.devicePixelRatio);
lienzo.appendChild(render.domElement);
const escena = new THREE.Scene();
escena.background = new THREE.Color(0xb9cde0);
const camara = new THREE.PerspectiveCamera(50, 1, 0.5, 20000);
camara.position.set(0, 900, 900);
const controles = new OrbitControls(camara, render.domElement);
controles.maxPolarAngle = Math.PI * 0.49;
escena.add(new THREE.HemisphereLight(0xffffff, 0x445533, 1.6));
const sol = new THREE.DirectionalLight(0xffffff, 1.6);
sol.position.set(400, 800, 300);
escena.add(sol);

function ajustar() {
  const r = lienzo.getBoundingClientRect();
  render.setSize(r.width, r.height);
  camara.aspect = r.width / Math.max(r.height, 1);
  camara.updateProjectionMatrix();
}
window.addEventListener("resize", ajustar);
ajustar();

const cargador = new GLTFLoader();
const cargarGLB = (url) => new Promise((ok) => cargador.load(url, (g) => ok(g.scene), undefined, () => ok(null)));

// ------------------------------------------------------------------ estado
const E = {
  pista: null, pistaArchivo: null, grupoPista: null, grupoEntorno: null, grupoCP: null,
  modeloCarro: null, carros: [], dt: 1 / 30, duracion: 1, t: 0, jugando: false, sel: null,
  rayosSel: null, corrida: null,
};

async function json(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
async function jsonGz(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  if (!url.endsWith(".gz")) return r.json();
  const flujo = r.body.pipeThrough(new DecompressionStream("gzip"));
  return new Response(flujo).json();
}
const aviso = (t) => ($("aviso").textContent = t || "");

// ------------------------------------------------------------------ pista
function lineaCerrada(puntos, color, h) {
  const v = puntos.map(([x, y]) => aEscena(x, y, h));
  v.push(v[0].clone());
  return new THREE.Line(new THREE.BufferGeometry().setFromPoints(v), new THREE.LineBasicMaterial({ color }));
}

async function cargarPista(archivo) {
  if (E.pistaArchivo === archivo) return;
  for (const g of [E.grupoPista, E.grupoEntorno, E.grupoCP]) if (g) escena.remove(g);
  E.pista = await json(archivo);
  E.pistaArchivo = archivo;
  const p = E.pista;
  E.grupoPista = new THREE.Group();
  E.grupoPista.add(lineaCerrada(p.borde_izquierdo, 0xffffff, 0.15));
  E.grupoPista.add(lineaCerrada(p.borde_derecho, 0xffffff, 0.15));
  // suelo de respaldo por si no hay modelo glTF
  const xs = p.linea_central.map((c) => c[0]), ys = p.linea_central.map((c) => c[1]);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const tam = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) + 600;
  const suelo = new THREE.Mesh(new THREE.PlaneGeometry(tam, tam), new THREE.MeshLambertMaterial({ color: 0x4f6b3a }));
  suelo.rotation.x = -Math.PI / 2;
  suelo.position.copy(aEscena(cx, cy, -0.3));
  E.grupoPista.add(suelo);
  escena.add(E.grupoPista);

  E.grupoCP = new THREE.Group();
  const mat = new THREE.LineBasicMaterial({ color: 0xff3b30, transparent: true, opacity: 0.7 });
  for (const cp of p.checkpoints) {
    const a = cp.izq || cp[0], b = cp.der || cp[1];
    E.grupoCP.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([aEscena(a[0], a[1], 0.2), aEscena(b[0], b[1], 0.2)]), mat));
  }
  E.grupoCP.visible = $("chkCP").checked;
  escena.add(E.grupoCP);

  const carpeta = archivo.split("/").slice(-2, -1)[0];
  const glb = MODELO_PISTA[carpeta] || `${carpeta}.glb`;
  aviso("cargando modelo 3D de la pista…");
  E.grupoEntorno = await cargarGLB(`modelos/${glb}`);
  aviso("");
  E.aereos = [];
  if (E.grupoEntorno) {
    suelo.visible = false;
    E.grupoEntorno.visible = $("chkEntorno").checked;
    escena.add(E.grupoEntorno);
    // el dirigible y el helicóptero dan vueltas sobre el circuito (rad/s; signo = sentido)
    for (const [nombre, vel] of [["Dirigible", 0.02], ["Helicoptero", -0.06]]) {
      const ob = E.grupoEntorno.getObjectByName(nombre);
      if (!ob) continue;
      const pivote = new THREE.Group();
      pivote.position.copy(aEscena(cx, cy));
      E.grupoEntorno.add(pivote);
      pivote.attach(ob);
      E.aereos.push({ pivote, vel });
    }
  }
  controles.target.copy(aEscena(cx, cy));
  camara.position.copy(aEscena(cx, cy - tam * 0.45, tam * 0.55));
  controles.update();
}

// ------------------------------------------------------------------ sonido
// Los choques se sintetizan con Web Audio (ruido filtrado + golpe grave): no hay
// archivos de audio que descargar ni licencias que citar. El navegador solo deja
// sonar después de un gesto del usuario (clic o tecla), por eso el contexto se crea ahí.
let ctxAudio = null, ruidoChoque = null, ultimoChoque = 0;
function audio() {
  if (!ctxAudio) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    ctxAudio = new Ctx();
    ruidoChoque = ctxAudio.createBuffer(1, Math.floor(ctxAudio.sampleRate * 0.6), ctxAudio.sampleRate);
    const d = ruidoChoque.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-5 * i / d.length);
  }
  if (ctxAudio.state === "suspended") ctxAudio.resume();
  return ctxAudio;
}
for (const ev of ["pointerdown", "keydown"]) window.addEventListener(ev, () => { if ($("chkSonido").checked) audio(); }, { passive: true });

function sonarChoque(volumen, paneo) {
  const ctx = $("chkSonido").checked ? audio() : null;
  if (!ctx || ctx.state === "closed") return false;            // si aún está arrancando, el sonido queda en cola
  const t = ctx.currentTime;
  const salida = ctx.createGain();
  salida.gain.value = Math.max(0, Math.min(1, volumen));
  const pan = ctx.createStereoPanner ? ctx.createStereoPanner() : null;
  if (pan) { pan.pan.value = Math.max(-1, Math.min(1, paneo)); salida.connect(pan); pan.connect(ctx.destination); }
  else salida.connect(ctx.destination);
  const ruido = ctx.createBufferSource();                       // crujido: ruido que se apaga y se oscurece
  ruido.buffer = ruidoChoque;
  ruido.playbackRate.value = 0.8 + Math.random() * 0.5;
  const filtro = ctx.createBiquadFilter();
  filtro.type = "lowpass";
  filtro.frequency.setValueAtTime(3200, t);
  filtro.frequency.exponentialRampToValueAtTime(250, t + 0.45);
  ruido.connect(filtro); filtro.connect(salida);
  ruido.start(t);
  const golpe = ctx.createOscillator(), gGolpe = ctx.createGain();   // golpe grave del impacto
  golpe.type = "sine";
  golpe.frequency.setValueAtTime(120, t);
  golpe.frequency.exponentialRampToValueAtTime(38, t + 0.22);
  gGolpe.gain.setValueAtTime(0.9, t);
  gGolpe.gain.exponentialRampToValueAtTime(0.001, t + 0.28);
  golpe.connect(gGolpe); gGolpe.connect(salida);
  golpe.start(t); golpe.stop(t + 0.3);
  return true;
}

// Suena un choque por cada carro que termina por "choque" entre los tiempos t0 y t1 de la
// repetición. Más fuerte cuanto más rápido iba y más cerca está de la cámara; con muchos
// choques juntos (primeras generaciones) se dejan al menos 70 ms entre uno y otro.
function sonidosDeChoque(t0, t1) {
  if (t1 <= t0 || t1 - t0 > 1.5) return;                        // saltos de la barra de tiempo: nada
  const lado = new THREE.Vector3().setFromMatrixColumn(camara.matrixWorld, 0);
  for (const c of E.carros) {
    if (c.datos.resultado?.motivo !== "choque") continue;
    const tc = (c.n - 1) * E.dt;
    if (tc <= t0 || tc > t1) continue;
    const ahora = performance.now();
    if (ahora - ultimoChoque < 70) continue;
    const v = c.datos.pasos.v[c.n - 1] ?? 20;
    const rel = c.raiz.position.clone().sub(camara.position);
    const dist = rel.length();
    const volumen = (0.25 + 0.75 * Math.min(v / 45, 1)) * 0.7 / (1 + dist / 220);
    if (sonarChoque(volumen, rel.dot(lado) / Math.max(dist, 1))) { ultimoChoque = ahora; E.choquesSonados = (E.choquesSonados || 0) + 1; }
  }
}

// ------------------------------------------------------------------ carros
function colorCarro(k, n) {
  return new THREE.Color().setHSL((k / Math.max(n, 1)) * 0.85, 0.75, 0.55);
}

function crearCarro(datos, color, etiqueta) {
  let obj;
  if (E.modeloCarro) {
    obj = E.modeloCarro.clone(true);
    obj.traverse((m) => {
      if (m.isMesh) {
        m.material = m.material.clone();
        if (/carroc/i.test(m.material.name)) m.material.color.copy(color);
        m.material.transparent = true;
        m.userData.carro = datos.id + "|" + etiqueta;
      }
    });
  } else {
    obj = new THREE.Mesh(new THREE.BoxGeometry(4, 1, 1.8), new THREE.MeshLambertMaterial({ color, transparent: true }));
    obj.position.y = 0.5;
    obj.userData.carro = datos.id + "|" + etiqueta;
  }
  const raiz = new THREE.Group();
  raiz.add(obj);
  escena.add(raiz);
  return { datos, raiz, color, etiqueta, n: datos.pasos.x.length };
}

function limpiarCarros() {
  for (const c of E.carros) escena.remove(c.raiz);
  E.carros = [];
  seleccionar(null);
}

function recalcularDuracion() {
  E.duracion = Math.max(1, ...E.carros.map((c) => c.n * E.dt));
  $("tiempo").max = E.duracion;
}

async function cargarGeneracion(corrida, archivo) {
  const doc = await jsonGz(`datos/${corrida}/${archivo}`);
  await cargarPista(doc.pista_archivo);
  E.dt = doc.dt;
  limpiarCarros();
  doc.carros.forEach((c, k) => E.carros.push(crearCarro(c, colorCarro(k, doc.carros.length), `gen ${doc.generacion}`)));
  recalcularDuracion();
  ponerTiempo(0);
  aviso(`${doc.carros.length} carros · generación ${doc.generacion} · ${doc.condicion}`);
}

async function agregarFantasma(corrida, archivo) {
  const doc = await jsonGz(`datos/${corrida}/${archivo}`);
  const c = doc.carros[0];
  const car = crearCarro(c, new THREE.Color().setHSL(Math.random(), 0.9, 0.6), `gen ${doc.generacion} (campeón)`);
  E.carros.push(car);
  recalcularDuracion();
  aviso(`fantasma: campeón de la generación ${doc.generacion}`);
}

function ubicar(c, t) {
  const p = c.datos.pasos;
  const f = Math.min(t / E.dt, c.n - 1);
  const k = Math.floor(f), k1 = Math.min(k + 1, c.n - 1), a = f - k;
  const x = p.x[k] + (p.x[k1] - p.x[k]) * a, y = p.y[k] + (p.y[k1] - p.y[k]) * a;
  let dr = p.rumbo[k1] - p.rumbo[k];
  dr = ((dr + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  c.raiz.position.copy(aEscena(x, y, 0));
  c.raiz.rotation.y = p.rumbo[k] + dr * a;
  const muerto = t / E.dt >= c.n - 1;
  c.raiz.visible = !muerto || $("chkMuertos").checked;
  c.raiz.traverse((m) => { if (m.isMesh) m.material.opacity = muerto ? 0.3 : 1; });
  return k;
}

function ponerTiempo(t) {
  E.t = Math.max(0, Math.min(t, E.duracion));
  $("tiempo").value = E.t;
  $("reloj").textContent = `${E.t.toFixed(1)} s`;
  for (const c of E.carros) {
    const k = ubicar(c, E.t);
    if (c === E.sel) actualizarFicha(c, k);
  }
  camaraCarro();
}

// ------------------------------------------------------------------ cámaras: libre, seguir, a bordo
const modoCamara = () => $("selCamara").value;
function camaraCarro() {
  const modo = modoCamara();
  controles.enabled = modo !== "bordo";
  if (!E.sel || modo === "libre") return;
  const p = E.sel.raiz.position, th = E.sel.raiz.rotation.y;
  // frente del carro en la escena: sim (cos th, sin th) -> escena (cos th, 0, -sin th)
  const frente = new THREE.Vector3(Math.cos(th), 0, -Math.sin(th));
  if (modo === "seguir") {
    const d = camara.position.clone().sub(controles.target);
    controles.target.copy(p);
    camara.position.copy(p.clone().add(d.setLength(Math.min(Math.max(d.length(), 25), 140))));
  } else {   // a bordo: atrás y arriba del carro, mirando hacia donde apunta; los rayos quedan a la vista
    camara.position.copy(p).addScaledVector(frente, -7).add(new THREE.Vector3(0, 3.2, 0));
    camara.lookAt(p.clone().addScaledVector(frente, 25).add(new THREE.Vector3(0, 0.5, 0)));
  }
}
$("selCamara").onchange = () => {
  if (modoCamara() !== "libre" && !E.sel && E.carros.length) seleccionar(E.carros[0]);  // el mejor carro
  if (modoCamara() === "bordo") camara.fov = 70; else camara.fov = 50;
  camara.updateProjectionMatrix();
  camaraCarro();
};

// ------------------------------------------------------------------ selección y ficha
const rayoCaster = new THREE.Raycaster();
render.domElement.addEventListener("click", (ev) => {
  const r = render.domElement.getBoundingClientRect();
  const m = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  rayoCaster.setFromCamera(m, camara);
  const objetos = E.carros.flatMap((c) => (c.raiz.visible ? [c.raiz] : []));
  const hit = rayoCaster.intersectObjects(objetos, true)[0];
  if (!hit) return;
  let o = hit.object;
  while (o && !E.carros.some((c) => c.raiz === o)) o = o.parent;
  seleccionar(E.carros.find((c) => c.raiz === o) || null);
});
$("cerrarFicha").onclick = () => seleccionar(null);

function seleccionar(c) {
  if (E.rayosSel) { escena.remove(E.rayosSel); E.rayosSel = null; }
  if (!c && modoCamara() === "bordo") { $("selCamara").value = "libre"; controles.enabled = true; }
  E.sel = c;
  $("ficha").classList.toggle("oculto", !c);
  if (!c) return;
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(5 * 2 * 3), 3));
  E.rayosSel = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xf5a524, depthTest: false }));
  E.rayosSel.renderOrder = 10;
  E.puntasRayos = [...Array(5)].map(() => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.45, 10, 8), new THREE.MeshBasicMaterial({ color: 0xff3b30, depthTest: false }));
    m.renderOrder = 11; E.rayosSel.add(m); return m;
  });
  escena.add(E.rayosSel);
  const ctrl = c.datos.controlador;
  $("fCerebroTitulo").textContent = ctrl.tipo === "red" ? "Cerebro: red 7 : 8 : 2" : "Cerebro: código del piloto";
  const codigo = ctrl.tipo === "red" ? codigoDeRed(ctrl.pesos) : ctrl.codigo;
  const pre = `<pre class="codigo">${codigo.split("\n").map((l, i) => `<span class="linea" data-l="${i + 1}">${escapar(l) || " "}</span>`).join("")}</pre>`;
  const red = ctrl.tipo === "red" ? svgRed() :
    `<p style="color:var(--tenue)">Este piloto no tiene red neuronal: su cerebro es el código (condición D).</p>`;
  $("fCerebro").innerHTML =
    `<div class="pestanas"><button data-p="red" class="activa">Red</button><button data-p="codigo">Código</button></div>` +
    `<div data-panel="red">${red}</div><div data-panel="codigo" hidden>${pre}</div>`;
  $("fCerebro").querySelectorAll(".pestanas button").forEach((b) => (b.onclick = () => {
    E.pestana = b.dataset.p;
    $("fCerebro").querySelectorAll(".pestanas button").forEach((x) => x.classList.toggle("activa", x === b));
    $("fCerebro").querySelectorAll("[data-panel]").forEach((p) => (p.hidden = p.dataset.panel !== E.pestana));
  }));
  const inicial = E.pestana || (ctrl.tipo === "red" ? "red" : "codigo");
  $("fCerebro").querySelector(`.pestanas button[data-p="${inicial}"]`).click();
  actualizarFicha(c, Math.min(Math.floor(E.t / E.dt), c.n - 1));
}
const escapar = (s) => s.replace(/[&<>]/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m]));

function actualizarFicha(c, k) {
  const d = c.datos, p = d.pasos, r = d.resultado;
  const fila = (a, b) => `<tr><td>${a}</td><td>${b}</td></tr>`;
  $("fTitulo").textContent = `${d.id} · ${c.etiqueta}`;
  $("fDatos").innerHTML =
    fila("rango", d.rango ?? "—") + fila("padre", d.padre ?? "—") +
    fila("aptitud", r.aptitud.toFixed(2)) + fila("avance", `${r.avance_m.toFixed(0)} m (${(100 * r.fraccion_vuelta).toFixed(1)}% de vuelta)`) +
    fila("velocidad", `${(p.v[k] * 3.6).toFixed(1)} km/h (máx. ${(d.cuerpo.v_max * 3.6).toFixed(0)})`) +
    fila("tiempo", `${(k * E.dt).toFixed(1)} s`) +
    fila("final", r.vuelta_completa ? `vuelta en ${r.t_vuelta.toFixed(2)} s` : `${r.motivo} a los ${r.t_final.toFixed(1)} s`);
  // sentidos: barras y rayos dibujados
  const ent = p.entradas[k];
  $("fRayos").innerHTML = d.cuerpo.angulos_rad.map((a, i) =>
    `<div class="barra"><span>${(a * 180 / Math.PI).toFixed(0)}° · ${d.cuerpo.alcances_m[i].toFixed(0)} m</span>` +
    `<div class="b"><i style="width:${(100 * ent[i]).toFixed(0)}%"></i></div><span>${(ent[i] * d.cuerpo.alcances_m[i]).toFixed(1)} m</span></div>`).join("");
  const pos = E.rayosSel.geometry.attributes.position;
  for (let i = 0; i < 5; i++) {
    const th = p.rumbo[k] + d.cuerpo.angulos_rad[i], L = ent[i] * d.cuerpo.alcances_m[i];
    const a = aEscena(p.x[k], p.y[k], 0.6), b = aEscena(p.x[k] + L * Math.cos(th), p.y[k] + L * Math.sin(th), 0.6);
    pos.setXYZ(2 * i, a.x, a.y, a.z); pos.setXYZ(2 * i + 1, b.x, b.y, b.z);
    // punta roja si el rayo toca un borde; se oculta si no ve nada dentro de su alcance
    E.puntasRayos[i].position.copy(b);
    E.puntasRayos[i].visible = ent[i] < 0.999;
  }
  pos.needsUpdate = true;
  if (d.controlador.tipo === "red") pintarRed(d.controlador.pesos, ent);
  else if (p.lineas) {
    const act = new Set(p.lineas[k] || []);
    document.querySelectorAll("pre.codigo .linea").forEach((s) => s.classList.toggle("activa", act.has(+s.dataset.l)));
  }
}

// ------------------------------------------------------------------ la red escrita como código
// La red 7:8:2 es una función: se muestra como el código Python equivalente, con sus pesos reales.
// Cada línea lleva al inicio el valor que toma en este instante (se actualiza al reproducir);
// las líneas con |valor| > 0.8 se resaltan.
const NOM_ENT = ["rayo1", "rayo2", "rayo3", "rayo4", "rayo5", "vel", "ang_cp"];
function codigoDeRed(w) {
  const f = (x) => (x >= 0 ? "+" : "-") + Math.abs(x).toFixed(2);
  const L = ["def manejar(rayo1, rayo2, rayo3, rayo4, rayo5, vel, ang_cp):"];
  for (let j = 0; j < 8; j++) {
    const t = NOM_ENT.map((n, i) => `${f(w[j * 7 + i])}*${n}`).join(" ");
    L.push(`    h${j + 1} = tanh(${t} ${f(w[56 + j])})`);
  }
  for (let o = 0; o < 2; o++) {
    const t = [...Array(8)].map((_, j) => `${f(w[64 + o * 8 + j])}*h${j + 1}`).join(" ");
    L.push(`    ${ETIQ_SAL[o]} = tanh(${t} ${f(w[80 + o])})`);
  }
  L.push("    return volante, pedal");
  return L.join("\n");
}
function valoresEnCodigo(vals) {   // vals: 10 números (8 ocultas + 2 salidas), uno por línea
  document.querySelectorAll('[data-panel="codigo"] pre.codigo .linea').forEach((s) => {
    const k = +s.dataset.l - 2;
    if (k < 0 || k >= vals.length) return;
    if (!s.dataset.base) s.dataset.base = s.textContent;
    s.textContent = `${vals[k] >= 0 ? "+" : "-"}${Math.abs(vals[k]).toFixed(2)} │${s.dataset.base}`;
    s.classList.toggle("activa", Math.abs(vals[k]) > 0.8);
  });
}

// ------------------------------------------------------------------ red neuronal brillando
const XC = [78, 190, 290], H = 250;
const yN = (i, n) => 20 + (i + 0.5) * ((H - 40) / n);
function svgRed() {
  let s = `<svg id="red" viewBox="0 0 345 ${H}" width="100%">`;
  for (let j = 0; j < 8; j++) for (let i = 0; i < 7; i++) s += `<line id="w1_${j}_${i}" x1="${XC[0]}" y1="${yN(i, 7)}" x2="${XC[1]}" y2="${yN(j, 8)}"/>`;
  for (let o = 0; o < 2; o++) for (let j = 0; j < 8; j++) s += `<line id="w2_${o}_${j}" x1="${XC[1]}" y1="${yN(j, 8)}" x2="${XC[2]}" y2="${yN(o, 2)}"/>`;
  for (let i = 0; i < 7; i++) s += `<circle id="n0_${i}" cx="${XC[0]}" cy="${yN(i, 7)}" r="8"/><text x="${XC[0] - 12}" y="${yN(i, 7) + 4}" text-anchor="end" font-size="10" fill="#8a96a3">${ETIQ_ENT[i]}</text>`;
  for (let j = 0; j < 8; j++) s += `<circle id="n1_${j}" cx="${XC[1]}" cy="${yN(j, 8)}" r="8"/>`;
  for (let o = 0; o < 2; o++) s += `<circle id="n2_${o}" cx="${XC[2]}" cy="${yN(o, 2)}" r="10"/><text x="${XC[2] + 14}" y="${yN(o, 2) + 4}" font-size="10" fill="#8a96a3">${ETIQ_SAL[o]}</text>`;
  return s + "</svg>";
}
function colorAct(a) {  // cálido = positivo (excita), frío = negativo (inhibe); brillo = |a|
  const m = Math.min(1, Math.abs(a));
  const [r, g, b] = a >= 0 ? [255, 106, 61] : [63, 167, 255];
  const f = 0.15 + 0.85 * m;
  return `rgb(${Math.round(r * f)},${Math.round(g * f)},${Math.round(b * f)})`;
}
function pintarRed(w, x) {
  const W1 = (j, i) => w[j * 7 + i], b1 = (j) => w[56 + j], W2 = (o, j) => w[64 + o * 8 + j], b2 = (o) => w[80 + o];
  const h = [...Array(8)].map((_, j) => Math.tanh(x.reduce((s, xi, i) => s + W1(j, i) * xi, b1(j))));
  const y = [0, 1].map((o) => Math.tanh(h.reduce((s, hj, j) => s + W2(o, j) * hj, b2(o))));
  const linea = (id, peso, act) => {
    const el = document.getElementById(id); if (!el) return;
    el.setAttribute("stroke", peso >= 0 ? "#ff6a3d" : "#3fa7ff");
    el.setAttribute("stroke-width", (0.3 + Math.min(3, Math.abs(peso)) * 0.9).toFixed(2));
    el.setAttribute("stroke-opacity", (0.08 + 0.8 * Math.min(1, Math.abs(peso * act))).toFixed(2));
  };
  for (let j = 0; j < 8; j++) for (let i = 0; i < 7; i++) linea(`w1_${j}_${i}`, W1(j, i), x[i]);
  for (let o = 0; o < 2; o++) for (let j = 0; j < 8; j++) linea(`w2_${o}_${j}`, W2(o, j), h[j]);
  const nodo = (id, a) => { const el = document.getElementById(id); if (el) { el.setAttribute("fill", colorAct(a)); el.setAttribute("stroke", "#222"); } };
  x.forEach((a, i) => nodo(`n0_${i}`, a));
  h.forEach((a, j) => nodo(`n1_${j}`, a));
  y.forEach((a, o) => nodo(`n2_${o}`, a));
  valoresEnCodigo([...h, ...y]);
}

// ------------------------------------------------------------------ controles de la interfaz
$("btnInicio").onclick = () => {
  ponerTiempo(0);
  if (!(E.sel && modoCamara() !== "libre") && E.pista) {   // cámara a la línea de salida
    const c = E.pista.linea_central, a = c[0], b = c[3 % c.length];
    const meta = aEscena(a[0], a[1]);
    const dx = b[0] - a[0], dy = b[1] - a[1], L = Math.hypot(dx, dy) || 1;
    controles.target.copy(meta);
    camara.position.copy(aEscena(a[0] - (dx / L) * 45, a[1] - (dy / L) * 45, 28));
    controles.update();
  }
};
$("btnPlay").onclick = () => { E.jugando = !E.jugando; $("btnPlay").textContent = E.jugando ? "❚❚" : "▶"; };
$("tiempo").oninput = (e) => ponerTiempo(+e.target.value);
$("chkEntorno").onchange = (e) => { if (E.grupoEntorno) E.grupoEntorno.visible = e.target.checked; };
$("chkCP").onchange = (e) => { if (E.grupoCP) E.grupoCP.visible = e.target.checked; };
$("chkMuertos").onchange = () => ponerTiempo(E.t);
$("selGen").onchange = () => cargarGeneracion(E.corrida, $("selGen").value).catch((e) => aviso(e.message));
$("btnFantasma").onclick = () => agregarFantasma(E.corrida, $("selFantasma").value).catch((e) => aviso(e.message));
$("selCorrida").onchange = () => elegirCorrida($("selCorrida").value);

async function elegirCorrida(id) {
  E.corrida = id;
  const gens = await json(`datos/${id}/replays.json`);
  const opc = gens.map((g) => `<option value="${g}">${g.replace(/\D+/g, " ").trim() * 1}</option>`).join("");
  $("selGen").innerHTML = opc;
  $("selFantasma").innerHTML = opc;
  if (gens.length) await cargarGeneracion(id, gens[0]);
  else aviso("esta corrida todavía no tiene repeticiones");
}

async function iniciar() {
  E.modeloCarro = await cargarGLB("modelos/carro.glb");
  const corridas = (await json("datos/corridas.json")).filter((c) => c.n_replays > 0);
  if (!corridas.length) { aviso("No hay corridas con repeticiones. Genera la demo: python scripts/demo_repeticiones.py"); return; }
  $("selCorrida").innerHTML = corridas.map((c) => `<option>${c.id}</option>`).join("");
  const pedida = new URLSearchParams(location.search).get("corrida");
  const id = corridas.some((c) => c.id === pedida) ? pedida : corridas[0].id;
  $("selCorrida").value = id;
  await elegirCorrida(id);
}

let ultimo = performance.now();
function bucle(ahora) {
  const dt = (ahora - ultimo) / 1000; ultimo = ahora;
  if (E.jugando) {
    const antes = E.t;
    ponerTiempo(E.t + dt * +$("selVel").value);
    sonidosDeChoque(antes, E.t);
    if (E.t >= E.duracion) { E.jugando = false; $("btnPlay").textContent = "▶"; }
  }
  for (const a of E.aereos || []) a.pivote.rotation.y += a.vel * Math.min(dt, 0.1);
  if (modoCamara() === "bordo" && E.sel) camaraCarro(); else controles.update();
  render.render(escena, camara);
  requestAnimationFrame(bucle);
}
requestAnimationFrame(bucle);
iniciar().catch((e) => aviso("Error: " + e.message));
window.__visor = { E, seleccionar, ponerTiempo, sonarChoque, sonidosDeChoque, audio: () => ctxAudio };   // acceso de depuración (pruebas automáticas)
