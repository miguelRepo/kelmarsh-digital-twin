// Simulación de colonia de hormigas con navegación por feromonas.
//
// Las hormigas son "ciegas": no conocen el mapa ni la posición de la comida.
// Solo perciben el entorno con sus antenas (3 sensores cortos delante) que
// detectan dos feromonas depositadas en el suelo:
//   - Feromona CASA  (azul): la deposita toda hormiga que sale del nido.
//     Las hormigas que vuelven la siguen para encontrar el camino de regreso.
//   - Feromona COMIDA (roja): la deposita la hormiga que vuelve cargada.
//     Las exploradoras la siguen para llegar a la fuente de comida.
// Ambas se evaporan con el tiempo, así que un camino solo sobrevive si otras
// hormigas siguen pasando por él y lo refuerzan. Los caminos cortos se
// recorren más a menudo, se refuerzan más y acaban dominando: el camino
// "óptimo" emerge solo, nadie lo calcula.

'use strict';

// ---------------------------------------------------------------- constantes

const W = 1280, H = 800;          // tamaño del lienzo en píxeles
const CELL = 4;                   // tamaño de celda de la rejilla de feromonas
const GW = W / CELL, GH = H / CELL;
const NCELLS = GW * GH;

const ANT_SPEED       = 1.3;      // px por paso
const CARRY_SLOWDOWN  = 0.85;     // cargada va un poco más lenta
const WANDER          = 0.22;     // ruido aleatorio del rumbo (rad)
const TURN_SPEED      = 0.3;      // giro hacia un sensor (rad)
const SENSOR_DIST     = 14;       // alcance de las antenas (px)
const SENSOR_ANGLE    = 0.55;     // separación angular de las antenas (rad)
const SENSE_THRESHOLD = 0.005;    // por debajo de esto el rastro no se percibe
const DEPOSIT         = 0.06;     // feromona depositada por paso (a reserva llena)
const RESERVE_DECAY   = 0.9975;   // la reserva se agota al alejarse del origen
const DIFFUSION       = 0.06;     // difusión de feromona entre celdas vecinas
const GIVE_UP_STEPS   = 2600;     // pasos explorando antes de volver de vacío
const FOOD_PER_CELL   = 5;        // unidades de comida por celda de la fuente
const FOOD_SMELL      = 0.5;      // la comida en sí desprende un poco de olor

const NEST = { x: 0, y: 0, r: 18 };

const SEARCHING = 0, RETURNING = 1;

// --------------------------------------------------------------- estado mapa

const walls    = new Uint8Array(NCELLS);
const food     = new Float32Array(NCELLS);
const pherHome = new Float32Array(NCELLS);
const pherFood = new Float32Array(NCELLS);
const tmpA     = new Float32Array(NCELLS);
const tmpB     = new Float32Array(NCELLS);

let nestCells = [];               // celdas del nido (emiten feromona casa fija)
let totalFood = 0;
let collected = 0;
let ants = [];
let paused = false;
let finished = false;

const idx = (cx, cy) => cy * GW + cx;
const cellAt = (x, y) => idx((x / CELL) | 0, (y / CELL) | 0);

function isWall(x, y) {
  if (x < 0 || y < 0 || x >= W || y >= H) return true;
  return walls[cellAt(x, y)] === 1;
}

// --------------------------------------------------------- generación de mapa

function generateMap() {
  walls.fill(0);
  food.fill(0);
  pherHome.fill(0);
  pherFood.fill(0);
  totalFood = 0;
  collected = 0;
  finished = false;

  // bordes
  for (let cx = 0; cx < GW; cx++)
    for (let cy = 0; cy < GH; cy++)
      if (cx < 2 || cy < 2 || cx >= GW - 2 || cy >= GH - 2) walls[idx(cx, cy)] = 1;

  // el nido en una zona aleatoria, no demasiado pegado al borde
  NEST.x = (0.15 + Math.random() * 0.7) * W;
  NEST.y = (0.15 + Math.random() * 0.7) * H;

  // rocas: manchas circulares aleatorias
  const nBlobs = 26 + (Math.random() * 10 | 0);
  for (let i = 0; i < nBlobs; i++) {
    const cx = Math.random() * GW, cy = Math.random() * GH;
    const r = 3 + Math.random() * 13;
    stampWallDisk(cx, cy, r);
  }
  // muros alargados que forman pasillos
  const nWallsLines = 10 + (Math.random() * 6 | 0);
  for (let i = 0; i < nWallsLines; i++) {
    let cx = Math.random() * GW, cy = Math.random() * GH;
    const a = Math.random() * Math.PI * 2;
    const len = 20 + Math.random() * 50;
    const thick = 1.5 + Math.random() * 2.5;
    for (let t = 0; t < len; t++) {
      stampWallDisk(cx, cy, thick);
      cx += Math.cos(a); cy += Math.sin(a);
    }
  }

  // despejar la zona del nido
  clearDisk(NEST.x / CELL, NEST.y / CELL, NEST.r / CELL + 5);

  // comprobar qué celdas son alcanzables desde el nido
  const reachable = floodFill((NEST.x / CELL) | 0, (NEST.y / CELL) | 0);
  let freeCells = 0, reachCount = 0;
  for (let i = 0; i < NCELLS; i++) {
    if (!walls[i]) freeCells++;
    if (reachable[i]) reachCount++;
  }
  if (reachCount < freeCells * 0.5 || reachCount < NCELLS * 0.3) {
    generateMap();               // mapa demasiado cerrado: probar otro
    return;
  }
  // las zonas inalcanzables se rellenan como muro (no engañan a nadie)
  for (let i = 0; i < NCELLS; i++) if (!walls[i] && !reachable[i]) walls[i] = 1;

  // fuentes de comida: lejos del nido y alcanzables
  const nSources = 5 + (Math.random() * 3 | 0);
  const minDist = Math.min(W, H) * 0.32;
  let placed = 0, tries = 0;
  while (placed < nSources && tries < 4000) {
    tries++;
    const cx = 4 + (Math.random() * (GW - 8) | 0);
    const cy = 4 + (Math.random() * (GH - 8) | 0);
    if (!reachable[idx(cx, cy)]) continue;
    const dx = cx * CELL - NEST.x, dy = cy * CELL - NEST.y;
    if (dx * dx + dy * dy < minDist * minDist) continue;
    const r = 3 + Math.random() * 3.5;
    for (let ox = -Math.ceil(r); ox <= r; ox++)
      for (let oy = -Math.ceil(r); oy <= r; oy++) {
        if (ox * ox + oy * oy > r * r) continue;
        const i = idx(cx + ox, cy + oy);
        if (!walls[i] && reachable[i] && food[i] === 0) {
          food[i] = FOOD_PER_CELL;
          totalFood += FOOD_PER_CELL;
        }
      }
    placed++;
  }

  // celdas del nido: emiten feromona casa de forma constante (el nido huele)
  nestCells = [];
  const ncx = NEST.x / CELL, ncy = NEST.y / CELL, nr = NEST.r / CELL;
  for (let cx = Math.floor(ncx - nr); cx <= ncx + nr; cx++)
    for (let cy = Math.floor(ncy - nr); cy <= ncy + nr; cy++) {
      const dx = cx - ncx, dy = cy - ncy;
      if (dx * dx + dy * dy <= nr * nr) nestCells.push(idx(cx, cy));
    }
}

function stampWallDisk(cx, cy, r) {
  for (let ox = -Math.ceil(r); ox <= r; ox++)
    for (let oy = -Math.ceil(r); oy <= r; oy++) {
      if (ox * ox + oy * oy > r * r) continue;
      const x = (cx + ox) | 0, y = (cy + oy) | 0;
      if (x >= 0 && y >= 0 && x < GW && y < GH) walls[idx(x, y)] = 1;
    }
}

function clearDisk(cx, cy, r) {
  for (let ox = -Math.ceil(r); ox <= r; ox++)
    for (let oy = -Math.ceil(r); oy <= r; oy++) {
      if (ox * ox + oy * oy > r * r) continue;
      const x = (cx + ox) | 0, y = (cy + oy) | 0;
      if (x >= 2 && y >= 2 && x < GW - 2 && y < GH - 2) walls[idx(x, y)] = 0;
    }
}

function floodFill(scx, scy) {
  const seen = new Uint8Array(NCELLS);
  const stack = [idx(scx, scy)];
  seen[stack[0]] = 1;
  while (stack.length) {
    const i = stack.pop();
    const cx = i % GW, cy = (i / GW) | 0;
    for (const [ox, oy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nx = cx + ox, ny = cy + oy;
      if (nx < 0 || ny < 0 || nx >= GW || ny >= GH) continue;
      const j = idx(nx, ny);
      if (!walls[j] && !seen[j]) { seen[j] = 1; stack.push(j); }
    }
  }
  return seen;
}

// ------------------------------------------------------------------- hormigas

function createAnts(n) {
  ants = [];
  for (let i = 0; i < n; i++) {
    ants.push({
      x: NEST.x, y: NEST.y,
      angle: Math.random() * Math.PI * 2,
      state: SEARCHING,
      carrying: false,
      reserve: 1,                       // cuánta feromona le queda por soltar
      steps: 0,                         // pasos desde que salió a explorar
      delay: (i / n) * 900 | 0,         // salen escalonadas, no en bloque
    });
  }
}

// muestrea lo que percibe una antena en (x, y); los muros "huelen" fatal
function sense(field, x, y, withFood) {
  if (x < 0 || y < 0 || x >= W || y >= H) return -1;
  const c = cellAt(x, y);
  if (walls[c]) return -1;
  let v = field[c];
  if (withFood && food[c] > 0) v += 10;   // comida al alcance de la antena
  return v;
}

function updateAnt(ant) {
  if (ant.delay > 0) { ant.delay--; return; }

  const searching = ant.state === SEARCHING;
  const field = searching ? pherFood : pherHome;

  // --- antenas: izquierda, centro, derecha
  const aL = ant.angle - SENSOR_ANGLE, aC = ant.angle, aR = ant.angle + SENSOR_ANGLE;
  const vL = sense(field, ant.x + Math.cos(aL) * SENSOR_DIST, ant.y + Math.sin(aL) * SENSOR_DIST, searching);
  const vC = sense(field, ant.x + Math.cos(aC) * SENSOR_DIST, ant.y + Math.sin(aC) * SENSOR_DIST, searching);
  const vR = sense(field, ant.x + Math.cos(aR) * SENSOR_DIST, ant.y + Math.sin(aR) * SENSOR_DIST, searching);

  const best = Math.max(vL, vC, vR);
  if (best > SENSE_THRESHOLD) {
    if (vL === best)      ant.angle -= TURN_SPEED;
    else if (vR === best) ant.angle += TURN_SPEED;
    // si gana el centro, sigue recto
    if (searching) ant.steps = Math.max(0, ant.steps - 2);  // un rastro le da ánimo
  }
  // ruido de rumbo: siempre hay algo de exploración
  ant.angle += (Math.random() - 0.5) * WANDER;

  // --- movimiento con rebote en muros
  const spd = ANT_SPEED * (ant.carrying ? CARRY_SLOWDOWN : 1);
  let moved = false;
  for (const da of [0, 0.6, -0.6, 1.2, -1.2, 2.0, -2.0, Math.PI]) {
    const a = ant.angle + da;
    const nx = ant.x + Math.cos(a) * spd, ny = ant.y + Math.sin(a) * spd;
    if (!isWall(nx, ny)) {
      if (da !== 0) ant.angle = a + (Math.random() - 0.5) * 0.4;
      ant.x = nx; ant.y = ny;
      moved = true;
      break;
    }
  }
  if (!moved) ant.angle += Math.PI;     // encajonada: media vuelta

  const c = cellAt(ant.x, ant.y);

  // --- depositar feromona (solo si le queda reserva: marca caminos cortos)
  if (ant.reserve > 0.01) {
    if (searching) {
      pherHome[c] = Math.min(1, pherHome[c] + DEPOSIT * ant.reserve);
    } else if (ant.carrying) {
      pherFood[c] = Math.min(1, pherFood[c] + DEPOSIT * ant.reserve);
    }
    ant.reserve *= RESERVE_DECAY;
  }

  // --- interacción con comida y nido
  if (searching) {
    ant.steps++;
    if (food[c] > 0) {                      // ¡comida bajo las patas!
      food[c] -= 1;
      ant.carrying = true;
      ant.state = RETURNING;
      ant.reserve = 1;
      ant.angle += Math.PI;                 // da media vuelta hacia casa
    } else if (ant.steps > GIVE_UP_STEPS) { // demasiado tiempo fuera: a casa
      ant.state = RETURNING;
      ant.carrying = false;
      ant.reserve = 0;
    }
  } else {
    const dx = ant.x - NEST.x, dy = ant.y - NEST.y;
    if (dx * dx + dy * dy < NEST.r * NEST.r) {
      if (ant.carrying) collected++;
      ant.carrying = false;
      ant.state = SEARCHING;
      ant.reserve = 1;
      ant.steps = 0;
      ant.angle += Math.PI + (Math.random() - 0.5);  // sale otra vez
    }
  }
}

// ------------------------------------------------ evaporación + difusión

let evapRate = 0.995;

function updatePheromones() {
  diffuseEvaporate(pherHome, tmpA);
  diffuseEvaporate(pherFood, tmpB);
  pherHome.set(tmpA);
  pherFood.set(tmpB);

  // el nido siempre huele a casa; la comida desprende un olor tenue
  for (const i of nestCells) pherHome[i] = 1;
  for (let i = 0; i < NCELLS; i++)
    if (food[i] > 0) pherFood[i] = Math.max(pherFood[i], FOOD_SMELL);
}

function diffuseEvaporate(src, dst) {
  for (let cy = 1; cy < GH - 1; cy++) {
    const row = cy * GW;
    for (let cx = 1; cx < GW - 1; cx++) {
      const i = row + cx;
      if (walls[i]) { dst[i] = 0; continue; }
      const nb = (src[i - 1] + src[i + 1] + src[i - GW] + src[i + GW]) * 0.25;
      let v = src[i] * (1 - DIFFUSION) + nb * DIFFUSION;
      v *= evapRate;
      dst[i] = v < 0.0005 ? 0 : v;
    }
  }
}

// ----------------------------------------------------------------- renderizado

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const gridCanvas = document.createElement('canvas');
gridCanvas.width = GW; gridCanvas.height = GH;
const gctx = gridCanvas.getContext('2d');
const img = gctx.createImageData(GW, GH);

let showHome = true, showFood = true;

function render() {
  // capa de rejilla: suelo, muros, feromonas y comida
  const d = img.data;
  for (let i = 0, p = 0; i < NCELLS; i++, p += 4) {
    let r = 16, g = 16, b = 20;
    if (walls[i]) {
      r = 58; g = 58; b = 68;
    } else {
      if (showHome) { const v = pherHome[i]; b += v * 200; g += v * 60; }
      if (showFood) { const v = pherFood[i]; r += v * 220; g += v * 70; }
      if (food[i] > 0) {
        const f = food[i] / FOOD_PER_CELL;
        r = 40; g = 120 + f * 130; b = 50;
      }
    }
    d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = 255;
  }
  gctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(gridCanvas, 0, 0, W, H);

  // nido
  ctx.beginPath();
  ctx.arc(NEST.x, NEST.y, NEST.r, 0, Math.PI * 2);
  ctx.fillStyle = '#7a5230';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(NEST.x, NEST.y, NEST.r * 0.45, 0, Math.PI * 2);
  ctx.fillStyle = '#2a1c10';
  ctx.fill();

  // hormigas
  let carrying = 0;
  for (const ant of ants) {
    if (ant.delay > 0) continue;
    if (ant.carrying) { ctx.fillStyle = '#8ee860'; carrying++; }
    else ctx.fillStyle = '#e8e3d8';
    ctx.fillRect(ant.x - 1, ant.y - 1, 2.4, 2.4);
  }

  document.getElementById('stCollected').textContent = collected;
  document.getElementById('stTotal').textContent = totalFood;
  document.getElementById('stAnts').textContent = ants.length;
  document.getElementById('stCarrying').textContent = carrying;

  if (!finished && totalFood > 0 && collected >= totalFood) {
    finished = true;
    document.getElementById('overlay').style.display = 'flex';
  }
}

// ----------------------------------------------------------------- bucle

let stepsPerFrame = 1;

function frame() {
  if (!paused) {
    for (let s = 0; s < stepsPerFrame; s++) {
      for (const ant of ants) updateAnt(ant);
      updatePheromones();
    }
  }
  render();
  requestAnimationFrame(frame);
}

// ----------------------------------------------------------------- controles

const $ = (id) => document.getElementById(id);

$('inAnts').addEventListener('input', e => $('lbAnts').textContent = e.target.value);
$('inSpeed').addEventListener('input', e => {
  stepsPerFrame = +e.target.value;
  $('lbSpeed').textContent = e.target.value + '×';
});
$('inEvap').addEventListener('input', e => {
  evapRate = +e.target.value;
  $('lbEvap').textContent = (+e.target.value).toFixed(3);
});
$('inShowHome').addEventListener('change', e => showHome = e.target.checked);
$('inShowFood').addEventListener('change', e => showFood = e.target.checked);
$('btnPause').addEventListener('click', () => {
  paused = !paused;
  $('btnPause').textContent = paused ? '▶ Continuar' : '⏸ Pausa';
});
$('btnReset').addEventListener('click', reset);

function reset() {
  generateMap();
  createAnts(+$('inAnts').value);
  document.getElementById('overlay').style.display = 'none';
}

reset();
requestAnimationFrame(frame);
