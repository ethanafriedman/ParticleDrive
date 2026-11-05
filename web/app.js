// Browser-only drawing and controls. All robotics math runs in Python.
const $ = id => document.getElementById(id);
const map = $('map'), ctx = map.getContext('2d');
const plot = $('plot'), pc = plot.getContext('2d');
let state = null, playing = false, busy = false, timer = null;
// These are display settings only; changing them does not reset the filter.
let showCarMarkers = true;
let particleSize = 2.8;

async function request(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: {'Content-Type': 'application/json'},
    ...(body === undefined ? {} : {body: JSON.stringify(body)})
  });
  if (!response.ok) throw Error('Simulation request failed');
  return response.json();
}
function stroke(color, width=1) { ctx.strokeStyle=color; ctx.lineWidth=width; }
function circle(x,y,r,color) { ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill(); }
function car(pose, color, size=12) {
  const [x,y,a] = pose;
  ctx.save();
  ctx.translate(x,y);
  ctx.rotate(a);
  // A dark outline keeps each car visible without hiding the particle cloud.
  ctx.fillStyle=color;
  ctx.strokeStyle='#071521';
  ctx.lineWidth=2.5;
  ctx.lineJoin='round';
  ctx.beginPath();
  ctx.moveTo(size*1.5,0);
  ctx.lineTo(-size,-size*.85);
  ctx.lineTo(-size,size*.85);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}
function drawMap(s) {
  ctx.clearRect(0,0,900,560);
  // Quiet map: intersections and the road loop are visible, not photorealistic.
  ctx.fillStyle='#0d1727';ctx.fillRect(0,0,900,560);
  ctx.strokeStyle='#15283b';ctx.lineWidth=1;
  for (let x=0;x<900;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,560);ctx.stroke()}
  for (let y=0;y<560;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(900,y);ctx.stroke()}
  ctx.strokeStyle='#24374c';ctx.lineWidth=60;ctx.lineJoin='round';ctx.beginPath();
  s.waypoints.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();
  ctx.strokeStyle='#496078';ctx.lineWidth=1.5;ctx.setLineDash([13,11]);ctx.beginPath();
  s.waypoints.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();ctx.setLineDash([]);
  if(s.sensors) {
    stroke('rgba(91,226,210,.10)'); ctx.setLineDash([5,8]);
    s.landmarks.forEach(([x,y])=>{ctx.beginPath();ctx.moveTo(...s.true.slice(0,2));ctx.lineTo(x,y);ctx.stroke()});ctx.setLineDash([]);
  }
  s.landmarks.forEach(([x,y],i)=>{circle(x,y,6,'#57718e');circle(x,y,2.8,'#b8cee6')});
  // Show the divergence between truth, particle estimate and odometry.
  stroke('#f6c665',1.5);ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(...s.true.slice(0,2));ctx.lineTo(...s.estimated.slice(0,2));ctx.stroke();ctx.setLineDash([]);
  // Draw the particle cloud first, so dots never cover the car arrows.
  // Small outlined markers remain readable even when particles converge tightly.
  s.particles.forEach(([x,y])=>circle(x,y,particleSize,'rgba(115,179,255,.65)'));
  if (showCarMarkers) {
    car(s.dead,'#fc718a',6);
    car(s.estimated,'#f7c965',7);
    car(s.true,'#55e4df',8);
  }
  ctx.fillStyle='#dce8f6';ctx.font='12px system-ui';ctx.fillText('LANDMARK MAP · TOP DOWN',19,27);
}
function drawPlot(s) {
  pc.clearRect(0,0,900,175); const data=s.history;
  pc.strokeStyle='#273b53';pc.lineWidth=1;
  [28,74,120,164].forEach(y=>{pc.beginPath();pc.moveTo(38,y);pc.lineTo(875,y);pc.stroke()});
  pc.fillStyle='#8094ac';pc.font='11px system-ui';
  if(!data.length){pc.fillText('Run the simulation to see localization errors over time.',45,90);return}
  const maxVal=Math.max(30,...data.flatMap(x=>[x[1],x[2]]));
  pc.fillText(`${Math.round(maxVal)} px`,4,30);pc.fillText('0',22,164);
  const drawSeries=(index,color)=>{pc.strokeStyle=color;pc.lineWidth=2.5;pc.beginPath();
    data.forEach((d,i)=>{const x=40+i/Math.max(1,data.length-1)*826,y=161-d[index]/maxVal*132;i?pc.lineTo(x,y):pc.moveTo(x,y)});pc.stroke()};
  drawSeries(2,'#f37b90');drawSeries(1,'#f6c665');
}
function render(s){state=s;drawMap(s);drawPlot(s);$('tick').textContent=s.tick;$('error').textContent=s.error+' px';$('drift').textContent=s.drift+' px';$('count').textContent=s.count.toLocaleString();$('sensor-state').textContent=s.sensors?'ON':'OFF';}
async function advance(){if(busy)return;busy=true;try{render(await request('/api/step',{}))}catch(e){pause();alert(e.message)}finally{busy=false}}
function pause(){playing=false;clearInterval(timer);$('play').textContent='▶ Start'}
$('play').onclick=()=>{if(playing){pause();return}playing=true;$('play').textContent='Ⅱ Pause';timer=setInterval(advance,100)};
$('step').onclick=()=>{pause();advance()};
async function reset(){pause();const body={particles:Number($('particles').value),noise:Number($('noise').value),sensors:$('sensors').checked};render(await request('/api/reset',body))}
$('reset').onclick=reset;$('particles').oninput=()=>{$('particle-value').textContent=$('particles').value};$('noise').oninput=()=>{$('noise-value').textContent=$('noise').value+' px'};
$('particles').onchange=reset;$('noise').onchange=reset;
$('sensors').onchange=async()=>render(await request('/api/sensors?enabled='+$('sensors').checked,{}));
$('kidnap').onclick=async()=>render(await request('/api/kidnap',{}));
$('show-markers').onchange = () => {
  showCarMarkers = $('show-markers').checked;
  if (state) drawMap(state);
};
$('particle-size').oninput = () => {
  particleSize = Number($('particle-size').value);
  $('particle-size-value').textContent = particleSize.toFixed(1) + ' px';
  if (state) drawMap(state);
};
request('/api/state').then(render).catch(e=>alert(e.message));
