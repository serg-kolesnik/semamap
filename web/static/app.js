const $=s=>document.querySelector(s);
let detail=new Map(),detailTok=0,detailTimer=null;
const DETAIL_PX=150;   // с какого размера клетки на экране грузим HD
let INFO,viewer,marks=[],results=[],ri=-1,cur=null,note='',LB=null,activeLevel=-1,labelsOn=true;
const status=t=>$('#st').textContent=t||'';

fetch('/api/info').then(r=>r.json()).then(i=>{INFO=i;init();})
  .catch(e=>status('ОШИБКА: '+e));

function init(){
  if(!INFO.search){$('#q').disabled=true;$('#q').placeholder='Поиск отключён';}
  if(INFO.labels)$('#lblbtn').style.display='';
  if(INFO.has3d)$('#to3d').style.display='';   // раньше кнопка «ЗОНЫ» оставалась скрытой навсегда
  status(INFO.n.toLocaleString('ru')+' ОБЪЕКТОВ');

  viewer=OpenSeadragon({
    id:'osd',
    tileSources: INFO.tileSource,                   // конфиг тайлов приходит из /api/info
    prefixUrl:'https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/images/',
    showNavigator:true, navigatorPosition:'BOTTOM_RIGHT',
    showRotationControl:true, showHomeControl:false,
    zoomControlAnchor: OpenSeadragon.ControlAnchor.BOTTOM_LEFT,
    maxZoomPixelRatio:8, minZoomImageRatio:0.8, visibilityRatio:0.4,
    animationTime:0.5, springStiffness:9, blendTime:0.12, imageLoaderLimit:10,
    gestureSettingsMouse:{clickToZoom:false, dblClickToZoom:false, flickEnabled:true, pinchRotate:true},
    gestureSettingsTouch:{clickToZoom:false, dblClickToZoom:true, flickEnabled:true, pinchRotate:true}
  });

  viewer.addHandler('open',()=>{if(INFO.labels)loadLabels();});
  viewer.addHandler('open-failed', e => { status("ОШИБКА КАРТЫ: " + e.message); });

  ['zoom','resize'].forEach(ev=>viewer.addHandler(ev,updLabels));
  ['zoom','pan','resize','animation-finish'].forEach(ev=>viewer.addHandler(ev,scheduleDetail));
  viewer.addHandler('canvas-click',e=>{
    if(!e.quick)return;
    const p=viewer.viewport.viewportToImageCoordinates(viewer.viewport.pointFromPixel(e.position));
    openCell(Math.floor(p.x/INFO.cell),Math.floor(p.y/INFO.cell));
  });
}

function goTo(x,y,cellPx){
  const vp=viewer.viewport;
  vp.panTo(vp.imageToViewportCoordinates((x+.5)*INFO.cell,(y+.5)*INFO.cell));
  vp.zoomTo(vp.imageToViewportZoom(cellPx/INFO.cell));
}
function clearMarks(){marks.forEach(m=>viewer.removeOverlay(m));marks=[];results=[];ri=-1;note='';}
function showMarks(items){
  clearMarks();results=items;
  items.forEach(it=>{
    const el=document.createElement('div');el.className='hit';
    viewer.addOverlay({element:el,location:viewer.viewport.imageToViewportCoordinates((it.x+.5)*INFO.cell,(it.y+.5)*INFO.cell),placement:OpenSeadragon.Placement.CENTER});
    marks.push(el);
  });
}
function step(d){
  if(!results.length)return;
  ri=(ri+d+results.length)%results.length;
  const r=results[ri];goTo(r.x,r.y,150);
  status(`ЦЕЛЬ ${ri+1}/${results.length}`+(note?' | '+note:''));
}

async function search(t,en){
  t=(t||$('#q').value).trim();if(!t)return;
  status('АНАЛИЗ...');

  let x1=0, y1=0, x2=INFO.w, y2=INFO.h;
  if (viewer && viewer.viewport) {
      const bounds = viewer.viewport.getBoundsNoRotate();
      const rect = viewer.viewport.viewportToImageRectangle(bounds);
      const padX = rect.width * 0.15;
      const padY = rect.height * 0.15;

      x1 = Math.max(0, Math.floor((rect.x - padX) / INFO.cell));
      y1 = Math.max(0, Math.floor((rect.y - padY) / INFO.cell));
      x2 = Math.min(INFO.w, Math.ceil((rect.x + rect.width + padX) / INFO.cell));
      y2 = Math.min(INFO.h, Math.ceil((rect.y + rect.height + padY) / INFO.cell));
  }

  const url = `/api/search?k=300&q=${encodeURIComponent(t)}` +
              (en ? `&en=${encodeURIComponent(en)}` : '') +
              `&x1=${x1}&y1=${y1}&x2=${x2}&y2=${y2}`;

  const r=await fetch(url);
  const j=await r.json();
  if(j.error){status(j.error);return;}

  showMarks(j.items);
  note=j.q_en?'['+j.q_en+']':'';ri=-1;

  if (j.items.length === 0) {
      status('В ЭТОЙ ОБЛАСТИ НИЧЕГО НЕ НАЙДЕНО (попробуйте отдалить карту)');
  } else {
      step(1);
  }
}

async function loadTags(id){
  const sfwBox=$('#sfw-tags');
  const nsfwBox=$('#nsfw-tags');
  sfwBox.innerHTML=''; nsfwBox.innerHTML='';

  if(!INFO.tags)return;

  try{
    const resp = await (await fetch('/api/tags?id='+id)).json();
    if(!cur||cur.id!==id)return;

    if(resp.nsfw){
      const b=document.createElement('span');
      b.className='nsfw-badge';
      b.textContent='18+';
      b.title='NSFW score: '+(resp.score||0).toFixed(3);
      nsfwBox.appendChild(b);
    }

    const nsfwTags = (resp.tags||[]).filter(t => t.nsfw);
    const sfwTags = (resp.tags||[]).filter(t => !t.nsfw);

    nsfwTags.forEach(it => {
      const b=document.createElement('button');
      b.className='chip nsfw';
      b.textContent=it.ru;
      b.title=it.en+' · '+Math.round(it.p*100)+'%';
      b.onclick=()=>{closeLb();$('#q').value=it.ru;search(it.ru,it.en);};
      nsfwBox.appendChild(b);
    });

    sfwTags.forEach(it => {
      const b=document.createElement('button');
      b.className='chip';
      b.textContent=it.ru;
      b.title=it.en+' · '+Math.round(it.p*100)+'%';
      b.onclick=()=>{closeLb();$('#q').value=it.ru;search(it.ru,it.en);};
      sfwBox.appendChild(b);
    });
  }catch(e){}
}

function loadLabels(){
  fetch('/api/labels').then(r=>r.json()).then(j=>{
    LB=j.levels.map(l=>({bs:l.bs,items:l.items,els:null}));
    activeLevel=-1;updLabels();
  });
}
function showLevel(L){
  if(!L.els)L.els=L.items.map(it=>{
    const el=document.createElement('div');el.className='lbl';el.textContent=it.t;
    el.style.fontSize=Math.min(26,12+3*Math.sqrt(it.n))+'px';return el;
  });
  L.els.forEach((el,i)=>viewer.addOverlay({element:el,
    location:viewer.viewport.imageToViewportCoordinates(L.items[i].x*INFO.cell,L.items[i].y*INFO.cell),
    placement:OpenSeadragon.Placement.CENTER,checkResize:false}));
}
function updLabels(){
  if(!LB||!LB.length||!viewer||!viewer.viewport)return;
  const cellPx=viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true))*INFO.cell;
  let want=-1;
  if(labelsOn&&cellPx<45){
    want=LB.length-1;
    for(let i=0;i<LB.length;i++)if(LB[i].bs*cellPx>=110){want=i;break;}
  }
  if(want===activeLevel)return;
  if(activeLevel>=0)LB[activeLevel].els.forEach(e=>viewer.removeOverlay(e));
  activeLevel=want;
  if(want>=0)showLevel(LB[want]);
}
async function openCell(x,y){
  const j=await (await fetch(`/api/cell?x=${x}&y=${y}`)).json();
  if(j.id<0)return false;
  cur=j;
  $('#img').src='/file/'+j.id;
  $('#cap').textContent=j.path+'  ';
  $('#lb').classList.add('on');

  loadTags(j.id);
  return true;
}
async function nav(dx,dy){
  if(!cur)return;
  for(let k=1;k<=3;k++) if(await openCell(cur.x+dx*k,cur.y+dy*k)) return;
}
function closeLb(){$('#lb').classList.remove('on');$('#img').removeAttribute('src');}

$('#q').addEventListener('keydown',e=>{if(e.key==='Enter')search();e.stopPropagation();});
$('#clr').onclick=()=>{clearMarks();status(INFO.n.toLocaleString('ru')+' ОБЪЕКТОВ');};
$('#cls').onclick=closeLb;
$('#lblbtn').onclick=()=>{labelsOn=!labelsOn;$('#lblbtn').style.opacity=labelsOn?1:.5;updLabels();};
$('#here').onclick=()=>{closeLb();goTo(cur.x,cur.y,160);};
$('#sim').onclick=async()=>{
  const id=cur.id,x=cur.x,y=cur.y;closeLb();goTo(x,y,110);
  showMarks(await (await fetch('/api/similar?k=200&id='+id)).json());
  status('СИНХРОНИЗАЦИЯ: '+results.length);
};

document.addEventListener('keydown',e=>{
  if(!viewer)return;
  const lb=$('#lb').classList.contains('on');
  if(e.key==='Escape'){lb?closeLb():clearMarks();return;}
  if(lb){
    const m={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[e.key];
    if(m){e.preventDefault();nav(...m);}
    return;
  }
  if(e.key===']')step(1);
  else if(e.key==='[')step(-1);
  else if(e.key==='j')goTo(Math.floor(Math.random()*INFO.w),Math.floor(Math.random()*INFO.h),160);
});

// ---- HD-превью при приближении ----
function scheduleDetail(){clearTimeout(detailTimer);detailTimer=setTimeout(updateDetail,200);}
function clearDetail(){detailTok++;detail.forEach(el=>viewer.removeOverlay(el));detail.clear();}
async function updateDetail(){
  if(!viewer||!viewer.viewport||!INFO)return;
  const cellPx=viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true))*INFO.cell;
  if(cellPx<DETAIL_PX){clearDetail();return;}
  const tok=++detailTok;
  const rect=viewer.viewport.viewportToImageRectangle(viewer.viewport.getBoundsNoRotate());
  const padX=rect.width*0.1,padY=rect.height*0.1;
  const x1=Math.floor((rect.x-padX)/INFO.cell),y1=Math.floor((rect.y-padY)/INFO.cell);
  const x2=Math.ceil((rect.x+rect.width+padX)/INFO.cell),y2=Math.ceil((rect.y+rect.height+padY)/INFO.cell);
  let items;
  try{items=await (await fetch(`/api/visible?x1=${x1}&y1=${y1}&x2=${x2}&y2=${y2}`)).json();}catch(e){return;}
  if(tok!==detailTok||!Array.isArray(items))return;   // пока ждали, вид уже изменился
  const want=new Set(items.map(i=>i.id));
  for(const [id,el] of detail)if(!want.has(id)){viewer.removeOverlay(el);detail.delete(id);}
  for(const it of items){
    if(detail.has(it.id))continue;
    const img=document.createElement('img');
    img.className='detail-img';img.draggable=false;
    img.onload=()=>{img.style.opacity=1;};
    img.onerror=()=>{viewer.removeOverlay(img);detail.delete(it.id);};
    img.src='/api/thumb512?id='+it.id;
    // Rect (а не точка): картинка масштабируется вместе с картой
    viewer.addOverlay({element:img,location:viewer.viewport.imageToViewportRectangle(it.x*INFO.cell,it.y*INFO.cell,INFO.cell,INFO.cell),checkResize:false});
    detail.set(it.id,img);
  }
}
