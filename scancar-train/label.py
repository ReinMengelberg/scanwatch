"""Local review UI. python label.py [--port 8765]  ->  http://localhost:8765

Frames view: the full camera frame with every detected car boxed, labelled and scored.
Crops view:  every stored crop as a grid.
Keys: s scancar · o other · h hard negative · k skip · f/c switch view · u unreviewed only (crops)
      frames: ←/→ previous/next frame, Tab or ↑/↓ next box, click a box to select it
      crops:  arrows move
Labels are appended to labels.csv (source=human). A box whose crop was a near-duplicate labels the
crop it duplicates. Shows the detector confidence, and P(scancar) once a model is trained
(highest first, so likely positives come up first).
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from scancar.common import DATA, LABEL_VALUES, ROOT, append_labels, latest_model, read_index, read_labels

_scores: dict = {"model": None, "mtime": None, "s": {}}


def scores(crops: list[str]) -> dict[str, float]:
    """P(scancar) per crop from the newest model; cached until a newer model or new crops show up."""
    model = latest_model()
    if not model or not crops:
        return {}
    key = (str(model), model.stat().st_mtime)
    if (_scores["model"], _scores["mtime"]) != key:
        _scores.update(model=key[0], mtime=key[1], s={})
    todo = [c for c in crops if c not in _scores["s"]]
    if todo:
        from scancar.score import score_crops

        print(f"scoring {len(todo)} crops with {model}")
        _scores["s"].update(zip(todo, score_crops(model, [ROOT / c for c in todo])))
    return _scores["s"]


def boxes() -> list[dict]:
    labels = read_labels()
    rows = [r for r in read_index() if r.get("x1")]
    sc = scores(sorted({r["crop"] for r in rows if r["crop"]}))
    out = []
    for r in rows:
        crop = r["crop"] or r["dup_of"]  # a duplicate box shares the label of the crop it duplicates
        lab = labels.get(crop, {})
        out.append(dict(crop=crop, dup=not r["crop"], frame=r["frame"], track_id=r["track_id"], event_id=r["event_id"],
                        synthetic=r["synthetic"] == "1", box=[float(r[k]) for k in ("x1", "y1", "x2", "y2")],
                        conf=float(r["conf"]) if r["conf"] else None, score=sc.get(crop),
                        label=lab.get("label", ""), by=lab.get("source", "")))
    return out


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Scancar labels</title><style>
body{font:13px system-ui;margin:0;background:#111;color:#ddd}
header{position:sticky;top:0;background:#1b1b1b;padding:8px 12px;border-bottom:1px solid #333;z-index:1}
kbd{background:#333;border-radius:3px;padding:1px 5px}
button{background:#2a2a2a;color:#ddd;border:1px solid #444;border-radius:4px;padding:3px 10px;cursor:pointer}
button.on{background:#ddd;color:#111}
#st{color:#999}
#frames{padding:12px}
#frames svg{width:100%;max-height:calc(100vh - 110px);display:block;background:#000}
#frames .info{padding:6px 0;color:#999}
#frames rect{fill:transparent;stroke-width:3;cursor:pointer}
#frames rect.dup{stroke-dasharray:8 5}
#frames rect.sel{stroke-width:6}
#frames .tag{font:600 15px system-ui;paint-order:stroke;stroke:#000;stroke-width:4px;pointer-events:none}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;padding:12px}
.c{background:#1b1b1b;border:3px solid #333;border-radius:6px;overflow:hidden;cursor:pointer}
.c img{width:100%;height:150px;object-fit:contain;background:#000;display:block}
.c .m{padding:4px 6px;font-size:11px;color:#999;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.c.sel{outline:3px solid #fff}
.l{font-weight:600;font-size:12px}
.scancar{border-color:#e5484d}.scancar .l{color:#e5484d}
.other{border-color:#30a46c}.other .l{color:#30a46c}
.hard_neg{border-color:#f5a623}.hard_neg .l{color:#f5a623}
.skip{border-color:#666;opacity:.55}.skip .l{color:#888}
.auto .l::after{content:" (auto)";font-weight:400;color:#888}
</style></head><body>
<header><button id="bf">Frames <kbd>f</kbd></button> <button id="bc">Crops <kbd>c</kbd></button> &nbsp;
<kbd>s</kbd> scancar <kbd>o</kbd> other <kbd>h</kbd> hard neg <kbd>k</kbd> skip &nbsp;
<span id="help"></span> &nbsp; <span id="st"></span></header>
<div id="frames"></div><div id="grid"></div><script>
const COL={scancar:'#e5484d',other:'#30a46c',hard_neg:'#f5a623',skip:'#888','':'#4aa3ff'};
const KEYS={s:'scancar',o:'other',h:'hard_neg',k:'skip'};
let B=[],frames=[],fi=0,bi=0,mode='frames',crops=[],view=[],sel=0,onlyNew=false;
const pct=v=>Math.round(v*100)+'%';
const esc=s=>s.replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

async function load(){
  B=await (await fetch('/api/boxes')).json();
  const by={};B.forEach(b=>(by[b.frame]=by[b.frame]||[]).push(b));
  // frames with the most likely scan car first
  const rank=f=>Math.max(...by[f].map(b=>b.score??(b.label==='scancar'?1:0)));
  frames=Object.keys(by).sort((a,b)=>rank(b)-rank(a)||a.localeCompare(b)).map(f=>({f,boxes:by[f]}));
  const seen=new Set();crops=B.filter(b=>!b.dup&&!seen.has(b.crop)&&seen.add(b.crop));
  crops.sort((a,b)=>(b.score??-1)-(a.score??-1)||(b.label==='scancar')-(a.label==='scancar'));
  render();
}
function status(){
  const n=l=>crops.filter(i=>i.label===l).length,h=crops.filter(i=>i.by==='human').length;
  document.getElementById('st').textContent=`${crops.length} crops · ${n('scancar')} scancar · ${n('other')} other · ${n('hard_neg')} hard neg · ${n('skip')} skip · ${h} reviewed`;
}
function render(){
  document.getElementById('bf').className=mode==='frames'?'on':'';
  document.getElementById('bc').className=mode==='crops'?'on':'';
  document.getElementById('frames').style.display=mode==='frames'?'':'none';
  document.getElementById('grid').style.display=mode==='crops'?'':'none';
  document.getElementById('help').innerHTML=mode==='frames'
    ?'<kbd>←→</kbd> frame <kbd>Tab</kbd>/<kbd>↑↓</kbd> box'
    :'<kbd>arrows</kbd> move <kbd>u</kbd> unreviewed only';
  status(); mode==='frames'?renderFrame():renderGrid();
}
function renderFrame(){
  const F=frames[fi];if(!F)return;
  bi=Math.min(bi,F.boxes.length-1);
  const W=960,H=540;
  const r=F.boxes.map((b,k)=>{
    const [x1,y1,x2,y2]=b.box,c=COL[b.label];
    const txt=`${b.label||'?'}${b.by==='auto'?'*':''}`+(b.score!=null?` · scan ${pct(b.score)}`:'')+(b.conf!=null?` · car ${pct(b.conf)}`:'');
    const ty=y1>22?y1-6:y2+18;
    return `<rect data-k="${k}" class="${b.dup?'dup':''} ${k===bi?'sel':''}" x="${x1}" y="${y1}" width="${x2-x1}" height="${y2-y1}" stroke="${c}"/>
      <text class="tag" x="${Math.min(x1,W-260)}" y="${ty}" fill="${c}">${k===bi?'▶ ':''}${esc(txt)}</text>`;}).join('');
  const b=F.boxes[bi];
  document.getElementById('frames').innerHTML=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
    <image href="/frame?p=${encodeURIComponent(F.f)}" width="${W}" height="${H}" preserveAspectRatio="none"/>${r}</svg>
    <div class="info">frame ${fi+1}/${frames.length} · ${esc(F.f.split('/').pop())} · ${esc(F.boxes[0].event_id)} &nbsp;|&nbsp;
    selected: ${esc(b.track_id)}${b.dup?' (near-duplicate of '+esc(b.crop.split('/').pop())+')':''} &nbsp;|&nbsp; * = auto label, dashed = duplicate crop</div>`;
}
function renderGrid(){
  view=onlyNew?crops.filter(i=>i.by!=='human'):crops;
  sel=Math.min(sel,Math.max(0,view.length-1));
  document.getElementById('grid').innerHTML=view.map((i,k)=>`<div class="c ${i.label} ${i.by==='auto'?'auto':''} ${k===sel?'sel':''}" data-k="${k}">
    <img loading="lazy" src="/img?p=${encodeURIComponent(i.crop)}"><div class="m"><span class="l">${i.label||'unlabelled'}</span>
    ${i.score!=null?' · scan '+pct(i.score):''}${i.conf!=null?' · car '+pct(i.conf):''}${i.synthetic?' · synth':''}<br>${esc(i.event_id)} · ${esc(i.track_id)}</div></div>`).join('');
  document.querySelector('.c.sel')?.scrollIntoView({block:'nearest'});
}
async function setLabel(crop,lab){
  await fetch('/api/label',{method:'POST',body:JSON.stringify({crop,label:lab})});
  B.forEach(b=>{if(b.crop===crop){b.label=lab;b.by='human'}});
}
document.getElementById('bf').onclick=()=>{mode='frames';render()};
document.getElementById('bc').onclick=()=>{mode='crops';render()};
document.getElementById('frames').onclick=e=>{const k=e.target.dataset?.k;if(k!==undefined){bi=+k;render()}};
document.getElementById('grid').onclick=e=>{const c=e.target.closest('.c');if(c){sel=+c.dataset.k;render()}};
const cols=()=>getComputedStyle(document.getElementById('grid')).gridTemplateColumns.split(' ').length;
document.onkeydown=async e=>{
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  if(e.key==='f'||e.key==='c'){mode=e.key==='f'?'frames':'crops';render();return}
  if(mode==='frames'){
    const F=frames[fi],n=F.boxes.length;
    if(e.key==='ArrowRight'||e.key==='ArrowLeft'){fi=(fi+(e.key==='ArrowRight'?1:-1)+frames.length)%frames.length;bi=0;render();e.preventDefault();return}
    if(e.key==='Tab'||e.key==='ArrowDown'||e.key==='ArrowUp'){bi=(bi+((e.key==='ArrowUp'||e.shiftKey)?-1:1)+n)%n;render();e.preventDefault();return}
    const lab=KEYS[e.key];if(!lab)return;
    await setLabel(F.boxes[bi].crop,lab);
    if(bi<n-1)bi++;render();return;
  }
  const mv={ArrowRight:1,ArrowLeft:-1,ArrowDown:cols(),ArrowUp:-cols()}[e.key];
  if(mv!==undefined){sel=Math.max(0,Math.min(view.length-1,sel+mv));render();e.preventDefault();return}
  if(e.key==='u'){onlyNew=!onlyNew;render();return}
  const lab=KEYS[e.key],it=view[sel];if(!lab||!it)return;
  await setLabel(it.crop,lab);if(!onlyNew)sel=Math.min(view.length-1,sel+1);render();
};
load();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query).get("p", [""])[0]
        if u.path == "/":
            return self._send(PAGE.encode(), "text/html; charset=utf-8")
        if u.path == "/api/boxes":
            return self._send(json.dumps(boxes()).encode(), "application/json")
        if u.path == "/img":
            p = (ROOT / q).resolve()
            if p.is_relative_to(DATA.resolve()) and p.suffix == ".jpg" and p.exists():
                return self._send(p.read_bytes(), "image/jpeg")
        if u.path == "/frame" and q in {r["frame"] for r in read_index()}:  # only frames we indexed
            p = ROOT / q
            if os.path.exists(p):
                return self._send(p.read_bytes(), "image/jpeg")
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/label":
            return self._send(b"not found", "text/plain", 404)
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        row = next((r for r in read_index() if r["crop"] and r["crop"] == d.get("crop")), None)
        if not row or d.get("label") not in LABEL_VALUES:
            return self._send(b"bad request", "text/plain", 400)
        append_labels([dict(crop=row["crop"], track_id=row["track_id"], event_id=row["event_id"],
                            label=d["label"], source="human")])
        self._send(b"ok", "text/plain")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
