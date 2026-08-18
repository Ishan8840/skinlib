"""Label a folder of faces in one continuous pass: paint regions, add notes, next.

    python tools/label_batch.py ~/Pictures/tolabel -o labels/

Two modes, and the first is usually the right one.

**Region** (default). Drag to paint over an affected area. A face with forty
marks is two or three strokes rather than forty clicks, which is the difference
between labelling ten faces and two hundred. Regions are what the pipeline can
actually deliver — per-spot detection tops out near F1 0.42 and does not improve
with tuning, whereas regional detection density agrees with hand labels at
r = +0.56. Painted masks also train a segmentation model directly, which is a
better match for "roughly where" than a box detector.

**Point.** Click individual marks. Slower, and only needed to score per-spot
precision and recall.

Each region carries a **count and a note** — "about 8 here", "old scarring, not
active" — because a painted area alone cannot say how dense it is, and the
person labelling usually knows something the mask does not record.

Saves per image as you advance, so a closed tab costs you only the current one,
and reopening skips anything already labelled. Runs as a local file with the
images inlined: nothing is uploaded, which matters for a folder of photographs
of people.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>Labelling &mdash; __N__ images</title>
<style>
 :root{--bg:#14110E;--panel:#1E1A15;--ink:#F0EAE1;--dim:#9A8E82;--line:#332B23;
       --mark:#E8A44C;--lesion:#E0575C;--ok:#79C2A0;--other:#6FA8D0;}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);height:100vh;display:flex;
      font:15px/1.5 ui-sans-serif,system-ui,sans-serif;overflow:hidden}
 #stage{flex:1;display:grid;place-items:center;padding:16px;overflow:auto;position:relative}
 #wrap{position:relative;line-height:0;max-height:100%;touch-action:none}
 #photo{max-width:100%;max-height:calc(100vh - 32px);display:block}
 #paint{position:absolute;inset:0;width:100%;height:100%;opacity:.42;pointer-events:none}
 #ov{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
 #hit{position:absolute;inset:0;cursor:crosshair}
 aside{width:310px;flex:none;background:var(--panel);border-left:1px solid var(--line);
       padding:18px;display:flex;flex-direction:column;gap:14px;overflow:auto}
 h1{font-size:13.5px;margin:0;word-break:break-all}
 .prog{font:12px ui-monospace,monospace;color:var(--dim)}
 .bar{height:4px;background:#000;border-radius:99px;overflow:hidden}
 .bar span{display:block;height:100%;background:var(--ok);transition:width .2s}
 .hint{color:var(--dim);font-size:12.5px;margin:0}
 .seg{display:flex;gap:6px}
 .seg button{flex:1;padding:7px;border-radius:7px;border:1px solid var(--line);
   background:transparent;color:var(--dim);font:inherit;font-size:13px;cursor:pointer}
 .seg button[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
 .kind{display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:8px;
       border:1px solid var(--line);cursor:pointer;background:transparent;color:var(--ink);
       font:inherit;text-align:left;width:100%;font-size:14px}
 .kind[aria-pressed="true"]{border-color:currentColor}
 .kind.m{color:var(--mark)} .kind.l{color:var(--lesion)} .kind.o{color:var(--other)}
 .sw{width:11px;height:11px;border-radius:3px;background:currentColor;flex:none}
 label.fld{display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--dim)}
 input,textarea{background:#000;border:1px solid var(--line);border-radius:6px;
   color:var(--ink);font:inherit;font-size:14px;padding:7px 9px;width:100%}
 textarea{resize:vertical;min-height:52px}
 input[type=range]{padding:0}
 button.act{padding:9px 11px;border-radius:8px;border:1px solid var(--line);
   background:transparent;color:var(--ink);font:inherit;cursor:pointer;width:100%}
 button.act:hover{border-color:var(--dim)}
 button.go{background:var(--ok);color:#0B0907;border-color:var(--ok);font-weight:600}
 .row{display:flex;gap:7px}
 .tally{font:12px ui-monospace,monospace;color:var(--dim);display:flex;gap:12px;flex-wrap:wrap}
 .keys{margin-top:auto;border-top:1px solid var(--line);padding-top:12px;
   display:flex;flex-direction:column;gap:4px;font-size:12px;color:var(--dim)}
 kbd{background:#000;border:1px solid var(--line);border-radius:4px;padding:1px 5px;
   font:11px ui-monospace,monospace}
 #done{position:absolute;inset:0;display:none;place-items:center;background:var(--bg);text-align:center}
 .saved{color:var(--ok)}
</style>
<div id="stage">
  <div id="wrap">
    <img id="photo" alt="">
    <canvas id="paint"></canvas>
    <svg id="ov" preserveAspectRatio="none"></svg>
    <div id="hit"></div>
  </div>
  <div id="done"><div><h2>All done</h2>
    <p class="hint">Every image labelled. JSON files are in your downloads folder.</p></div></div>
</div>
<aside>
  <h1 id="name"></h1>
  <div class="prog"><span id="pos"></span> &middot; <span id="left"></span> left</div>
  <div class="bar"><span id="fill" style="width:0%"></span></div>

  <div class="seg" role="group" aria-label="Mode">
    <button id="mRegion" aria-pressed="true" onclick="setTool('region')">Paint region</button>
    <button id="mPoint" aria-pressed="false" onclick="setTool('point')">Click spots</button>
  </div>

  <button class="kind m" id="kMark" aria-pressed="true" onclick="setKind('mark')">
    <span class="sw"></span> Pigmented / dark marks</button>
  <button class="kind l" id="kLesion" aria-pressed="false" onclick="setKind('lesion')">
    <span class="sw"></span> Active / inflamed</button>
  <button class="kind o" id="kOther" aria-pressed="false" onclick="setKind('other')">
    <span class="sw"></span> Other (texture, pores)</button>

  <label class="fld" id="brushBox">Brush size <span id="bsz" class="prog"></span>
    <input type="range" id="brush" min="8" max="120" value="34" oninput="onBrush()"></label>

  <label class="fld">Roughly how many in what you painted (optional)
    <input type="number" id="count" min="0" placeholder="e.g. 8"></label>
  <label class="fld">Note (optional)
    <textarea id="note" placeholder="old scarring, not active"></textarea></label>

  <button class="act" onclick="commit()">Add this area &crarr;</button>
  <div class="tally" id="tally"></div>

  <div class="row">
    <button class="act" onclick="undo()">Undo</button>
    <button class="act" onclick="clearAll()">Clear</button>
  </div>
  <button class="act go" onclick="next()">Save &amp; next &rarr;</button>
  <button class="act" onclick="skip()">Skip (no face / unusable)</button>
  <p class="hint" id="status"></p>

  <div class="keys">
    <div><kbd>1</kbd> marks &nbsp;<kbd>2</kbd> active &nbsp;<kbd>3</kbd> other</div>
    <div><kbd>r</kbd> paint &nbsp;<kbd>p</kbd> points &nbsp;<kbd>[</kbd> <kbd>]</kbd> brush</div>
    <div><kbd>enter</kbd> add area &nbsp;<kbd>space</kbd> save &amp; next</div>
    <div><kbd>u</kbd> undo &nbsp;<kbd>s</kbd> skip</div>
  </div>
</aside>
<script>
const IMAGES = __IMAGES__;
const COLOUR = {mark:'#E8A44C', lesion:'#E0575C', other:'#6FA8D0'};
let i = 0, tool = 'region', kind = 'mark', saved = 0;
let areas = [], points = [], drawing = false, strokes = [], current = null;
const el = id => document.getElementById(id);
const photo = el('photo'), paint = el('paint'), ov = el('ov'), hit = el('hit'), wrap = el('wrap');
const ctx = paint.getContext('2d');

function setTool(t){ tool = t;
  el('mRegion').setAttribute('aria-pressed', String(t==='region'));
  el('mPoint').setAttribute('aria-pressed', String(t==='point'));
  el('brushBox').style.display = t==='region' ? '' : 'none'; }
function setKind(k){ kind = k;
  el('kMark').setAttribute('aria-pressed', String(k==='mark'));
  el('kLesion').setAttribute('aria-pressed', String(k==='lesion'));
  el('kOther').setAttribute('aria-pressed', String(k==='other')); }
function onBrush(){ el('bsz').textContent = el('brush').value + 'px'; }

function at(e){ const r = hit.getBoundingClientRect(), im = IMAGES[i];
  return { x:(e.clientX-r.left)/r.width*im.w, y:(e.clientY-r.top)/r.height*im.h }; }

hit.addEventListener('pointerdown', e => {
  if(i >= IMAGES.length) return;
  const p = at(e);
  if(tool === 'point'){
    const R = Math.max(9, Math.round(IMAGES[i].w*0.011));
    const idx = points.findIndex(m => Math.hypot(m.x-p.x, m.y-p.y) < m.r*1.15);
    if(idx>=0) points.splice(idx,1); else points.push({x:p.x,y:p.y,r:R,type:kind});
    draw(); return;
  }
  drawing = true; hit.setPointerCapture(e.pointerId);
  current = { type: kind, brush: +el('brush').value, path: [[p.x,p.y]] };
  draw();
});
hit.addEventListener('pointermove', e => {
  if(!drawing || tool!=='region') return;
  const p = at(e); current.path.push([p.x,p.y]); draw();
});
hit.addEventListener('pointerup', () => {
  if(!drawing) return;
  drawing = false;
  if(current && current.path.length) strokes.push(current);
  current = null; draw();
});

function commit(){
  if(!strokes.length){ flash('Paint an area first.'); return; }
  areas.push({ type: strokes[0].type, strokes: strokes,
    count: el('count').value ? +el('count').value : null,
    note: el('note').value.trim() || null });
  strokes = []; el('count').value=''; el('note').value=''; draw();
}
function undo(){
  if(strokes.length) strokes.pop();
  else if(tool==='point' && points.length) points.pop();
  else if(areas.length) areas.pop();
  draw();
}
function clearAll(){ areas=[]; strokes=[]; points=[]; draw(); }

function draw(){
  if(i >= IMAGES.length) return;
  const im = IMAGES[i];
  paint.width = im.w; paint.height = im.h;
  ctx.clearRect(0,0,im.w,im.h);
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  const all = areas.flatMap(a => a.strokes).concat(strokes, current ? [current] : []);
  for(const s of all){
    ctx.strokeStyle = COLOUR[s.type]; ctx.lineWidth = s.brush;
    ctx.beginPath();
    s.path.forEach(([x,y],n) => n ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
    if(s.path.length===1) ctx.lineTo(s.path[0][0]+0.1, s.path[0][1]);
    ctx.stroke();
  }
  ov.setAttribute('viewBox', `0 0 ${im.w} ${im.h}`);
  ov.innerHTML = points.map(m =>
    `<circle cx="${m.x.toFixed(1)}" cy="${m.y.toFixed(1)}" r="${m.r}" fill="none"
      stroke="${COLOUR[m.type]}" stroke-width="${Math.max(2,im.w*0.003)}"/>`).join('');
  const by = t => areas.filter(a=>a.type===t).length;
  el('tally').innerHTML =
    `<span>${by('mark')} mark areas</span><span>${by('lesion')} active</span>` +
    `<span>${by('other')} other</span>` + (points.length?`<span>${points.length} points</span>`:'') +
    (strokes.length?`<span style="color:var(--ok)">unsaved stroke</span>`:'');
}
function flash(t){ el('status').textContent = t; setTimeout(()=>el('status').textContent='',2200); }

function load(){
  if(i >= IMAGES.length){ el('done').style.display='grid'; wrap.style.display='none'; return; }
  const im = IMAGES[i];
  areas=[]; strokes=[]; points=[]; current=null;
  photo.src = 'data:image/'+im.ext+';base64,'+im.data;
  el('name').textContent = im.name;
  el('pos').textContent = `${i+1} / ${IMAGES.length}`;
  el('left').textContent = IMAGES.length-i-1;
  el('fill').style.width = (100*i/IMAGES.length)+'%';
  el('count').value=''; el('note').value='';
  el('status').innerHTML = saved ? `<span class="saved">${saved} saved</span>` : '';
  draw();
}
function save(){
  if(strokes.length) commit();
  const im = IMAGES[i];
  const payload = { image: im.name, width: im.w, height: im.h,
    // Painted areas: each is a set of strokes with a brush width, so the mask
    // can be rebuilt exactly. count/note carry what the mask cannot.
    areas: areas.map(a => ({ type:a.type, count:a.count, note:a.note,
      strokes: a.strokes.map(s => ({ brush:s.brush,
        path: s.path.map(([x,y]) => [Math.round(x), Math.round(y)]) })) })),
    // Point labels stay under `marks`, unchanged, so evaluate.py still reads them.
    marks: points.map(m => ({x:Math.round(m.x), y:Math.round(m.y), r:m.r, type:m.type})) };
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(payload,null,1)],{type:'application/json'}));
  a.download = im.stem + '.labels.json'; a.click(); URL.revokeObjectURL(a.href); saved++;
}
function next(){ if(i>=IMAGES.length) return; save(); i++; load(); }
function skip(){ if(i>=IMAGES.length) return; i++; load(); }
addEventListener('keydown', e => {
  const typing = ['INPUT','TEXTAREA'].includes(document.activeElement.tagName);
  if(e.key==='Enter' && typing){ e.preventDefault(); commit(); return; }
  if(typing && e.key!=='Escape') return;
  if(e.key==='1') setKind('mark'); else if(e.key==='2') setKind('lesion');
  else if(e.key==='3') setKind('other');
  else if(e.key==='r') setTool('region'); else if(e.key==='p') setTool('point');
  else if(e.key==='[' ){ el('brush').value = Math.max(8, +el('brush').value-6); onBrush(); }
  else if(e.key===']' ){ el('brush').value = Math.min(120, +el('brush').value+6); onBrush(); }
  else if(e.key==='u') undo(); else if(e.key==='s') skip();
  else if(e.key==='Enter') commit();
  else if(e.key===' ' || e.key==='ArrowRight'){ e.preventDefault(); next(); }
});
onBrush(); load();
</script>
"""


def build(images: list[Path], out: Path) -> Path:
    import cv2

    entries = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        entries.append({
            "name": path.name, "stem": path.stem,
            "ext": "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else path.suffix.lstrip("."),
            "w": width, "h": height,
            "data": base64.b64encode(path.read_bytes()).decode(),
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace("__IMAGES__", json.dumps(entries)).replace("__N__", str(len(entries))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("-o", "--labels", type=Path, default=Path("labels"))
    parser.add_argument("--limit", type=int, default=60,
                        help="images per page; each is inlined, so run several pages "
                             "rather than one enormous file")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    done = set()
    for folder in (args.labels, Path.home() / "Downloads"):
        if folder.exists():
            done |= {f.name.replace(".labels.json", "") for f in folder.glob("*.labels.json")}

    images = sorted(f for f in args.folder.iterdir()
                    if f.suffix.lower() in SUFFIXES and f.stem not in done)
    if not images:
        raise SystemExit(f"nothing left to label in {args.folder} ({len(done)} already done)")

    batch = images[: args.limit]
    target = args.out or (args.folder / "label_batch.html")
    build(batch, target)
    print(f"{len(images)} unlabelled, {len(done)} already done")
    print(f"this page: {len(batch)} images -> {target}")
    print("\nDrag to paint an area, add a count/note, ENTER to add it, SPACE for the next image.")
    print(f"When finished:  mv ~/Downloads/*.labels.json {args.labels}/")
    if not args.no_open:
        import webbrowser
        webbrowser.open(target.resolve().as_uri())


if __name__ == "__main__":
    main()
