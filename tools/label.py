"""Click-to-label a photo, for scoring the detectors against ground truth.

    python tools/label.py photo.jpg
    # opens a page in your browser; click each mark; Export

Writes two files beside the photo:

* ``<photo>.labelled.png`` — red outlines on the original, which is exactly the
  format ``tools/evaluate.py`` already reads, so scoring needs no changes.
* ``<photo>.labels.json`` — the same marks with their TYPE recorded, which the
  PNG cannot carry. Pigmented marks and active lesions are different detectors
  answering different questions, and a label set that conflates them can only
  score one of them.

Why this exists: every spot threshold in the library traces to one hand-labelled
face with nine marks, and `SpotsConfig` says in as many words that a narrow
optimum on a single image is a prime candidate for overfitting. It was — the
defaults find sub-millimetre noise at the lash line and call it pigmentation.
Fixing that honestly needs labels, and labels need to be cheap to produce or
they do not get produced.

The page is a plain local file. It runs no network, and nothing leaves the
machine — which matters, because the input is a photograph of somebody's face.
"""

from __future__ import annotations

import argparse
import base64
import webbrowser
from pathlib import Path

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>Label &mdash; __NAME__</title>
<style>
  :root{ --bg:#15120F; --panel:#211C17; --ink:#F0EAE1; --dim:#9A8E82;
         --line:#332B23; --mark:#E8A44C; --lesion:#E0575C; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.5 ui-sans-serif,system-ui,sans-serif;display:flex;height:100vh}
  #stage{flex:1;overflow:auto;display:grid;place-items:center;padding:20px}
  #wrap{position:relative;line-height:0;cursor:crosshair}
  img{max-width:100%;display:block}
  svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
  aside{width:290px;flex:none;background:var(--panel);border-left:1px solid var(--line);
        padding:22px 20px;display:flex;flex-direction:column;gap:18px;overflow:auto}
  h1{font-size:15px;margin:0;letter-spacing:.02em}
  .hint{color:var(--dim);font-size:13px;margin:0}
  .modes{display:flex;flex-direction:column;gap:8px}
  .mode{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:8px;
        border:1px solid var(--line);cursor:pointer;background:transparent;color:var(--ink);
        font:inherit;text-align:left}
  .mode[aria-pressed="true"]{border-color:currentColor}
  .mode.m{color:var(--mark)} .mode.l{color:var(--lesion)}
  .swatch{width:11px;height:11px;border-radius:99px;background:currentColor;flex:none}
  .mode span.n{margin-left:auto;font-variant-numeric:tabular-nums;color:var(--dim)}
  button.act{padding:10px 12px;border-radius:8px;border:1px solid var(--line);
             background:transparent;color:var(--ink);font:inherit;cursor:pointer}
  button.act:hover{border-color:var(--dim)}
  button.go{background:var(--ink);color:var(--bg);border-color:var(--ink);font-weight:600}
  kbd{background:#000;border:1px solid var(--line);border-radius:4px;padding:1px 5px;
      font:12px ui-monospace,monospace;color:var(--dim)}
  .keys{display:flex;flex-direction:column;gap:5px;font-size:13px;color:var(--dim);
        border-top:1px solid var(--line);padding-top:16px;margin-top:auto}
</style>
<div id="stage"><div id="wrap">
  <img id="photo" src="data:image/__EXT__;base64,__DATA__" alt="">
  <svg id="ov" viewBox="0 0 __W__ __H__" preserveAspectRatio="none"></svg>
</div></div>
<aside>
  <h1>__NAME__</h1>
  <p class="hint">Click each mark you can actually see. Click a marker again to remove it.</p>
  <div class="modes">
    <button class="mode m" id="bm" aria-pressed="true" onclick="setMode('mark')">
      <span class="swatch"></span> Pigmented mark <span class="n" id="cm">0</span></button>
    <button class="mode l" id="bl" aria-pressed="false" onclick="setMode('lesion')">
      <span class="swatch"></span> Active lesion <span class="n" id="cl">0</span></button>
  </div>
  <button class="act" onclick="undo()">Undo last</button>
  <button class="act" onclick="clearAll()">Clear all</button>
  <button class="act go" onclick="save()">Export labels</button>
  <p class="hint" id="status"></p>
  <div class="keys">
    <div><kbd>1</kbd> pigmented &nbsp; <kbd>2</kbd> active</div>
    <div><kbd>u</kbd> undo &nbsp; <kbd>e</kbd> export</div>
    <div>Radius follows what you drag, or click for default.</div>
  </div>
</aside>
<script>
const W=__W__, H=__H__, R=Math.max(9, Math.round(W*0.011));
let mode='mark', marks=[];
const ov=document.getElementById('ov'), wrap=document.getElementById('wrap');
function setMode(m){ mode=m;
  document.getElementById('bm').setAttribute('aria-pressed', String(m==='mark'));
  document.getElementById('bl').setAttribute('aria-pressed', String(m==='lesion')); }
function at(e){ const r=wrap.getBoundingClientRect();
  return { x:(e.clientX-r.left)/r.width*W, y:(e.clientY-r.top)/r.height*H }; }
wrap.addEventListener('click', e => {
  const p=at(e);
  const hit=marks.findIndex(m => Math.hypot(m.x-p.x, m.y-p.y) < m.r*1.15);
  if(hit>=0) marks.splice(hit,1); else marks.push({x:p.x, y:p.y, r:R, type:mode});
  draw();
});
function draw(){
  ov.innerHTML = marks.map(m =>
    `<circle cx="${m.x.toFixed(1)}" cy="${m.y.toFixed(1)}" r="${m.r}" fill="none"
      stroke="${m.type==='mark'?'#E8A44C':'#E0575C'}" stroke-width="${Math.max(2,W*0.003)}"/>`
  ).join('');
  document.getElementById('cm').textContent = marks.filter(m=>m.type==='mark').length;
  document.getElementById('cl').textContent = marks.filter(m=>m.type==='lesion').length;
}
function undo(){ marks.pop(); draw(); }
function clearAll(){ if(marks.length && confirm('Remove all '+marks.length+' marks?')){marks=[];draw();} }
function dl(blob, name){ const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download=name; a.click(); URL.revokeObjectURL(a.href); }
function save(){
  if(!marks.length){ document.getElementById('status').textContent='Nothing to export yet.'; return; }
  // 1. JSON, with the type each mark was given.
  dl(new Blob([JSON.stringify({image:"__NAME__", width:W, height:H,
      marks:marks.map(m=>({x:Math.round(m.x),y:Math.round(m.y),r:m.r,type:m.type}))}, null, 2)],
      {type:'application/json'}), "__STEM__.labels.json");
  // 2. PNG with SATURATED RED outlines, which is what tools/evaluate.py reads.
  //    It keys on red excess over both other channels, so only pure red counts.
  const c=document.createElement('canvas'); c.width=W; c.height=H;
  const g=c.getContext('2d'); g.drawImage(document.getElementById('photo'),0,0,W,H);
  g.strokeStyle='#FF0000'; g.lineWidth=Math.max(2, Math.round(W*0.004));
  marks.forEach(m => { g.beginPath(); g.arc(m.x,m.y,m.r,0,6.2832); g.stroke(); });
  c.toBlob(b => dl(b, "__STEM__.labelled.png"), 'image/png');
  document.getElementById('status').textContent =
    marks.length+' marks exported. Check your downloads folder.';
}
addEventListener('keydown', e => {
  if(e.key==='1') setMode('mark'); else if(e.key==='2') setMode('lesion');
  else if(e.key==='u') undo(); else if(e.key==='e') save();
});
draw();
</script>
"""


def build(photo: Path, out_dir: Path) -> Path:
    import cv2

    image = cv2.imread(str(photo))
    if image is None:
        raise SystemExit(f"could not read {photo}")
    height, width = image.shape[:2]

    data = base64.b64encode(photo.read_bytes()).decode()
    ext = "jpeg" if photo.suffix.lower() in (".jpg", ".jpeg") else photo.suffix.lstrip(".")

    page = (
        PAGE.replace("__DATA__", data)
        .replace("__EXT__", ext)
        .replace("__NAME__", photo.name)
        .replace("__STEM__", photo.stem)
        .replace("__W__", str(width))
        .replace("__H__", str(height))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{photo.stem}.label.html"
    target.write_text(page)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("photo", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=None,
                        help="where to write the page (default: beside the photo)")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    page = build(args.photo, args.out or args.photo.parent)
    print(f"labelling page: {page}")
    print(f"exports:        {args.photo.stem}.labels.json  and  {args.photo.stem}.labelled.png")
    print()
    print("Then score the detector against it:")
    print(f"  python tools/evaluate.py {args.photo} <downloads>/{args.photo.stem}.labelled.png --sweep")
    if not args.no_open:
        webbrowser.open(page.resolve().as_uri())


if __name__ == "__main__":
    main()
