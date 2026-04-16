"""
Annotation tool for curating scene cuts.

For each auto-detected cut, shows frames from both sides and asks:
  - Approve (good cut, open play → set piece close-up)
  - Reject  (bad cut, wrong direction, or not the right type)
  - Skip    (unsure)

Approved cuts are saved to annotations.json. A separate script
(sample_pairs.py) then samples training pairs from approved cuts.

Usage:
    conda activate base
    python annotate.py [--port 5050]

SSH tunnel:
    ssh -L 5050:localhost:5050 <your-server>
Then open: http://localhost:5050
"""

import argparse
import json
from pathlib import Path
from flask import Flask, jsonify, request, send_file, render_template_string

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FRAMES_ROOT    = Path(__file__).parent / "frames"
CUTS_JSON      = Path(__file__).parent / "cuts.json"
ANNOT_JSON     = Path(__file__).parent / "annotations.json"
CONTEXT_FRAMES = 8   # frames shown on each side

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load cuts
# ---------------------------------------------------------------------------
with open(CUTS_JSON) as f:
    cuts_map: dict[str, list[int]] = json.load(f)

all_cuts: list[tuple[str, int, int]] = []
for half_key, cut_list in sorted(cuts_map.items()):
    for ci, frame_no in enumerate(cut_list):
        all_cuts.append((half_key, ci, frame_no))


def load_annotations() -> dict:
    if ANNOT_JSON.exists():
        with open(ANNOT_JSON) as f:
            return json.load(f)
    return {}


def save_annotations(data: dict):
    with open(ANNOT_JSON, "w") as f:
        json.dump(data, f, indent=2)


def annot_key(half_key: str, cut_idx: int) -> str:
    return f"{half_key}::{cut_idx}"


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cut Annotator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: monospace; background: #111; color: #eee; padding: 16px; }
  h1   { font-size: 1.1rem; margin-bottom: 6px; }
  .info     { color: #aaa; font-size: 0.85rem; margin-bottom: 4px; }
  .progress { font-size: 0.85rem; color: #7bf; margin-bottom: 12px; }
  .hint     { font-size: 0.75rem; color: #555; margin-bottom: 12px; }

  .scenes { display: flex; gap: 10px; }
  .scene  { flex: 1; min-width: 0; }
  .scene-label {
    font-size: 0.8rem; font-weight: bold; margin-bottom: 6px;
    padding: 4px 8px; border-radius: 4px;
  }
  .scene1-label { background: #1a3a5c; color: #7bf; }
  .scene2-label { background: #3a1a1a; color: #f87; }

  .frames-grid { display: flex; flex-wrap: wrap; gap: 4px; }
  .frame-cell  { border-radius: 3px; overflow: hidden; }
  .frame-cell img { display: block; width: 160px; height: 90px; object-fit: cover; }

  .divider {
    width: 4px; background: #333; border-radius: 3px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    writing-mode: vertical-lr; color: #555; font-size: 0.7rem;
    letter-spacing: 3px; padding: 6px 2px; user-select: none;
  }

  /* big action buttons */
  .actions { margin-top: 16px; display: flex; gap: 12px; align-items: center; }
  .btn {
    padding: 10px 28px; border: none; border-radius: 6px;
    cursor: pointer; font-family: monospace; font-size: 1rem; font-weight: bold;
    transition: filter 0.1s;
  }
  .btn:hover    { filter: brightness(1.2); }
  .btn:disabled { opacity: 0.35; cursor: default; filter: none; }
  .btn-approve  { background: #27ae60; color: #fff; }
  .btn-reject   { background: #c0392b; color: #fff; }
  .btn-skip     { background: #444;    color: #aaa; }
  .btn-nav      { background: #222; color: #888; border: 1px solid #333; font-size: 0.85rem; padding: 8px 16px; }
  .status { font-size: 0.85rem; margin-left: 8px; color: #666; }

  /* cut list */
  .cut-list { margin-top: 18px; border-top: 1px solid #222; padding-top: 12px; }
  .cut-list h2 { font-size: 0.85rem; color: #555; margin-bottom: 8px; }
  .cut-list-inner {
    display: flex; flex-wrap: wrap; gap: 4px;
    max-height: 150px; overflow-y: auto;
  }
  .pill {
    padding: 2px 8px; border-radius: 20px; font-size: 0.7rem;
    cursor: pointer; background: #1a1a1a; border: 1px solid #2a2a2a; color: #666;
    text-decoration: none;
  }
  .pill:hover     { border-color: #555; color: #bbb; }
  .pill.approved  { border-color: #27ae60; color: #2ecc71; }
  .pill.rejected  { border-color: #c0392b; color: #e74c3c; }
  .pill.skipped   { border-color: #333;    color: #444; }
  .pill.active    { outline: 2px solid #7bf; }
</style>
</head>
<body>

<h1>Cut Annotator</h1>
<div class="info"     id="info">Loading…</div>
<div class="progress" id="progress"></div>
<div class="hint"><b>A</b> = approve &nbsp;·&nbsp; <b>R</b> = reject &nbsp;·&nbsp; <b>S</b> = skip &nbsp;·&nbsp; <b>[ ]</b> or <b>← →</b> = prev / next</div>

<div class="scenes">
  <div class="scene">
    <div class="scene-label scene1-label">Before cut</div>
    <div class="frames-grid" id="grid1"></div>
  </div>
  <div class="divider">CUT</div>
  <div class="scene">
    <div class="scene-label scene2-label">After cut</div>
    <div class="frames-grid" id="grid2"></div>
  </div>
</div>

<div class="actions">
  <button class="btn btn-nav"     id="btnPrev"    onclick="navigate(-1)">← Prev</button>
  <button class="btn btn-nav"     id="btnNext"    onclick="navigate(+1)">Next →</button>
  <button class="btn btn-approve" id="btnApprove" onclick="decide('approved')">✓ Approve</button>
  <button class="btn btn-reject"  id="btnReject"  onclick="decide('rejected')">✗ Reject</button>
  <button class="btn btn-skip"    id="btnSkip"    onclick="decide('skipped')">Skip</button>
  <span class="status" id="msg"></span>
</div>

<div class="cut-list">
  <h2>All cuts</h2>
  <div class="cut-list-inner" id="cutList"></div>
</div>

<script>
let state       = null;
let allCuts     = [];
let annotations = {};

async function init() {
  const meta  = await fetch('/api/meta').then(r => r.json());
  allCuts     = meta.cuts;
  annotations = meta.annotations;
  renderPills();
  const first = allCuts.findIndex(c => !annotations[c.key]);
  await loadCut(first >= 0 ? first : 0);
}

function renderPills() {
  const el = document.getElementById('cutList');
  el.innerHTML = '';
  allCuts.forEach((c, i) => {
    const a   = document.createElement('a');
    const ann = annotations[c.key];
    a.className   = 'pill' + (ann ? ' ' + ann.decision : '');
    a.id          = `pill-${i}`;
    a.href        = '#';
    a.textContent = `${c.half_key.split('/').slice(-2).join('/')} #${c.cut_idx}`;
    a.title       = `frame ${c.frame_no}`;
    a.onclick     = e => { e.preventDefault(); loadCut(i); };
    el.appendChild(a);
  });
}

async function loadCut(idx) {
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
  const pill = document.getElementById(`pill-${idx}`);
  if (pill) { pill.classList.add('active'); pill.scrollIntoView({block:'nearest'}); }

  document.getElementById('info').textContent = 'Loading…';
  state = await fetch(`/api/cut/${idx}`).then(r => r.json());
  state.global_idx = idx;

  const done = Object.values(annotations).filter(v => v.decision === 'approved').length;
  const rej  = Object.values(annotations).filter(v => v.decision === 'rejected').length;
  const skip = Object.values(annotations).filter(v => v.decision === 'skipped').length;
  const rem  = allCuts.length - done - rej - skip;

  document.getElementById('info').textContent =
    `Cut ${idx + 1} / ${allCuts.length}  —  ${state.half_key}  |  frame ${state.frame_no}`;
  document.getElementById('progress').textContent =
    `${done} approved  ·  ${rej} rejected  ·  ${skip} skipped  ·  ${rem} remaining`;

  renderGrid('grid1', state.scene1_frames);
  renderGrid('grid2', state.scene2_frames);

  // Show current decision if already annotated
  const ann = annotations[state.key];
  document.getElementById('msg').textContent = ann ? `[${ann.decision}]` : '';

  document.getElementById('btnPrev').disabled = idx === 0;
  document.getElementById('btnNext').disabled = idx === allCuts.length - 1;
}

function renderGrid(gridId, frames) {
  const grid = document.getElementById(gridId);
  grid.innerHTML = '';
  frames.forEach(frameNo => {
    const cell   = document.createElement('div');
    cell.className = 'frame-cell';
    const padded   = String(frameNo).padStart(6, '0');
    cell.innerHTML = `<img src="/frame/${encodeURIComponent(state.half_key)}/${padded}.jpg" loading="lazy">`;
    grid.appendChild(cell);
  });
}

async function decide(decision) {
  const payload = {
    key:      state.key,
    half_key: state.half_key,
    cut_frame: state.frame_no,
    decision,
  };
  await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload),
  });
  annotations[state.key] = payload;
  renderPills();
  document.getElementById('msg').textContent = `[${decision}]`;

  // Auto-advance to next unannotated
  const next = allCuts.findIndex((c, i) => i > state.global_idx && !annotations[c.key]);
  if (next >= 0) await loadCut(next);
}

async function navigate(delta) {
  const next = state.global_idx + delta;
  if (next >= 0 && next < allCuts.length) await loadCut(next);
}

document.addEventListener('keydown', e => {
  if (e.key === 'a' || e.key === 'A') decide('approved');
  else if (e.key === 'r' || e.key === 'R') decide('rejected');
  else if (e.key === 's' || e.key === 'S') decide('skipped');
  else if (e.key === ']' || e.key === 'ArrowRight') navigate(+1);
  else if (e.key === '[' || e.key === 'ArrowLeft')  navigate(-1);
});

init();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/meta")
def api_meta():
    annotations = load_annotations()
    cuts = [
        {"key": annot_key(hk, ci), "half_key": hk, "cut_idx": ci, "frame_no": fn}
        for hk, ci, fn in all_cuts
    ]
    return jsonify({"cuts": cuts, "annotations": annotations})


@app.route("/api/cut/<int:idx>")
def api_cut(idx: int):
    if idx < 0 or idx >= len(all_cuts):
        return jsonify({"error": "out of range"}), 404

    half_key, cut_idx, frame_no = all_cuts[idx]
    half_dir   = FRAMES_ROOT / half_key
    all_frames = sorted(int(f.stem) for f in half_dir.glob("*.jpg"))
    n = len(all_frames)

    try:
        cut_pos = all_frames.index(frame_no)
    except ValueError:
        cut_pos = min(range(n), key=lambda i: abs(all_frames[i] - frame_no))

    s1_start = max(0, cut_pos - CONTEXT_FRAMES + 1)
    scene1   = all_frames[s1_start : cut_pos + 1]

    s2_end = min(n, cut_pos + 1 + CONTEXT_FRAMES)
    scene2 = all_frames[cut_pos + 1 : s2_end]

    return jsonify({
        "key":           annot_key(half_key, cut_idx),
        "half_key":      half_key,
        "cut_idx":       cut_idx,
        "frame_no":      frame_no,
        "scene1_frames": scene1,
        "scene2_frames": scene2,
    })


@app.route("/frame/<path:half_key>/<filename>")
def serve_frame(half_key: str, filename: str):
    path = FRAMES_ROOT / half_key / filename
    if not path.exists():
        return "not found", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json()
    if not data or "key" not in data:
        return jsonify({"ok": False, "error": "missing key"}), 400
    annotations = load_annotations()
    annotations[data["key"]] = data
    save_annotations(annotations)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    print(f"\n  Open: http://localhost:{args.port}")
    print(f"  SSH:  ssh -L {args.port}:localhost:{args.port} <your-server>\n")
    app.run(host=args.host, port=args.port, debug=False)
