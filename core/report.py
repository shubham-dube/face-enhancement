"""
core/report.py — Self-contained A/B HTML report generator.

Produces a single enhancement_report.html with:
  • No external CDN dependencies (all CSS + JS inline, images as base64)
  • Dark surveillance-system aesthetic
  • Summary KPI dashboard
  • Sortable / filterable face grid with before/after comparison
  • Per-face metrics: sharpness bar, SSIM badge, PQ score, identity badge
  • Identity-cluster section (groups of faces sharing an identity)
  • Pipeline metadata footer
"""

from __future__ import annotations

import base64
import datetime
from pathlib import Path

import cv2
import numpy as np

from core.config import TARGET_SIZE
from core.metrics import compute_pq_score


# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark forensics / surveillance theme
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg0:      #07090e;
  --bg1:      #0d1117;
  --bg2:      #161b22;
  --bg3:      #1c2230;
  --border:   #21293b;
  --accent:   #00e676;
  --accent2:  #00b0ff;
  --red:      #ff5252;
  --amber:    #ffab40;
  --text:     #d0d8e8;
  --muted:    #5a6a82;
  --mono:     'Courier New', Courier, monospace;
  --sans:     -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  --radius:   6px;
  --card-w:   340px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg0);
  color: var(--text);
  font-family: var(--sans);
  font-size: 13px;
  line-height: 1.5;
  min-height: 100vh;
}

/* ── scanline overlay (aesthetic) ── */
body::before {
  content: '';
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,.06) 2px,
    rgba(0,0,0,.06) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ── header ── */
header {
  background: var(--bg1);
  border-bottom: 1px solid var(--border);
  padding: 18px 32px;
  display: flex; align-items: center; gap: 18px;
}
.logo {
  display: flex; align-items: center; gap: 10px;
}
.logo-icon {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
}
.logo-text { font-size: 17px; font-weight: 700; letter-spacing: .3px; }
.logo-sub  { font-size: 11px; color: var(--muted); font-family: var(--mono); }
header .meta {
  margin-left: auto;
  text-align: right;
  color: var(--muted);
  font-family: var(--mono);
  font-size: 11px;
  line-height: 1.7;
}

/* ── KPI bar ── */
.kpi-bar {
  display: flex; flex-wrap: wrap; gap: 1px;
  background: var(--border);
  border-bottom: 1px solid var(--border);
}
.kpi {
  flex: 1 1 160px;
  background: var(--bg1);
  padding: 16px 22px;
}
.kpi-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }
.kpi-value { font-size: 28px; font-weight: 700; font-family: var(--mono); margin: 4px 0 2px; }
.kpi-sub   { font-size: 11px; color: var(--muted); }
.kpi.green .kpi-value { color: var(--accent); }
.kpi.blue  .kpi-value { color: var(--accent2); }
.kpi.amber .kpi-value { color: var(--amber); }

/* ── improvement banner ── */
.banner {
  background: linear-gradient(90deg, #003d1a, #001e33);
  border-bottom: 1px solid var(--border);
  padding: 14px 32px;
  display: flex; align-items: center; gap: 32px; flex-wrap: wrap;
}
.banner-stat { font-family: var(--mono); font-size: 13px; }
.banner-stat span { color: var(--accent); font-weight: 700; font-size: 15px; }
.banner-tag {
  margin-left: auto;
  background: rgba(0,230,118,.12);
  border: 1px solid rgba(0,230,118,.3);
  color: var(--accent);
  font-family: var(--mono);
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 20px;
}

/* ── controls ── */
.controls {
  padding: 14px 32px;
  background: var(--bg1);
  border-bottom: 1px solid var(--border);
  display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
}
.ctrl-label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.btn {
  background: var(--bg2);
  border: 1px solid var(--border);
  color: var(--text);
  font-family: var(--mono);
  font-size: 11px;
  padding: 5px 12px;
  border-radius: var(--radius);
  cursor: pointer;
  transition: border-color .15s, color .15s;
}
.btn:hover { border-color: var(--accent2); color: var(--accent2); }
.btn.active { border-color: var(--accent); color: var(--accent); background: rgba(0,230,118,.08); }
.sep { width: 1px; height: 20px; background: var(--border); }

/* ── grid ── */
.grid-section { padding: 24px 32px; }
.section-title { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--muted); margin-bottom: 16px; }
.grid {
  display: flex; flex-wrap: wrap; gap: 16px;
}

/* ── face card ── */
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  width: var(--card-w);
  overflow: hidden;
  transition: border-color .15s, transform .15s;
}
.card:hover { border-color: var(--accent2); transform: translateY(-2px); }
.card.hidden { display: none; }

.card-header {
  padding: 8px 12px;
  background: var(--bg3);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.card-filename { font-family: var(--mono); font-size: 11px; color: var(--muted); }
.pq-badge {
  font-family: var(--mono);
  font-size: 10px;
  padding: 2px 7px;
  border-radius: 20px;
  font-weight: 700;
}
.pq-badge.high   { background: rgba(0,230,118,.15); color: var(--accent);  border: 1px solid rgba(0,230,118,.3); }
.pq-badge.mid    { background: rgba(0,176,255,.12); color: var(--accent2); border: 1px solid rgba(0,176,255,.3); }
.pq-badge.low    { background: rgba(255,82,82,.10); color: var(--red);     border: 1px solid rgba(255,82,82,.3); }

.images {
  display: flex;
}
.img-wrap {
  flex: 1;
  position: relative;
  overflow: hidden;
}
.img-wrap img {
  width: 100%; height: 160px;
  object-fit: cover;
  display: block;
}
.img-label {
  position: absolute; bottom: 0; left: 0; right: 0;
  background: rgba(0,0,0,.65);
  color: #fff;
  font-family: var(--mono);
  font-size: 9px;
  text-align: center;
  padding: 3px;
  letter-spacing: 1px;
}
.img-divider { width: 2px; background: var(--bg0); flex-shrink: 0; }

.metrics {
  padding: 10px 12px;
  display: flex; flex-direction: column; gap: 7px;
}
.metric-row {
  display: flex; align-items: center; gap: 8px;
}
.metric-key {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--muted);
  width: 62px;
  flex-shrink: 0;
}
.metric-bar-wrap {
  flex: 1;
  height: 5px;
  background: var(--bg3);
  border-radius: 3px;
  overflow: hidden;
}
.metric-bar {
  height: 100%;
  border-radius: 3px;
  background: linear-gradient(90deg, var(--accent2), var(--accent));
}
.metric-val {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text);
  width: 58px;
  text-align: right;
}
.badge {
  display: inline-flex; align-items: center; gap: 4px;
  font-family: var(--mono); font-size: 10px;
  padding: 2px 8px; border-radius: 20px;
}
.badge.match    { background: rgba(0,230,118,.12); color: var(--accent);  border: 1px solid rgba(0,230,118,.25); }
.badge.no-match { background: rgba(255,82,82,.10); color: var(--red);     border: 1px solid rgba(255,82,82,.25); }
.badge.improved { background: rgba(0,176,255,.10); color: var(--accent2); border: 1px solid rgba(0,176,255,.25); }

/* ── cluster section ── */
.cluster-section {
  padding: 0 32px 32px;
}
.cluster {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 12px;
  overflow: hidden;
}
.cluster-head {
  background: var(--bg3);
  padding: 8px 14px;
  font-family: var(--mono);
  font-size: 11px;
  display: flex; gap: 12px; align-items: center;
}
.cluster-body { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px 14px; }
.cluster-thumb img { width: 60px; height: 60px; object-fit: cover; border-radius: 4px; border: 1px solid var(--border); }

/* ── footer ── */
footer {
  background: var(--bg1);
  border-top: 1px solid var(--border);
  padding: 18px 32px;
  display: flex; align-items: center; gap: 24px; flex-wrap: wrap;
}
.footer-tag {
  font-family: var(--mono); font-size: 10px; color: var(--muted);
}
.pipeline-stages { display: flex; gap: 6px; }
.stage-chip {
  background: var(--bg3);
  border: 1px solid var(--border);
  font-family: var(--mono); font-size: 10px;
  padding: 2px 8px; border-radius: 20px; color: var(--accent2);
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# JS — sort / filter (vanilla, no CDN)
# ─────────────────────────────────────────────────────────────────────────────

_JS = """
const grid = document.getElementById('face-grid');
const cards = () => [...grid.querySelectorAll('.card')];

function setFilter(key) {
  document.querySelectorAll('[data-filter]').forEach(b => b.classList.remove('active'));
  document.querySelector('[data-filter="'+key+'"]').classList.add('active');
  cards().forEach(c => {
    if (key === 'all') c.classList.remove('hidden');
    else if (key === 'matched')   c.classList.toggle('hidden', c.dataset.match === 'false');
    else if (key === 'unmatched') c.classList.toggle('hidden', c.dataset.match === 'true');
    else if (key === 'improved')  c.classList.toggle('hidden', c.dataset.improved === 'false');
  });
}

function setSort(key) {
  document.querySelectorAll('[data-sort]').forEach(b => b.classList.remove('active'));
  document.querySelector('[data-sort="'+key+'"]').classList.add('active');
  const all = cards().sort((a, b) => {
    const va = parseFloat(a.dataset[key] || 0);
    const vb = parseFloat(b.dataset[key] || 0);
    return vb - va;  // descending
  });
  all.forEach(c => grid.appendChild(c));
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pct_color_class(pq: float) -> str:
    if pq >= 60:  return "high"
    if pq >= 35:  return "mid"
    return "low"


def _bar_width(value: float, cap: float) -> int:
    return int(min(max(value / cap, 0.0), 1.0) * 100)


def _card_html(r: dict) -> str:
    pq       = compute_pq_score(r["sharpness_after"], r["ssim_improvement"], r["match_after"])
    pq_cls   = _pct_color_class(pq)
    improved = r["sharpness_after"] > r["sharpness_before"]
    delta_s  = r["sharpness_after"] - r["sharpness_before"]

    sharp_bar_b = _bar_width(r["sharpness_before"], 300)
    sharp_bar_a = _bar_width(r["sharpness_after"],  300)
    ssim_bar    = _bar_width(max(r["ssim_improvement"], 0), 0.5)

    match_badge = (
        f'<span class="badge match">✓ {r["matched_identity"]}</span>'
        if r["match_after"]
        else '<span class="badge no-match">✗ no match</span>'
    )
    improved_badge = (
        f'<span class="badge improved">▲ {delta_s:+.1f} sharp</span>'
        if improved else ""
    )

    return f"""
<div class="card" data-match="{str(r['match_after']).lower()}"
     data-improved="{str(improved).lower()}"
     data-pq="{pq}" data-sharp="{r['sharpness_after']}" data-ssim="{r['ssim_improvement']}">
  <div class="card-header">
    <span class="card-filename">{r['filename']}</span>
    <span class="pq-badge {pq_cls}">PQ {pq}</span>
  </div>
  <div class="images">
    <div class="img-wrap">
      <img src="data:image/jpeg;base64,{r['raw_b64']}" loading="lazy" alt="before">
      <div class="img-label">BEFORE</div>
    </div>
    <div class="img-divider"></div>
    <div class="img-wrap">
      <img src="data:image/jpeg;base64,{r['enhanced_b64']}" loading="lazy" alt="after">
      <div class="img-label">AFTER</div>
    </div>
  </div>
  <div class="metrics">
    <div class="metric-row">
      <span class="metric-key">Sharp ▸</span>
      <div class="metric-bar-wrap"><div class="metric-bar" style="width:{sharp_bar_b}%; opacity:.35"></div></div>
      <span class="metric-val" style="color:var(--muted)">{r['sharpness_before']:.1f}</span>
    </div>
    <div class="metric-row">
      <span class="metric-key">Sharp ▸</span>
      <div class="metric-bar-wrap"><div class="metric-bar" style="width:{sharp_bar_a}%"></div></div>
      <span class="metric-val">{r['sharpness_after']:.1f}</span>
    </div>
    <div class="metric-row">
      <span class="metric-key">SSIM △</span>
      <div class="metric-bar-wrap"><div class="metric-bar" style="width:{ssim_bar}%"></div></div>
      <span class="metric-val">{r['ssim_improvement']:+.4f}</span>
    </div>
    <div class="metric-row" style="margin-top:3px;">
      {match_badge} {improved_badge}
    </div>
  </div>
</div>"""


def _cluster_html(clusters: dict[int, list[str]], results: list[dict]) -> str:
    if not clusters:
        return ""

    enc_map = {r["filename"]: r.get("enhanced_b64", "") for r in results}

    parts = ['<div class="cluster-section">',
             '<div class="section-title">⬡ Detected identity clusters (unverified)</div>']

    for cid, files in clusters.items():
        parts.append(
            f'<div class="cluster">'
            f'<div class="cluster-head">'
            f'<span style="color:var(--accent2)">CLUSTER {cid + 1}</span>'
            f'<span style="color:var(--muted)">{len(files)} faces appear to share an identity</span>'
            f'</div>'
            f'<div class="cluster-body">'
        )
        for fname in files[:12]:   # cap at 12 thumbnails per cluster
            b64 = enc_map.get(fname, "")
            if b64:
                parts.append(
                    f'<div class="cluster-thumb" title="{fname}">'
                    f'<img src="data:image/jpeg;base64,{b64}" loading="lazy" alt="{fname}">'
                    f'</div>'
                )
        parts.append("</div></div>")

    parts.append("</div>")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_ab_report(results: list[dict], output_path: Path) -> None:
    """
    Build a fully self-contained, offline-capable HTML report and write it to
    output_path.

    Parameters
    ----------
    results     : list of per-face dicts as assembled in the main loop of
                  solution.py.  Must include raw_b64 and enhanced_b64 keys.
    output_path : destination .html file path
    """
    if not results:
        output_path.write_text("<html><body>No faces processed.</body></html>")
        return

    n = len(results)
    n_matched_before = sum(1 for r in results if r["match_before"])
    n_matched_after  = sum(1 for r in results if r["match_after"])
    acc_before = round(n_matched_before / n * 100, 1) if n else 0.0
    acc_after  = round(n_matched_after  / n * 100, 1) if n else 0.0
    acc_delta  = round(acc_after - acc_before, 1)

    avg_sharp_b = round(float(np.mean([r["sharpness_before"] for r in results])), 1)
    avg_sharp_a = round(float(np.mean([r["sharpness_after"]  for r in results])), 1)
    sharp_delta = round(avg_sharp_a - avg_sharp_b, 1)
    sharp_pct   = round((avg_sharp_a - avg_sharp_b) / max(avg_sharp_b, 1) * 100, 0)

    avg_ssim = round(float(np.mean([r["ssim_improvement"] for r in results])), 4)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── cards ────────────────────────────────────────────────────────────────
    cards_html = "\n".join(_card_html(r) for r in results)

    # ── identity clusters ─────────────────────────────────────────────────────
    from core.metrics import cluster_by_identity  # avoid circular at module level
    # Inject encodings into results for clustering (stored temporarily)
    cluster_map = cluster_by_identity(results, tolerance=0.60)
    clusters_html = _cluster_html(cluster_map, results)

    # ── assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentio Mind · Face Enhancement Report · {ts}</title>
<style>{_CSS}</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">◉</div>
    <div>
      <div class="logo-text">Sentio Mind</div>
      <div class="logo-sub">CCTV Face Enhancement · Project 4</div>
    </div>
  </div>
  <div class="meta">
    Generated: {ts}<br>
    Faces processed: {n} &nbsp;|&nbsp; Pipeline: denoise → clahe → upscale → zone-sharpen
  </div>
</header>

<div class="kpi-bar">
  <div class="kpi green">
    <div class="kpi-label">Recognition after</div>
    <div class="kpi-value">{acc_after}%</div>
    <div class="kpi-sub">was {acc_before}% before</div>
  </div>
  <div class="kpi blue">
    <div class="kpi-label">Accuracy gain</div>
    <div class="kpi-value">{acc_delta:+.1f}pp</div>
    <div class="kpi-sub">{n_matched_after} / {n} matched</div>
  </div>
  <div class="kpi amber">
    <div class="kpi-label">Avg sharpness after</div>
    <div class="kpi-value">{avg_sharp_a}</div>
    <div class="kpi-sub">was {avg_sharp_b} ({sharp_pct:+.0f}%)</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg SSIM gain</div>
    <div class="kpi-value">{avg_ssim:+.4f}</div>
    <div class="kpi-sub">Structural Similarity Index</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Total faces</div>
    <div class="kpi-value">{n}</div>
    <div class="kpi-sub">→ enhanced_faces/</div>
  </div>
</div>

<div class="banner">
  <div class="banner-stat">Recognition: <span>{acc_before}%</span> → <span>{acc_after}%</span></div>
  <div class="banner-stat">Sharpness: <span>{avg_sharp_b}</span> → <span>{avg_sharp_a}</span> (<span>{sharp_delta:+.1f}</span>)</div>
  <div class="banner-stat">SSIM improvement: <span>{avg_ssim:+.4f}</span></div>
  <div class="banner-tag">◉ PIPELINE COMPLETE</div>
</div>

<div class="controls">
  <span class="ctrl-label">Filter:</span>
  <button class="btn active" data-filter="all"      onclick="setFilter('all')">All ({n})</button>
  <button class="btn" data-filter="matched"          onclick="setFilter('matched')">Matched ({n_matched_after})</button>
  <button class="btn" data-filter="unmatched"        onclick="setFilter('unmatched')">Unmatched ({n - n_matched_after})</button>
  <button class="btn" data-filter="improved"         onclick="setFilter('improved')">Improved</button>
  <div class="sep"></div>
  <span class="ctrl-label">Sort by:</span>
  <button class="btn" data-sort="pq"     onclick="setSort('pq')">PQ Score</button>
  <button class="btn" data-sort="sharp"  onclick="setSort('sharp')">Sharpness</button>
  <button class="btn" data-sort="ssim"   onclick="setSort('ssim')">SSIM</button>
</div>

<div class="grid-section">
  <div class="section-title">◈ Face enhancement results — {n} images</div>
  <div class="grid" id="face-grid">
{cards_html}
  </div>
</div>

{clusters_html}

<footer>
  <div class="footer-tag">Sentio Mind · Project 4 · CCTV Face Enhancement · 2026</div>
  <div class="pipeline-stages">
    <span class="stage-chip">① denoise</span>
    <span class="stage-chip">② clahe</span>
    <span class="stage-chip">③ upscale</span>
    <span class="stage-chip">④ zone-sharpen</span>
  </div>
  <div class="footer-tag" style="margin-left:auto">Output: 240×240 JPEG q=95 · No DL models · CPU only</div>
</footer>

<script>{_JS}</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"  HTML report → {output_path}  ({output_path.stat().st_size // 1024} KB)")