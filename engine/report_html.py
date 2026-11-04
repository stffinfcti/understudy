"""Self-contained HTML report: one file, inline CSS/JS, record embedded as
JSON (``</`` escaped so the script tag stays inside)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

_CSS = """
:root{color-scheme:light dark}
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:32px;
  background:canvas;color:canvastext;max-width:1080px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}.sub{color:#888;font-size:13px;margin-bottom:28px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-bottom:28px}
.card{border:1px solid #8884;border-radius:12px;padding:16px 18px}
.card h2{font-size:16px;margin:0 0 10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.pill{font-size:11px;font-weight:600;padding:2px 10px;border-radius:99px;color:#fff}
.ready{background:#3FB27F}.partial{background:#E3A43B}.not_ready{background:#E5534B}.insufficient{background:#7C8591}
.big{font-size:34px;font-weight:700}.ci{height:8px;background:#8883;border-radius:4px;position:relative;margin:10px 0 4px}
.ci i{position:absolute;top:-4px;height:16px;border-radius:4px;background:#5E9BD3;display:block}
.miss{font-size:13px;color:#888}.cost{font-size:13px;color:#888;margin-top:8px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:28px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #8883}
.dec{border:1px solid #8883;border-radius:10px;margin-bottom:8px;overflow:hidden}
.dec>summary{cursor:pointer;padding:10px 14px;font-size:13px;list-style:none;display:flex;gap:8px;align-items:center}
.dec>summary::-webkit-details-marker{display:none}
.dec.open>summary{border-bottom:1px solid #8883}
.chip{font-size:10px;font-weight:700;padding:1px 8px;border-radius:99px;background:#8884}
.chip.EXACT,.chip.AGREE,.chip.ALTERNATIVE,.chip.BETTER{background:#3FB27F44}
.chip.CRITICAL_MISS,.chip.INVALID{background:#E5534B44}.chip.MINOR_MISS{background:#E3A43B44}
.body{padding:12px 14px;font-size:13px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}
pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:#8881;padding:10px;border-radius:8px;margin:0}
.filters{display:flex;gap:8px;align-items:center;margin-bottom:14px;font-size:13px;flex-wrap:wrap}
.filters button{border:1px solid #8884;background:transparent;color:inherit;border-radius:99px;
  padding:4px 14px;cursor:pointer}.filters button.on{background:#5E9BD3;color:#fff;border-color:#5E9BD3}
.filters select{background:canvas;color:canvastext;border:1px solid #8884;border-radius:8px;padding:4px 8px}
.reason{color:#888;font-size:12px;margin-top:8px}
"""

_JS = """
const data = JSON.parse(document.getElementById('record').textContent);
const MISS = new Set(['CRITICAL_MISS','MINOR_MISS','INVALID']);
const TAKE = new Set(['EXACT','AGREE','ALTERNATIVE','BETTER']);
const esc = s => (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const money = v => typeof v==='number' ? '$'+v.toFixed(2) : null;
const pct = v => typeof v==='number' ? Math.round(v*100)+'%' : '—';
let filter='all', cand=null;
function pill(l){const L={ready:'Ready',partial:'Partial',not_ready:'Not ready',insufficient:'Not enough data'}[l]||l;
  return `<span class="pill ${l}">${L}</span>`}
function card(c){
  const ci=c.ci95||[];const k=c.counts||{};
  const cost=[typeof c.cost_ratio==='number'?Math.round(c.cost_ratio*100)+'% of the cost':null,
    money(c.projected_monthly_usd)?'~'+money(c.projected_monthly_usd)+'/mo':null,
    (c.savings_monthly_usd>0)?'saves ~'+money(c.savings_monthly_usd)+'/mo':null].filter(Boolean).join(' · ')||'cost unknown';
  return `<div class="card"><h2>${esc(c.model||c.spec)} ${pill((c.recommendation||{}).level)}</h2>
   <div class="big">${pct(c.takeover_rate)}</div>
   <div class="ci">${ci.length===2?`<i style="left:${ci[0]*100}%;width:${(ci[1]-ci[0])*100}%"></i>`:''}</div>
   <div class="miss">${(k.CRITICAL_MISS||0)+(k.INVALID||0)} critical · ${k.MINOR_MISS||0} minor · ${c.scored||0} scored</div>
   <div class="cost">${esc(cost)}</div></div>`}
function row(d){
  const rep=d.replays||{};
  if(cand && !rep[cand]) return '';
  const keys=cand?[cand]:Object.keys(rep);
  const show=keys.filter(k=>filter==='all'||(filter==='misses'&&MISS.has(rep[k].verdict))||(filter==='critical'&&['CRITICAL_MISS','INVALID'].includes(rep[k].verdict)));
  if(!show.length) return '';
  const chips=show.map(k=>`<span class="chip ${rep[k].verdict}">${rep[k].verdict}</span>`).join('');
  const cols=show.map(k=>`<div><div class="sub" style="margin:0 0 4px">What ${esc(k)} would do — ${rep[k].verdict}</div><pre>${esc(rep[k].rendered||'(no output)')}</pre><div class="reason">${esc(rep[k].reason||'')}</div></div>`).join('');
  return `<details class="dec"><summary><b>${esc(d.id)}</b><span class="chip">${esc(d.category)}</span>${chips}</summary>
   <div class="body"><pre>${esc(d.excerpt||'')}</pre>
   <div class="cols" style="margin-top:10px"><div><div class="sub" style="margin:0 0 4px">What your agent did</div><pre>${esc(d.production||'')}</pre></div>${cols}</div></div></details>`}
function paint(){
  const dc=(data.decisions||[]);
  document.getElementById('cards').innerHTML=(data.candidates||[]).map(card).join('');
  const cats={};(data.candidates||[]).forEach(c=>Object.assign(cats,c.by_category||{}));
  document.getElementById('cats').innerHTML='<tr><th>Category</th>'+(data.candidates||[]).map(c=>`<th>${esc(c.model||c.spec)}</th>`).join('')+'</tr>'
   +Object.entries(cats).map(([cat,slot])=>`<tr><td>${esc(cat)}</td>`+(data.candidates||[]).map(c=>{const s=(c.by_category||{})[cat];return `<td>${s&&typeof s.takeover_rate==='number'?pct(s.takeover_rate)+' ('+s.n+')':'—'}</td>`}).join('')+'</tr>').join('');
  document.getElementById('decs').innerHTML=dc.map(row).join('')||'<div class="sub">Nothing matches this filter.</div>';
}
document.querySelectorAll('#fbar button').forEach(b=>b.onclick=()=>{filter=b.dataset.f;
  document.querySelectorAll('#fbar button').forEach(x=>x.classList.toggle('on',x===b));paint()});
document.getElementById('fsel').onchange=e=>{cand=e.target.value||null;paint()};
document.getElementById('fsel').innerHTML='<option value="">All candidates</option>'+
  (data.candidates||[]).map(c=>`<option>${esc(c.spec)}</option>`).join('');
paint();
"""


def render_html(record: dict, path: Path) -> Path:
    """Write the standalone HTML report next to the run record."""
    path = Path(path)
    record_json = json.dumps(record, ensure_ascii=False, default=str).replace("</", "<\\/")
    corpus = record.get("corpus") or {}
    current = record.get("current_model") or {}
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Understudy — {html.escape(str(record.get('id') or ''))}</title>
<style>{_CSS}</style></head><body>
<h1>Could a cheaper model do your agent's job?</h1>
<div class="sub">{len(record.get('decisions') or [])} real decisions ·
 {html.escape(str(corpus.get('sessions', '?')))} conversations ·
 last {html.escape(str(corpus.get('window_days', '?')))} days ·
 current model {html.escape(str(current.get('model') or '?'))} ·
 audition {html.escape(str(record.get('id') or ''))}</div>
<div class="grid" id="cards"></div>
<h2 style="font-size:15px;margin:0 0 10px">By category</h2>
<table id="cats"></table>
<h2 style="font-size:15px;margin:0 0 10px">Decisions</h2>
<div class="filters" id="fbar">
 <button data-f="all" class="on">All</button><button data-f="misses">Misses</button>
 <button data-f="critical">Critical</button><select id="fsel"></select></div>
<div id="decs"></div>
<script id="record" type="application/json">{record_json}</script>
<script>{_JS}</script>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
