// Understudy — desktop half. Plain ESM, no build step.
// Backend: dashboard/plugin_api.py at /api/plugins/understudy (ctx.rest).
// Every identifier below was verified against apps/desktop/src/sdk and
// components/ui/*.tsx.

import { jsx } from 'react/jsx-runtime'
import { useEffect, useRef, useState } from 'react'
import {
  atom, host,
  Badge, Button, Codicon,
  Dialog, DialogContent, DialogHeader, DialogTitle,
  EmptyState, GlyphSpinner, Input, SandboxedFrame,
  SegmentedControl, Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
  PALETTE_AREA, ROUTES_AREA, SIDEBAR_NAV_AREA,
} from '@hermes/plugin-sdk'
import * as sdk from '@hermes/plugin-sdk'

const VERDICT_VARIANT = { ready: 'success', partial: 'warn', not_ready: 'destructive', insufficient: 'muted' }
const VERDICT_LABEL = { ready: 'Ready', partial: 'Partial', not_ready: 'Not ready', insufficient: 'Not enough data' }
const TAKEOVER = new Set(['EXACT', 'AGREE', 'ALTERNATIVE', 'BETTER'])
const MISS = new Set(['CRITICAL_MISS', 'MINOR_MISS', 'INVALID'])
const CRITICAL = new Set(['CRITICAL_MISS', 'INVALID'])
const chipVariant = v => TAKEOVER.has(v) ? 'success' : CRITICAL.has(v) ? 'destructive' : v === 'MINOR_MISS' ? 'warn' : 'muted'

const pct = v => typeof v === 'number' ? `${Math.round(v * 100)}%` : '—'
const money = v => typeof v === 'number' ? `$${v.toFixed(2)}` : null
const modelOf = c => (c && (c.model || c.spec)) || '?'

// Palette → page channel: the command sets the atom and navigates; the page
// subscribes and opens the dialog. A number (nonce) so repeats refire.
const openNewAudition = atom(0)

const DIM = 'var(--ui-text-tertiary)'
const DIM2 = 'var(--ui-text-secondary)'
const STROKE = 'var(--ui-stroke-secondary)'
const FILL = 'var(--ui-bg-tertiary)'
const ACCENT = 'var(--primary)'

function VerdictPill({ level }) {
  return jsx(Badge, { variant: VERDICT_VARIANT[level] || 'muted', children: VERDICT_LABEL[level] || level || '—' })
}

function Bar({ frac, children }) {
  return jsx('div', {
    style: { height: 6, borderRadius: 3, background: FILL, position: 'relative', overflow: 'hidden' },
    children: jsx('div', { style: { height: '100%', width: `${Math.max(0, Math.min(1, frac || 0)) * 100}%`, background: ACCENT, borderRadius: 3 } }),
  })
}

function CandidateCard({ cand }) {
  const counts = cand.counts || {}
  const critical = (counts.CRITICAL_MISS || 0) + (counts.INVALID || 0)
  const minor = counts.MINOR_MISS || 0
  const ci = cand.ci95 || []
  const costBits = [
    typeof cand.cost_ratio === 'number' ? `${Math.round(cand.cost_ratio * 100)}% of the cost` : null,
    money(cand.projected_monthly_usd) ? `~${money(cand.projected_monthly_usd)}/mo` : null,
    cand.savings_monthly_usd > 0 ? `saves ~${money(cand.savings_monthly_usd)}/mo` : null,
  ].filter(Boolean).join(' · ')
  const cats = Object.entries(cand.by_category || {})
  return jsx('div', {
    style: { border: `1px solid ${STROKE}`, borderRadius: 10, padding: '14px 16px', marginBottom: 10 },
    children: [
      jsx('div', { style: { display: 'flex', gap: 10, alignItems: 'center', marginBottom: 6, flexWrap: 'wrap' }, children: [
        jsx('b', { children: modelOf(cand) }),
        jsx(VerdictPill, { level: (cand.recommendation || {}).level }),
        cand.provider && jsx('span', { style: { color: DIM, fontSize: 12 }, children: cand.provider }),
      ] }),
      jsx('div', { style: { display: 'flex', gap: 16, alignItems: 'baseline' }, children: [
        jsx('span', { style: { fontSize: 30, fontWeight: 700 }, children: pct(cand.takeover_rate) }),
        jsx('span', { style: { fontSize: 12, color: DIM }, children: `could take over · ${critical} critical · ${minor} minor · ${cand.scored || 0} scored` }),
      ] }),
      ci.length === 2 && jsx('div', { style: { margin: '8px 0 4px' }, children: jsx('div', {
        style: { height: 6, borderRadius: 3, background: FILL, position: 'relative' },
        children: jsx('div', { style: { position: 'absolute', left: `${ci[0] * 100}%`, width: `${Math.max(1, (ci[1] - ci[0]) * 100)}%`, top: -2, height: 10, borderRadius: 3, background: ACCENT } }),
      }) }),
      jsx('div', { style: { fontSize: 12, color: DIM, marginTop: 6 }, children: costBits || 'cost unknown' }),
      cats.length > 0 && jsx('div', { style: { display: 'flex', gap: 10, marginTop: 8, flexWrap: 'wrap' }, children:
        cats.map(([cat, slot]) => jsx('div', { key: cat, style: { flex: '1 1 90px' }, children: [
          jsx('div', { style: { fontSize: 11, color: DIM }, children: `${cat} ${pct(slot.takeover_rate)}` }),
          jsx(Bar, { frac: slot.takeover_rate || 0 }),
        ] })) }),
      (cand.recommendation || {}).text && jsx('div', { style: { fontSize: 12, marginTop: 8, color: DIM2 }, children: cand.recommendation.text }),
    ],
  })
}

function ReportDialog({ ctx, record, open, onClose }) {
  const [src, setSrc] = useState(null)
  useEffect(() => {
    if (!open || !record) return
    setSrc(null)
    ctx.rest(`/runs/${record.id}/html`).then(r => {
      if (r && r.html) setSrc(`data:text/html;base64,${btoa(unescape(encodeURIComponent(r.html)))}`)
    }).catch(() => {})
  }, [open, record && record.id])
  return jsx(Dialog, { open, onOpenChange: v => !v && onClose(), children:
    jsx(DialogContent, { style: { maxWidth: 960, width: '90vw' }, children: [
      jsx(DialogHeader, { children: jsx(DialogTitle, { children: `Report — ${record && record.id}` }) }),
      jsx('div', { style: { height: '70vh' }, children:
        src ? jsx(SandboxedFrame, { src, style: { width: '100%', height: '100%', border: `1px solid ${STROKE}`, borderRadius: 8 } })
            : jsx('div', { style: { display: 'grid', placeItems: 'center', height: '100%' }, children: jsx(GlyphSpinner, {}) }) }),
    ] }) })
}

function DecisionDialog({ decision, candSpec, open, onClose }) {
  if (!decision) return null
  const replays = decision.replays || {}
  const specs = candSpec && replays[candSpec] ? [candSpec] : Object.keys(replays)
  const pre = { whiteSpace: 'pre-wrap', fontSize: 12, maxHeight: 300, overflow: 'auto', margin: 0 }
  return jsx(Dialog, { open, onOpenChange: v => !v && onClose(), children:
    jsx(DialogContent, { style: { maxWidth: 900 }, children: [
      jsx(DialogHeader, { children: jsx(DialogTitle, { children: `${decision.id} · ${decision.category} · ${decision.kind}` }) }),
      jsx('pre', { style: { ...pre, opacity: 0.8, maxHeight: 140 }, children: decision.excerpt || '' }),
      jsx('div', { style: { display: 'grid', gridTemplateColumns: `repeat(${specs.length + 1}, 1fr)`, gap: 10, marginTop: 10 }, children: [
        jsx('div', { children: [
          jsx('div', { style: { fontSize: 12, marginBottom: 4, color: DIM }, children: 'What your agent did' }),
          jsx('pre', { style: pre, children: decision.production || '' }),
        ] }),
        ...specs.map(spec => {
          const r = replays[spec] || {}
          return jsx('div', { key: spec, children: [
            jsx('div', { style: { fontSize: 12, marginBottom: 4, display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }, children: [
              `What ${spec} would do`, jsx(Badge, { variant: chipVariant(r.verdict), children: r.verdict || '—' }),
            ] }),
            jsx('pre', { style: pre, children: r.rendered || '(no output)' }),
            jsx('div', { style: { fontSize: 11, color: DIM, marginTop: 4 }, children: [
              r.reason, money(r.cost_usd) ? ` · ${money(r.cost_usd)}` : '', r.latency_ms ? ` · ${Math.round(r.latency_ms)}ms` : '',
            ].join('') }),
          ] })
        }),
      ] }),
    ] }) })
}

function NewAuditionDialog({ ctx, open, onClose, onStarted, recent }) {
  const [cands, setCands] = useState('')
  const [sample, setSample] = useState(20)
  const [days, setDays] = useState(14)
  const [budget, setBudget] = useState(1.5)
  const [est, setEst] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  useEffect(() => { if (open) ctx.rest('/settings').then(s => { if (s && s.budget_usd) setBudget(s.budget_usd) }).catch(() => {}) }, [open])
  const specs = cands.split(/[,\s]+/).map(s => s.trim()).filter(Boolean)
  const check = async () => {
    setBusy(true); setErr(''); setEst(null)
    try {
      const q = `/estimate?candidates=${encodeURIComponent(specs.join(','))}&sample=${sample}&days=${days}`
      setEst(await ctx.rest(q))
    } catch (e) { setErr('Could not reach the Understudy backend — is the plugin enabled on this profile?') }
    setBusy(false)
  }
  const run = async () => {
    setBusy(true); setErr('')
    try {
      const res = await ctx.rest('/runs', { method: 'POST', body: { candidates: specs, sample, days, budget_usd: budget } })
      onStarted(res && res.run_id)
      onClose()
    } catch (e) { setErr('The audition could not be started.') }
    setBusy(false)
  }
  return jsx(Dialog, { open, onOpenChange: v => !v && onClose(), children:
    jsx(DialogContent, { children: [
      jsx(DialogHeader, { children: jsx(DialogTitle, { children: 'New audition' }) }),
      jsx('div', { style: { fontSize: 12, color: DIM, marginBottom: 10 }, children: 'Candidate models, comma or space separated. provider:model or a bare model id.' }),
      jsx(Input, { value: cands, onChange: e => setCands(e.target.value), placeholder: 'openrouter:moonshotai/kimi-k3' }),
      recent.length > 0 && jsx('div', { style: { display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }, children:
        recent.map(r => jsx(Badge, { key: r, variant: 'outline', style: { cursor: 'pointer' }, onClick: () => setCands(prev => prev ? `${prev}, ${r}` : r), children: r })) }),
      jsx('div', { style: { display: 'flex', gap: 10, marginTop: 12, alignItems: 'center', flexWrap: 'wrap', fontSize: 12 }, children: [
        'decisions', jsx(Input, { type: 'number', min: 8, max: 60, value: sample, style: { width: 70 }, onChange: e => setSample(Number(e.target.value) || 20) }),
        'days', jsx(Select, { value: String(days), onValueChange: v => setDays(Number(v)), children: [
          jsx(SelectTrigger, { children: jsx(SelectValue, {}) }),
          jsx(SelectContent, { children: [7, 14, 30, 60].map(d => jsx(SelectItem, { key: d, value: String(d), children: d })) }),
        ] }),
        'budget $', jsx(Input, { type: 'number', min: 0.5, max: 10, step: 0.5, value: budget, style: { width: 80 }, onChange: e => setBudget(Number(e.target.value) || 1.5) }),
      ] }),
      est && jsx('div', { style: { fontSize: 12, marginTop: 10 }, children: [
        `${est.decisions_available} past decisions found (${Object.entries(est.by_category || {}).map(([c, n]) => `${c} ${n}`).join(', ')}) · ${est.sample_per_candidate} per candidate · ~${est.planned_calls_total} calls`,
        est.estimated_cost_usd != null && ` · est. ~$${est.estimated_cost_usd.toFixed(2)} at list prices`,
      ] }),
      err && jsx('div', { style: { fontSize: 12, marginTop: 10, color: 'var(--destructive)' }, children: err }),
      jsx('div', { style: { display: 'flex', gap: 8, marginTop: 14, justifyContent: 'flex-end' }, children: [
        jsx(Button, { variant: 'secondary', onClick: check, disabled: busy || !specs.length, children: 'Check' }),
        jsx(Button, { onClick: run, disabled: busy || !specs.length, children: 'Run audition' }),
      ] }),
    ] }) })
}

function UnderstudyPage({ ctx }) {
  const [runs, setRuns] = useState([])
  const [selected, setSelected] = useState(null)
  const [record, setRecord] = useState(null)
  const [filter, setFilter] = useState('all')
  const [candSel, setCandSel] = useState('')
  const [dialog, setDialog] = useState(null)
  const [reportOpen, setReportOpen] = useState(false)
  const [newOpen, setNewOpen] = useState(false)
  const [backendErr, setBackendErr] = useState(false)
  const [progress, setProgress] = useState(null)
  const selectedRef = useRef(null)
  selectedRef.current = selected

  const refresh = async () => {
    try {
      const res = await ctx.rest('/runs')
      setRuns(res.runs || [])
      setBackendErr(false)
    } catch (e) { setBackendErr(true) }
  }
  const load = async id => {
    try { setRecord(await ctx.rest(`/runs/${id}`)); ctx.storage.set('understudy.lastRun', id) }
    catch (e) { setRecord(null) }
  }
  useEffect(() => {
    refresh()
    const last = ctx.storage.get('understudy.lastRun', null)
    if (last) { setSelected(last); load(last) }
    const offOpen = openNewAudition.subscribe(() => setNewOpen(true))
    return offOpen
  }, [])
  const running = record && record.status === 'running'
  useEffect(() => {
    const off = ctx.onEvent('plugin.understudy.progress', ev => {
      const p = ev && ev.payload
      if (!p) return
      if (p.run_id === selectedRef.current) { setProgress(p); load(p.run_id) } else refresh()
    })
    const off2 = ctx.onEvent('plugin.understudy.finished', ev => {
      setProgress(null); refresh()
      const p = ev && ev.payload
      if (p && p.run_id === selectedRef.current) load(p.run_id)
    })
    return () => { off && off(); off2 && off2() }
  }, [])
  useEffect(() => {
    if (!running) return
    const t = setInterval(() => selectedRef.current && load(selectedRef.current), 2000)
    return () => clearInterval(t)
  }, [running])

  const recentCands = [...new Set(runs.flatMap(r => (r.candidates || []).map(c => c.spec || c.model)).filter(Boolean))].slice(0, 8)
  const decisions = (record && record.decisions) || []
  const shown = decisions.filter(d => {
    const reps = d.replays || {}
    const keys = candSel && reps[candSel] ? [candSel] : Object.keys(reps)
    if (filter === 'all') return keys.length > 0
    if (filter === 'misses') return keys.some(k => MISS.has((reps[k] || {}).verdict))
    return keys.some(k => CRITICAL.has((reps[k] || {}).verdict))
  })
  const candSpecs = record ? [...new Set(decisions.flatMap(d => Object.keys(d.replays || {})))] : []
  const tallies = (progress && progress.tallies) || {}
  const progressFrac = progress && progress.total ? (progress.done || 0) / progress.total : 0

  return jsx('div', { style: { padding: '24px 28px', maxWidth: 1100 }, children: [
    jsx('div', { style: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }, children: [
      jsx('div', { children: [
        jsx('h1', { style: { fontSize: 20, margin: 0 }, children: 'Understudy' }),
        jsx('div', { style: { fontSize: 12, color: DIM }, children: 'Audition cheaper models on your real history' }),
      ] }),
      jsx(Button, { onClick: () => setNewOpen(true), children: 'New audition' }),
    ] }),
    backendErr && jsx('div', { style: { fontSize: 13, marginBottom: 14, color: DIM2 }, children: 'The Understudy backend is not answering — enable the understudy plugin on this profile and reload.' }),
    jsx('div', { style: { display: 'grid', gridTemplateColumns: '240px 1fr', gap: 20 }, children: [
      jsx('div', { children: [
        runs.length === 0 && !backendErr && jsx(EmptyState, {
          title: 'No auditions yet',
          description: 'Understudy replays a sample of your real past decisions on candidate models, judges each against what your current model did, and reports how often it could have taken over — plus projected cost.',
        }),
        runs.map(r => jsx('div', {
          key: r.id, onClick: () => { setSelected(r.id); setProgress(null); load(r.id) },
          style: { padding: '8px 10px', borderRadius: 8, cursor: 'pointer', marginBottom: 4,
            background: r.id === selected ? 'var(--ui-control-active-background)' : 'transparent' },
          children: [
            jsx('div', { style: { fontSize: 12, display: 'flex', gap: 6, alignItems: 'center' }, children: [
              r.status === 'running' && jsx(GlyphSpinner, {}),
              (r.candidates || []).map(c => c.model || c.spec).join(', ') || r.id,
            ] }),
            jsx('div', { style: { fontSize: 11, color: DIM, display: 'flex', gap: 6, alignItems: 'center' }, children: [
              r.created_at || '', r.status,
              r.candidates && r.candidates[0] && r.candidates[0].level && jsx(VerdictPill, { level: r.candidates[0].level }),
            ] }),
          ] })),
      ] }),
      jsx('div', { children: [
        !record && !running && jsx('div', { style: { color: DIM2, fontSize: 13 }, children: 'Pick a run on the left, or start a new audition.' }),
        record && jsx('div', { children: [
          jsx('div', { style: { fontSize: 12, color: DIM, marginBottom: 14 }, children:
            `${decisions.length} decisions · ${(record.corpus || {}).sessions || '?'} conversations · last ${(record.corpus || {}).window_days || '?'} days · current ${(record.current_model || {}).model || '?'}` }),
          running && jsx('div', { style: { marginBottom: 14 }, children: [
            jsx(Bar, { frac: progressFrac }),
            jsx('div', { style: { display: 'flex', gap: 10, marginTop: 6, fontSize: 12, color: DIM2, flexWrap: 'wrap', alignItems: 'center' }, children: [
              progress && `${progress.done || 0}/${progress.total || 0} replays`,
              progress && money(progress.spent_usd) && ` · ${money(progress.spent_usd)} spent`,
              ...Object.entries(tallies).map(([label, t]) => jsx('span', { key: label, children: `${label === 'baseline' ? 'your model' : label}: ${t.ok || 0}/${t.scored || 0}` })),
              jsx(Button, { variant: 'secondary', onClick: () => ctx.rest(`/runs/${record.id}/cancel`, { method: 'POST' }).then(refresh), children: 'Cancel' }),
            ] }),
          ] }),
          (record.candidates || []).map(c => jsx(CandidateCard, { key: c.spec, cand: c })),
          (record.baseline || {}).scored > 0 && jsx('div', { style: { fontSize: 12, color: DIM, marginBottom: 14 }, children:
            `Your current model matched itself on ${record.baseline.scored} re-asked decisions (${pct(record.baseline.takeover_rate)}).` }),
          (record.summary || record.report_html) && jsx('div', { style: { display: 'flex', gap: 8, marginBottom: 14 }, children: [
            record.report_html && jsx(Button, { variant: 'secondary', onClick: () => setReportOpen(true), children: 'Open report' }),
            record.report_html && jsx(Button, { variant: 'ghost', onClick: () => ctx.os.revealPath(record.report_html), children: 'Reveal file' }),
            record.summary && jsx(Button, { variant: 'ghost', onClick: () => ctx.os.writeClipboard(record.summary).then(ok => ok && host.notify && host.notify({ title: 'Understudy', body: 'Summary copied' })).catch(() => {}), children: 'Copy summary' }),
          ] }),
          decisions.length > 0 && jsx('div', { children: [
            jsx('div', { style: { display: 'flex', gap: 10, marginBottom: 10, alignItems: 'center' }, children: [
              jsx(SegmentedControl, { value: filter, onChange: setFilter, sizing: 'content', options: [
                { id: 'all', label: 'All' }, { id: 'misses', label: 'Misses' }, { id: 'critical', label: 'Critical' },
              ] }),
              jsx(Select, { value: candSel, onValueChange: setCandSel, children: [
                jsx(SelectTrigger, { children: jsx(SelectValue, { placeholder: 'All candidates' }) }),
                jsx(SelectContent, { children: [
                  jsx(SelectItem, { value: '', children: 'All candidates' }),
                  ...candSpecs.map(s => jsx(SelectItem, { key: s, value: s, children: s })),
                ] }),
              ] }),
            ] }),
            shown.map(d => {
              const reps = d.replays || {}
              const chips = Object.entries(reps).map(([k, r]) => jsx(Badge, { key: k, variant: chipVariant((r || {}).verdict), children: `${k.split(':').pop()}: ${(r || {}).verdict}` }))
              return jsx('div', {
                key: d.id, onClick: () => setDialog(d),
                style: { padding: '8px 10px', borderRadius: 8, cursor: 'pointer', border: `1px solid ${STROKE}`, marginBottom: 6, fontSize: 12, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' },
                children: [jsx('b', { children: d.id }), jsx(Badge, { variant: 'muted', children: d.category }), ...chips],
              })
            }),
          ] }),
        ] }),
      ] }),
    ] }),
    jsx(NewAuditionDialog, { ctx, open: newOpen, onClose: () => setNewOpen(false), recent: recentCands,
      onStarted: id => { if (id) { setSelected(id); refresh(); load(id) } } }),
    jsx(DecisionDialog, { decision: dialog, candSpec: candSel || null, open: !!dialog, onClose: () => setDialog(null) }),
    record && jsx(ReportDialog, { ctx, record, open: reportOpen, onClose: () => setReportOpen(false) }),
  ] })
}

function AuditionCard({ ctx, file, fallback }) {
  const [record, setRecord] = useState(null)
  const name = (file && (file.name || file.path || '').split('/').pop()) || ''
  const runId = name.replace(/\.png$|\.html$|\.json$/g, '')
  useEffect(() => { if (runId) ctx.rest(`/runs/${runId}`).then(setRecord).catch(() => {}) }, [runId])
  if (!record) return fallback || null
  return jsx('div', {
    style: { border: `1px solid ${STROKE}`, borderRadius: 10, padding: '12px 14px', maxWidth: 520 },
    children: [
      jsx('div', { style: { fontSize: 11, color: DIM, marginBottom: 8 }, children: `Understudy · ${runId}` }),
      ...(record.candidates || []).map(c => {
        const counts = c.counts || {}
        return jsx('div', { key: c.spec, style: { display: 'flex', gap: 10, alignItems: 'center', marginBottom: 6, fontSize: 12, flexWrap: 'wrap' }, children: [
          jsx('b', { children: modelOf(c) }),
          jsx(VerdictPill, { level: (c.recommendation || {}).level }),
          jsx('span', { children: pct(c.takeover_rate) }),
          jsx('span', { style: { color: DIM }, children: `${(counts.CRITICAL_MISS || 0) + (counts.INVALID || 0)} critical` }),
          typeof c.cost_ratio === 'number' && jsx('span', { style: { color: DIM }, children: `${Math.round(c.cost_ratio * 100)}% of the cost` }),
        ] })
      }),
      jsx(Button, { variant: 'secondary', onClick: () => {
        ctx.storage.set('understudy.lastRun', runId)
        host.navigate('/understudy')
      }, children: 'Open report' }),
    ],
  })
}

export default {
  id: 'understudy',
  name: 'Understudy',
  register(ctx) {
    ctx.registerMany([
      { id: 'understudy.page', area: ROUTES_AREA, data: { path: '/understudy' }, render: () => jsx(UnderstudyPage, { ctx }) },
      { id: 'understudy.nav', area: SIDEBAR_NAV_AREA, data: { path: '/understudy', label: 'Understudy', codicon: 'beaker' } },
      { id: 'understudy.open', area: PALETTE_AREA, data: { id: 'understudy.open', label: 'Understudy: Open', run: () => host.navigate('/understudy') } },
      { id: 'understudy.new', area: PALETTE_AREA, data: { id: 'understudy.new', label: 'Understudy: New audition', run: () => { openNewAudition.set(Date.now()); host.navigate('/understudy') } } },
    ])
    if (typeof sdk.TOOL_MEDIA_RENDERER_AREA === 'string') {
      try {
        ctx.register({
          id: 'understudy.card', area: sdk.TOOL_MEDIA_RENDERER_AREA,
          data: {
            tools: ['understudy_audition'],
            render: ({ file, fallback }) => jsx(AuditionCard, { ctx, file, fallback }),
          },
        })
      } catch (e) { /* older hosts: MEDIA: still shows the PNG */ }
    }
  },
}
