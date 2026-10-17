/* Shared code of the AtmosTransport v0.5 CATRINE benchmark pages.
   Data files register themselves with SITE.put(key, value) and are loaded on demand by adding a
   <script> element, so the pages work from a web server and from disk. */
"use strict";
const SITE = (() => {
  const store = {}, waiting = {};
  const version = (document.querySelector('meta[name="build"]') || {}).content || "";

  function put(key, value) {
    store[key] = value;
    (waiting[key] || []).forEach(w => w.resolve(value));
    delete waiting[key];
  }
  function load(key, file) {
    if (key in store) return Promise.resolve(store[key]);
    return new Promise((resolve, reject) => {
      const first = !waiting[key];
      (waiting[key] = waiting[key] || []).push({resolve, reject});
      if (!first) return;
      const s = document.createElement("script");
      s.src = `data/${file}?v=${version}`;
      s.onerror = () => { (waiting[key] || []).forEach(w => w.reject(new Error(`missing data/${file}`))); delete waiting[key]; };
      document.head.appendChild(s);
    });
  }
  const get = key => store[key];
  const ready = fn => (document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", fn) : fn());

  // ---- request tokens: a redraw started later wins; an older one that finishes afterwards is dropped
  function gate() {
    let n = 0;
    return () => { const t = ++n; return () => t === n; };
  }

  // ---- formatting
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const monthLabel = m => m === "period" ? "Period mean" : `${MONTHS[+m.slice(5, 7) - 1]} ${m.slice(0, 4)}`;
  const midMonth = m => `${m}-15`;
  const fmt = (x, d = 3, sign = false) => {   // fixed decimals, typographic minus
    if (x === null || x === undefined || !isFinite(x)) return "–";
    const s = x.toFixed(d);
    const zero = +s === 0;
    return (zero ? "" : x < 0 ? "−" : sign ? "+" : "") + s.replace("-", "");
  };
  const sigfmt = (x, s = 3, sign = false) => {   // significant figures, no exponent for ordinary magnitudes
    if (x === null || x === undefined || !isFinite(x)) return "–";
    if (x === 0) return "0";
    const e = Math.floor(Math.log10(Math.abs(x)));
    if (e < -3 || e > 5) {
      const [mant, ex] = x.toExponential(s - 1).split("e");
      return `${sign && x > 0 ? "+" : ""}${mant.replace("-", "−")}×10<sup>${(+ex < 0 ? "−" : "") + Math.abs(+ex)}</sup>`;
    }
    return (sign && x > 0 ? "+" : "") + x.toFixed(Math.max(0, s - 1 - e));
  };
  const niceCeil = x => {
    if (!(x > 0) || !isFinite(x)) return 1;
    const e = Math.pow(10, Math.floor(Math.log10(x)));
    for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * e >= x * 0.999) return m * e;
    return 10 * e;
  };
  const decimalsFor = (range, n = 4) => Math.max(0, Math.min(6, Math.ceil(-Math.log10(Math.max(range, 1e-12) / n)) + 1));

  // ---- page state in the URL hash (#tracer=co2&band=column), so a view can be linked
  // allowed: {key: [values]} for keys with a fixed set of values; anything else falls back to the default
  function state(defaults, allowed = {}) {
    const h = new URLSearchParams(location.hash.slice(1));
    const s = {...defaults};
    for (const k of Object.keys(defaults)) if (h.has(k) && (!allowed[k] || allowed[k].includes(h.get(k)))) s[k] = h.get(k);
    return s;
  }
  const keys = list => list.map(x => x.key);
  function saveState(s) {
    history.replaceState(null, "", "#" + new URLSearchParams(Object.entries(s).filter(([, v]) => v !== null && v !== undefined)).toString());
  }

  // ---- controls
  function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (k === "style") e.setAttribute("style", v);
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat(Infinity)) if (c !== null && c !== undefined) e.append(c.nodeType ? c : document.createTextNode(String(c)));
    return e;
  }
  function html(markup) { const e = document.createElement("span"); e.innerHTML = markup; return e; }   // trusted generator strings only
  function control(label, ...inner) { return el("div", {class: "control"}, el("span", {class: "lab"}, label), ...inner); }
  function seg(options, value, onChange, aria) {
    const box = el("div", {class: "seg", role: "group", "aria-label": aria || ""});
    for (const o of options) {
      const b = el("button", {type: "button", "aria-pressed": String(o.key === value), title: o.title || null});
      b.innerHTML = o.name;       // trusted labels from the generator (may hold <sub>/<sup>)
      b.addEventListener("click", () => {
        box.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", "false"));
        b.setAttribute("aria-pressed", "true");
        onChange(o.key);
      });
      box.append(b);
    }
    box.set = key => box.querySelectorAll("button").forEach((x, i) => x.setAttribute("aria-pressed", String(options[i].key === key)));
    return box;
  }
  function monthSelect(months, value, onChange, withPeriod = true) {
    const opts = (withPeriod ? ["period"] : []).concat(months);
    const sel = el("select", {"aria-label": "month"}, opts.map(m => el("option", {value: m, selected: m === value}, monthLabel(m))));
    sel.addEventListener("change", () => onChange(sel.value));
    const step = d => { const i = opts.indexOf(sel.value) + d; if (i >= 0 && i < opts.length) { sel.value = opts[i]; onChange(sel.value); } };
    return el("span", {class: "stepper"},
      el("button", {type: "button", title: "previous month", "aria-label": "previous month", onclick: () => step(-1)}, "‹"), sel,
      el("button", {type: "button", title: "next month", "aria-label": "next month", onclick: () => step(1)}, "›"));
  }
  // Dataset legend that doubles as the visibility toggle for every chart on the page.
  function legend(datasets, hidden, onChange) {
    const box = el("div", {class: "legend", role: "group", "aria-label": "datasets"});
    for (const d of datasets) {
      const b = el("button", {type: "button", "aria-pressed": String(!hidden.has(d.key)), title: d.title || "show or hide"},
        el("span", {class: "key" + (d.dash === "dot" ? " dot" : ""), style: `border-top-color:${d.color}`}), html(d.name));
      b.addEventListener("click", () => {
        hidden.has(d.key) ? hidden.delete(d.key) : hidden.add(d.key);
        b.setAttribute("aria-pressed", String(!hidden.has(d.key)));
        onChange();
      });
      box.append(b);
    }
    return box;
  }

  // ---- datasets
  const meta = () => store.meta;
  const tracer = k => meta().tracers.find(t => t.key === k);
  const band = k => meta().bands.find(b => b.key === k);
  const run = k => meta().runs.find(r => r.key === k);
  const mainRuns = () => meta().runs.filter(r => r.role === "signoff");
  const GC = () => meta().gc;
  const hasMonths = r => r && r.months.length > 0;

  // ---- charts (D3 v7, assets/d3.min.js)
  // Line and bar charts are SVG; heatmaps draw their cells into a <canvas> under an SVG layer that holds the
  // axes, coastlines and tropopause. A chart redraws itself when its container changes width (ResizeObserver).
  // Pointer input: hovering (mouse) or tapping and sliding sideways (touch) shows every series at the nearest
  // point; vertical swipes scroll the page (touch-action: pan-y), so a chart never traps the page on a phone.
  // Time axes of daily and 3-hourly charts zoom by dragging across them with the mouse (on a touch screen only
  // after "Zoom" is switched on); "Reset zoom" or a double-click restores the full period.
  const INK = "#0b0b0b", INK2 = "#52514e", MUTED = "#898781", GRID = "#e1e0d9", AXIS = "#c3c2b7", REF = "#1f1f1d";
  const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', TICK = 11.5;
  const NS = "http://www.w3.org/2000/svg";
  let uid = 0;
  const measure = document.createElement("canvas").getContext("2d");
  const textWidth = (s, size = TICK) => { measure.font = `${size}px ${FONT}`; return measure.measureText(String(s).replace(/<[^>]*>/g, "")).width; };
  const latLabel = v => { const r = Math.round(v); return r === 0 ? "Eq" : `${Math.abs(r)}° ${r > 0 ? "N" : "S"}`; };
  const lonLabel = v => { const r = Math.round(v); return r === 0 || Math.abs(r) === 180 ? `${Math.abs(r)}°` : `${Math.abs(r)}° ${r > 0 ? "E" : "W"}`; };
  const within = (v, d) => v >= Math.min(d[0], d[1]) - 1e-12 && v <= Math.max(d[0], d[1]) + 1e-12;
  const valFmt = (digits, unit = "", sign = false) => v => `${fmt(v, digits, sign)}${unit ? " " + unit : ""}`;
  const shortName = s => String(s.name || "").replace(" v0.5", "");

  // Text with <sub>, <sup>, <b>, <i> (trusted generator strings) into an SVG <text> element as tspans.
  function rich(text, markup, size = TICK) {
    const t = text.node ? text.node() : text;
    while (t.firstChild) t.removeChild(t.firstChild);
    let dy = 0, small = 0, bold = 0, ital = 0;
    for (const part of String(markup).split(/(<\/?(?:sub|sup|b|i)>)/)) {
      if (!part) continue;
      const m = part.match(/^<(\/?)(sub|sup|b|i)>$/);
      if (m) {
        const s = m[1] ? -1 : 1;
        if (m[2] === "sup") { dy -= s * 0.42 * size; small += s; }
        else if (m[2] === "sub") { dy += s * 0.25 * size; small += s; }
        else if (m[2] === "b") bold += s; else ital += s;
        continue;
      }
      const ts = document.createElementNS(NS, "tspan");
      if (dy) ts.setAttribute("dy", dy);          // relative to the previous tspan
      dy = 0;
      if (small > 0) ts.setAttribute("font-size", 0.74 * size);
      if (bold > 0) ts.setAttribute("font-weight", 600);
      if (ital > 0) ts.setAttribute("font-style", "italic");
      ts.textContent = part.replace(/&nbsp;/g, " ").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
      t.appendChild(ts);
    }
    return text;
  }
  // Tick labels: fixed decimals from the tick step; mantissa × 10^n for very small or large numbers.
  function tickFormat(ticks, suffix = "") {
    const big = d3.max(ticks, Math.abs) || 0;
    if (big && (big < 1e-3 || big >= 1e6)) return v => {
      if (v === 0) return "0";
      const e = Math.floor(Math.log10(Math.abs(v)) + 1e-9), mant = +(v / Math.pow(10, e)).toPrecision(3);
      return `${String(mant).replace("-", "−")}×10<sup>${e < 0 ? "−" : ""}${Math.abs(e)}</sup>${suffix}`;
    };
    const step = +(ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : Math.abs(ticks[0]) || 1).toPrecision(6);   // 0.09999… -> 0.1
    const d = Math.min(8, d3.precisionFixed(step));
    return v => (v === 0 ? "0" : fmt(v, d)) + suffix;
  }
  // Time ticks for a domain [t0, t1] (ms) and a plot width: month starts (every 1, 2, 3, 4, 6 or 12 months),
  // days or hours when zoomed in. Two lines: month (or day) and, where it changes, the year (or month and year).
  function timeTicks(d0, d1, width) {
    const maxN = Math.max(2, Math.floor(width / Math.min(90, Math.max(46, width / 12)))), days = (d1 - d0) / 864e5, out = [];
    if (days > 75) {
      const k = [1, 2, 3, 6, 12].find(s => days / 30.44 / s <= maxN) || 24;
      const iv = k === 24 ? d3.utcYear.every(2) : d3.utcMonth.every(k);
      let prev = null;
      for (const t of iv.range(new Date(d0), new Date(d1 + 1))) {
        const y = t.getUTCFullYear();
        out.push({v: +t, l1: k >= 12 ? String(y) : MONTHS[t.getUTCMonth()], l2: k < 12 && y !== prev ? String(y) : ""});
        prev = y;
      }
    } else if (days > 2.5) {
      const k = [1, 2, 3, 5, 7, 10, 15].find(s => days / s <= maxN) || 15;
      let prev = null;
      for (const t of d3.utcDay.every(k).range(new Date(d0), new Date(d1 + 1))) {
        if (k > 1 && t.getUTCDate() > 31 - k / 2) continue;       // skip the crowded tick before a month start
        const mo = d3.utcFormat("%b %Y")(t);
        out.push({v: +t, l1: String(t.getUTCDate()), l2: mo !== prev ? mo : ""});
        prev = mo;
      }
    } else {
      const k = [1, 3, 6, 12].find(s => days * 24 / s <= maxN) || 12;
      let prev = null;
      for (const t of d3.utcHour.every(k).range(new Date(d0), new Date(d1 + 1))) {
        const day = d3.utcFormat("%-d %b %Y")(t);
        out.push({v: +t, l1: d3.utcFormat("%H:%M")(t), l2: day !== prev ? day : ""});
        prev = day;
      }
    }
    return out;
  }
  const toTime = v => v === null || v === undefined ? null : typeof v === "number" ? v : v instanceof Date ? +v
    : Date.parse(String(v).replace(" ", "T") + (String(v).length <= 10 ? "T00:00Z" : "Z"));
  function autoDomain(vals, ax = {}) {
    let lo = d3.min(vals), hi = d3.max(vals);
    if (lo === undefined) { lo = 0; hi = 1; }
    if (ax.tozero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    let pad = (hi - lo) * 0.06;
    if (!(pad > 0)) pad = Math.abs(hi) * 0.05 || 1;           // a flat line
    return [ax.tozero && lo === 0 ? 0 : lo - pad, ax.tozero && hi === 0 ? 0 : hi + pad];
  }
  function nearest(arr, v, ix) {
    if (!arr.length) return null;
    const i = d3.bisector(q => q[ix]).left(arr, v);
    const a = arr[Math.max(0, i - 1)], b = arr[Math.min(arr.length - 1, i)];
    return Math.abs(a[ix] - v) <= Math.abs(b[ix] - v) ? a : b;
  }

  // Chart host: the container div gets an SVG, a tooltip, the zoom buttons and (heatmaps) a canvas.
  const shown = new Set();
  const widthOf = div => Math.floor(div.getBoundingClientRect().width);
  const resizer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(entries => {
    for (const e of entries) {
      const div = e.target, C = div._chart;
      if (!C || widthOf(div) === C.width) continue;
      cancelAnimationFrame(C.raf);
      C.raf = requestAnimationFrame(() => { if (div._chart === C) C.render(); });
    }
  });
  const armed = new Set();
  document.addEventListener("pointerdown", ev => {
    for (const C of [...shown]) if (!C.div.contains(ev.target)) C.clear();
    for (const C of [...armed]) if (!C.div.contains(ev.target)) C.arm(false);
  }, {capture: true, passive: true});
  // a new device pixel ratio (window moved to another screen, browser zoom): redraw the canvases sharply
  (function watchDpr() {
    if (!window.matchMedia) return;
    const mq = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`), on = () => {
      mq.removeEventListener("change", on);
      document.querySelectorAll(".chart").forEach(d => d._chart && d._chart.render && d._chart.render());
      watchDpr();
    };
    mq.addEventListener("change", on);
  })();
  function host(div, kind) {
    div.classList.remove("loading");
    const note = div.querySelector(".empty");
    if (note) note.remove();
    if (div._chart && div._chart.kind === kind) return div._chart;
    if (div._chart) { div._chart.clear(); armed.delete(div._chart); }
    div.innerHTML = "";
    div.classList.add("chart");
    const C = {kind, div, width: -1, zoom: null, zoomMode: false};
    C.arm = on => {
      C.zoomMode = on;
      on ? armed.add(C) : armed.delete(C);
      div.classList.toggle("zoom-armed", on);
      const b = C.tools.querySelector(".arm");
      if (b) b.setAttribute("aria-pressed", String(on));
    };
    if (kind === "heat") C.canvas = div.appendChild(el("canvas", {class: "cells", "aria-hidden": "true"}));
    C.svg = d3.select(div).append("svg").attr("class", "chart-svg");
    C.tools = div.appendChild(el("div", {class: "chart-tools"}));
    C.tip = div.appendChild(el("div", {class: "tip", hidden: true}));
    C.clear = () => { C.tip.hidden = true; shown.delete(C); if (C.unmark) C.unmark(); };
    C.showTip = (markup, x, y, touch) => {     // x, y: pointer position in the container
      const tip = C.tip;
      tip.innerHTML = markup;
      tip.hidden = false;
      shown.add(C);
      const W = div.clientWidth, H = div.clientHeight, w = tip.offsetWidth, h = tip.offsetHeight;
      let left = x + 14;
      if (left + w > W - 2) left = x - 14 - w;
      if (left < 2) left = Math.max(2, Math.min(W - w - 2, x - w / 2));
      const top = touch ? 2 : Math.max(2, Math.min(H - h - 2, y - h / 2));
      tip.style.left = `${left}px`;
      tip.style.top = `${top}px`;
    };
    div._chart = C;
    if (resizer) resizer.observe(div);
    return C;
  }
  function dropChart(div) {
    if (!div._chart) return;
    div._chart.clear();
    armed.delete(div._chart);
    if (resizer) resizer.unobserve(div);
    div._chart = null;
    div.classList.remove("chart");
  }
  // Zoom buttons: "Reset zoom" while zoomed; "Zoom" (touch screens only, see site.css) arms drag-to-zoom once.
  function zoomTools(C, zoomable) {
    C.tools.innerHTML = "";
    if (!zoomable) return;
    if (C.zoom) C.tools.append(el("button", {type: "button", class: "reset", onclick: () => { C.zoom = null; C.arm(false); C.render(); }}, "Reset zoom"));
    C.tools.append(el("button", {type: "button", class: "arm", "aria-pressed": String(C.zoomMode), title: "then drag across the chart",
                                 onclick: () => C.arm(!C.zoomMode)}, "Zoom"));
    C.arm(C.zoomMode);
  }

  // Dataset series for S.chart: colour, dash and name come from the dataset (run, GCHP, observations).
  function series(ds, x, y, extra = {}) { return {key: ds.key, name: ds.name, color: ds.color, dash: ds.dash, x, y, ...extra}; }
  // Time axis of the months on the site; res sets the hover label: "month", "day" or "3h".
  function timeAxis(res = "month", extra = {}) {
    return {type: "time", domain: xRange(), zoom: res !== "month",
            hover: res === "month" ? "%B %Y" : res === "day" ? "%-d %b %Y" : "%-d %b %Y, %H:%M UTC", ...extra};
  }

  // Line chart.
  // spec: {height, x: {type: "time"|"linear", domain, ticks, tickFmt, label, hover (d3 utcFormat or function), zoom, zero},
  //        y | panels: [{y, series, refs, name, weight}], series, refs, hoverAxis: "x" | "y" (profiles: values on x)}
  // axis: {domain, log, ticks, tickFmt, suffix, label, tozero, zero: true | "ref" (the reference's zero line), zeroColor, fmt}
  // series: {key, name, label, color, dash, width, x, y, line (default true), markers: true | "open", size, fmt}
  // refs: [{value, label}] dotted horizontal reference lines.
  function chart(div, spec) {
    const C = host(div, "line"), time = (spec.x || {}).type === "time", ix = spec.hoverAxis === "y" ? "y" : "x";
    const x = {...(spec.x || {})};
    if (time && x.domain) x.domain = x.domain.map(toTime);
    const panels = (spec.panels || [{y: spec.y, series: spec.series, refs: spec.refs}]).map(p => ({...p, y: {...(p.y || {})}, refs: p.refs || []}));
    for (const p of panels) p.series = (p.series || []).map(s => {
      const xs = s.x || [], ys = s.y || [], n = Math.min(xs.length, ys.length), pts = new Array(n);
      for (let k = 0; k < n; k++) {
        const xv = time ? toTime(xs[k]) : xs[k], yv = ys[k];
        pts[k] = {x: xv === null || xv === undefined || !isFinite(xv) ? null : +xv, y: yv === null || yv === undefined || !isFinite(yv) ? null : +yv};
      }
      const sorted = pts.filter(q => q.x !== null && q.y !== null).sort((a, b) => a[ix] - b[ix]);
      const gaps = [];
      for (let k = 1; k < sorted.length; k++) gaps.push(sorted[k][ix] - sorted[k - 1][ix]);
      return {...s, pts, sorted, spacing: gaps.length ? d3.median(gaps) : 0};
    });
    const base = JSON.stringify(x.domain || null);
    if (C.base !== base) { C.zoom = null; C.base = base; }
    C.spec = {...spec, x, panels, time, ix};
    C.render = () => renderLine(C);
    C.render();
    return C;
  }
  function renderLine(C) {
    const div = C.div, sp = C.spec, W = widthOf(div);
    C.width = W;
    C.clear();
    if (!W) { C.svg.selectAll("*").remove(); C.tools.innerHTML = ""; return; }   // hidden (closed <details>): drawn when it opens
    const H = sp.height || 320, P = sp.panels, ix = sp.ix, vx = ix === "x" ? "y" : "x", xa = sp.x, time = sp.time;
    const zoomable = time && ix === "x" && xa.zoom !== false;
    const xd = (zoomable && C.zoom) || xa.domain || autoDomain(P.flatMap(p => p.series.flatMap(s => s.sorted.map(q => q.x))), xa);
    const inView = q => !(zoomable && C.zoom) || (q.x >= xd[0] && q.x <= xd[1]);
    // vertical layout
    const mt = 10, mb = (time ? 36 : 24) + (xa.label ? 18 : 0), gap = 18;
    const avail = H - mt - mb - gap * (P.length - 1), wsum = d3.sum(P, p => p.weight || 1);
    let top = mt;
    for (const p of P) {
      p.h = Math.max(40, avail * (p.weight || 1) / wsum);
      p.top = top;
      top += p.h + gap;
      const ya = p.y;
      p.yd = ya.domain || autoDomain(p.series.flatMap(s => s.sorted.filter(inView).map(q => q.y)).concat(p.refs.map(r => r.value)), ya);
      p.Y = (ya.log ? d3.scaleLog() : d3.scaleLinear()).domain(p.yd).range([p.top + p.h, p.top]);
      p.ticks = ya.ticks ? ya.ticks.filter(t => within(t, p.yd)) : p.Y.ticks(Math.max(2, Math.round(p.h / 46)));
      const f = ya.tickFmt || tickFormat(p.ticks, ya.suffix || "");
      p.labels = p.ticks.map(f);
    }
    const bottom = P[P.length - 1].top + P[P.length - 1].h;
    const labW = d3.max(P.flatMap(p => p.labels.map(l => textWidth(l)))) || 16;
    const ml = Math.ceil(labW + 9 + (P.some(p => p.y.label) ? 21 : 0)), mr = 14, pw = Math.max(60, W - ml - mr);
    const X = (time ? d3.scaleUtc() : xa.log ? d3.scaleLog() : d3.scaleLinear()).domain(xd).range([0, pw]);
    let xt;
    if (time) xt = timeTicks(+xd[0], +xd[1], pw);
    else {
      const vals = xa.ticks ? (typeof xa.ticks === "function" ? xa.ticks(pw) : xa.ticks).filter(t => within(t, xd)) : X.ticks(Math.max(3, Math.round(pw / 75)));
      const f = xa.tickFmt || tickFormat(vals, xa.suffix || "");
      xt = vals.map(v => ({v, l1: f(v)}));
    }
    const svg = C.svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${ml},0)`);
    const hline = (sel, y, color, width, dash) => sel.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y).attr("y2", y)
      .attr("stroke", color).attr("stroke-width", width).attr("stroke-dasharray", dash || null);
    for (const p of P) {
      const pg = g.append("g");
      p.ticks.forEach((t, k) => {
        const y = Math.round(p.Y(t)) + 0.5;
        hline(pg, y, GRID, 1);
        rich(pg.append("text").attr("class", "tick").attr("x", -8).attr("y", y).attr("dy", "0.34em").attr("text-anchor", "end"), p.labels[k]);
      });
      for (const t of xt) {
        const x = Math.round(X(t.v)) + 0.5;
        pg.append("line").attr("x1", x).attr("x2", x).attr("y1", p.top).attr("y2", p.top + p.h).attr("stroke", GRID);
      }
      pg.append("line").attr("x1", 0).attr("x2", pw).attr("y1", p.top + p.h + 0.5).attr("y2", p.top + p.h + 0.5).attr("stroke", AXIS);
      const zc = ya => ya.zeroColor || (ya.zero === "ref" ? REF : AXIS), zw = ya => ya.zero === "ref" ? 1.6 : 1;
      if (ix === "x" && p.y.zero && within(0, p.yd)) hline(pg, Math.round(p.Y(0)) + 0.5, zc(p.y), zw(p.y));
      if (ix === "y" && xa.zero && within(0, xd)) {
        const x0 = Math.round(X(0)) + 0.5;
        pg.append("line").attr("x1", x0).attr("x2", x0).attr("y1", p.top).attr("y2", p.top + p.h).attr("stroke", zc(xa)).attr("stroke-width", zw(xa));
      }
      for (const r of p.refs) {
        const y = p.Y(r.value);
        hline(pg, y, r.color || MUTED, 1.2, "2 3");
        if (!r.label) continue;
        // label above or below the line, right or left: wherever it covers the fewest data points
        const lw = textWidth(r.label, 11) + 6, spots = [[pw - 2, y - 5, "end"], [pw - 2, y + 14, "end"], [2, y - 5, "start"], [2, y + 14, "start"]];
        const pts = p.series.flatMap(s => s.sorted.filter(inView).map(q => [X(q.x), p.Y(q.y)]));
        const cover = ([x, yb, a]) => { const x0 = a === "end" ? x - lw : x, x1 = x0 + lw;
          return pts.filter(([px, py]) => px >= x0 - 4 && px <= x1 + 4 && py >= yb - 13 && py <= yb + 4).length + (yb - 12 < p.top || yb > p.top + p.h ? 1e6 : 0); };
        const [lx, ly, la] = spots.reduce((b, c) => cover(c) < cover(b) ? c : b);
        rich(pg.append("text").attr("class", "note").attr("x", lx).attr("y", ly).attr("text-anchor", la), r.label, 11);
      }
      if (p.y.label) rich(pg.append("text").attr("class", "axis-title").attr("text-anchor", "middle")
        .attr("transform", `translate(${-(ml - 13)},${p.top + p.h / 2}) rotate(-90)`), p.y.label, 12);
      const clip = `clip-${++uid}`;
      pg.append("clipPath").attr("id", clip).append("rect").attr("x", -5).attr("y", p.top - 5).attr("width", pw + 10).attr("height", p.h + 10);
      const sg = pg.append("g").attr("clip-path", `url(#${clip})`);
      // GCHP last and thinner, so a run on top of it does not hide it (and vice versa)
      for (const s of [...p.series.filter(s => s.key !== "gc"), ...p.series.filter(s => s.key === "gc")]) {
        const w = s.key === "gc" ? Math.min(s.width || 2, 1.5) : (s.width || 2);
        if (s.line !== false) sg.append("path")
          .attr("d", d3.line().defined(q => q.x !== null && q.y !== null).x(q => X(q.x)).y(q => p.Y(q.y))(s.pts))
          .attr("fill", "none").attr("stroke", s.color).attr("stroke-width", w).attr("stroke-linejoin", "round")
          .attr("stroke-linecap", "round").attr("stroke-dasharray", s.dash === "dot" ? "1 4" : null);
        if (s.markers) {
          const open = s.markers === "open", r = (s.size || 5) / 2;
          sg.append("g").selectAll("circle").data(s.sorted.filter(inView)).join("circle")
            .attr("cx", q => X(q.x)).attr("cy", q => p.Y(q.y)).attr("r", r)
            .attr("fill", open ? "none" : s.color).attr("stroke", open ? s.color : null).attr("stroke-width", open ? 1.4 : null);
        }
      }
    }
    for (const t of xt) {
      const x = X(t.v);
      rich(g.append("text").attr("class", "tick").attr("x", x).attr("y", bottom + 16).attr("text-anchor", "middle"), t.l1);
      if (t.l2) g.append("text").attr("class", "tick").attr("x", x).attr("y", bottom + 30).attr("text-anchor", "middle").text(t.l2);
    }
    if (xa.label) rich(g.append("text").attr("class", "axis-title").attr("x", pw / 2).attr("y", H - 5).attr("text-anchor", "middle"), xa.label, 12);

    // hover / tap
    const top0 = P[0].top, mark = g.append("g").attr("class", "mark").attr("pointer-events", "none");
    const brush = g.append("rect").attr("class", "brush").attr("y", top0).attr("height", bottom - top0).attr("display", "none");
    const ov = g.append("rect").attr("class", "overlay").attr("x", 0).attr("y", top0).attr("width", pw).attr("height", bottom - top0).attr("fill", "transparent");
    const IS = ix === "x" ? X : P[0].Y;
    const head = v => ix === "x" ? (typeof xa.hover === "function" ? xa.hover(v) : time ? d3.utcFormat(xa.hover || "%-d %b %Y")(new Date(v)) : fmt(v, 2))
      : (P[0].y.hover ? P[0].y.hover(v) : fmt(v, 2));
    const valueFmt = (s, p) => s.fmt || (ix === "x" ? p.y.fmt : xa.fmt) || (v => fmt(v, 3));
    C.unmark = () => mark.selectAll("*").remove();
    C.hover = (mx, my, touch) => {
      const pos = ix === "x" ? mx : my, at = IS.invert(pos);
      let best = null;
      for (const p of P) for (const s of p.series) {
        const q = nearest(s.sorted, +at, ix);
        if (!q || (ix === "x" && !within(q.x, [+xd[0], +xd[1]]))) continue;
        const d = Math.abs(IS(q[ix]) - pos);
        if (!best || d < best.d) best = {d, v: q[ix]};
      }
      if (!best) { C.clear(); return; }
      const sv = best.v, spx = IS(sv), scale = ix === "x" && !xa.log ? pw / (xd[1] - xd[0]) : 0;
      let body = "";
      mark.selectAll("*").remove();
      if (ix === "x") mark.append("line").attr("class", "guide").attr("x1", spx).attr("x2", spx).attr("y1", top0).attr("y2", bottom);
      else mark.append("line").attr("class", "guide").attr("x1", 0).attr("x2", pw).attr("y1", spx).attr("y2", spx);
      for (const p of P) {
        const rows = [];
        for (const s of [...p.series.filter(s => s.key === "gc"), ...p.series.filter(s => s.key !== "gc")]) {
          const q = nearest(s.sorted, sv, ix);
          if (!q || Math.abs(IS(q[ix]) - spx) > Math.max(14, 0.6 * s.spacing * scale)) continue;
          const when = time && Math.abs(q.x - sv) > 6e4 ? ` <span class="when">${d3.utcFormat(s.whenFmt || "%-d %b %H:%M")(new Date(q.x))}</span>` : "";
          const sw = s.line === false ? `<span class="sw pt${s.markers === "open" ? " open" : ""}" style="color:${s.color}"></span>`
            : `<span class="sw${s.dash === "dot" ? " dot" : ""}" style="color:${s.color}"></span>`;
          rows.push(`<div class="row">${sw}<span class="n">${s.label || shortName(s)}${when}</span><span class="v">${valueFmt(s, p)(q[vx])}</span></div>`);
          mark.append("circle").attr("cx", X(q.x)).attr("cy", p.Y(q.y)).attr("r", 3.6).attr("fill", s.color).attr("stroke", "#fff").attr("stroke-width", 1.5);
        }
        if (rows.length) body += (P.length > 1 && p.name ? `<div class="sec">${p.name}</div>` : "") + rows.join("");
      }
      if (!body) { C.clear(); return; }
      C.showTip(`<div class="h">${head(sv)}</div>${body}`, ml + (ix === "x" ? spx : mx), ix === "x" ? my : spx, touch);
    };
    // pointer handling: mouse hover + drag-to-zoom; touch tap / sideways slide, zoom only when armed
    let drag = null;
    const clampX = v => Math.max(0, Math.min(pw, v));
    ov.on("pointerdown", ev => {
      const [mx, my] = d3.pointer(ev, g.node());
      const mouse = ev.pointerType !== "touch";
      if (zoomable && ((mouse && ev.button === 0) || (!mouse && C.zoomMode))) {
        drag = {x0: clampX(mx)};
        try { ov.node().setPointerCapture(ev.pointerId); } catch (e) { /* capture is optional */ }
        if (!mouse) C.clear();
        if (!mouse) ev.preventDefault();
        return;
      }
      if (!mouse) C.hover(mx, my, true);
    });
    ov.on("pointermove", ev => {
      const [mx, my] = d3.pointer(ev, g.node());
      if (drag) {
        const a = Math.min(drag.x0, clampX(mx)), b = Math.max(drag.x0, clampX(mx));
        if (b - a > 3) { brush.attr("display", null).attr("x", a).attr("width", b - a); if (ev.pointerType !== "touch") C.clear(); }
        if (ev.pointerType !== "touch" && b - a <= 3) C.hover(mx, my, false);
        return;
      }
      if (ev.pointerType !== "touch") C.hover(mx, my, false);
      else if (ev.buttons) C.hover(mx, my, true);
    });
    const endDrag = ev => {
      if (!drag) return;
      const [mx, my] = d3.pointer(ev, g.node());
      const a = Math.min(drag.x0, clampX(mx)), b = Math.max(drag.x0, clampX(mx));
      drag = null;
      brush.attr("display", "none");
      if (b - a > 6) {
        C.zoom = [+X.invert(a), +X.invert(b)];
        C.arm(false);
        C.render();
      } else if (ev.pointerType !== "touch") C.hover(mx, my, false);
      else { C.arm(false); C.hover(mx, my, true); }
    };
    ov.on("pointerup", endDrag);
    ov.on("pointercancel", () => { drag = null; brush.attr("display", "none"); C.clear(); });
    ov.on("pointerleave", ev => { if (ev.pointerType !== "touch" && !drag) C.clear(); });
    ov.on("dblclick", () => { if (C.zoom) { C.zoom = null; C.render(); } });
    zoomTools(C, zoomable);
  }

  // Grouped bar chart. opt: {height, categories, series: [{key, name, label, color, values, extra: [html]}], y: {label, fmt}}
  function bars(div, opt) {
    const C = host(div, "bars");
    C.spec = opt;
    C.render = () => renderBars(C);
    C.render();
    return C;
  }
  function renderBars(C) {
    const div = C.div, o = C.spec, W = widthOf(div);
    C.width = W;
    C.clear();
    if (!W) { C.svg.selectAll("*").remove(); return; }
    const H = o.height || 240, mt = 10, mb = 26, ph = H - mt - mb;
    const vals = o.series.flatMap(s => s.values.filter(v => v !== null && isFinite(v)));
    const Y = d3.scaleLinear().domain([0, (d3.max(vals) || 1) * 1.08]).range([mt + ph, mt]);
    const ticks = Y.ticks(Math.max(2, Math.round(ph / 46))), f = tickFormat(ticks);
    const ml = Math.ceil((d3.max(ticks, t => textWidth(f(t))) || 16) + 9 + (o.y && o.y.label ? 21 : 0)), mr = 10, pw = Math.max(60, W - ml - mr);
    const X0 = d3.scaleBand().domain(o.categories).range([0, pw]).paddingInner(0.3).paddingOuter(0.15);
    const X1 = d3.scaleBand().domain(o.series.map(s => s.key)).range([0, X0.bandwidth()]).padding(0.08);
    const svg = C.svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${ml},0)`);
    const hl = g.append("rect").attr("class", "band-hl").attr("y", mt).attr("height", ph).attr("display", "none");
    for (const t of ticks) {
      const y = Math.round(Y(t)) + 0.5;
      g.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y).attr("y2", y).attr("stroke", t === 0 ? AXIS : GRID);
      rich(g.append("text").attr("class", "tick").attr("x", -8).attr("y", y).attr("dy", "0.34em").attr("text-anchor", "end"), f(t));
    }
    if (o.y && o.y.label) rich(g.append("text").attr("class", "axis-title").attr("text-anchor", "middle")
      .attr("transform", `translate(${-(ml - 13)},${mt + ph / 2}) rotate(-90)`), o.y.label, 12);
    for (const s of o.series) g.append("g").selectAll("rect").data(o.categories.map((c, k) => [c, s.values[k]]).filter(d => d[1] !== null && isFinite(d[1])))
      .join("rect").attr("x", d => X0(d[0]) + X1(s.key)).attr("width", X1.bandwidth()).attr("y", d => Y(d[1])).attr("height", d => Y(0) - Y(d[1])).attr("fill", s.color);
    for (const c of o.categories) g.append("text").attr("class", "tick").attr("x", X0(c) + X0.bandwidth() / 2).attr("y", mt + ph + 17).attr("text-anchor", "middle").text(c);
    const ov = g.append("rect").attr("class", "overlay").attr("x", 0).attr("y", mt).attr("width", pw).attr("height", ph).attr("fill", "transparent");
    C.unmark = () => hl.attr("display", "none");
    const hover = (mx, my, touch) => {
      const k = d3.minIndex(o.categories, c => Math.abs(X0(c) + X0.bandwidth() / 2 - mx));
      const c = o.categories[k];
      hl.attr("display", null).attr("x", X0(c) - X0.step() * X0.paddingInner() / 2).attr("width", X0.step());
      const rows = o.series.map(s => s.values[k] === null || !isFinite(s.values[k]) ? "" :
        `<div class="row"><span class="sw pt sq" style="color:${s.color}"></span><span class="n">${s.label || shortName(s)}</span>` +
        `<span class="v">${(o.y && o.y.fmt ? o.y.fmt : v => fmt(v, 3))(s.values[k])}</span>${s.extra ? `<span class="x">${s.extra[k]}</span>` : ""}</div>`).join("");
      C.showTip(`<div class="h">${c}</div>${rows}`, ml + X0(c) + X0.bandwidth() / 2, my, touch);
    };
    ov.on("pointermove", ev => { const [mx, my] = d3.pointer(ev, g.node()); if (ev.pointerType !== "touch" || ev.buttons) hover(mx, my, ev.pointerType === "touch"); });
    ov.on("pointerdown", ev => { if (ev.pointerType === "touch") { const [mx, my] = d3.pointer(ev, g.node()); hover(mx, my, true); } });
    ov.on("pointerleave", ev => { if (ev.pointerType !== "touch") C.clear(); });
    ov.on("pointercancel", () => C.clear());
  }

  // Heatmap: cells drawn into a canvas (device pixels, cell edges rounded to whole pixels, so no seams and no
  // smoothing), axes, frame, coastlines and lines in SVG on top, colour bar on the right.
  // opt: {xc, yc (cell centres; z is row-major, row = y index), z, zmin, zmax, scale, unit, digits, title,
  //       x: {domain, ticks, tickFmt, label}, y: {domain, ticks, tickFmt, label, log}, aspect | height(W),
  //       coast: {lon, lat}, lines: [{x, y, color, width}], readout: (i, j) => html (where), note: (i, j) => html (after the value)}
  const LUTS = new WeakMap();
  function lut(scale, n = 512) {
    if (!LUTS.has(scale)) {
      const s = d3.scaleLinear().domain(scale.map(d => d[0])).range(scale.map(d => d[1])).interpolate(d3.interpolateRgb).clamp(true);
      LUTS.set(scale, Array.from({length: n}, (_, k) => s(k / (n - 1))));
    }
    return LUTS.get(scale);
  }
  function cellEdges(c, log) {       // midpoints between centres; half a spacing beyond the ends (in log space if log)
    const f = log ? Math.log : v => v, g = log ? Math.exp : v => v, t = c.map(f), n = t.length, e = [g(t[0] - (t[1] - t[0]) / 2)];
    for (let k = 1; k < n; k++) e.push(g((t[k - 1] + t[k]) / 2));
    e.push(g(t[n - 1] + (t[n - 1] - t[n - 2]) / 2));
    return e;
  }
  const cellOf = (e, v) => { for (let k = 0; k < e.length - 1; k++) if (within(v, [e[k], e[k + 1]])) return k; return -1; };
  function heatmap(div, opt) {
    const C = host(div, "heat");
    C.spec = opt;
    C.render = () => renderHeat(C);
    C.render();
    return C;
  }
  function renderHeat(C) {
    const div = C.div, o = C.spec, W = widthOf(div);
    C.width = W;
    C.clear();
    if (!W) { C.svg.selectAll("*").remove(); C.canvas.width = C.canvas.height = 0; return; }
    const nx = o.xc.length, ny = o.yc.length;
    const cs = d3.scaleLinear().domain([o.zmin, o.zmax]), cbGap = 10, cbW = 10;
    const yticks0 = typeof o.y.ticks === "function" ? o.y.ticks(1000) : o.y.ticks, ylab0 = yticks0.map(o.y.tickFmt || String);
    const ml = Math.ceil((d3.max(ylab0, l => textWidth(l)) || 16) + 9 + (o.y.label ? 21 : 0));
    const mt = 6, mb = 22 + (o.x.label ? 18 : 0);
    // colour-bar ticks follow the bar height, the right margin their labels: settle both (two passes at most)
    const labW = ts => { const f = tickFormat(ts); return Math.ceil(d3.max(ts, t => textWidth(f(t), 11)) || 20); };
    let cbTicks = cs.ticks(5), mr, pw, ph;
    for (let pass = 0; pass < 2; pass++) {
      mr = cbGap + cbW + 6 + labW(cbTicks) + 20;
      pw = Math.max(80, W - ml - mr);
      ph = o.aspect ? Math.round(pw * o.aspect) : Math.max(140, o.height(W) - mt - mb);
      const want = cs.ticks(Math.max(3, Math.round(ph * 0.92 / 45))), wider = labW(want) > labW(cbTicks);
      cbTicks = want;
      if (!wider) break;
    }
    const H = ph + mt + mb, cbf = tickFormat(cbTicks);
    const xticks = typeof o.x.ticks === "function" ? o.x.ticks(pw) : o.x.ticks;
    const yticks = typeof o.y.ticks === "function" ? o.y.ticks(ph) : o.y.ticks, ylab = yticks.map(o.y.tickFmt || String);
    const X = d3.scaleLinear().domain(o.x.domain).range([0, pw]), Y = (o.y.log ? d3.scaleLog() : d3.scaleLinear()).domain(o.y.domain).range([ph, 0]);
    // cells
    const dpr = window.devicePixelRatio || 1, cv = C.canvas;
    Object.assign(cv.style, {left: `${ml}px`, top: `${mt}px`, width: `${pw}px`, height: `${ph}px`});
    cv.width = Math.round(pw * dpr);
    cv.height = Math.round(ph * dpr);
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    const xe = cellEdges(o.xc), ye = cellEdges(o.yc, o.y.log);
    const xp = xe.map(v => Math.round(X(v) * dpr)), yp = ye.map(v => Math.round(Y(v) * dpr));
    const L = lut(o.scale), n = L.length - 1, span = (o.zmax - o.zmin) || 1;
    for (let j = 0; j < ny; j++) {
      const y0 = Math.min(yp[j], yp[j + 1]), h = Math.max(1, Math.abs(yp[j + 1] - yp[j]));
      for (let i = 0; i < nx; i++) {
        const v = o.z[j * nx + i];
        if (v === null || !isFinite(v)) continue;
        ctx.fillStyle = L[Math.max(0, Math.min(n, Math.round((v - o.zmin) / span * n)))];
        ctx.fillRect(Math.min(xp[i], xp[i + 1]), y0, Math.max(1, Math.abs(xp[i + 1] - xp[i])), h);
      }
    }
    // overlay
    const svg = C.svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${ml},${mt})`);
    const clip = `clip-${++uid}`;
    g.append("clipPath").attr("id", clip).append("rect").attr("width", pw).attr("height", ph);
    const og = g.append("g").attr("clip-path", `url(#${clip})`);
    if (o.coast && o.coast.lon.length) og.append("path").attr("class", "coast")
      .attr("d", d3.line().defined((_, k) => o.coast.lon[k] !== null && o.coast.lat[k] !== null).x((_, k) => X(o.coast.lon[k])).y((_, k) => Y(o.coast.lat[k]))(o.coast.lon));
    for (const ln of o.lines || []) og.append("path").attr("fill", "none").attr("stroke", ln.color || "rgba(20,20,20,0.75)").attr("stroke-width", ln.width || 1.4)
      .attr("d", d3.line().defined((_, k) => ln.y[k] !== null && ln.y[k] > 0).x((_, k) => X(ln.x[k])).y((_, k) => Y(ln.y[k]))(ln.x));
    g.append("rect").attr("width", pw).attr("height", ph).attr("fill", "none").attr("stroke", AXIS);
    yticks.forEach((t, k) => {
      const y = Math.round(Y(t)) + 0.5;
      g.append("line").attr("x1", -4).attr("x2", 0).attr("y1", y).attr("y2", y).attr("stroke", AXIS);
      g.append("text").attr("class", "tick").attr("x", -7).attr("y", y).attr("dy", "0.34em").attr("text-anchor", "end").text(ylab[k]);
    });
    for (const t of xticks) {
      const x = Math.round(X(t)) + 0.5;
      g.append("line").attr("x1", x).attr("x2", x).attr("y1", ph).attr("y2", ph + 4).attr("stroke", AXIS);
      g.append("text").attr("class", "tick").attr("x", x).attr("y", ph + 16).attr("text-anchor", "middle").text((o.x.tickFmt || String)(t));
    }
    if (o.x.label) rich(g.append("text").attr("class", "axis-title").attr("x", pw / 2).attr("y", ph + 34).attr("text-anchor", "middle"), o.x.label, 12);
    if (o.y.label) rich(g.append("text").attr("class", "axis-title").attr("text-anchor", "middle")
      .attr("transform", `translate(${-(ml - 13)},${ph / 2}) rotate(-90)`), o.y.label, 12);
    // colour bar (92 % of the plot height, as before)
    const gid = `grad-${++uid}`, cb0 = ph * 0.04, cbH = ph * 0.92, cx = pw + cbGap;
    const grad = g.append("linearGradient").attr("id", gid).attr("x1", 0).attr("y1", 1).attr("x2", 0).attr("y2", 0);
    for (const [off, col] of o.scale) grad.append("stop").attr("offset", off).attr("stop-color", col);
    g.append("rect").attr("x", cx).attr("y", cb0).attr("width", cbW).attr("height", cbH).attr("fill", `url(#${gid})`);
    cs.range([cb0 + cbH, cb0]);
    for (const t of cbTicks) {
      const y = Math.round(cs(t)) + 0.5;
      g.append("line").attr("x1", cx + cbW).attr("x2", cx + cbW + 3).attr("y1", y).attr("y2", y).attr("stroke", INK2);
      rich(g.append("text").attr("class", "tick cb").attr("x", cx + cbW + 6).attr("y", y).attr("dy", "0.34em"), cbf(t), 11);
    }
    rich(g.append("text").attr("class", "axis-title cb").attr("text-anchor", "middle")
      .attr("transform", `translate(${W - ml - 6},${cb0 + cbH / 2}) rotate(90)`), o.unit, 11.5);
    // hover / tap: cell value
    const cell = g.append("rect").attr("class", "cell-hl").attr("display", "none");
    C.unmark = () => cell.attr("display", "none");
    const hover = (mx, my, touch) => {
      const i = cellOf(xe, X.invert(mx)), j = cellOf(ye, Y.invert(my));
      const v = i < 0 || j < 0 ? NaN : o.z[j * nx + i];
      if (v === null || !isFinite(v)) { C.clear(); return; }
      const x0 = X(xe[i]), x1 = X(xe[i + 1]), y0 = Y(ye[j]), y1 = Y(ye[j + 1]);
      cell.attr("display", null).attr("x", Math.min(x0, x1)).attr("y", Math.min(y0, y1)).attr("width", Math.abs(x1 - x0)).attr("height", Math.abs(y1 - y0));
      C.showTip(`<div class="h">${o.title}</div><div>${o.readout ? o.readout(i, j) : `${latLabel(o.yc[j])}, ${lonLabel(o.xc[i])}`}</div>` +
                `<div><b>${fmt(v, o.digits)}</b> ${o.unit}</div>` + (o.note && o.note(i, j) ? `<div class="when">${o.note(i, j)}</div>` : ""), ml + mx, mt + my, touch);
    };
    const ov = g.append("rect").attr("class", "overlay").attr("width", pw).attr("height", ph).attr("fill", "transparent");
    ov.on("pointermove", ev => { const [mx, my] = d3.pointer(ev, g.node()); if (ev.pointerType !== "touch" || ev.buttons) hover(mx, my, ev.pointerType === "touch"); });
    ov.on("pointerdown", ev => { if (ev.pointerType === "touch") { const [mx, my] = d3.pointer(ev, g.node()); hover(mx, my, true); } });
    ov.on("pointerleave", ev => { if (ev.pointerType !== "touch") C.clear(); });
    ov.on("pointercancel", () => C.clear());
    zoomTools(C, false);
  }
  function emptyNote(div, text) {
    div.classList.remove("loading");
    dropChart(div);
    div.innerHTML = "";
    div.append(el("div", {class: "empty"}, text));
  }
  function clearEmpty(div) { const e = div.querySelector(".empty"); if (e) e.remove(); }
  // Colour scales: sequential = yellow-green-blue (analogous hues, light to dark) for absolute fields;
  // diverging = blue - neutral grey - red for differences (zero is the neutral midpoint).
  const SEQ = [[0, "#fbfbd9"], [0.125, "#eef7b8"], [0.25, "#c9e9b4"], [0.375, "#86cfbb"], [0.5, "#45b5c3"],
               [0.625, "#1f92c0"], [0.75, "#2361a8"], [0.875, "#243696"], [1, "#0c1f5c"]];
  const DIV = [[0, "#0d366b"], [0.15, "#2a78d6"], [0.32, "#86b6ef"], [0.45, "#d9e6f5"], [0.5, "#f0efec"],
               [0.55, "#f8dcda"], [0.68, "#f19c9a"], [0.85, "#e34948"], [1, "#8c1d1c"]];

  function decode(enc) {
    const bin = atob(enc.b64), n = bin.length / 2, out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let c = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
      if (c >= 32768) c -= 65536;
      out[i] = c === -32768 ? NaN : enc.lo + (c + 32767) * enc.step;
    }
    return out;
  }
  function percentile(values, p) {
    const v = Array.from(values).filter(x => isFinite(x)).sort((a, b) => a - b);
    if (!v.length) return NaN;
    return v[Math.min(v.length - 1, Math.max(0, Math.round(p / 100 * (v.length - 1))))];
  }

  // A lat-lon map with coastlines. opt: {z (flat, row-major latitude × longitude), zmin, zmax, scale, unit, digits, title}
  function mapPlot(div, opt) {
    const m = meta();
    return heatmap(div, {...opt, xc: m.mlon, yc: m.mlat, aspect: 0.5, coast: get("coast"),
      x: {domain: [-180, 180], ticks: pw => pw < 400 ? [-120, 0, 120] : [-120, -60, 0, 60, 120], tickFmt: lonLabel},
      y: {domain: [-90, 90], ticks: ph => ph < 150 ? [-60, 0, 60] : [-60, -30, 0, 30, 60], tickFmt: latLabel}});
  }
  // Latitude-pressure section: z rows = pressure levels (meta.plevels), columns = latitude bins (meta.latc).
  function sectionPlot(div, opt) {
    const m = meta(), trop = opt.trop;
    return heatmap(div, {...opt, xc: m.latc, yc: m.plevels, height: W => Math.max(260, Math.round(W * 0.5)),
      lines: trop ? [{x: m.latc, y: trop}] : [],
      readout: (i, j) => `${latLabel(m.latc[i])}, ${m.plevels[j]} hPa`,
      note: i => trop && trop[i] ? `tropopause ${Math.round(trop[i])} hPa` : "",
      x: {domain: [-90, 90], ticks: pw => pw < 260 ? [-60, 0, 60] : [-90, -60, -30, 0, 30, 60, 90], tickFmt: latLabel, label: "latitude"},
      y: {domain: [925, 10], log: true, ticks: [900, 700, 500, 300, 200, 100, 50, 30, 20, 10], label: "dry pressure (hPa)"}});
  }
  // Status strip under the header: data time, progress per run, signoff verdict.
  const STATUS = {PASS: "pass", FAIL: "fail", PENDING: "pending", REPORTED: "reported"};
  function badge(status) {      // icon shape + label: never colour alone
    const cls = STATUS[status] || "pending";
    return el("span", {class: `badge ${cls}`}, el("span", {class: "ico", "aria-hidden": "true"}), cls);
  }
  function statusStrip() {
    const box = document.getElementById("status-strip");
    if (!box || !meta()) return;
    const m = meta();
    // "MERRA-2 v0.5 run: data through May 2022; updated 2026-10-13 16:40 UTC"
    const runs = mainRuns().filter(hasMonths);
    runs.forEach((r, i) => box.append(el("span", {class: "run", title: `${r.months.length} of ${m.all_months.length} protocol months compared with GCHP`},
      el("span", {class: "key", style: `border-top-color:${r.color}`}),
      `${r.name} run: data through ${monthLabel(r.months[r.months.length - 1])}${i === runs.length - 1 ? `; updated ${m.generated}` : ""}`)));
  }
  // Text and elements that belong to one run (data-run="era5") are removed when that run is not on the site.
  function pruneRuns() {
    document.querySelectorAll("[data-run]").forEach(e => { if (!hasMonths(run(e.dataset.run))) e.remove(); });
    document.querySelectorAll("[data-runs]").forEach(e => { if (!e.dataset.runs.split(" ").every(k => hasMonths(run(k)))) e.remove(); });
  }
  // Time axis: the months on the site (the protocol period once the runs are complete), padded by half a month.
  const xRange = () => {
    const ms = meta().months, [y, mo] = ms[ms.length - 1].split("-").map(Number);
    const next = mo === 12 ? `${y + 1}-01` : `${y}-${String(mo + 1).padStart(2, "0")}`;
    return [`${ms[0]}-01`, `${next}-01`];
  };
  const periodText = () => {
    const m = meta(), c = m.common;
    if (!c.length) return "";
    return `Period mean = ${mainRuns().filter(hasMonths).length > 1 ? "months both runs have compared" : "the compared months"}, without the ` +
           `December 2021 spin-up (${monthLabel(c[0])} – ${monthLabel(c[c.length - 1])}).`;
  };

  // Area-weighted mean over latitude rows of a lat-lon grid (cos-latitude weights, NaN skipped).
  function zonalOfMap(flat, nx, lats) {
    const out = [];
    for (let j = 0; j < lats.length; j++) {
      let s = 0, n = 0;
      for (let i = 0; i < nx; i++) { const v = flat[j * nx + i]; if (isFinite(v)) { s += v; n++; } }
      out.push(n ? s / n : null);
    }
    return out;
  }
  function weightedMean(values, weights) {
    let s = 0, w = 0;
    values.forEach((v, i) => { if (v !== null && isFinite(v)) { s += v * weights[i]; w += weights[i]; } });
    return w > 0 ? s / w : null;
  }

  // ---- 2 x 2 panel grids shared by the map, near-surface and vertical pages
  // Panels: each run minus GCHP (top row), GCHP itself and ERA5 minus MERRA-2 (bottom row). One diverging scale
  // for the three differences, so their colours compare directly.
  // Panels: each run on the site minus GCHP, then GCHP, then ERA5 - MERRA-2 when both runs are on the site.
  const gridRuns = () => ["era5", "merra2"].filter(k => hasMonths(run(k)));
  const gridSpecs = () => { const r = gridRuns(); return [...r, "gc", ...(r.length === 2 ? ["e_m"] : [])]; };
  function panelGrid(box, specs) {
    if (box.panels) return box.panels;
    box.innerHTML = "";
    box.classList.add("grid2");
    box.panels = {};
    for (const s of specs) {
      const title = el("div", {class: "panel-title"});
      const sub = el("div", {class: "panel-sub"});
      const plot = el("div", {class: "plot"});
      box.append(el("div", {}, title, sub, plot));
      box.panels[s] = {title, sub, plot};
    }
    return box.panels;
  }
  function panelHead(p, keys, text, stats) {
    p.title.innerHTML = "";
    for (const k of keys) p.title.append(el("span", {class: "key", style: `border-top-color:${k}`}));
    p.title.append(text);
    p.sub.innerHTML = stats || "";
  }
  function fieldStats(values, weights) {      // area-weighted mean and RMS of a field (weights per value)
    let s = 0, s2 = 0, w = 0;
    for (let i = 0; i < values.length; i++) { const v = values[i]; if (isFinite(v)) { s += v * weights[i]; s2 += v * v * weights[i]; w += weights[i]; } }
    return w > 0 ? [s / w, Math.sqrt(s2 / w)] : [NaN, NaN];
  }
  function mapWeights() {
    const m = meta();
    if (!mapWeights.w) {
      const nx = m.mlon.length;
      mapWeights.w = new Float32Array(nx * m.mlat.length);
      m.mlat.forEach((lat, j) => { const c = Math.cos(lat * Math.PI / 180); for (let i = 0; i < nx; i++) mapWeights.w[j * nx + i] = c; });
    }
    return mapWeights.w;
  }
  const runName = r => r.name.replace(" v0.5", "");
  // Decoded fields of one tracer/band/month: {gc, <run>: run - GCHP, e_m: ERA5 - MERRA-2}
  async function mapFields(tracerKey, bandKey, month) {
    const raw = await load(`map|${tracerKey}|${bandKey}|${month}`, `maps/${tracerKey}_${bandKey}_${month}.js`);
    const out = {};
    for (const [k, v] of Object.entries(raw)) out[k] = decode(v);
    if (out.era5 && out.merra2) out.e_m = out.era5.map((x, i) => x - out.merra2[i]);
    return out;
  }
  async function mapGrid(box, tracerKey, bandKey, month, isCurrent = () => true) {
    const m = meta(), t = tracer(tracerKey), b = band(bandKey);
    const P = panelGrid(box, gridSpecs());
    Object.values(P).forEach(p => p.plot.classList.add("loading"));
    let F;
    try { F = await mapFields(tracerKey, bandKey, month); }
    catch (e) { if (isCurrent()) Object.values(P).forEach(p => emptyNote(p.plot, "No data for this month.")); return null; }
    if (!isCurrent()) return undefined;          // a newer selection is being drawn
    const W = mapWeights(), nx = m.mlon.length;
    const diffs = gridRuns().filter(k => F[k]);
    const D = niceCeil(percentile(diffs.flatMap(k => Array.from(F[k], Math.abs)), 98));
    const lo = percentile(F.gc, 2), hi = percentile(F.gc, 98);
    const dd = decimalsFor(D, 1) + 1, da = decimalsFor(hi - lo, 4);
    const stat = (f, sign) => { const [mu, rms] = fieldStats(f, W); return `mean ${fmt(mu, dd, sign)}, ${sign ? "RMS" : "spatial range shown"} ${sign ? fmt(rms, dd) : `${fmt(lo, da)}–${fmt(hi, da)}`} ${t.unit}`; };
    for (const k of gridRuns()) {
      const r = run(k);
      panelHead(P[k], [r.color], `${r.name} − GCHP`, F[k] ? stat(F[k], true) : "");
      if (!F[k]) { emptyNote(P[k].plot, `${r.name} has not reached ${monthLabel(month)} yet.`); continue; }
      clearEmpty(P[k].plot);
      mapPlot(P[k].plot, {z: F[k], zmin: -D, zmax: D, scale: DIV, unit: t.unit, digits: dd, title: `${runName(r)} − GCHP`});
    }
    panelHead(P.gc, [GC().color], "GCHP (reference)", stat(F.gc, false));
    clearEmpty(P.gc.plot);
    mapPlot(P.gc.plot, {z: F.gc, zmin: lo, zmax: hi, scale: SEQ, unit: t.unit, digits: da, title: "GCHP"});
    if (P.e_m) {
      panelHead(P.e_m, [run("era5").color, run("merra2").color], "ERA5 − MERRA-2 (spread of the two v0.5 runs)", F.e_m ? stat(F.e_m, true) : "");
      if (F.e_m) { clearEmpty(P.e_m.plot); mapPlot(P.e_m.plot, {z: F.e_m, zmin: -D, zmax: D, scale: DIV, unit: t.unit, digits: dd, title: "ERA5 − MERRA-2"}); }
      else emptyNote(P.e_m.plot, "Needs both runs for this month.");
    }
    return F;
  }
  // Zonal mean (by latitude row) of the band field: values or run minus GCHP.
  function latitudeChart(div, F, tracerKey, view, hidden) {
    const m = meta(), t = tracer(tracerKey), nx = m.mlon.length, lat = m.mlat;
    const zm = f => zonalOfMap(f, nx, lat);
    const list = [];
    if (view === "abs" && !hidden.has("gc")) list.push(series(GC(), lat, zm(F.gc)));
    for (const r of mainRuns()) {
      if (!F[r.key] || hidden.has(r.key)) continue;
      const v = view === "diff" ? F[r.key] : F[r.key].map((x, i) => x + F.gc[i]);
      list.push(series(r, lat, zm(v)));
    }
    chart(div, {height: 300, series: list,
      x: {domain: [-90, 90], ticks: pw => pw < 380 ? [-90, -45, 0, 45, 90] : [-90, -60, -30, 0, 30, 60, 90], tickFmt: latLabel, label: "latitude", hover: latLabel},
      y: {label: view === "diff" ? `run − GCHP (${t.unit})` : t.unit, zero: view === "diff" ? "ref" : false,
          fmt: valFmt(t.digits + (view === "diff" ? 1 : 0), t.unit)}});
  }
  // Latitude-pressure sections from zonal_<tracer>.js for one month (or "period").
  async function sectionGrid(box, tracerKey, month, isCurrent = () => true) {
    const m = meta(), t = tracer(tracerKey);
    const Z = await load(`zonal_${tracerKey}`, `zonal_${tracerKey}.js`);
    if (!isCurrent()) return undefined;
    const P = panelGrid(box, gridSpecs());
    const e = Z.months[month];
    if (!e) { Object.values(P).forEach(p => emptyNote(p.plot, "No data for this month.")); return null; }
    const flat = a => a.flat().map(v => v === null ? NaN : v);
    const F = {gc: flat(e.gc)};
    for (const k of gridRuns()) if (e[k]) F[k] = flat(e[k]);
    if (F.era5 && F.merra2) F.e_m = F.era5.map((x, i) => x - F.merra2[i]);
    const nl = m.latc.length, W = new Float32Array(F.gc.length).fill(1);
    const D = niceCeil(percentile(gridRuns().filter(k => F[k]).flatMap(k => F[k].map(Math.abs)), 98));
    const lo = percentile(F.gc, 2), hi = percentile(F.gc, 98);
    const dd = decimalsFor(D, 1) + 1, da = decimalsFor(hi - lo, 4);
    const trop = Z.tropopause[month];
    const opts = (f, scale, zmin, zmax, digits, title) => ({z: f, zmin, zmax, scale, unit: t.unit, digits, title, trop});
    for (const k of gridRuns()) {
      const r = run(k);
      panelHead(P[k], [r.color], `${r.name} − GCHP`, "");
      if (!F[k]) { emptyNote(P[k].plot, `${r.name} has not reached ${monthLabel(month)} yet.`); continue; }
      clearEmpty(P[k].plot);
      sectionPlot(P[k].plot, opts(F[k], DIV, -D, D, dd, `${runName(r)} − GCHP`));
    }
    panelHead(P.gc, [GC().color], "GCHP (reference)", "");
    clearEmpty(P.gc.plot);
    sectionPlot(P.gc.plot, opts(F.gc, SEQ, lo, hi, da, "GCHP"));
    if (P.e_m) {
      panelHead(P.e_m, [run("era5").color, run("merra2").color], "ERA5 − MERRA-2 (spread of the two v0.5 runs)", "");
      if (F.e_m) { clearEmpty(P.e_m.plot); sectionPlot(P.e_m.plot, opts(F.e_m, DIV, -D, D, dd, "ERA5 − MERRA-2")); }
      else emptyNote(P.e_m.plot, "Needs both runs for this month.");
    }
    return {F, e};
  }

  ready(() => { statusStrip(); pruneRuns(); });
  return {put, load, get, ready, el, control, seg, monthSelect, legend, state, saveState, meta, tracer, band, run, mainRuns, GC,
          hasMonths, html, keys, gate, chart, bars, heatmap, series, timeAxis, valFmt, latLabel, lonLabel, emptyNote, clearEmpty,
          mapPlot, sectionPlot, SEQ, DIV, decode, percentile, niceCeil, decimalsFor, fmt, sigfmt, monthLabel, midMonth, badge,
          zonalOfMap, weightedMean, INK, INK2, MUTED, GRID, REF,
          mapGrid, mapFields, latitudeChart, sectionGrid, runName, periodText, xRange};
})();
