/* Shared code of the AtmosTransport performance benchmark pages (same patterns as the CATRINE benchmark site).
   data/bench.js registers the whole data set with SITE.put("bench", ...); every page reads it with SITE.bench().
   Charts are D3 v7 SVG (assets/d3.min.js, served from this site). Pointer input: hover with the mouse, or tap
   (and slide) on a touch screen; a tap elsewhere closes the tooltip. Charts never capture vertical page
   scrolling (touch-action: pan-y). There is no zoom: every chart shows all of its data. */
"use strict";
const SITE = (() => {
  const store = {};
  const put = (key, value) => { store[key] = value; };
  const get = key => store[key];
  const bench = () => store.bench;
  const ready = fn => (document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", fn) : fn());

  // ---- formatting
  const fin = x => x !== null && x !== undefined && isFinite(x);
  const fmt = (x, d = 2) => fin(x) ? (x < 0 ? "−" : "") + Math.abs(x).toFixed(d) : "–";
  const sig = (x, s = 3) => {          // significant figures, no exponent for ordinary magnitudes
    if (!fin(x)) return "–";
    if (x === 0) return "0";
    const e = Math.floor(Math.log10(Math.abs(x)));
    const t = Math.abs(x).toFixed(Math.max(0, s - 1 - e));
    return (x < 0 ? "−" : "") + (e >= 4 ? (+t).toLocaleString("en-US") : t);
  };
  const times = x => fin(x) ? `${x < 10 ? x.toFixed(2) : x.toFixed(1)}×` : "–";
  const pct = x => fin(x) ? `${(100 * x).toFixed(0)} %`.replace(/^-/, "−").replace(/^−0 %$/, "0 %") : "–";
  const sypd = s => 86400 / (365 * s);
  const tr = t => `${t} tracer${+t === 1 ? "" : "s"}`;
  const niceUtc = s => {                // 20261026T155725 -> 26 Oct 15:57 UTC
    const m = String(s || "").match(/^(\d{4})(\d\d)(\d\d)T(\d\d)(\d\d)/);
    if (!m) return s || "";
    return `${+m[3]} ${["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][+m[2] - 1]} ${m[4]}:${m[5]} UTC`;
  };

  // ---- data lookups
  let index = null;
  const key = (d, v, g, t, o) => `${d}|${v}|${g}|${t}|${o}`;
  function cfg(d, v, g, t, o) {
    if (!index) { index = new Map(); for (const c of bench().configs) index.set(key(c.d, c.v, c.g, c.t, c.o), c); }
    return index.get(key(d, v, g, +t, o)) || null;
  }
  const configs = (f = {}) => bench().configs.filter(c => Object.entries(f).every(([k, v]) => v === undefined || String(c[k]) === String(v)));
  const version = k => bench().versions.find(v => v.key === k);
  const device = k => bench().devices.find(d => d.key === k);
  const grid = k => bench().grids.find(g => g.key === k);
  const op = k => bench().ops.find(o => o.key === k);
  const gridLabel = k => { const g = grid(k); return g && g.met !== "MERRA-2" ? `${k} (${g.met})` : k; };
  const uniq = xs => [...new Set(xs)];
  const passText = c => c && c.p.length > 1 ? c.p.map(p => sig(p.s)).join(" / ") : "";
  const flagPill = c => c && c.flag ? `<span class="flag" title="passes ${passText(c)} s/day differ by ${pct(c.spread)} of their mean">Δ${pct(c.spread)}</span>` : "";

  // ---- page state in the URL hash (#device=L40S&ops=all), so a view can be linked
  function state(defaults) {
    const h = new URLSearchParams(location.hash.slice(1)), s = {...defaults};
    for (const k of Object.keys(defaults)) if (h.has(k)) s[k] = h.get(k);
    return s;
  }
  function saveState(s) {
    history.replaceState(null, "", "#" + new URLSearchParams(Object.entries(s).filter(([, v]) => v !== null && v !== undefined)).toString());
  }

  // ---- DOM helpers and controls
  function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") e.className = v;
      else if (k === "style") e.setAttribute("style", v);
      else if (k === "html") e.innerHTML = v;              // trusted generator strings only
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat(Infinity)) if (c !== null && c !== undefined) e.append(c.nodeType ? c : document.createTextNode(String(c)));
    return e;
  }
  const html = markup => el("span", {html: markup});
  const control = (label, ...inner) => el("div", {class: "control"}, el("span", {class: "lab"}, label), ...inner);
  function seg(options, value, onChange, aria) {
    const box = el("div", {class: "seg", role: "group", "aria-label": aria || ""});
    for (const o of options) {
      const b = el("button", {type: "button", "aria-pressed": String(String(o.key) === String(value)), title: o.title || null, html: o.name});
      b.addEventListener("click", () => onChange(String(o.key)));
      box.append(b);
    }
    return box;
  }
  // One control row per page. defs: [{key, label, options: st => [{key, name, title}], prefer: [keys]}].
  // Options come from the data present for the earlier choices; a choice that has no data snaps to the first
  // preferred (or first) option that has. tail (the legend) sits at the end of the row.
  function controlRow(box, defs, st, onChange, tail) {
    function build() {
      box.innerHTML = "";
      for (const d of defs) {
        const opts = d.options(st);
        if (!opts.length) { st[d.key] = null; continue; }
        if (!opts.some(o => String(o.key) === String(st[d.key]))) {
          const p = (d.prefer || []).find(k => opts.some(o => String(o.key) === String(k)));
          st[d.key] = String(p !== undefined ? p : opts[0].key);
        }
        box.append(control(d.label, seg(opts, st[d.key], k => { st[d.key] = k; build(); saveState(st); onChange(); }, d.label)));
      }
      if (tail) box.append(tail);
    }
    build();
    saveState(st);
    return build;
  }
  // Legend that doubles as the visibility toggle. items: {key, name, color, kind: "line"|"sq"|"shape"|"hatch", shape, dash}
  function legend(items, hidden, onChange) {
    const box = el("div", {class: "legend", role: "group", "aria-label": "series"});
    for (const d of items) {
      let key;
      if (d.kind === "shape") {
        key = el("span", {class: "key shape"});
        key.innerHTML = `<svg width="14" height="14" viewBox="-7 -7 14 14"><path d="${symbolPath(d.shape, 64)}" fill="${d.color}"/></svg>`;
      } else if (d.kind === "hatch") key = el("span", {class: "key hatch"});
      else if (d.kind === "sq") key = el("span", {class: "key sq", style: `background:${d.color}`});
      else key = el("span", {class: "key", style: `border-top-color:${d.color};border-top-style:${d.dash === "dot" ? "dotted" : d.dash === "dash" ? "dashed" : "solid"}`});
      const b = el("button", {type: "button", "aria-pressed": String(!hidden.has(d.key)), title: "show or hide"}, key, html(d.name));
      b.addEventListener("click", () => {
        hidden.has(d.key) ? hidden.delete(d.key) : hidden.add(d.key);
        b.setAttribute("aria-pressed", String(!hidden.has(d.key)));
        onChange();
      });
      box.append(b);
    }
    return box;
  }
  // Data table. head: [[cell, ...]] rows of {text|html, cls, colspan}; body likewise ({grp: text} = group row).
  function table(box, head, body) {
    const cell = (tag, c) => {
      if (c === null || c === undefined || typeof c !== "object") c = {text: c === null || c === undefined ? "" : String(c)};
      return el(tag, {class: c.cls || null, colspan: c.colspan || null, title: c.title || null, html: c.html !== undefined ? c.html : null},
        c.html !== undefined ? null : c.text);
    };
    const n = Math.max(...head.map(r => r.reduce((a, c) => a + ((c && c.colspan) || 1), 0)));
    box.innerHTML = "";
    box.append(el("table", {class: "data"},
      el("thead", {}, head.map(r => el("tr", {}, r.map(c => cell("th", c))))),
      el("tbody", {}, body.map(r => r.grp !== undefined ? el("tr", {class: "grp"}, el("td", {colspan: n, html: r.grp}))
        : el("tr", {}, r.map(c => cell("td", c)))))));
  }

  // ---- status strip: progress per device against the planned matrix
  function statusStrip() {
    const box = document.getElementById("status-strip"), B = bench();
    if (!box || !B) return;
    for (const d of B.devices) {
      const p = d.progress;
      box.append(el("span", {class: "run", title: `${p.done} of ${p.planned} planned runs (release, operator and scaling matrices)` +
          (p.extra ? `; ${p.extra} extra C360/C720 runs` : "") + (p.failed ? `; ${p.failed} failed` : "")},
        el("span", {class: "bar-key", style: `display:inline-block;width:9px;height:9px;border-radius:2px;background:${d.color}`}),
        `${d.name} ${p.done}/${p.planned}`,
        el("span", {class: "bar"}, el("span", {style: `width:${Math.min(100, 100 * p.done / p.planned)}%;background:${d.color}`})),
        el("span", {class: "when"}, p.last ? niceUtc(p.last) : "")));
    }
    for (const w of B.waiting) box.append(el("span", {class: "run waiting"}, `${w.name}: waiting for data`));
    box.append(el("span", {class: "run when"}, `updated ${B.generated}`));
  }

  // ---- charts
  const INK = "#0b0b0b", INK2 = "#52514e", MUTED = "#898781", GRID = "#e1e0d9", AXIS = "#c3c2b7", SURFACE = "#fcfcfb";
  const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', TICK = 11.5;
  let uid = 0;
  const measure = document.createElement("canvas").getContext("2d");
  const textWidth = (s, size = TICK, weight = 400) => { measure.font = `${weight} ${size}px ${FONT}`; return measure.measureText(String(s).replace(/<[^>]*>/g, "")).width; };
  const SYMBOLS = () => ({circle: d3.symbolCircle, square: d3.symbolSquare, diamond: d3.symbolDiamond, triangle: d3.symbolTriangle});
  const symbolPath = (shape, size) => d3.symbol(SYMBOLS()[shape] || d3.symbolCircle, size)();
  const tickLabel = v => v >= 1e4 ? d3.format(",")(+v.toPrecision(3)) : String(+v.toPrecision(3));
  // Log-axis ticks: 1-2-5, 1-3 or 1 per decade, whichever fits; inside one decade, linear ticks.
  function logTicks(lo, hi, px) {
    const maxN = Math.max(2, Math.floor(px / 46));
    const make = mults => {
      const t = [];
      for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++)
        for (const m of mults) { const v = +(m * Math.pow(10, e)).toPrecision(6); if (v >= lo * 0.999 && v <= hi * 1.001) t.push(v); }
      return t;
    };
    let t = make([1, 2, 5]);
    if (t.length < 2) return d3.ticks(lo, hi, Math.min(maxN, 4)).filter(v => v > 0);
    if (t.length <= maxN) return t;
    t = make([1, 3]);
    if (t.length <= maxN) return t;
    t = make([1]);
    const k = Math.ceil(t.length / maxN);
    return t.filter((_, i) => i % k === 0);
  }
  const logDomain = (vals, pad = 1.18) => {
    const v = vals.filter(x => fin(x) && x > 0);
    if (!v.length) return [0.1, 1];
    let lo = d3.min(v) / pad, hi = d3.max(v) * pad;
    if (hi / lo < 3) { const m = Math.sqrt(lo * hi); lo = m / 1.8; hi = m * 1.8; }
    return [lo, hi];
  };

  const shown = new Set();
  const widthOf = div => Math.floor(div.getBoundingClientRect().width);
  const resizer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(entries => {
    for (const e of entries) {
      const C = e.target._chart;
      if (!C || widthOf(e.target) === C.width) continue;
      cancelAnimationFrame(C.raf);
      C.raf = requestAnimationFrame(() => { if (e.target._chart === C) C.render(); });
    }
  });
  document.addEventListener("pointerdown", ev => { for (const C of [...shown]) if (!C.div.contains(ev.target)) C.clear(); }, {capture: true, passive: true});
  function host(div, kind) {
    const note = div.querySelector(".empty");
    if (note) note.remove();
    if (div._chart && div._chart.kind === kind) return div._chart;
    if (div._chart) div._chart.clear();
    div.innerHTML = "";
    div.classList.add("chart");
    const C = {kind, div, width: -1};
    C.svg = d3.select(div).append("svg");
    C.tip = div.appendChild(el("div", {class: "tip", hidden: true}));
    C.clear = () => { C.tip.hidden = true; shown.delete(C); if (C.unmark) C.unmark(); };
    C.showTip = (markup, x, y, touch) => {
      const tip = C.tip;
      tip.innerHTML = markup;
      tip.hidden = false;
      shown.add(C);
      const W = div.clientWidth, H = div.clientHeight, w = tip.offsetWidth, h = tip.offsetHeight;
      let left = x + 14;
      if (left + w > W - 2) left = x - 14 - w;
      if (left < 2) left = Math.max(2, Math.min(W - w - 2, x - w / 2));
      let top = Math.max(2, Math.min(H - h - 2, y - h / 2));
      if (touch) top = y - h - 18 >= 0 ? y - h - 18 : Math.min(H - h - 2, y + 22);    // keep it clear of the finger
      tip.style.left = `${left}px`;
      tip.style.top = `${Math.max(2, top)}px`;
    };
    div._chart = C;
    if (resizer) resizer.observe(div);
    return C;
  }
  function emptyNote(div, text) {
    if (div._chart) { div._chart.clear(); if (resizer) resizer.unobserve(div); div._chart = null; }
    div.classList.remove("chart");
    div.innerHTML = "";
    div.append(el("div", {class: "empty"}, text));
  }
  // pointer wiring of an overlay: mouse hover, touch tap and slide; fn(mx, my, touch)
  function pointer(C, ov, g, fn) {
    ov.on("pointermove", ev => { const [mx, my] = d3.pointer(ev, g.node()); if (ev.pointerType !== "touch" || ev.buttons) fn(mx, my, ev.pointerType === "touch"); });
    ov.on("pointerdown", ev => { if (ev.pointerType === "touch") { const [mx, my] = d3.pointer(ev, g.node()); fn(mx, my, true); } });
    ov.on("pointerleave", ev => { if (ev.pointerType !== "touch") C.clear(); });
    ov.on("pointercancel", () => C.clear());
  }
  const swatch = (color, shape) => shape
    ? `<svg class="sw" width="12" height="12" viewBox="-6 -6 12 12" style="flex:none;border:0;width:12px;height:12px"><path d="${symbolPath(shape, 52)}" fill="${color}"/></svg>`
    : `<span class="sw pt sq" style="color:${color}"></span>`;
  const tipRow = (sw, name, value, extra = "") => `<div class="row">${sw}<span class="n">${name}</span><span class="v">${value}</span>${extra ? `<span class="x">${extra}</span>` : ""}</div>`;
  const axisTitle = (g, x, y, text, rotate) => g.append("text").attr("class", "axis-title").attr("text-anchor", "middle")
    .attr("transform", `translate(${x},${y})${rotate ? " rotate(-90)" : ""}`).text(text);

  // Dot rows: one row per category, one marker per series on a log x axis (mean of the passes), hollow dots
  // and a thin line for the two passes, a grey bar from the fastest to the slowest series.
  // opt: {rows: [{key, label, sep}], series: [{key, name, color, shape}], get(row, s) -> config, hidden,
  //       x: {label}, value: c -> number (default c.s), passValue: (c, p) -> number, unit, tipExtra(row) -> html}
  function dotRows(div, opt) {
    const C = host(div, "dots");
    C.spec = opt;
    C.render = () => renderDots(C);
    C.render();
    return C;
  }
  function renderDots(C) {
    const o = C.spec, div = C.div, W = widthOf(div);
    C.width = W;
    C.clear();
    if (!W) return;
    const val = o.value || (c => c.s), pv = o.passValue || ((c, p) => p.s), unit = o.unit || "s/day";
    const S = o.series.filter(s => !o.hidden || !o.hidden.has(s.key));
    const cell = (r, s) => o.get(r, s);
    const all = [];
    for (const r of o.rows) for (const s of S) { const c = cell(r, s); if (c) { all.push(val(c)); c.p.forEach(p => all.push(pv(c, p))); } }
    const flagged = r => S.some(s => { const c = cell(r, s); return c && c.flag; });
    const labels = o.rows.map(r => r.label + (flagged(r) ? " †" : ""));
    const rowH = 30, mt = 6, mb = 44, labW = Math.ceil(d3.max(labels, l => textWidth(l, 12)) || 20) + 12, mr = 14;
    const pw = Math.max(80, W - labW - mr), H = mt + o.rows.length * rowH + mb;
    const [lo, hi] = logDomain(all);
    const X = d3.scaleLog().domain([lo, hi]).range([0, pw]);
    const ticks = logTicks(lo, hi, pw);
    const svg = C.svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${labW},0)`);
    const hl = g.append("rect").attr("class", "row-hl").attr("x", -labW).attr("width", W).attr("height", rowH).attr("display", "none");
    const bottom = mt + o.rows.length * rowH;
    for (const t of ticks) {
      const x = Math.round(X(t)) + 0.5;
      g.append("line").attr("x1", x).attr("x2", x).attr("y1", mt).attr("y2", bottom).attr("stroke", GRID);
      g.append("text").attr("class", "tick").attr("x", x).attr("y", bottom + 15).attr("text-anchor", "middle").text(tickLabel(t));
    }
    g.append("line").attr("x1", 0).attr("x2", pw).attr("y1", bottom + 0.5).attr("y2", bottom + 0.5).attr("stroke", AXIS);
    axisTitle(g, pw / 2, H - 8, o.x && o.x.label ? o.x.label : `${unit} (log scale)`);
    o.rows.forEach((r, i) => {
      const y = mt + i * rowH + rowH / 2;
      if (r.sep && i > 0) g.append("line").attr("x1", -labW).attr("x2", pw).attr("y1", mt + i * rowH + 0.5).attr("y2", mt + i * rowH + 0.5).attr("stroke", AXIS);
      g.append("text").attr("class", "rowlab").attr("x", -10).attr("y", y).attr("dy", "0.34em").attr("text-anchor", "end").text(labels[i]);
      const cs = S.map(s => [s, cell(r, s)]).filter(([, c]) => c);
      if (cs.length > 1) {
        const xs = cs.map(([, c]) => X(val(c)));
        g.append("line").attr("x1", d3.min(xs)).attr("x2", d3.max(xs)).attr("y1", y).attr("y2", y).attr("stroke", AXIS).attr("stroke-width", 3).attr("stroke-linecap", "round");
      }
      for (const [s, c] of cs) {
        if (c.p.length > 1) {
          const px = c.p.map(p => X(pv(c, p)));
          g.append("line").attr("x1", d3.min(px)).attr("x2", d3.max(px)).attr("y1", y).attr("y2", y).attr("stroke", s.color).attr("stroke-width", 1.5).attr("opacity", 0.55);
          for (const x of px) g.append("circle").attr("cx", x).attr("cy", y).attr("r", 2.6).attr("fill", SURFACE).attr("stroke", s.color).attr("stroke-width", 1.3);
        }
      }
      for (const [s, c] of cs) g.append("path").attr("d", symbolPath(s.shape || "circle", 78)).attr("transform", `translate(${X(val(c))},${y})`)
        .attr("fill", s.color).attr("stroke", SURFACE).attr("stroke-width", 1.5);
    });
    const ov = g.append("rect").attr("class", "overlay").attr("x", -labW).attr("y", mt).attr("width", W).attr("height", bottom - mt).attr("fill", "transparent");
    C.unmark = () => hl.attr("display", "none");
    pointer(C, ov, g, (mx, my, touch) => {
      const i = Math.max(0, Math.min(o.rows.length - 1, Math.floor((my - mt) / rowH))), r = o.rows[i];
      hl.attr("display", null).attr("y", mt + i * rowH);
      const rows = S.map(s => {
        const c = cell(r, s);
        if (!c) return "";
        const passes = c.p.length > 1 ? `passes ${c.p.map(p => sig(pv(c, p))).join(" / ")}${c.flag ? " †" : ""}` : "1 pass";
        return tipRow(swatch(s.color, s.shape), s.name, `${sig(val(c))} ${unit}`, passes);
      }).join("");
      if (!rows) { C.clear(); return; }
      C.showTip(`<div class="h">${r.title || r.label}</div>${rows}${o.tipExtra ? o.tipExtra(r, S) : ""}`, labW + Math.min(mx, pw * 0.5), mt + i * rowH + rowH / 2, touch);
    });
  }

  // Grouped vertical bars with the pass values as small dots. opt: {categories: [{key, label}], series: [{key, name, color}],
  // get(cat, s) -> config, value(c), passValue(c, p), hidden, y: {label}, unit, tipExtra(cat, S)}
  function groupBars(div, opt) {
    const C = host(div, "bars");
    C.spec = opt;
    C.render = () => renderBars(C);
    C.render();
    return C;
  }
  function renderBars(C) {
    const o = C.spec, div = C.div, W = widthOf(div);
    C.width = W;
    C.clear();
    if (!W) return;
    const S = o.series.filter(s => !o.hidden || !o.hidden.has(s.key));
    const H = o.height || 300, mt = 18, mb = 30, ph = H - mt - mb;
    const vals = [];
    for (const k of o.categories) for (const s of S) { const c = o.get(k, s); if (c) { vals.push(o.value(c)); c.p.forEach(p => vals.push(o.passValue(c, p))); } }
    const Y = d3.scaleLinear().domain([0, (d3.max(vals) || 1) * 1.04]).range([mt + ph, mt]).nice(4);
    const ticks = Y.ticks(Math.max(2, Math.round(ph / 50)));
    const ml = Math.ceil((d3.max(ticks, t => textWidth(tickLabel(t))) || 16) + 9 + 21), mr = 8, pw = Math.max(60, W - ml - mr);
    const span = Math.min(pw, o.categories.length * (S.length * 64 + 36)), off = (pw - span) / 2;   // bars at most ~60 px wide
    const X0 = d3.scaleBand().domain(o.categories.map(c => c.key)).range([off, off + span]).paddingInner(0.28).paddingOuter(0.12);
    const X1 = d3.scaleBand().domain(S.map(s => s.key)).range([0, X0.bandwidth()]).padding(0.1);
    const svg = C.svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${ml},0)`);
    const hl = g.append("rect").attr("class", "band-hl").attr("y", mt).attr("height", ph).attr("display", "none");
    for (const t of ticks) {
      const y = Math.round(Y(t)) + 0.5;
      g.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y).attr("y2", y).attr("stroke", t === 0 ? AXIS : GRID);
      g.append("text").attr("class", "tick").attr("x", -8).attr("y", y).attr("dy", "0.34em").attr("text-anchor", "end").text(tickLabel(t));
    }
    axisTitle(g, -(ml - 13), mt + ph / 2, o.y.label, true);
    const bw = X1.bandwidth();
    // value labels on every bar or on none (they must not collide with the neighbouring bar's label)
    const labelsFit = o.categories.every(k => S.every(s => { const c = o.get(k, s); return !c || textWidth(sig(o.value(c), o.value(c) >= 100 ? 3 : 2), 10.5) <= X1.step() - 2; }));
    for (const k of o.categories) for (const s of S) {
      const c = o.get(k, s);
      if (!c) continue;
      const x = X0(k.key) + X1(s.key), v = o.value(c), r = Math.min(4, bw / 2);
      g.append("path").attr("fill", s.color)
        .attr("d", `M${x},${Y(0)}V${Y(v) + r}a${r},${r} 0 0 1 ${r},${-r}H${x + bw - r}a${r},${r} 0 0 1 ${r},${r}V${Y(0)}Z`);
      for (const p of c.p.length > 1 ? c.p : []) g.append("circle").attr("cx", x + bw / 2).attr("cy", Y(o.passValue(c, p))).attr("r", Math.min(2.8, bw / 4))
        .attr("fill", INK).attr("stroke", SURFACE).attr("stroke-width", 1.2);
      const lab = sig(v, v >= 100 ? 3 : 2);
      if (labelsFit)
        g.append("text").attr("class", "val").attr("x", x + bw / 2).attr("y", Y(Math.max(v, ...c.p.map(p => o.passValue(c, p)))) - 5).attr("text-anchor", "middle").text(lab);
    }
    for (const k of o.categories) g.append("text").attr("class", "rowlab").attr("x", X0(k.key) + X0.bandwidth() / 2).attr("y", mt + ph + 19).attr("text-anchor", "middle").text(k.label);
    const ov = g.append("rect").attr("class", "overlay").attr("x", 0).attr("y", mt).attr("width", pw).attr("height", ph + mb).attr("fill", "transparent");
    C.unmark = () => hl.attr("display", "none");
    pointer(C, ov, g, (mx, my, touch) => {
      const i = d3.minIndex(o.categories, c => Math.abs(X0(c.key) + X0.bandwidth() / 2 - mx)), k = o.categories[i];
      hl.attr("display", null).attr("x", X0(k.key) - X0.step() * X0.paddingInner() / 2).attr("width", X0.step());
      const rows = S.map(s => {
        const c = o.get(k, s);
        return c ? tipRow(swatch(s.color), s.name, `${sig(o.value(c))} ${o.unit}`, `${sig(c.s)} s/day${c.p.length > 1 ? `, passes ${passText(c)}` : ""}${c.flag ? " †" : ""}`) : "";
      }).join("");
      if (!rows) { C.clear(); return; }
      C.showTip(`<div class="h">${k.title || k.label}</div>${rows}${o.tipExtra ? o.tipExtra(k, S) : ""}`, ml + X0(k.key) + X0.bandwidth() / 2, Math.min(my, mt + ph * 0.6), touch);
    });
  }

  // Line chart on log-log or log-linear axes with markers; x ticks at named positions (grids).
  // opt: {series: [{key, name, color, dash, pts: [{x, y, open}]}], hidden, x: {ticks: [{v, label}], label},
  //       y: {label, log, fmt}, bands: [{x0, x1, label}], height, head(x) -> html, rowFmt(series, pt) -> [value, extra]}
  function lines(div, opt) {
    const C = host(div, "lines");
    C.spec = opt;
    C.render = () => renderLines(C);
    C.render();
    return C;
  }
  function renderLines(C) {
    const o = C.spec, div = C.div, W = widthOf(div);
    C.width = W;
    C.clear();
    if (!W) return;
    const S = o.series.filter(s => !o.hidden || !o.hidden.has(s.key));
    const H = o.height || 300, mt = 16, mb = 46, ph = H - mt - mb;
    const ys = S.flatMap(s => s.pts.map(p => p.y)), xs = o.x.ticks.map(t => t.v);
    const ylog = o.y.log !== false;
    const yd = ylog ? logDomain(ys, 1.25) : [0, (d3.max(ys) || 1) * 1.12];
    const Y = (ylog ? d3.scaleLog() : d3.scaleLinear()).domain(yd).range([mt + ph, mt]);
    if (!ylog) Y.nice();
    const yt = ylog ? logTicks(yd[0], yd[1], ph * 1.25) : Y.ticks(Math.max(2, Math.round(ph / 50)));
    const ml = Math.ceil((d3.max(yt, t => textWidth(tickLabel(t))) || 16) + 9 + 21), mr = 18, pw = Math.max(60, W - ml - mr);
    const xd = xs.length > 1 ? [d3.min(xs) / 1.6, d3.max(xs) * 1.6] : [xs[0] / 3, xs[0] * 3];
    const X = d3.scaleLog().domain(xd).range([0, pw]);
    const svg = C.svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const g = svg.append("g").attr("transform", `translate(${ml},0)`);
    for (const b of o.bands || []) {
      const x0 = Math.max(0, X(b.x0)), x1 = Math.min(pw, X(b.x1));
      g.append("rect").attr("class", "ib-band").attr("x", x0).attr("y", mt).attr("width", x1 - x0).attr("height", ph);
      if (b.label && x1 - x0 > textWidth(b.label, 11) + 6) g.append("text").attr("class", "ib").attr("x", (x0 + x1) / 2).attr("y", mt + 12).attr("text-anchor", "middle").text(b.label);
    }
    for (const t of yt) {
      const y = Math.round(Y(t)) + 0.5;
      g.append("line").attr("x1", 0).attr("x2", pw).attr("y1", y).attr("y2", y).attr("stroke", GRID);
      g.append("text").attr("class", "tick").attr("x", -8).attr("y", y).attr("dy", "0.34em").attr("text-anchor", "end").text(o.y.tickFmt ? o.y.tickFmt(t) : tickLabel(t));
    }
    const minGap = d3.max(o.x.ticks, t => textWidth(t.label)) + 6;
    let lastX = -1e9;
    for (const t of o.x.ticks) {
      const x = Math.round(X(t.v)) + 0.5;
      g.append("line").attr("x1", x).attr("x2", x).attr("y1", mt).attr("y2", mt + ph).attr("stroke", GRID);
      if (x - lastX >= minGap) { g.append("text").attr("class", "tick").attr("x", x).attr("y", mt + ph + 16).attr("text-anchor", "middle").text(t.label); lastX = x; }
    }
    g.append("line").attr("x1", 0).attr("x2", pw).attr("y1", mt + ph + 0.5).attr("y2", mt + ph + 0.5).attr("stroke", AXIS);
    axisTitle(g, -(ml - 13), mt + ph / 2, o.y.label, true);
    axisTitle(g, pw / 2, H - 8, o.x.label);
    const dash = d => d === "dot" ? "1.5 4" : d === "dash" ? "6 4" : null;
    for (const s of S) {
      const pts = s.pts.filter(p => fin(p.y) && p.y > 0).sort((a, b) => a.x - b.x);
      g.append("path").attr("d", d3.line().x(p => X(p.x)).y(p => Y(p.y))(pts)).attr("fill", "none").attr("stroke", s.color)
        .attr("stroke-width", 2).attr("stroke-dasharray", dash(s.dash)).attr("stroke-linecap", "round").attr("stroke-linejoin", "round");
      for (const p of pts) g.append("path").attr("d", symbolPath(s.shape || "circle", 60)).attr("transform", `translate(${X(p.x)},${Y(p.y)})`)
        .attr("fill", p.open ? SURFACE : s.color).attr("stroke", p.open ? s.color : SURFACE).attr("stroke-width", 1.5);
    }
    const mark = g.append("g").attr("pointer-events", "none");
    const ov = g.append("rect").attr("class", "overlay").attr("x", 0).attr("y", mt).attr("width", pw).attr("height", ph).attr("fill", "transparent");
    C.unmark = () => mark.selectAll("*").remove();
    pointer(C, ov, g, (mx, my, touch) => {
      const t = o.x.ticks[d3.minIndex(o.x.ticks, t => Math.abs(X(t.v) - mx))];
      mark.selectAll("*").remove();
      mark.append("line").attr("class", "guide").attr("x1", X(t.v)).attr("x2", X(t.v)).attr("y1", mt).attr("y2", mt + ph);
      const rows = S.map(s => {
        const p = s.pts.find(q => q.x === t.v);
        if (!p || !fin(p.y)) return "";
        const [v, extra] = o.rowFmt ? o.rowFmt(s, p) : [sig(p.y), ""];
        return tipRow(`<span class="sw${s.dash === "dot" ? " dot" : ""}" style="color:${s.color}"></span>`, s.name, v, extra);
      }).join("");
      if (!rows) { C.clear(); return; }
      C.showTip(`<div class="h">${o.head ? o.head(t) : t.label}</div>${rows}`, ml + X(t.v), my, touch);
    });
  }

  // Horizontal stacked bars of the operator breakdown; negative components stack left of zero, the measured
  // all-operator time is a black tick. opt: {rows: [{key, label, color, parts: {k: s}, total}], comps: [{key, name, color}], hidden}
  function stack(div, opt) {
    const C = host(div, "stack");
    C.spec = opt;
    C.render = () => renderStack(C);
    C.render();
    return C;
  }
  function renderStack(C) {
    const o = C.spec, div = C.div, W = widthOf(div);
    C.width = W;
    C.clear();
    if (!W) return;
    const K = o.comps.filter(k => !o.hidden || !o.hidden.has(k.key));
    const rowH = 34, gap = 12, mt = 6, mb = 40;
    const ext = o.rows.map(r => {
      let pos = 0, neg = 0;
      for (const k of K) { const v = r.parts[k.key]; if (v >= 0) pos += v; else neg += v; }
      return [neg, Math.max(pos, r.total)];
    });
    const lo = Math.min(0, d3.min(ext, e => e[0])), hi = d3.max(ext, e => e[1]) || 1;
    const labW = Math.ceil(d3.max(o.rows, r => textWidth(r.label, 12)) || 20) + 26, mr = 16, pw = Math.max(80, W - labW - mr);
    const X = d3.scaleLinear().domain([lo, hi * 1.04]).range([0, pw]).nice();
    const ticks = X.ticks(Math.max(2, Math.round(pw / 70)));
    const H = mt + o.rows.length * (rowH + gap) - gap + mb;
    const svg = C.svg.attr("width", W).attr("height", H).attr("viewBox", `0 0 ${W} ${H}`);
    svg.selectAll("*").remove();
    const pid = `hatch-${++uid}`;
    svg.append("defs").html(`<pattern id="${pid}" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">` +
      `<rect width="5" height="5" fill="#f0efec"/><line x1="0" y1="0" x2="0" y2="5" stroke="${INK2}" stroke-width="1.6"/></pattern>`);
    const fill = k => k.color === "hatch" ? `url(#${pid})` : k.color;
    const g = svg.append("g").attr("transform", `translate(${labW},0)`);
    const bottom = mt + o.rows.length * (rowH + gap) - gap;
    const hl = g.append("rect").attr("class", "row-hl").attr("x", -labW).attr("width", W).attr("height", rowH + 8).attr("display", "none");
    for (const t of ticks) {
      const x = Math.round(X(t)) + 0.5;
      g.append("line").attr("x1", x).attr("x2", x).attr("y1", mt).attr("y2", bottom).attr("stroke", t === 0 ? AXIS : GRID);
      g.append("text").attr("class", "tick").attr("x", x).attr("y", bottom + 15).attr("text-anchor", "middle").text(t === 0 ? "0" : String(+t.toPrecision(3)).replace("-", "−"));
    }
    axisTitle(g, pw / 2, H - 8, "seconds per simulated day");
    o.rows.forEach((r, i) => {
      const y = mt + i * (rowH + gap);
      g.append("rect").attr("x", -labW + 2).attr("y", y + rowH / 2 - 5).attr("width", 10).attr("height", 10).attr("rx", 2).attr("fill", r.color);
      g.append("text").attr("class", "rowlab").attr("x", -labW + 18).attr("y", y + rowH / 2).attr("dy", "0.34em").text(r.label);
      let pos = 0, neg = 0;
      for (const k of K) {
        const v = r.parts[k.key];
        if (!fin(v) || v === 0) continue;
        const a = v >= 0 ? pos : neg + v, b = v >= 0 ? pos + v : neg;
        if (v >= 0) pos += v; else neg += v;
        g.append("rect").attr("x", X(a)).attr("y", y).attr("width", Math.max(0, X(b) - X(a))).attr("height", rowH)
          .attr("fill", fill(k)).attr("stroke", SURFACE).attr("stroke-width", 1.5);
      }
      const tx = X(r.total);
      g.append("line").attr("x1", tx).attr("x2", tx).attr("y1", y - 4).attr("y2", y + rowH + 4).attr("stroke", INK).attr("stroke-width", 2);
    });
    const ov = g.append("rect").attr("class", "overlay").attr("x", -labW).attr("y", mt).attr("width", W).attr("height", bottom - mt).attr("fill", "transparent");
    C.unmark = () => hl.attr("display", "none");
    pointer(C, ov, g, (mx, my, touch) => {
      const i = Math.max(0, Math.min(o.rows.length - 1, Math.floor((my - mt + gap / 2) / (rowH + gap)))), r = o.rows[i];
      hl.attr("display", null).attr("y", mt + i * (rowH + gap) - 4);
      const rows = K.map(k => tipRow(k.color === "hatch" ? `<span class="sw pt sq" style="color:${INK2}"></span>` : swatch(k.color), k.name,
        `${sig(r.parts[k.key], 2)} s`, pct(r.parts[k.key] / r.total))).join("");
      C.showTip(`<div class="h">${r.title || r.label}</div>${rows}${tipRow(`<span class="sw" style="color:${INK}"></span>`, "all operators (measured)", `${sig(r.total)} s`)}`,
        labW + Math.min(mx, pw * 0.4), mt + i * (rowH + gap) + rowH / 2, touch);
    });
  }

  ready(statusStrip);
  return {put, get, bench, ready, fin, fmt, sig, times, pct, sypd, tr, niceUtc, cfg, configs, version, device, grid, op, gridLabel, uniq,
          passText, flagPill, state, saveState, el, html, control, seg, controlRow, legend, table, emptyNote,
          dotRows, groupBars, lines, stack, symbolPath, INK, INK2, MUTED};
})();
