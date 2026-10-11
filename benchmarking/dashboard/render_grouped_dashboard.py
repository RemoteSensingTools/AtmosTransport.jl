#!/usr/bin/env python3
"""Render a compact benchmark dashboard from github-action-benchmark data."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


DATA_PREFIX = "window.BENCHMARK_DATA = "


def _load_data(path: Path) -> dict:
    text = path.read_text()
    if not text.startswith(DATA_PREFIX):
        raise SystemExit(f"{path} does not look like github-action-benchmark data.js")
    payload = text[len(DATA_PREFIX):].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    return json.loads(payload)


def _write_html(path: Path) -> None:
    html = r"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>AtmosTransport Benchmarks</title>
    <style>
      :root {
        color-scheme: light;
        --border: #d6d9de;
        --text: #20242a;
        --muted: #59616d;
        --band: #f6f8fa;
      }
      body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text);
        background: white;
      }
      header {
        padding: 16px 20px 10px;
        border-bottom: 1px solid var(--border);
      }
      h1 {
        margin: 0 0 6px;
        font-size: 24px;
        font-weight: 650;
      }
      .meta {
        color: var(--muted);
        font-size: 13px;
      }
      main {
        padding: 16px 20px 28px;
      }
      .toolbar {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        align-items: center;
        margin-bottom: 14px;
        font-size: 13px;
      }
      select {
        font: inherit;
        padding: 4px 8px;
      }
      .row-title {
        margin: 18px 0 8px;
        font-size: 18px;
        font-weight: 650;
      }
      .grid-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        gap: 14px;
        align-items: stretch;
      }
      .panel {
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px 12px 12px;
        min-height: 340px;
      }
      .panel h3 {
        margin: 0 0 8px;
        font-size: 15px;
        font-weight: 650;
      }
      .chart-wrap {
        position: relative;
        height: 300px;
      }
      .summary {
        margin-top: 10px;
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
      }
      .summary th,
      .summary td {
        padding: 5px 6px;
        border-bottom: 1px solid #e8eaed;
        text-align: right;
      }
      .summary th:first-child,
      .summary td:first-child {
        text-align: left;
      }
      .summary tbody tr:nth-child(odd) {
        background: var(--band);
      }
      .empty {
        color: var(--muted);
        font-size: 13px;
        padding: 20px 0;
      }
      a {
        color: #0969da;
        text-decoration: none;
      }
      a:hover {
        text-decoration: underline;
      }
      @media (max-width: 900px) {
        .grid-row {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <header>
      <h1>AtmosTransport Benchmarks</h1>
      <div class="meta">
        Last update: <span id="last-update"></span> ·
        <a id="repo-link" rel="noopener"></a> ·
        <a href="data.js">raw dashboard data</a>
      </div>
    </header>
    <main>
      <div class="toolbar">
        <label>Operator <select id="operator-select"></select></label>
        <label>Tracers <select id="tracer-select"></select></label>
        <label>Float type <select id="float-select"></select></label>
        <label>Metric <select id="metric-select">
          <option value="time">Time per step</option>
          <option value="throughput">Cell-tracer updates/s</option>
        </select></label>
      </div>
      <section>
        <div class="row-title">Up to C180</div>
        <div class="grid-row">
          <div class="panel"><h3>CPU</h3><div class="chart-wrap"><canvas id="small-cpu"></canvas></div><table class="summary" id="small-cpu-table"></table></div>
          <div class="panel"><h3>GPU</h3><div class="chart-wrap"><canvas id="small-gpu"></canvas></div><table class="summary" id="small-gpu-table"></table></div>
        </div>
      </section>
      <section>
        <div class="row-title">Above C180</div>
        <div class="grid-row">
          <div class="panel"><h3>CPU</h3><div class="chart-wrap"><canvas id="large-cpu"></canvas></div><table class="summary" id="large-cpu-table"></table></div>
          <div class="panel"><h3>GPU</h3><div class="chart-wrap"><canvas id="large-gpu"></canvas></div><table class="summary" id="large-gpu-table"></table></div>
        </div>
      </section>
    </main>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@2.9.2/dist/Chart.min.js"></script>
    <script src="data.js"></script>
    <script>
      "use strict";

      const colors = {
        C24: "#0969da",
        C90: "#1a7f37",
        C180: "#bf8700",
        C360: "#8250df",
        C720: "#cf222e",
      };
      const fallbackColors = ["#57606a", "#0550ae", "#116329", "#953800", "#6639ba", "#a40e26"];
      const dashes = [[], [6, 4], [2, 3], [10, 3, 2, 3]];
      const pointStyles = ["circle", "triangle", "rect", "rectRot", "star", "cross"];
      const smallGridMax = 180;
      const charts = {};
      // Series name written by benchmarking/src/labels.jl (case_series_name):
      // "<group> / C<N> L<levels> <operator> <tracers>tr[ <scheme>] / <backend label> / <float type>",
      // where GPU backend labels carry the device ("GPU-CUDA L40S") and the
      // scheme is omitted for the default "ppm".
      const mainCaseRe = /^(?!Operator Breakdown)([^/]+?) \/ (C(\d+)) L(\d+) ([^ ]+) (\d+)tr(?: ([^ /]+))? \/ (CPU-[^/]+|GPU-[^/]+) \/ ([^/]+)$/;

      function parseExtra(extra) {
        if (!extra || extra === "[object Object]") return {};
        if (typeof extra === "object") return extra;
        try { return JSON.parse(extra); } catch (_) { return {}; }
      }

      // Environment fingerprint written by benchmarking/src/records.jl: a
      // change of Julia version or resolved Manifest marks a discontinuity.
      function environmentText(extra) {
        if (!extra.julia_version && !extra.manifest_sha256) return "environment not recorded";
        return `Julia ${extra.julia_version || "?"}, Manifest ${String(extra.manifest_sha256 || "?").slice(0, 12)}`;
      }

      function records() {
        const out = [];
        for (const entries of Object.values(window.BENCHMARK_DATA.entries)) {
          for (const entry of entries) {
            for (const bench of entry.benches) {
              const m = bench.name.match(mainCaseRe);
              if (!m) continue;
              const backendLabel = m[8].trim();
              const extra = parseExtra(bench.extra);
              const levels = m[4];
              const scheme = m[7] || "";
              out.push({
                commit: entry.commit.id.slice(0, 7),
                commitUrl: entry.commit.url,
                date: entry.date,
                name: bench.name,
                group: m[1],
                backendGroup: backendLabel.startsWith("CPU-") ? "CPU" : "GPU",
                backendLabel,
                grid: m[2],
                gridN: Number(m[3]),
                levels,
                operator: m[5],
                tracers: m[6],
                scheme,
                floatType: m[9].trim(),
                // A chart line: every case field except the ones the toolbar
                // selects (operator, tracers, float type).
                seriesKey: [m[1], m[2], levels, scheme, backendLabel].join("|"),
                time: Number(bench.value),
                throughput: Number(extra.updates_per_second || extra.updatesPerSecond || NaN),
                device: extra.device || backendLabel,
                environment: environmentText(extra),
              });
            }
          }
        }
        return out;
      }

      function latestByCase(rows) {
        const map = new Map();
        for (const row of rows) {
          const key = [row.seriesKey, row.operator, row.tracers, row.floatType].join("|");
          const old = map.get(key);
          if (!old || row.date > old.date) map.set(key, row);
        }
        return Array.from(map.values());
      }

      function unique(values) {
        return Array.from(new Set(values)).sort((a, b) => {
          const na = Number(a), nb = Number(b);
          return Number.isFinite(na) && Number.isFinite(nb) ? na - nb : String(a).localeCompare(String(b));
        });
      }

      function bySeries(a, b) {
        return a.gridN - b.gridN || Number(a.levels) - Number(b.levels) ||
               a.backendLabel.localeCompare(b.backendLabel) || a.scheme.localeCompare(b.scheme) ||
               a.group.localeCompare(b.group);
      }

      // Case text of a row: grid, levels, and a non-default scheme, plus the
      // group when the panel shows more than one group.
      function caseLabel(row, showGroup) {
        const text = [row.grid, "L" + row.levels, row.scheme].filter(Boolean).join(" ");
        return showGroup ? `${row.group}: ${text}` : text;
      }

      function hasSeveralGroups(rows) {
        return new Set(rows.map(row => row.group)).size > 1;
      }

      function fillSelect(select, values) {
        const previous = select.value;
        select.innerHTML = "";
        for (const value of values) {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = value;
          select.appendChild(option);
        }
        if (values.includes(previous)) select.value = previous;
      }

      function escapeHtml(text) {
        return String(text).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
      }

      function renderTable(id, rows, metric) {
        const table = document.getElementById(id);
        if (rows.length === 0) {
          table.innerHTML = '<tbody><tr><td class="empty">No data</td></tr></tbody>';
          return;
        }
        const valueLabel = metric === "time" ? "s/step" : "updates/s";
        const showGroup = hasSeveralGroups(rows);
        const body = rows.map(row => {
          const value = metric === "time" ? row.time : row.throughput;
          const formatted = metric === "time" ? value.toFixed(4) : value.toExponential(3);
          return `<tr><td>${escapeHtml(caseLabel(row, showGroup))}</td>` +
                 `<td title="${escapeHtml(row.device)}">${escapeHtml(row.backendLabel)}</td>` +
                 `<td title="${escapeHtml(row.environment)}"><a href="${escapeHtml(row.commitUrl)}">${escapeHtml(row.commit)}</a></td><td>${formatted}</td></tr>`;
        }).join("");
        table.innerHTML = `<thead><tr><th>Case</th><th>Device</th><th>Commit</th><th>${valueLabel}</th></tr></thead><tbody>${body}</tbody>`;
      }

      // History per series (group, grid, levels, scheme, device); commits in
      // date order. Colour follows the grid, dash the device, and point style
      // the group and scheme, so series sharing a grid and device stay apart.
      function renderChart(id, rows, metric) {
        const ctx = document.getElementById(id).getContext("2d");
        if (charts[id]) charts[id].destroy();
        const firstSeen = new Map();
        for (const row of rows) {
          const old = firstSeen.get(row.commit);
          if (old === undefined || row.date < old) firstSeen.set(row.commit, row.date);
        }
        const labels = Array.from(firstSeen.keys()).sort((a, b) => firstSeen.get(a) - firstSeen.get(b));
        const seriesRows = new Map();
        for (const row of rows.slice().sort(bySeries)) {
          if (!seriesRows.has(row.seriesKey)) seriesRows.set(row.seriesKey, []);
          seriesRows.get(row.seriesKey).push(row);
        }
        const devices = unique(rows.map(row => row.backendLabel));
        const variants = unique(rows.map(row => row.group + "|" + row.scheme));
        const showGroup = hasSeveralGroups(rows);
        const datasets = Array.from(seriesRows.values()).map((list, i) => {
          const byCommit = new Map();
          for (const row of list) {
            const old = byCommit.get(row.commit);
            if (!old || row.date > old.date) byCommit.set(row.commit, row);
          }
          const first = list[0];
          const color = colors[first.grid] || fallbackColors[i % fallbackColors.length];
          return {
            label: `${caseLabel(first, showGroup)} ${first.backendLabel}`,
            data: labels.map(label => {
              const row = byCommit.get(label);
              return row ? (metric === "time" ? row.time : row.throughput) : null;
            }),
            environments: labels.map(label => (byCommit.get(label) || {}).environment || ""),
            borderColor: color,
            backgroundColor: color + "33",
            borderDash: dashes[devices.indexOf(first.backendLabel) % dashes.length],
            pointStyle: pointStyles[variants.indexOf(first.group + "|" + first.scheme) % pointStyles.length],
            fill: false,
            spanGaps: true,
          };
        });
        charts[id] = new Chart(ctx, {
          type: "line",
          data: { labels, datasets },
          options: {
            maintainAspectRatio: false,
            legend: { position: "bottom" },
            scales: {
              yAxes: [{
                ticks: { beginAtZero: true },
                scaleLabel: {
                  display: true,
                  labelString: metric === "time" ? "s/step" : "cell-tracer updates/s",
                },
              }],
              xAxes: [{ scaleLabel: { display: true, labelString: "commit" } }],
            },
            tooltips: {
              callbacks: {
                label: (item, data) => {
                  const dataset = data.datasets[item.datasetIndex];
                  const unit = metric === "time" ? "s/step" : "updates/s";
                  return `${dataset.label}: ${item.yLabel} ${unit} (${dataset.environments[item.index]})`;
                },
              },
            },
          },
        });
      }

      function render() {
        const operator = document.getElementById("operator-select").value;
        const tracers = document.getElementById("tracer-select").value;
        const floatType = document.getElementById("float-select").value;
        const metric = document.getElementById("metric-select").value;
        const rows = records().filter(row => row.operator === operator && row.tracers === tracers &&
                                             row.floatType === floatType);
        const groups = [
          ["small-cpu", row => row.gridN <= smallGridMax, "CPU"],
          ["small-gpu", row => row.gridN <= smallGridMax, "GPU"],
          ["large-cpu", row => row.gridN > smallGridMax, "CPU"],
          ["large-gpu", row => row.gridN > smallGridMax, "GPU"],
        ];
        for (const [id, inPanel, backend] of groups) {
          const panelRows = rows.filter(row => row.backendGroup === backend && inPanel(row));
          renderChart(id, panelRows, metric);
          renderTable(id + "-table", latestByCase(panelRows).sort(bySeries), metric);
        }
      }

      const all = latestByCase(records());
      document.getElementById("last-update").textContent = new Date(window.BENCHMARK_DATA.lastUpdate).toString();
      const repo = document.getElementById("repo-link");
      repo.href = window.BENCHMARK_DATA.repoUrl;
      repo.textContent = window.BENCHMARK_DATA.repoUrl;
      fillSelect(document.getElementById("operator-select"), unique(all.map(row => row.operator)));
      fillSelect(document.getElementById("tracer-select"), unique(all.map(row => row.tracers)));
      fillSelect(document.getElementById("float-select"), unique(all.map(row => row.floatType)));
      document.getElementById("operator-select").onchange = render;
      document.getElementById("tracer-select").onchange = render;
      document.getElementById("float-select").onchange = render;
      document.getElementById("metric-select").onchange = render;
      render();
    </script>
  </body>
</html>
"""
    path.write_text(html)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: render_grouped_dashboard.py DATA_JS OUTPUT_HTML", file=sys.stderr)
        return 2
    data_path = Path(argv[1])
    output_path = Path(argv[2])
    _load_data(data_path)
    _write_html(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
