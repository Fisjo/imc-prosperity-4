#!/usr/bin/env python3
"""
plot_backtests.py
=================

Dashboard generator for prosperity4btx ``.log`` files.

Reads one or more backtest logs (the JSON output of ``prosperity4btx``)
and produces a single self-contained HTML dashboard with:

    1. Total cumulative PnL chart -- one line per strategy.
    2. Per-product PnL chart with a product selector (also overlays
       all strategies for the chosen product).
    3. A sortable summary table of the final PnL per product per
       strategy, with a delta column when comparing two runs.

The output HTML loads Plotly via CDN, so it has no Python runtime or
local server requirement once generated -- just open it in a browser.

Usage
-----
Single run:
    python plot_backtests.py run.log

Compare strategies:
    python plot_backtests.py baseline.log v3_obi.log

With explicit names:
    python plot_backtests.py --names "Baseline,V3 OBI" baseline.log v3_obi.log

Custom output path:
    python plot_backtests.py run.log -o dashboard.html

Larger / smaller per-product resolution:
    python plot_backtests.py run.log --decim 3000
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_log(path: Path) -> pd.DataFrame:
    """Parse one ``.log`` file into a tidy (timestamp, product, pnl) frame."""
    with open(path, "r") as f:
        data = json.load(f)
    if "activitiesLog" not in data:
        raise ValueError(
            f"{path} does not look like a prosperity4btx log "
            "(missing 'activitiesLog' key)."
        )
    df = pd.read_csv(
        io.StringIO(data["activitiesLog"]),
        sep=";",
        usecols=["timestamp", "product", "profit_and_loss"],
        dtype={
            "timestamp": "int64",
            "product": "category",
            "profit_and_loss": "float64",
        },
    )
    df = df.rename(columns={"profit_and_loss": "pnl"})
    df = df.sort_values(["timestamp", "product"], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Aggregation / payload
# ---------------------------------------------------------------------------
def _decimate(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Subsample to at most ``max_points``, keeping endpoints."""
    n = len(x)
    if n <= max_points or max_points <= 2:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    return x[idx], y[idx]


def build_payload(strategies: dict[str, pd.DataFrame], *, decim: int = 1500) -> dict:
    """Aggregate per-strategy frames into the JSON payload consumed by the dashboard."""
    out: dict = {"strategies": []}
    all_products: set[str] = set()

    for name, df in strategies.items():
        # ---- Total PnL = sum across products at each timestamp ----------
        total = df.groupby("timestamp", observed=True)["pnl"].sum().sort_index()
        t_total = total.index.to_numpy()
        y_total = total.to_numpy()

        # The total chart can take ~2x the resolution -- it's just one line.
        td_total, yd_total = _decimate(t_total, y_total, decim * 2)

        # ---- Per-product PnL pivot --------------------------------------
        pivot = df.pivot_table(
            index="timestamp",
            columns="product",
            values="pnl",
            observed=True,
            aggfunc="last",
        ).sort_index()

        t_prod = pivot.index.to_numpy()
        # Decimate the shared time axis once.
        idx = (
            np.linspace(0, len(t_prod) - 1, decim).astype(np.int64)
            if len(t_prod) > decim
            else np.arange(len(t_prod))
        )
        td_prod = t_prod[idx]

        per_product: dict[str, list[float]] = {}
        for prod in pivot.columns:
            yv = pivot[prod].to_numpy(dtype=np.float64)
            yd = yv[idx]
            per_product[str(prod)] = [round(float(v), 2) for v in yd]
            all_products.add(str(prod))

        # Final PnL per product = last non-NaN value
        final_per_product = {
            str(p): float(pivot[p].dropna().iloc[-1]) if pivot[p].notna().any() else 0.0
            for p in pivot.columns
        }

        # Aggregate stats
        total_final = float(y_total[-1])
        running_max = np.maximum.accumulate(y_total)
        max_dd = float((y_total - running_max).min())
        peak = float(y_total.max())
        trough = float(y_total.min())

        out["strategies"].append({
            "name": name,
            "final_total": total_final,
            "max_drawdown": max_dd,
            "peak": peak,
            "trough": trough,
            "total": {
                "t": [int(v) for v in td_total],
                "y": [round(float(v), 2) for v in yd_total],
            },
            "per_product_t": [int(v) for v in td_prod],
            "per_product": per_product,
            "final_per_product": final_per_product,
        })

    out["products"] = sorted(all_products)
    return out


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Backtest Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {
    --bg: #f5f6f8;
    --card: #ffffff;
    --border: #e3e6eb;
    --text: #1f2937;
    --muted: #6b7280;
    --pos: #16a34a;
    --neg: #dc2626;
    --accent: #2563eb;
    --table-stripe: #fafbfc;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    line-height: 1.4;
  }
  header {
    padding: 22px 32px 14px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
  }
  header h1 {
    margin: 0 0 4px;
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.01em;
  }
  header .subtitle {
    color: var(--muted);
    font-size: 13px;
  }
  main {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px 32px 48px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  .kpi-grid {
    display: grid;
    gap: 16px;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  }
  .kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 8px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  .kpi-name {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
  }
  .kpi-value {
    font-size: 24px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  .kpi-sub {
    color: var(--muted);
    font-size: 12px;
    margin-top: 6px;
    font-variant-numeric: tabular-nums;
  }
  .pos { color: var(--pos); }
  .neg { color: var(--neg); }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  .card h2 {
    margin: 0 0 12px;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: -0.005em;
  }
  .toolbar {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }
  .toolbar label { font-size: 13px; color: var(--muted); }
  .toolbar select {
    font-size: 14px;
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: #fff;
    min-width: 280px;
    font-family: inherit;
  }
  .chart { width: 100%; height: 420px; }
  table.summary {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
  }
  table.summary th, table.summary td {
    text-align: left;
    padding: 7px 12px;
    border-bottom: 1px solid var(--border);
  }
  table.summary th {
    background: #fafbfc;
    font-weight: 600;
    cursor: pointer;
    user-select: none;
    position: sticky;
    top: 0;
  }
  table.summary th:hover { background: #f0f2f5; }
  table.summary td.num { text-align: right; }
  table.summary tr:nth-child(even) td { background: var(--table-stripe); }
  table.summary tr:hover td { background: #eef2ff; }
  .table-wrap { max-height: 540px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; }
  footer { padding: 12px 32px 24px; color: var(--muted); font-size: 11px; }
  .files-list { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 4px; }
  .file-chip { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Backtest Dashboard</h1>
  <div class="subtitle">__SUBTITLE__</div>
  <div class="files-list">__FILES_HTML__</div>
</header>
<main>
  <section id="kpi" class="kpi-grid"></section>

  <section class="card">
    <h2>Total Cumulative PnL</h2>
    <div id="total-chart" class="chart"></div>
  </section>

  <section class="card">
    <h2>Per-Product PnL</h2>
    <div class="toolbar">
      <label for="product-select">Product</label>
      <select id="product-select"></select>
    </div>
    <div id="product-chart" class="chart"></div>
  </section>

  <section class="card">
    <h2>Final PnL by Product</h2>
    <div class="table-wrap">
      <table id="summary-table" class="summary"></table>
    </div>
  </section>
</main>
<footer>Generated by plot_backtests.py - charts powered by Plotly.js. Click any column header in the table to sort.</footer>

<script>
const PAYLOAD = __PAYLOAD__;

const PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777"];
const SECONDS_PER_DAY = 1000000;

function fmt(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  const sign = n < 0 ? "-" : "";
  const abs = Math.abs(n);
  return sign + abs.toLocaleString("en-US", { maximumFractionDigits: 0 });
}
function fmtSigned(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "-";
  return (n >= 0 ? "+" : "") + fmt(n);
}
function color(i) { return PALETTE[i % PALETTE.length]; }

function dayBoundaryShapes(maxX) {
  const shapes = [];
  for (let d = SECONDS_PER_DAY; d < maxX; d += SECONDS_PER_DAY) {
    shapes.push({
      type: "line", x0: d, x1: d, y0: 0, y1: 1, yref: "paper",
      line: { color: "rgba(0,0,0,0.18)", width: 1, dash: "dash" }
    });
  }
  return shapes;
}

function commonLayout(yTitle, traces) {
  let maxX = 0;
  for (const tr of traces) {
    if (tr.x && tr.x.length) maxX = Math.max(maxX, tr.x[tr.x.length - 1]);
  }
  return {
    margin: { t: 16, b: 50, l: 75, r: 20 },
    xaxis: { title: "Timestamp", gridcolor: "#eef0f3", zerolinecolor: "#d4d8de" },
    yaxis: { title: yTitle, gridcolor: "#eef0f3", zerolinecolor: "#d4d8de", tickformat: ",d" },
    legend: { orientation: "h", y: 1.12, x: 0 },
    plot_bgcolor: "#fff",
    paper_bgcolor: "#fff",
    hovermode: "x unified",
    shapes: dayBoundaryShapes(maxX),
    font: { family: "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size: 12 }
  };
}

function renderTotal() {
  const traces = PAYLOAD.strategies.map((s, i) => ({
    x: s.total.t,
    y: s.total.y,
    type: "scattergl",
    mode: "lines",
    name: s.name,
    line: { color: color(i), width: 2 },
    hovertemplate: "%{y:,.0f}<extra>" + s.name + "</extra>"
  }));
  Plotly.react("total-chart", traces, commonLayout("Cumulative PnL", traces),
    { responsive: true, displaylogo: false });
}

function renderProduct(product) {
  const traces = PAYLOAD.strategies.map((s, i) => ({
    x: s.per_product_t,
    y: s.per_product[product] || [],
    type: "scattergl",
    mode: "lines",
    name: s.name,
    line: { color: color(i), width: 2 },
    hovertemplate: "%{y:,.0f}<extra>" + s.name + "</extra>"
  }));
  Plotly.react("product-chart", traces, commonLayout(product + " PnL", traces),
    { responsive: true, displaylogo: false });
}

function buildKpis() {
  const root = document.getElementById("kpi");
  root.innerHTML = "";
  PAYLOAD.strategies.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "kpi-card";
    div.style.borderTopColor = color(i);
    const cls = s.final_total >= 0 ? "pos" : "neg";
    div.innerHTML = `
      <div class="kpi-name">${s.name}</div>
      <div class="kpi-value ${cls}">${fmtSigned(s.final_total)}</div>
      <div class="kpi-sub">Peak ${fmt(s.peak)} &middot; Max DD ${fmt(s.max_drawdown)}</div>
    `;
    root.appendChild(div);
  });
}

function buildProductSelect() {
  const sel = document.getElementById("product-select");
  PAYLOAD.products.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p; opt.textContent = p;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", e => renderProduct(e.target.value));
}

function buildSummaryTable() {
  const t = document.getElementById("summary-table");
  const headers = ["Product", ...PAYLOAD.strategies.map(s => s.name)];
  const showDelta = PAYLOAD.strategies.length === 2;
  if (showDelta) headers.push("\u0394 (" + PAYLOAD.strategies[1].name + " - " + PAYLOAD.strategies[0].name + ")");

  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  headers.forEach((h, idx) => {
    const th = document.createElement("th");
    th.textContent = h;
    if (idx > 0) th.style.textAlign = "right";
    th.dataset.idx = idx;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  t.appendChild(thead);

  const tbody = document.createElement("tbody");
  PAYLOAD.products.forEach(p => {
    const row = document.createElement("tr");
    const parts = [`<td><strong>${p}</strong></td>`];
    const vals = PAYLOAD.strategies.map(s => s.final_per_product[p] ?? 0);
    vals.forEach(v => {
      const cls = v >= 0 ? "pos" : "neg";
      parts.push(`<td class="num ${cls}">${fmtSigned(v)}</td>`);
    });
    if (showDelta) {
      const delta = vals[1] - vals[0];
      const cls = delta >= 0 ? "pos" : "neg";
      parts.push(`<td class="num ${cls}">${fmtSigned(delta)}</td>`);
    }
    row.innerHTML = parts.join("");
    tbody.appendChild(row);
  });

  // Footer total row
  const tfoot = document.createElement("tfoot");
  const trf = document.createElement("tr");
  trf.style.fontWeight = "600";
  trf.style.borderTop = "2px solid var(--border)";
  const fparts = ["<td>Total</td>"];
  const totals = PAYLOAD.strategies.map(s => s.final_total);
  totals.forEach(v => {
    const cls = v >= 0 ? "pos" : "neg";
    fparts.push(`<td class="num ${cls}">${fmtSigned(v)}</td>`);
  });
  if (showDelta) {
    const d = totals[1] - totals[0];
    const cls = d >= 0 ? "pos" : "neg";
    fparts.push(`<td class="num ${cls}">${fmtSigned(d)}</td>`);
  }
  trf.innerHTML = fparts.join("");
  tfoot.appendChild(trf);

  t.appendChild(tbody);
  t.appendChild(tfoot);

  // Sorting
  const sortState = { col: -1, asc: true };
  thead.querySelectorAll("th").forEach((th) => {
    th.addEventListener("click", () => {
      const idx = parseInt(th.dataset.idx);
      sortState.asc = sortState.col === idx ? !sortState.asc : true;
      sortState.col = idx;
      const rows = Array.from(tbody.querySelectorAll("tr"));
      rows.sort((a, b) => {
        let av = a.children[idx].textContent.trim();
        let bv = b.children[idx].textContent.trim();
        if (idx === 0) {
          return sortState.asc ? av.localeCompare(bv) : bv.localeCompare(av);
        }
        const an = parseFloat(av.replace(/[+,]/g, "")) || 0;
        const bn = parseFloat(bv.replace(/[+,]/g, "")) || 0;
        return sortState.asc ? an - bn : bn - an;
      });
      tbody.innerHTML = "";
      rows.forEach(r => tbody.appendChild(r));
    });
  });
}

function init() {
  buildKpis();
  buildProductSelect();
  buildSummaryTable();
  renderTotal();
  renderProduct(PAYLOAD.products[0]);
}
document.addEventListener("DOMContentLoaded", init);
</script>
</body>
</html>
"""


def render_dashboard(payload: dict, files: list[Path], output: Path) -> None:
    subtitle = (
        f"{len(payload['strategies'])} strategy "
        + ("comparison" if len(payload["strategies"]) > 1 else "run")
        + f" - {len(payload['products'])} products"
    )
    files_html = " ".join(f'<span class="file-chip">{f.name}</span>' for f in files)
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    html = (
        HTML_TEMPLATE
        .replace("__SUBTITLE__", subtitle)
        .replace("__FILES_HTML__", files_html)
        .replace("__PAYLOAD__", payload_json)
    )
    output.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML dashboard from prosperity4btx .log files.",
    )
    parser.add_argument("files", nargs="+", type=Path, help="One or more .log files to load.")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("dashboard.html"),
        help="Path of the HTML to write (default: dashboard.html).",
    )
    parser.add_argument(
        "--names", type=str, default=None,
        help="Comma-separated names for the strategies (in the same order as files). "
             "Defaults to the file stem.",
    )
    parser.add_argument(
        "--decim", type=int, default=1500,
        help="Per-product series resolution (default: 1500 points). "
             "The total chart uses 2x this value.",
    )
    args = parser.parse_args(argv)

    # Resolve strategy names
    if args.names:
        names = [n.strip() for n in args.names.split(",")]
        if len(names) != len(args.files):
            parser.error(
                f"--names provided {len(names)} entries but {len(args.files)} files were given."
            )
    else:
        names = [f.stem for f in args.files]

    # De-duplicate names so they render distinctly in the legend.
    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for n in names:
        if n in seen:
            seen[n] += 1
            unique_names.append(f"{n} ({seen[n]})")
        else:
            seen[n] = 1
            unique_names.append(n)

    # Validate files
    for f in args.files:
        if not f.exists():
            parser.error(f"File not found: {f}")

    # Parse and aggregate
    strategies: dict[str, pd.DataFrame] = {}
    for name, path in zip(unique_names, args.files):
        t0 = time.time()
        df = parse_log(path)
        print(f"  parsed {path.name:40s}  {len(df):>10,} rows  in {time.time()-t0:.2f}s",
              file=sys.stderr)
        strategies[name] = df

    t0 = time.time()
    payload = build_payload(strategies, decim=args.decim)
    print(f"  built payload in {time.time()-t0:.2f}s", file=sys.stderr)

    t0 = time.time()
    render_dashboard(payload, list(args.files), args.output)
    size_mb = args.output.stat().st_size / 1e6
    print(
        f"  wrote {args.output}  ({size_mb:.2f} MB)  in {time.time()-t0:.2f}s",
        file=sys.stderr,
    )
    print(f"\nDone. Open {args.output} in your browser.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
