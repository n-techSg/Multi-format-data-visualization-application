"""
Plotbench for Streamlit
Plot columns from CSV and Excel files, add calculated series, and ask Claude
to explain charts or suggest new ones.

Run locally:   streamlit run app.py
AI features:   put ANTHROPIC_API_KEY in .streamlit/secrets.toml (see README)
"""

from __future__ import annotations

import datetime
import io
import json
import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

PALETTE = ["#0B64A0", "#E07A2E", "#2A9D8F", "#C23B4E", "#7B5EA7",
           "#B49A1F", "#4F7A8C", "#D0619E", "#5E9E3B", "#8A5A3C"]

CHART_TYPES = {
    "line": "Line", "area": "Area", "step": "Step line",
    "scatter": "Scatter", "bubble": "Bubble",
    "bar": "Bar (vertical)", "hbar": "Bar (horizontal)", "stacked": "Stacked bar",
    "histogram": "Histogram",
    "pie": "Pie", "doughnut": "Doughnut", "polar": "Polar area", "radar": "Radar",
}
SINGLE_Y = {"pie", "doughnut", "polar", "histogram"}   # one value column only
CIRCULAR = {"pie", "doughnut", "polar"}
NO_X = {"histogram"}
NUMERIC_X = {"line", "area", "step", "scatter", "bubble"}
BARS = {"bar", "hbar", "stacked"}
CALC_OK = lambda t: t not in SINGLE_Y                    # calculated series allowed
AGGS = {"none": "Don't combine", "sum": "Sum", "mean": "Average",
        "count": "Count", "min": "Minimum", "max": "Maximum"}

KINDS = ["prev", "cumsum", "mean", "min", "max", "sum", "first", "last"]
REF_RE = re.compile(r"(?:\b(" + "|".join(KINDS) + r")\s*)?\[([^\]]+)\]")

FORMULA_HELP = (
    "**[Column]** value in the same row  \n"
    "**prev[Column]** value in the previous row, e.g. `[Fuel] - prev[Fuel]`  \n"
    "**cumsum[Column]** running total  \n"
    "**mean[ ] min[ ] max[ ] sum[ ] first[ ] last[ ]** whole-column figures, "
    "e.g. `[Fuel] / mean[Fuel]`  \n"
    "**x** the X value (days since the first date for date columns), **i** the row number  \n"
    "Operators `+ - * / ^ %` and functions `sqrt abs log log10 exp sin cos tan "
    "arctan2`, plus `pi` and `e`. Rows that can't be computed are left as gaps."
)
FORMULA_SYNTAX = (
    "Plotbench formulas use [Column name] for the value in a row, prev[Column] for the "
    "previous row, cumsum[Column], mean[Column], min/max/sum/first/last[Column], x and i, "
    "and operators + - * / ^ with functions sqrt, abs, log (natural), log10, exp, sin, cos."
)

st.set_page_config(page_title="Plotbench", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

ss = st.session_state
ss.setdefault("charts", [])        # list of dicts: {id, calcs:[...], prev_series:[...]}
ss.setdefault("uid", 1)
ss.setdefault("sample", False)
ss.setdefault("suggestions", None)  # {"ds": name, "items": [...]} or {"error": str}


def next_id() -> int:
    ss.uid += 1
    return ss.uid


def k(cid: int, name: str) -> str:
    """Widget key for a chart setting."""
    return f"c{cid}_{name}"


# --------------------------------------------------------------------------
# Loading data
# --------------------------------------------------------------------------

def _clean_numeric(s: pd.Series) -> pd.Series:
    txt = s.astype("string").str.strip().str.replace(",", "", regex=False).str.rstrip("%")
    return pd.to_numeric(txt, errors="coerce")


def tidy_frame(df: pd.DataFrame, header: bool) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    # column names: strings, unique
    names, seen = [], {}
    for i, c in enumerate(df.columns):
        n = f"Column {i + 1}" if (not header or c is None or str(c).startswith("Unnamed") or str(c).strip() == "") else str(c).strip()
        if n in seen:
            seen[n] += 1
            n = f"{n} ({seen[n]})"
        else:
            seen[n] = 1
        names.append(n)
    df.columns = names
    # type detection: number, date, or text
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            df[c] = s.astype(float)
            continue
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        filled = s.dropna()
        filled = filled[filled.astype("string").str.strip() != ""]
        if filled.empty:
            continue
        nums = _clean_numeric(filled)
        if nums.notna().mean() >= 0.9:
            df[c] = _clean_numeric(s).astype(float)
            continue
        try:
            dates = pd.to_datetime(filled.astype("string"), errors="coerce", dayfirst=True, format="mixed")
        except (ValueError, TypeError):
            dates = pd.Series(pd.NaT, index=filled.index)
        if dates.notna().mean() >= 0.9:
            df[c] = pd.to_datetime(s.astype("string"), errors="coerce", dayfirst=True, format="mixed")
    return df


@st.cache_data(show_spinner=False)
def excel_sheets(data: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(data)).sheet_names


@st.cache_data(show_spinner="Reading file…")
def read_file(name: str, data: bytes, sheet: str | None, header: bool) -> pd.DataFrame:
    hdr = 0 if header else None
    if name.lower().endswith((".csv", ".tsv", ".txt")):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=hdr, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
    else:
        df = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=hdr)
    return tidy_frame(df, header)


def sample_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 60
    i = np.arange(n)
    speed = 11 + np.sin(i / 6) * 2.2 + (rng.random(n) - 0.5) * 1.2
    draft = np.where(i < 30, 12.4, 9.1)
    bf = np.clip(np.round(3 + np.sin(i / 4) * 2 + rng.random(n) * 1.5), 1, 8)
    foc = 0.0175 * speed ** 3 * (draft / 11) + bf * 0.9 + (rng.random(n) - 0.5) * 2
    return pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "Speed (kn)": speed.round(2), "ME fuel (t/day)": foc.round(2),
        "Draft (m)": draft, "Wind (BF)": bf, "Leg": np.where(i < 30, "Laden", "Ballast"),
    })


def col_type(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    return "text"


# --------------------------------------------------------------------------
# Sidebar: files
# --------------------------------------------------------------------------

datasets: dict[str, pd.DataFrame] = {}

with st.sidebar:
    st.title("📈 Plotbench")
    st.caption("Plot columns from CSV and Excel files.")
    files = st.file_uploader("Load data files", type=["csv", "tsv", "txt", "xls", "xlsx", "xlsm"],
                             accept_multiple_files=True)
    for f in files or []:
        data = f.getvalue()
        name = f.name
        while name in datasets:
            name += " (copy)"
        with st.expander(name, expanded=False):
            sheet = None
            if not f.name.lower().endswith((".csv", ".tsv", ".txt")):
                try:
                    sheets = excel_sheets(data)
                except Exception as e:  # noqa: BLE001
                    st.error(f"Couldn't open this workbook: {e}")
                    continue
                sheet = st.selectbox("Sheet", sheets, key=f"ds_{name}_sheet") if len(sheets) > 1 else sheets[0]
            header = st.checkbox("First row is header", True, key=f"ds_{name}_hdr")
        try:
            df = read_file(f.name, data, sheet, header)
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't read {f.name}: {e}")
            continue
        if df.empty:
            st.warning(f"{f.name} has no readable data.")
            continue
        label = name if sheet is None or len(excel_sheets(data)) == 1 else f"{name} · {sheet}"
        datasets[label] = df

    if ss.sample:
        datasets["Sample voyage log"] = sample_frame()

    if not datasets:
        if st.button("Try with sample voyage data", width="stretch"):
            ss.sample = True
            st.rerun()

    if datasets:
        st.divider()
        st.subheader("Data preview")
        prev_name = st.selectbox("File", list(datasets), key="preview_ds", label_visibility="collapsed")
        pdf = datasets[prev_name]
        st.caption(f"{len(pdf):,} rows × {pdf.shape[1]} columns")
        st.dataframe(pd.DataFrame({"Column": pdf.columns, "Type": [col_type(pdf[c]) for c in pdf.columns]}),
                     hide_index=True, width="stretch", height=min(35 * (pdf.shape[1] + 1) + 3, 260))
        if st.toggle("Show first rows", key="preview_rows"):
            st.dataframe(pdf.head(100), width="stretch", height=260)

# --------------------------------------------------------------------------
# AI client
# --------------------------------------------------------------------------

def _secret(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # noqa: BLE001  (no secrets file)
        pass
    return os.environ.get(name, default)


API_KEY = _secret("ANTHROPIC_API_KEY")
MODEL = _secret("ANTHROPIC_MODEL", "claude-sonnet-5-5")
AI_ON = bool(API_KEY)


@st.cache_resource
def get_client(key: str):
    import anthropic
    return anthropic.Anthropic(api_key=key)


def ai_error_text(e: Exception) -> str:
    import anthropic
    if isinstance(e, anthropic.AuthenticationError):
        return "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in your secrets."
    if isinstance(e, anthropic.RateLimitError):
        return "Too many AI requests right now. Wait a moment, then try again."
    if isinstance(e, anthropic.NotFoundError):
        return f"The model '{MODEL}' wasn't found. Set ANTHROPIC_MODEL in your secrets to a current model."
    if isinstance(e, anthropic.APIConnectionError):
        return "Couldn't reach the Anthropic API. Check the server's internet connection."
    return f"The AI request failed: {e}"


def ai_stream(system: str, messages: list[dict], max_tokens: int = 1200):
    client = get_client(API_KEY)
    with client.messages.stream(model=MODEL, max_tokens=max_tokens, system=system,
                                messages=messages) as stream:
        for text in stream.text_stream:
            yield text


def ai_json(system: str, prompt: str, max_tokens: int = 2500):
    client = get_client(API_KEY)
    msg = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                                 messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("The reply didn't contain a JSON list.")
    return json.loads(text[start:end + 1])


# --------------------------------------------------------------------------
# Calculated series
# --------------------------------------------------------------------------

class FormulaError(Exception):
    pass


def _match_col(df: pd.DataFrame, want: str) -> str:
    want = want.strip()
    if want in df.columns:
        return want
    for c in df.columns:
        if c.lower() == want.lower():
            return c
    raise FormulaError(f'No column called "{want}".')


def eval_formula(expr: str, df: pd.DataFrame, xcol: str | None) -> pd.Series:
    """Evaluate a Plotbench formula row by row over df (in file order)."""
    if not expr.strip():
        raise FormulaError("Enter a formula.")
    env: dict[str, object] = {}
    counter = [0]

    def repl(m: re.Match) -> str:
        kind, col = m.group(1) or "cur", _match_col(df, m.group(2))
        if col_type(df[col]) != "number":
            raise FormulaError(f'"{col}" isn\'t a numeric column.')
        v = pd.to_numeric(df[col], errors="coerce").astype(float)
        name = f"_v{counter[0]}"
        counter[0] += 1
        env[name] = {
            "cur": v, "prev": v.shift(1), "cumsum": v.fillna(0).cumsum(),
            "mean": v.mean(), "min": v.min(), "max": v.max(), "sum": v.sum(),
            "first": v.dropna().iloc[0] if v.notna().any() else np.nan,
            "last": v.dropna().iloc[-1] if v.notna().any() else np.nan,
        }[kind]
        return f" {name} "

    code = REF_RE.sub(repl, expr).replace("^", "**")
    n = len(df)
    env["i"] = pd.Series(np.arange(1, n + 1, dtype=float), index=df.index)
    if xcol and xcol in df.columns and col_type(df[xcol]) in ("number", "date"):
        xs = df[xcol]
        if col_type(xs) == "date":
            xs = (xs - xs.min()).dt.total_seconds() / 86400
        env["x"] = pd.to_numeric(xs, errors="coerce").astype(float)
    else:
        env["x"] = pd.Series(np.nan, index=df.index)
    env["pi"], env["e"] = np.pi, np.e
    try:
        with np.errstate(all="ignore"):
            out = pd.eval(code, local_dict=env, engine="python", parser="pandas")
    except Exception as err:  # noqa: BLE001
        msg = re.sub(r"_v\d+", "[column]", str(err))
        raise FormulaError(f"Formula error: {msg}") from None
    if np.isscalar(out):
        out = pd.Series(float(out), index=df.index)
    out = pd.to_numeric(pd.Series(out, index=df.index), errors="coerce").astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


def unique_name(existing: set[str], base: str) -> str:
    name, i = base, 2
    while name in existing:
        name = f"{base} ({i})"
        i += 1
    return name


# --------------------------------------------------------------------------
# Chart management (callbacks run before the script, so keys can be set freely)
# --------------------------------------------------------------------------

def default_columns(df: pd.DataFrame) -> tuple[str | None, list[str]]:
    cols = list(df.columns)
    if not cols:
        return None, []
    x = next((c for c in cols if col_type(df[c]) == "date"), cols[0])
    nums = [c for c in cols if col_type(df[c]) == "number" and c != x]
    return x, nums[:1]


def add_chart(ds_name: str | None = None, ctype: str = "line", preset: dict | None = None):
    names = list(datasets)
    if not names:
        return None
    ds_name = ds_name if ds_name in datasets else names[0]
    cid = next_id()
    x, ys = default_columns(datasets[ds_name])
    ss[k(cid, "ds")] = ds_name
    ss[k(cid, "type")] = ctype
    ss[k(cid, "x")] = x
    ss[k(cid, "ys")] = ys
    chart = {"id": cid, "calcs": [], "prev_series": [], "ai": [], "ai_sig": None}
    if preset:
        for key, val in preset.items():
            if key == "calcs":
                chart["calcs"] = [dict(c, id=next_id()) for c in val]
            else:
                ss[k(cid, key)] = val
    ss.charts.append(chart)
    return chart


def remove_chart(cid: int):
    ss.charts = [c for c in ss.charts if c["id"] != cid]


def duplicate_chart(cid: int):
    src = next(c for c in ss.charts if c["id"] == cid)
    preset = {}
    for name in ("type", "x", "ys", "size", "xlabel", "ylabel", "agg", "bins", "logy",
                 "legend", "sortx", "zero", "points", "zoom", "shown", "focus", "frange"):
        if k(cid, name) in ss:
            preset[name] = ss[k(cid, name)]
    title = ss.get(k(cid, "title"), "")
    preset["title"] = f"{title} (copy)" if title else ""
    preset["calcs"] = [{key: v for key, v in c.items() if key != "id"} for c in src["calcs"]]
    add_chart(ss[k(cid, "ds")], ss[k(cid, "type")], preset)


def add_subtraction(cid: int):
    chart = next(c for c in ss.charts if c["id"] == cid)
    a, b = ss.get(k(cid, "subA")), ss.get(k(cid, "subB"))
    if not a or not b:
        return
    taken = set(ss.get(k(cid, "ys"), [])) | {c["name"] for c in chart["calcs"]}
    chart["calcs"].append({"id": next_id(), "name": unique_name(taken, f"{a} − {b}"),
                           "expr": f"[{a}] - [{b}]", "right": False})


def add_custom(cid: int):
    chart = next(c for c in ss.charts if c["id"] == cid)
    expr = (ss.get(k(cid, "newexpr")) or "").strip()
    name = (ss.get(k(cid, "newname")) or "").strip()
    df = datasets.get(ss.get(k(cid, "ds")))
    try:
        if df is None:
            raise FormulaError("Load a file first.")
        eval_formula(expr, df, ss.get(k(cid, "x")))
    except FormulaError as e:
        ss[k(cid, "newerr")] = str(e)
        return
    taken = set(ss.get(k(cid, "ys"), [])) | {c["name"] for c in chart["calcs"]}
    chart["calcs"].append({"id": next_id(), "name": unique_name(taken, name or expr),
                           "expr": expr, "right": False})
    ss[k(cid, "newexpr")] = ""
    ss[k(cid, "newname")] = ""
    ss[k(cid, "newerr")] = ""


def remove_calc(cid: int, kid: int):
    chart = next(c for c in ss.charts if c["id"] == cid)
    chart["calcs"] = [c for c in chart["calcs"] if c["id"] != kid]


def request_explain(cid: int):
    ss[k(cid, "explain_req")] = True


def close_ai(cid: int):
    chart = next(c for c in ss.charts if c["id"] == cid)
    chart["ai"], chart["ai_sig"] = [], None


# --------------------------------------------------------------------------
# Building chart data
# --------------------------------------------------------------------------

def sanitize(key: str, options: list, default, multi: bool = False):
    """Make sure a stored widget value is still valid for its options."""
    if multi:
        cur = ss.get(key, default)
        ss[key] = [v for v in (cur or []) if v in options]
    elif ss.get(key) not in options:
        ss[key] = default if default in options else (options[0] if options else None)


def build_work(cfg: dict, df: pd.DataFrame, calcs: list[dict]):
    """Return (work frame, series list, info) for plotting and AI."""
    t, x = cfg["type"], cfg["x"]
    ys = list(cfg["ys"])
    series = [{"name": y, "calc": None} for y in ys]
    work = pd.DataFrame(index=df.index)
    if x and t not in NO_X:
        work["__x"] = df[x]
    for y in ys:
        work[y] = pd.to_numeric(df[y], errors="coerce").astype(float)
    if CALC_OK(t):
        for c in calcs:
            try:
                work[c["name"]] = eval_formula(c["expr"], df, x)
                c["error"] = ""
                series.append({"name": c["name"], "calc": c})
            except FormulaError as e:
                c["error"] = str(e)
    if t == "bubble" and cfg.get("size"):
        work["__size"] = pd.to_numeric(df[cfg["size"]], errors="coerce").astype(float)

    xtype = col_type(df[x]) if x and x in df.columns else None
    if "__x" in work:
        work = work[work["__x"].notna()]
        if xtype == "text":
            work["__x"] = work["__x"].astype(str)
    # focus range
    if cfg.get("focus") and cfg.get("frange") and xtype in ("number", "date") and "__x" in work:
        lo, hi = cfg["frange"]
        work = work[(work["__x"] >= lo) & (work["__x"] <= hi)]
    names = [s["name"] for s in series]
    agg = cfg.get("agg", "none")
    if agg != "none" and t not in ("scatter", "bubble") and "__x" in work and names:
        g = work.groupby("__x", sort=False)[names]
        work = (pd.DataFrame({n: g.size() for n in names}) if agg == "count" else g.agg(agg)).reset_index()
    if "__x" in work and (cfg.get("sortx") or (t in NUMERIC_X and xtype in ("number", "date"))):
        work = work.sort_values("__x", kind="stable")
    return work.reset_index(drop=True), series, xtype


def make_figure(cfg: dict, work: pd.DataFrame, series: list[dict], xtype: str | None, df: pd.DataFrame):
    t = cfg["type"]
    shown = set(cfg.get("shown") or [])
    fig = go.Figure()
    title = cfg["title_text"]
    n = len(work)
    color_of = {s["name"]: PALETTE[i % len(PALETTE)] for i, s in enumerate(series)}
    vis = lambda name: True if name in shown else False

    if t == "histogram":
        vals = pd.to_numeric(df[cfg["ys"][0]], errors="coerce").dropna()
        fig.add_trace(go.Histogram(x=vals, nbinsx=int(cfg.get("bins") or 20), name=cfg["ys"][0],
                                   marker=dict(color=PALETTE[0], line=dict(width=1, color="white"))))
        fig.update_layout(bargap=0.02)
    elif t in CIRCULAR:
        y = cfg["ys"][0]
        labels, vals = work["__x"].astype(str), work[y]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
        if t == "polar":
            fig.add_trace(go.Barpolar(r=vals, theta=labels, marker_color=colors, name=y))
        else:
            fig.add_trace(go.Pie(labels=labels, values=vals, hole=0.45 if t == "doughnut" else 0,
                                 marker=dict(colors=colors), sort=False, name=y))
    elif t == "radar":
        theta = work["__x"].astype(str).tolist()
        for s in series:
            r = work[s["name"]].tolist()
            fig.add_trace(go.Scatterpolar(
                r=r + r[:1], theta=theta + theta[:1], name=s["name"], visible=vis(s["name"]),
                fill="none" if s["calc"] else "toself", opacity=0.9,
                line=dict(color=color_of[s["name"]], dash="dash" if s["calc"] else "solid"),
                mode="lines+markers" if cfg.get("points") else "lines"))
    else:
        xv = work["__x"]
        big = n > 1500
        for s in series:
            name, col = s["name"], color_of[s["name"]]
            yaxis = "y2" if (s["calc"] and s["calc"].get("right") and t != "hbar") else "y"
            yv = work[name]
            common = dict(name=name, visible=vis(name), legendgroup=name)
            if s["calc"]:
                common["name"] = f"{name} (calc)"
            if t in BARS and not s["calc"]:
                if t == "hbar":
                    fig.add_trace(go.Bar(x=yv, y=xv, orientation="h", marker_color=col, **common))
                else:
                    fig.add_trace(go.Bar(x=xv, y=yv, marker_color=col, yaxis=yaxis, **common))
            elif t in BARS:  # calculated series as a line overlay
                xy = dict(x=yv, y=xv) if t == "hbar" else dict(x=xv, y=yv, yaxis=yaxis)
                fig.add_trace(go.Scatter(mode="lines+markers", line=dict(color=col, dash="dash", width=2),
                                         marker=dict(symbol="diamond", size=7), **xy, **common))
            elif t in ("scatter", "bubble"):
                marker = dict(color=col, opacity=0.75, size=4 if big else 8)
                if s["calc"]:
                    marker.update(symbol="triangle-up", opacity=0.9)
                elif t == "bubble" and "__size" in work:
                    sz = work["__size"]
                    rng = (sz.max() - sz.min()) or 1
                    marker["size"] = (8 + 32 * (sz - sz.min()) / rng).fillna(6)
                    marker["opacity"] = 0.5
                    common["customdata"] = sz
                    common["hovertemplate"] = f"%{{x}}<br>{name}: %{{y}}<br>{cfg.get('size')}: %{{customdata}}<extra></extra>"
                fig.add_trace(go.Scatter(x=xv, y=yv, mode="markers", marker=marker, yaxis=yaxis, **common))
            else:  # line, area, step
                show_pts = cfg.get("points") and n <= 300
                fig.add_trace(go.Scatter(
                    x=xv, y=yv, yaxis=yaxis, connectgaps=True,
                    mode="lines+markers" if show_pts or s["calc"] and n <= 300 else "lines",
                    line=dict(color=col, width=2 if s["calc"] else 1.75, dash="dash" if s["calc"] else "solid",
                              shape="hv" if t == "step" else "linear"),
                    marker=dict(size=6 if s["calc"] else 5, symbol="diamond" if s["calc"] else "circle"),
                    fill="tozeroy" if (t == "area" and not s["calc"]) else None,
                    **common))
        if t == "stacked":
            fig.update_layout(barmode="stack")

        # axes
        xlab, ylab = cfg["xlabel_text"], cfg["ylabel_text"]
        value_axis = dict(title=ylab, type="log" if cfg.get("logy") else "linear",
                          rangemode="tozero" if (cfg.get("zero") or t in BARS) else "normal")
        cat_axis = dict(title=xlab)
        if xtype == "text" and t in NUMERIC_X:
            cat_axis["type"] = "category"
        zoom = cfg.get("zoom", "x")
        if t == "hbar":
            fig.update_layout(xaxis=value_axis, yaxis=dict(cat_axis, autorange="reversed"))
            fig.update_xaxes(fixedrange=(zoom == "y"))
            fig.update_yaxes(fixedrange=(zoom == "x"))
        else:
            fig.update_layout(xaxis=cat_axis, yaxis=value_axis)
            fig.update_xaxes(fixedrange=(zoom == "y"))
            fig.update_yaxes(fixedrange=(zoom == "x"))
            right = [s["name"] for s in series if s["calc"] and s["calc"].get("right")]
            if right:
                fig.update_layout(yaxis2=dict(title=", ".join(right), overlaying="y", side="right",
                                              showgrid=False, fixedrange=(zoom == "x")))

    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=18)),
        showlegend=bool(cfg.get("legend", True)), height=460,
        margin=dict(l=10, r=10, t=60, b=10), hovermode="x unified" if t in ("line", "area", "step") else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        dragmode="zoom",
    )
    return fig


# --------------------------------------------------------------------------
# AI: chart explanation context
# --------------------------------------------------------------------------

def fmt(v) -> str:
    if isinstance(v, (pd.Timestamp, np.datetime64, datetime.datetime, datetime.date)):
        ts = pd.Timestamp(v)
        return ts.strftime("%Y-%m-%d %H:%M") if (ts.hour or ts.minute) else ts.strftime("%Y-%m-%d")
    if isinstance(v, (int, float, np.integer, np.floating)):
        if not np.isfinite(v):
            return ""
        a = abs(float(v))
        return f"{v:.3g}" if a and (a >= 1e7 or a < 1e-3) else f"{round(float(v), 4):g}"
    return str(v)


def chart_context(cfg: dict, work: pd.DataFrame, series: list[dict], xtype, df: pd.DataFrame, ds_name: str) -> str | None:
    t = cfg["type"]
    L = [f"Chart title: {cfg['title_text']}", f"Chart type: {CHART_TYPES[t]}",
         f"Data file: {ds_name}, {len(df)} rows"]
    if t == "histogram":
        vals = pd.to_numeric(df[cfg["ys"][0]], errors="coerce").dropna()
        if vals.empty:
            return None
        counts, edges = np.histogram(vals, bins=int(cfg.get("bins") or 20))
        L.append(f'Histogram of "{cfg["ys"][0]}". Bin range: count')
        L += [f"{fmt(edges[i])} to {fmt(edges[i + 1])}: {c}" for i, c in enumerate(counts)]
        L.append(f"Overall: n={len(vals)}, min={fmt(vals.min())}, max={fmt(vals.max())}, mean={fmt(vals.mean())}, "
                 f"median={fmt(vals.median())}, std dev={fmt(vals.std())}")
        return "\n".join(L)
    if t in CIRCULAR:
        y = cfg["ys"][0]
        L.append(f'Category column "{cfg["x"]}", value column "{y}"'
                 + (f", rows combined by {cfg['agg']}" if cfg.get("agg") != "none" else "") + ". Category: value")
        L += [f"{a}: {fmt(b)}" for a, b in zip(work["__x"].astype(str)[:200], work[y][:200])]
        return "\n".join(L)

    shown = set(cfg.get("shown") or [])
    vis = [s for s in series if s["name"] in shown and work[s["name"]].notna().any()]
    if not vis:
        return None
    L.append(f'X axis: "{cfg["xlabel_text"]}" (column "{cfg["x"]}", {xtype})')
    L.append(f'Y axis: "{cfg["ylabel_text"]}"' + (" (log scale)" if cfg.get("logy") else ""))
    if cfg.get("agg", "none") != "none" and t not in ("scatter", "bubble"):
        L.append(f"Rows sharing an X value are combined by: {cfg['agg']}")
    if cfg.get("focus") and cfg.get("frange"):
        L.append(f"The viewer has limited the chart to X from {fmt(cfg['frange'][0])} to {fmt(cfg['frange'][1])}.")
    xnum = None
    if xtype == "date":
        xnum = (work["__x"] - work["__x"].min()).dt.total_seconds() / 86400
    elif xtype == "number":
        xnum = work["__x"].astype(float)

    L += ["", "Visible series:"]
    for s in vis:
        v = work[s["name"]]
        ok = v.notna()
        vv = v[ok]
        line = (f'- "{s["name"]}"' + (f' (calculated with formula {s["calc"]["expr"]})' if s["calc"] else "")
                + (" [right-hand axis]" if s["calc"] and s["calc"].get("right") else "")
                + f": n={len(vv)}, min={fmt(vv.min())}, max={fmt(vv.max())}, mean={fmt(vv.mean())}, "
                  f"median={fmt(vv.median())}, std dev={fmt(vv.std())}")
        xs = work.loc[ok, "__x"]
        if xtype in ("number", "date"):
            line += f", first={fmt(vv.iloc[0])} at {fmt(xs.iloc[0])}, last={fmt(vv.iloc[-1])} at {fmt(xs.iloc[-1])}"
        line += f", lowest at {fmt(xs.loc[vv.idxmin()])}, highest at {fmt(xs.loc[vv.idxmax()])}"
        if xnum is not None and len(vv) > 2:
            r = np.corrcoef(xnum[ok], vv)[0, 1]
            if np.isfinite(r):
                line += f", correlation with X r={r:.2f}"
        sd = vv.std()
        if sd and np.isfinite(sd):
            out = vv[(vv - vv.mean()).abs() / sd > 2.5].head(5)
            if len(out):
                line += ". Points more than 2.5 std dev from the mean: " + "; ".join(
                    f"{fmt(val)} at {fmt(work.loc[idx, '__x'])}" for idx, val in out.items())
        L.append(line)
    if len(vis) > 1:
        pairs = []
        for a, b in combinations(vis, 2):
            pair = work[[a["name"], b["name"]]].dropna()
            if len(pair) > 2:
                r = pair.corr().iloc[0, 1]
                if np.isfinite(r):
                    pairs.append(f'- "{a["name"]}" vs "{b["name"]}": r={r:.2f} ({len(pair)} matched points)')
        if pairs:
            L += ["", "Correlation between series:"] + pairs[:15]
    stride = max(1, int(np.ceil(len(work) / 150)))
    sample = work.iloc[::stride]
    L += ["", f"Data points{f' (every {stride}th of {len(work)}, sampled)' if stride > 1 else ''}, CSV:"]
    L.append(",".join(f'"{c}"' for c in [cfg["x"]] + [s["name"] for s in vis]))
    for _, row in sample.iterrows():
        L.append(",".join([fmt(row["__x"])] + [fmt(row[s["name"]]) if pd.notna(row[s["name"]]) else "" for s in vis]))
    return "\n".join(L)


EXPLAIN_SYSTEM = f"""You are a data analyst helping an engineer read a chart made in Plotbench, a charting tool. You receive a summary of exactly what the chart shows: its configuration, statistics for each visible series, correlations, and a sample of the plotted points.

When asked to explain the chart:
- Start with a one-sentence headline of the main takeaway.
- Then 3 to 6 short bullet points covering trends, relationships between series, notable highs, lows, jumps or outliers (say where they happen on the X axis), and any change in behaviour across the range.
- Quote specific numbers from the data. Never invent values that are not in the data.
- Say briefly when a pattern could have more than one explanation, or when the data is too sparse to be sure. Correlation is not cause.
- Finish with one line starting "Try next:" suggesting a further chart or calculated series. {FORMULA_SYNTAX}
Keep it under about 220 words, in simple Markdown with "- " bullets. No tables or headings.
For follow-up questions, answer in under 150 words using numbers from the data where possible."""


# --------------------------------------------------------------------------
# AI: chart suggestions
# --------------------------------------------------------------------------

def dataset_summary(name: str, df: pd.DataFrame) -> str:
    L = [f"File: {name}, {len(df)} rows, {df.shape[1]} columns.", "", "Columns:"]
    for c in df.columns:
        s = df[c]
        ty = col_type(s)
        d = f'- "{c}" ({ty})'
        if ty == "number" and s.notna().any():
            d += f": min {fmt(s.min())}, max {fmt(s.max())}, mean {fmt(s.mean())}"
        elif ty == "date" and s.notna().any():
            d += f": {fmt(s.min())} to {fmt(s.max())}"
        elif ty == "text":
            vc = s.astype(str).value_counts()
            d += f": {len(vc)} distinct values, e.g. " + ", ".join(f"{v[:30]} ({n})" for v, n in vc.head(5).items())
        miss = int(s.isna().sum())
        if miss:
            d += f", {miss} blank"
        L.append(d)
    L += ["", "First rows (CSV):", df.head(8).to_csv(index=False)]
    return "\n".join(L)


def run_suggestions(ds_name: str):
    df = datasets[ds_name]
    prompt = f"""Suggest 4 charts that would reveal the most useful insights in this data: trends over time, relationships between variables, distributions, comparisons between groups, and derived quantities worth calculating.

{dataset_summary(ds_name, df)}

Chart types (use the id): {", ".join(f"{a} ({b})" for a, b in CHART_TYPES.items())}.
Rules: "x" and every "ys" entry must be an exact column name from the list. "ys" holds numeric columns. For histogram, leave "x" empty and put one value column in "ys". For pie, doughnut and polar use one numeric column in "ys" and a category column as "x". Use "agg" (none, sum, mean, count, min, max) when several rows share the same X value and should be combined. You may add calculated series in "calcs" for X/Y charts. {FORMULA_SYNTAX} Column names in formulas must match exactly. "calcs" may be empty, and "ys" may be empty when calcs carry the series.

Reply with only a JSON array like:
[{{"title":"Fuel use against speed","type":"scatter","x":"Speed","ys":["Fuel"],"calcs":[{{"name":"Fuel per mile","expr":"[Fuel] / [Speed]"}}],"agg":"none","why":"One sentence on what this chart could reveal."}}]"""
    raw = ai_json("You help an engineer explore a data file in Plotbench, a charting tool. Reply with JSON only.", prompt)
    items = []
    for sg in raw if isinstance(raw, list) else []:
        if not isinstance(sg, dict) or sg.get("type") not in CHART_TYPES:
            continue
        t = sg["type"]
        x = sg.get("x") if sg.get("x") in df.columns else None
        ys = [y for y in (sg.get("ys") or []) if y in df.columns and col_type(df[y]) == "number" and y != x]
        calcs = []
        if CALC_OK(t):
            for c in sg.get("calcs") or []:
                try:
                    eval_formula(str(c.get("expr", "")), df, x)
                    calcs.append({"name": str(c.get("name") or c["expr"])[:60], "expr": c["expr"], "right": False})
                except (FormulaError, KeyError, AttributeError):
                    pass
        if t not in NO_X and not x:
            continue
        if not ys and not calcs:
            continue
        items.append({"title": str(sg.get("title", ""))[:90], "type": t, "x": x,
                      "ys": ys[:1] if t in SINGLE_Y else ys, "calcs": calcs,
                      "agg": sg.get("agg") if sg.get("agg") in AGGS else "none", "why": str(sg.get("why", ""))})
    return items[:6]


def add_suggested(i: int):
    sg = ss.suggestions["items"][i]
    preset = {"ys": sg["ys"], "agg": sg["agg"], "title": sg["title"], "calcs": sg["calcs"]}
    if sg["x"]:
        preset["x"] = sg["x"]
    add_chart(ss.suggestions["ds"], sg["type"], preset)
    sg["added"] = True


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------

if not datasets:
    st.title("Plotbench")
    st.info("Load a CSV or Excel file from the sidebar to start plotting, or try the sample voyage data.")
    st.stop()

if not ss.charts:
    add_chart()

top = st.columns([1.2, 1.7, 2.2, 3])
top[0].button("➕ Add chart", type="primary", on_click=add_chart, width="stretch")
per_row = top[1].segmented_control("Layout", ["1 per row", "2 per row"], default="1 per row",
                                   key="layout", label_visibility="collapsed") or "1 per row"
if AI_ON:
    sg_ds = top[2].selectbox("File for suggestions", list(datasets), key="sg_ds", label_visibility="collapsed")
    if top[3].button("✨ Suggest charts with AI", width="content"):
        with st.spinner("Claude is looking at the columns…"):
            try:
                ss.suggestions = {"ds": sg_ds, "items": run_suggestions(sg_ds)}
            except Exception as e:  # noqa: BLE001
                ss.suggestions = {"error": ai_error_text(e) if not isinstance(e, (ValueError, json.JSONDecodeError))
                                  else "The suggestions came back in a form Plotbench couldn't read. Try again."}
else:
    top[2].caption("AI features are off. Add ANTHROPIC_API_KEY to secrets to turn on chart explanations and suggestions.")

if ss.suggestions:
    with st.container(border=True):
        h = st.columns([6, 1])
        h[0].subheader("Suggested charts" + (f" for {ss.suggestions['ds']}" if "ds" in ss.suggestions else ""))
        if h[1].button("Close", key="sg_close"):
            ss.suggestions = None
            st.rerun()
        if "error" in ss.suggestions:
            st.error(ss.suggestions["error"])
        elif not ss.suggestions["items"]:
            st.warning("None of the suggestions matched the columns in this file. Try again.")
        else:
            cols = st.columns(min(4, len(ss.suggestions["items"])))
            for i, sg in enumerate(ss.suggestions["items"]):
                with cols[i % len(cols)]:
                    st.markdown(f"**{sg['title'] or 'Suggested chart'}**")
                    st.caption(sg["why"])
                    desc = CHART_TYPES[sg["type"]] + (f" of {', '.join(sg['ys']) or 'calculated values'} by {sg['x']}" if sg["x"] else f" of {sg['ys'][0]}")
                    if sg["agg"] != "none":
                        desc += f", rows combined by {sg['agg']}"
                    st.caption(desc)
                    for c in sg["calcs"]:
                        st.code(c["expr"], language=None)
                    st.button("Added" if sg.get("added") else "Add this chart", key=f"sg_add_{i}",
                              disabled=bool(sg.get("added")), on_click=add_suggested, args=(i,))


def render_chart(chart: dict):
    cid = chart["id"]
    ds_names = list(datasets)
    sanitize(k(cid, "ds"), ds_names, ds_names[0])
    ds_name = ss[k(cid, "ds")]
    df = datasets[ds_name]
    cols = list(df.columns)
    num_cols = [c for c in cols if col_type(df[c]) == "number"]
    dx, dys = default_columns(df)

    with st.container(border=True):
        head = st.columns([5, 1, 1, 1])
        head[0].text_input("Chart title", key=k(cid, "title"), placeholder="Chart title",
                           label_visibility="collapsed")
        if AI_ON:
            head[1].button("✨ Explain", key=k(cid, "btn_explain"), on_click=request_explain, args=(cid,),
                           width="stretch", help="Ask Claude what this chart is telling you")
        head[2].button("Duplicate", key=k(cid, "btn_dup"), on_click=duplicate_chart, args=(cid,), width="stretch")
        head[3].button("Remove", key=k(cid, "btn_rm"), on_click=remove_chart, args=(cid,), width="stretch")

        if per_row == "1 per row":
            plot_area, set_area = st.columns([3, 1.25])
            settings = set_area.container(height=520, border=False)
        else:
            plot_area = st.container()
            settings = st.expander("Chart settings")

        # ---------------- settings ----------------
        with settings:
            if len(ds_names) > 1:
                st.selectbox("Data file", ds_names, key=k(cid, "ds"))
            sanitize(k(cid, "type"), list(CHART_TYPES), "line")
            t = st.selectbox("Chart type", list(CHART_TYPES), format_func=CHART_TYPES.get, key=k(cid, "type"))
            if chart.get("last_type") not in (None, t):
                ss[k(cid, "agg")] = "sum" if t in CIRCULAR else "none"
            chart["last_type"] = t
            if t not in NO_X:
                sanitize(k(cid, "x"), cols, dx)
                xl = "Category (labels) column" if t in CIRCULAR else "Spoke labels column" if t == "radar" else "X axis column"
                st.selectbox(xl, cols, key=k(cid, "x"), format_func=lambda c: f"{c} ({col_type(df[c])})")
            x = ss.get(k(cid, "x")) if t not in NO_X else None
            yopts = [c for c in num_cols if c != x]
            if t in SINGLE_Y:
                cur = ss.get(k(cid, "ys")) or dys
                sanitize(k(cid, "y1"), yopts, cur[0] if cur and cur[0] in yopts else (yopts[0] if yopts else None))
                y1 = st.selectbox("Value column", yopts, key=k(cid, "y1"))
                ss[k(cid, "ys")] = [y1] if y1 else []
            else:
                sanitize(k(cid, "ys"), yopts, dys, multi=True)
                st.multiselect("Y axis column(s)", yopts, key=k(cid, "ys"))
            if t == "bubble":
                sanitize(k(cid, "size"), [None] + num_cols, None)
                st.selectbox("Bubble size column", [None] + num_cols, key=k(cid, "size"),
                             format_func=lambda c: "— none —" if c is None else c)
            if t == "histogram":
                ss.setdefault(k(cid, "bins"), 20)
                st.number_input("Number of bins", 2, 200, key=k(cid, "bins"))

            ys = ss.get(k(cid, "ys")) or []
            if t not in CIRCULAR and t != "radar":
                st.text_input("X axis label", key=k(cid, "xlabel"), placeholder=(ys[0] if t == "histogram" else x) or "")
                st.text_input("Y axis label", key=k(cid, "ylabel"), placeholder="Count" if t == "histogram" else (ys[0] if len(ys) == 1 else "Value"))
            if t not in ("histogram", "scatter", "bubble"):
                sanitize(k(cid, "agg"), list(AGGS), "sum" if t in CIRCULAR else "none")
                st.selectbox("Combine rows with the same X", list(AGGS), format_func=AGGS.get, key=k(cid, "agg"))
            c1, c2 = st.columns(2)
            ss.setdefault(k(cid, "legend"), True)
            c1.checkbox("Legend", key=k(cid, "legend"))
            if t not in CIRCULAR and t != "radar":
                c2.checkbox("Log scale (Y)", key=k(cid, "logy"))
                c1.checkbox("Start Y at zero", key=k(cid, "zero"))
            if t != "histogram":
                c2.checkbox("Sort by X", key=k(cid, "sortx"))
            if t in ("line", "area", "step", "radar"):
                ss.setdefault(k(cid, "points"), True)
                c1.checkbox("Show points", key=k(cid, "points"))
            if t not in CIRCULAR and t != "radar":
                st.selectbox("Zoom direction", ["x", "y", "xy"], key=k(cid, "zoom"),
                             format_func={"x": "Horizontal (X) only", "y": "Vertical (Y) only", "xy": "Both directions"}.get)
            # focus range (applies to the chart and the AI explanation)
            xtype_now = col_type(df[x]) if x else None
            if t not in CIRCULAR and t != "histogram" and xtype_now in ("number", "date"):
                if st.checkbox("Focus on an X range", key=k(cid, "focus"),
                               help="Limits both the chart and the AI explanation to this range. "
                                    "Mouse zoom only changes the view."):
                    xs = df[x].dropna()
                    lo, hi = xs.min(), xs.max()
                    if xtype_now == "date":
                        lo, hi = lo.to_pydatetime(), hi.to_pydatetime()
                    else:
                        lo, hi = float(lo), float(hi)
                    if lo < hi:
                        cur = ss.get(k(cid, "frange"))
                        if not (isinstance(cur, (tuple, list)) and len(cur) == 2 and lo <= cur[0] <= hi and lo <= cur[1] <= hi):
                            ss[k(cid, "frange")] = (lo, hi)
                        st.slider("X range", lo, hi, key=k(cid, "frange"))

            # calculated series
            st.divider()
            st.markdown("**Calculated series**")
            if not CALC_OK(t):
                st.caption("Not available for pie, doughnut, polar area or histogram charts.")
            else:
                if num_cols:
                    sanitize(k(cid, "subA"), num_cols, num_cols[0])
                    sanitize(k(cid, "subB"), num_cols, num_cols[1] if len(num_cols) > 1 else num_cols[0])
                    s1, s2 = st.columns(2)
                    s1.selectbox("A", num_cols, key=k(cid, "subA"))
                    s2.selectbox("minus B", num_cols, key=k(cid, "subB"))
                    st.button("Add A − B", key=k(cid, "btn_sub"), on_click=add_subtraction, args=(cid,), width="stretch")
                st.text_input("Custom series name (optional)", key=k(cid, "newname"))
                st.text_area("Custom formula", key=k(cid, "newexpr"), height=70, help=FORMULA_HELP,
                             on_change=lambda key=k(cid, "newerr"): ss.__setitem__(key, ""),
                             placeholder="e.g. [ME fuel (t/day)] / [Speed (kn)]")
                if ss.get(k(cid, "newerr")):
                    st.error(ss[k(cid, "newerr")])
                st.button("Add series", key=k(cid, "btn_calc"), on_click=add_custom, args=(cid,), width="stretch")
                for c in chart["calcs"]:
                    kid = c["id"]
                    with st.container(border=True):
                        nk, ek, rk = k(cid, f"calc{kid}_name"), k(cid, f"calc{kid}_expr"), k(cid, f"calc{kid}_right")
                        ss.setdefault(nk, c["name"])
                        ss.setdefault(ek, c["expr"])
                        ss.setdefault(rk, c.get("right", False))
                        st.text_input("Name", key=nk)
                        st.text_input("Formula", key=ek, help=FORMULA_HELP)
                        old = c["name"]
                        others = set(ys) | {o["name"] for o in chart["calcs"] if o is not c}
                        c["name"] = unique_name(others, ss[nk].strip() or ss[ek]) if ss[nk].strip() != old else old
                        c["expr"], c["right"] = ss[ek], ss[rk] if t not in ("hbar", "radar") else False
                        if c.get("error"):
                            st.error(c["error"])
                        r1, r2 = st.columns([3, 2])
                        if t not in ("hbar", "radar"):
                            r1.checkbox("Right-hand axis", key=rk)
                        r2.button("Remove", key=k(cid, f"calc{kid}_rm"), on_click=remove_calc, args=(cid, kid))

        # ---------------- data + figure ----------------
        cfg = {"type": t, "x": x, "ys": ys, "size": ss.get(k(cid, "size")), "bins": ss.get(k(cid, "bins"), 20),
               "agg": ss.get(k(cid, "agg"), "sum" if t in CIRCULAR else "none") if t not in ("histogram", "scatter", "bubble") else "none",
               "legend": ss.get(k(cid, "legend"), True), "logy": ss.get(k(cid, "logy"), False),
               "zero": ss.get(k(cid, "zero"), False), "sortx": ss.get(k(cid, "sortx"), False),
               "points": ss.get(k(cid, "points"), True), "zoom": ss.get(k(cid, "zoom"), "x"),
               "focus": ss.get(k(cid, "focus"), False), "frange": ss.get(k(cid, "frange"))}

        problem = None
        if not ys and not (CALC_OK(t) and chart["calcs"]):
            problem = "Choose at least one Y column, or add a calculated series, in the settings."
        elif t not in NO_X and not x:
            problem = "Choose a column for the X axis."
        work, series, xtype = (pd.DataFrame(), [], None)
        if not problem:
            work, series, xtype = build_work(cfg, df, chart["calcs"])
            if t != "histogram" and work.empty:
                problem = "No rows have usable values for the selected columns."

        # visibility list (shown in chart and in the saved PNG)
        names = [s["name"] for s in series]
        with settings:
            if series and t not in CIRCULAR and t != "histogram":
                st.divider()
                new = [n for n in names if n not in chart["prev_series"]]
                cur = [n for n in ss.get(k(cid, "shown"), names) if n in names] + new
                ss[k(cid, "shown")] = list(dict.fromkeys(cur))
                st.multiselect("Show in chart and saved PNG", names, key=k(cid, "shown"),
                               help="Series you remove here are hidden and left out of the PNG. "
                                    "Clicking a legend entry hides it on screen only.")
        chart["prev_series"] = names
        cfg["shown"] = ss.get(k(cid, "shown"), names) if t not in CIRCULAR and t != "histogram" else names

        # titles
        if t == "histogram":
            auto_title = f"Distribution of {ys[0]}" if ys else ""
        else:
            lab = [s["name"] for s in series][:2]
            more = len(series) - 2
            auto_title = (" and ".join(lab) + (f" +{more}" if more > 0 else "") + (f" by {x}" if x else "")) if series else ""
        cfg["title_text"] = ss.get(k(cid, "title")) or auto_title
        cfg["xlabel_text"] = ss.get(k(cid, "xlabel")) or ((ys[0] if ys else "") if t == "histogram" else (x or ""))
        cfg["ylabel_text"] = ss.get(k(cid, "ylabel")) or ("Count" if t == "histogram" or cfg["agg"] == "count"
                                                          else (names[0] if len(names) == 1 else "Value"))

        with plot_area:
            if problem:
                st.info(problem)
            else:
                fig = make_figure(cfg, work, series, xtype, df)
                fname = re.sub(r"[^\w\- ]+", "", cfg["title_text"] or "chart").strip().replace(" ", "-")[:80] or "chart"
                st.plotly_chart(fig, key=k(cid, "plot"), theme="streamlit", config={
                    "displaylogo": False, "scrollZoom": False,
                    "toImageButtonOptions": {"format": "png", "filename": fname, "scale": 2},
                })
                st.caption("Drag to zoom, double-click to reset. Use the camera icon in the chart toolbar to save a PNG.")

        # ---------------- AI explanation ----------------
        if AI_ON and not problem:
            sig = json.dumps({**{kk: str(v) for kk, v in cfg.items()},
                              "calcs": [(c["name"], c["expr"], c.get("right")) for c in chart["calcs"]]}, sort_keys=True)
            want = ss.pop(k(cid, "explain_req"), False)
            if want or chart["ai"]:
                st.markdown("##### What this chart shows")
                ctx = chart_context(cfg, work, series, xtype, df, ds_name)
                if want:
                    if not ctx:
                        st.warning("There's nothing to explain yet. Make at least one series visible.")
                    else:
                        chart["ai"] = [{"role": "user", "content": "Explain this chart.\n\n" + ctx}]
                        chart["ai_sig"] = sig
                stale = chart["ai_sig"] is not None and chart["ai_sig"] != sig
                if stale and not want:
                    st.info("The chart has changed since this explanation. Click ✨ Explain to update it.")
                for i, m in enumerate(chart["ai"]):
                    if m["role"] == "assistant":
                        st.markdown(m["content"])
                    elif i > 0:
                        st.markdown(f"**› {m.get('display', m['content'])}**")
                pending = chart["ai"] and chart["ai"][-1]["role"] == "user"
                if pending:
                    try:
                        text = st.write_stream(ai_stream(EXPLAIN_SYSTEM, [{"role": m["role"], "content": m["content"]} for m in chart["ai"]]))
                        chart["ai"].append({"role": "assistant", "content": text if isinstance(text, str) else "".join(text)})
                    except Exception as e:  # noqa: BLE001
                        st.error(ai_error_text(e))
                        if len(chart["ai"]) == 1:
                            chart["ai"] = []
                        else:
                            chart["ai"].pop()
                if chart["ai"]:
                    with st.form(k(cid, "askform"), clear_on_submit=True, border=False):
                        a1, a2 = st.columns([5, 1])
                        q = a1.text_input("Ask a follow-up", placeholder="e.g. What could explain the peak?",
                                          label_visibility="collapsed")
                        sent = a2.form_submit_button("Ask", width="stretch")
                    if sent and q.strip():
                        content = q.strip()
                        if stale and ctx:
                            content = f"The chart has changed. Updated summary:\n\n{ctx}\n\nQuestion: {content}"
                            chart["ai_sig"] = sig
                        chart["ai"].append({"role": "user", "content": content, "display": q.strip()})
                        st.rerun()
                    b1, b2 = st.columns([1, 5])
                    b1.button("Close", key=k(cid, "ai_close"), on_click=close_ai, args=(cid,))
                    b2.caption("Generated by Claude from the data in this chart. Check important conclusions against the source data.")


charts = list(ss.charts)
if per_row == "2 per row":
    for i in range(0, len(charts), 2):
        row = st.columns(2)
        for j, ch in enumerate(charts[i:i + 2]):
            with row[j]:
                render_chart(ch)
else:
    for ch in charts:
        render_chart(ch)
