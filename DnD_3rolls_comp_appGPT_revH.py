import math
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
import graphviz as gv
import math

# -------------------------------
# Helpers
# -------------------------------

STATES = ["d0", "d4", "d6", "d8", "d12"]
STATE_DIE = {"d0": 0, "d4": 4, "d6": 6, "d8": 8, "d12": 12}
IDX = {s: i for i, s in enumerate(STATES)}

def chance_label(p, Ns=(2,3,4,5)):
    """Return '≈1 in N' or '< 1 in N' for p."""
    v = {N: 1.0/N for N in Ns}
    under = {N: vN for N, vN in v.items() if vN <= p + 1e-12}
    if under:
        N_star = max(under, key=under.get)
        return f"≈1 in {N_star}"
    N_star = min(v, key=lambda N: (v[N]-p) if v[N] > p else 1e9)
    return f"< 1 in {N_star}"

@lru_cache(None)
def dist_die(n):
    """Return pmf for a fair die: values 1..n. If n=0 => pmf at 0 only."""
    if n == 0:
        pmf = np.zeros(1)
        pmf[0] = 1.0
        return pmf, np.arange(1)  # support {0}
    pmf = np.ones(n) / n
    return pmf, np.arange(1, n+1)

def convolve(pmf_a, xs_a, pmf_b, xs_b):
    """Discrete convolution of two pmfs with supports xs_a, xs_b."""
    min_x = xs_a.min() + xs_b.min()
    max_x = xs_a.max() + xs_b.max()
    size = int(max_x - min_x + 1)
    res = np.zeros(size)
    for i, xa in enumerate(xs_a):
        res[(xa + xs_b - min_x).astype(int)] += pmf_a[i] * pmf_b
    xs = np.arange(min_x, max_x+1)
    return res, xs

def pmf_total_for_state(bonus, extra_die):
    """PMF of total = d20 + extra_die + bonus."""
    pmf20, xs20 = dist_die(20)
    pmf_extra, xs_extra = dist_die(extra_die)
    pmf_sum, xs_sum = convolve(pmf20, xs20, pmf_extra, xs_extra)
    # add bonus by shifting support
    xs_sum = xs_sum + bonus
    return pmf_sum, xs_sum

def success_buckets(pmf, xs, DD, T):
    """Return buckets: Succ_T2, Succ_T1, Succ_0, Fail_0, Fail_D1, Fail_D2."""
    # Thresholds
    t_dd = DD
    t_t1 = DD + T
    t_t2 = DD + 2*T
    f0_lo = DD - T                 # (DD-T, DD)
    f1_lo = DD - 2*T               # [DD-2T, DD-T]
    # Note: we align to integer totals: xs are ints.

    Succ_T2 = pmf[xs >= t_t2].sum()
    Succ_T1 = pmf[(xs >= t_t1) & (xs < t_t2)].sum()
    Succ_0  = pmf[(xs >= t_dd)  & (xs < t_t1)].sum()
    # Fail buckets per your new rule: no-stay => all failures drop
    Fail_0  = pmf[(xs > f0_lo) & (xs < t_dd)].sum()     # (DD-T, DD)
    Fail_D1 = pmf[(xs >= f1_lo) & (xs <= f0_lo)].sum()  # [DD-2T, DD-T]
    Fail_D2 = pmf[xs <  f1_lo].sum()                    # (-inf, DD-2T)

    # Normalize tiny numeric drift
    total = Succ_T2 + Succ_T1 + Succ_0 + Fail_0 + Fail_D1 + Fail_D2
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        Succ_T2 /= total; Succ_T1 /= total; Succ_0 /= total
        Fail_0  /= total; Fail_D1 /= total; Fail_D2 /= total

    return Succ_T2, Succ_T1, Succ_0, Fail_0, Fail_D1, Fail_D2

def build_tables(DD, B, T):
    """Compute the 6-row table for all states."""
    rows = []
    for s in STATES:
        pmf, xs = pmf_total_for_state(B, STATE_DIE[s])
        rows.append(success_buckets(pmf, xs, DD, T))
    arr = np.array(rows).T  # shape (6,5)
    df = pd.DataFrame(arr, index=["Succ_T2","Succ_T1","Succ_0","Fail_0","Fail_D1","Fail_D2"], columns=STATES)
    df_round = (100*df).round(1)
    return df, df_round

def matrices_from_table(df, floor_at_d0=True):
    """Build Ms and Mf from the 6-row table under 'no-stay' rule; optional floor at d0."""
    Succ_T2, Succ_T1, Succ_0, Fail_0, Fail_D1, Fail_D2 = [df.loc[row].values for row in df.index]
    p_succ = Succ_T2 + Succ_T1 + Succ_0
    Ms = np.diag(p_succ)
    # Mf: only subdiagonals -1,-2,-3
    Mf = np.zeros((5,5))
    for c in range(5):
        targets = []
        if floor_at_d0:
            r1 = max(c-1, 0); r2 = max(c-2, 0); r3 = max(c-3, 0)
        else:
            r1 = c-1; r2 = c-2; r3 = c-3
        if r1 >= 0: Mf[r1, c] += Fail_0[c]
        if r2 >= 0: Mf[r2, c] += Fail_D1[c]
        if r3 >= 0: Mf[r3, c] += Fail_D2[c]
    return p_succ, Ms, Mf

def ge1_on_two(Mf, start_idx):
    """P(>=1 success on remaining 2 rolls) = 1 - 1^T (Mf^2 e_s)"""
    e = np.zeros((5,1)); e[start_idx,0] = 1
    v = Mf @ (Mf @ e)
    return float(1.0 - v.sum())

def two_of_two(p_succ, start_idx):
    """P(2/2) = p_succ(state)^2 because Ms is diagonal."""
    p = p_succ[start_idx]
    return float(p*p)

def cap_idx(i): return min(max(i,0),4)






# --- Friendly fractions per your rules ---------------------------------------

# Only these denominators for >=10%
_FRACTION_DENOMS = (2, 3, 4, 5, 10)

def _quantize_5pct(p: float) -> float:
    """Nearest 5% (0.05 steps)."""
    return round(p * 20.0) / 20.0

def _best_simple_fraction(pq: float, denoms=_FRACTION_DENOMS):
    """Pick m/n closest to pq with n in allowed denominators (prefer smaller n on ties)."""
    best = None
    for n in denoms:
        m = int(round(pq * n))
        m = max(0, min(n, m))
        approx = m / n
        err = abs(pq - approx)
        key = (err, n)  # prefer smaller n on ties
        if best is None or key < best[0]:
            best = (key, (m, n), approx)
    m, n, approx = best[1][0], best[1][1], best[2]
    # Never show "1 in 1" unless pq ~ 1.00
    if m == n and pq < 0.995:
        m = n - 1
    # Bias 50% window to 1/2
    if abs(pq - 0.50) <= 0.025:
        return (1, 2)
    # Bias 90–95% window to 9/10
    if 0.875 <= pq < 0.975:
        return (9, 10)
    return (m, n)

def friendly_fraction_label(p: float) -> str:
    """
    - If p >= 10%: snap to nearest 5% then express with n in {2,3,4,5,10}.
      Examples: 0.53 -> 0.55 -> '≈ 1 in 2', 0.93 -> 0.95 -> '≈ 9 in 10'.
    - If p < 10%: ceil% then map:
      1→1/100, 2→1/50, 3→1/30, 4–6→1/20, 7–10→1/10.
    """
    p = max(0.0, min(1.0, p))
    if p <= 0.0:
        return "< 1 in 100"
    if p < 0.10:
        pct = math.ceil(p * 100.0)  # 1..9
        if   pct == 1: n = 100
        elif pct == 2: n = 50
        elif pct == 3: n = 30
        elif 4 <= pct <= 6: n = 20
        else: n = 10     # 7..9
        return f"≈ 1 in {n}"
    pq = _quantize_5pct(p)
    m, n = _best_simple_fraction(pq)
    return "≈ 1 in 1" if (m == n) else f"≈ {m} in {n}"

def pct_label(p: float) -> str:
    return f"{p*100:.0f}%"

def leaf_multiline(p: float, contrib: float | None, net_contrib: float | None = None) -> str:
    lines = [pct_label(p), f"({friendly_fraction_label(p)})"]
    if net_contrib is not None:
        lines.append(f"net {net_contrib*100:.1f}%")
    elif contrib is not None:
        lines.append(f"contrib {contrib*100:.1f}%")
    return "\\n".join(lines)








# -------------------------------
# UI
# -------------------------------

st.set_page_config(page_title="D&D Decision Tree Simulator", layout="wide")
st.title("🎲 Decision Tree Simulator — D20 + extra die")

with st.sidebar:
    st.header("Parameters")
    DD = st.number_input("Difficulty (DD)", 1, 60, 15, 1)
    B  = st.number_input("Bonus (B)", -20, 20, 6, 1)
    T  = st.number_input("Threshold T", 0, 20, 5, 1)
    start_state = st.selectbox("Start extra die", STATES, index=2)  # default d6
    start_idx = IDX[start_state]
    need_successes = st.selectbox("Success condition over 3 rolls", ["≥1", "≥2 (default)", "3/3"], index=1)
    floor_at_d0 = st.checkbox("Floor drops at d0 in Mf (recommended)", True)

st.caption(f"States = {STATES} (extra die = 0/4/6/8/12). Rule: no fail-stay, failures drop −1/−2/−3.")

# Compute tables
df, df_pct = build_tables(DD, B, T)
p_succ, Ms, Mf = matrices_from_table(df, floor_at_d0=floor_at_d0)

with st.expander("Per-state probabilities (from exact dice enumeration)"):
    st.dataframe((100*df).round(2))

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("First roll outcomes (from start state)")
    st.write(f"Start = **{start_state}**")
    s = start_idx
    fail1 = df.loc["Fail_0", STATES].values[s]
    fail2 = df.loc["Fail_D1", STATES].values[s]
    fail3 = df.loc["Fail_D2", STATES].values[s]
    succ  = p_succ[s]
    st.markdown(
        f"- Success: **{succ*100:.1f}%** ({chance_label(succ)})\n"
        f"- Fail −1: **{fail1*100:.1f}%**\n"
        f"- Fail −2: **{fail2*100:.1f}%**\n"
        f"- Fail −3: **{fail3*100:.1f}%**"
    )

with col2:
    st.subheader("Quick check (Ms + Mf columns should sum ≈ 1)")
    sums = (Ms + Mf).sum(axis=0)
    st.dataframe(pd.DataFrame({"state":STATES, "colsum Ms+Mf":sums.round(6)}))

st.markdown("---")
st.header("Decision tree — branches & push options")

def branch_table(s_idx, after="success"):
    rows = []
    # number of remaining successes needed after the first roll:
    if need_successes.startswith("≥1"):
        need_after_success = max(0, 1-1)  # not used; we directly handle n=2 cases
        need_after_fail    = 1            # ≥2 overall -> need 2/2 after a fail
        ge1 = lambda idx: 1.0            # if overall need ≥1 (special case)
    elif need_successes.startswith("≥2"):
        # after a success: need ≥1 on 2; after a fail: need 2/2
        need_after_fail = 2
    else:  # "3/3"
        # after a success: need 2/2; after a fail: impossible (0)
        need_after_fail = 3

    labels = []
    if after == "success":
        options = {
            "Keep": s_idx,
            "+1": cap_idx(s_idx+1),
            "+2": cap_idx(s_idx+2),
            "Max (d12)": 4,
        }
        for name, idx2 in options.items():
            if need_successes.startswith("≥2"):
                p_cond = ge1_on_two(Mf, idx2)
            elif need_successes.startswith("≥1"):
                # after a success with overall ≥1, already achieved goal
                p_cond = 1.0
            else: # 3/3
                p_cond = two_of_two(p_succ, idx2)
            rows.append([name, STATES[idx2], p_cond])
    else:  # after fail with severity k
        options = {
            "Keep": s_idx,
            "+1": cap_idx(s_idx+1),
            "+2": cap_idx(s_idx+2),
            "Max (d12)": 4,
        }
        for name, idx2 in options.items():
            if need_successes.startswith("≥2"):
                p_cond = two_of_two(p_succ, idx2)
            elif need_successes.startswith("≥1"):
                # after a fail with overall ≥1: need ≥1 on 2 rolls
                p_cond = ge1_on_two(Mf, idx2)
            else: # 3/3
                p_cond = 0.0  # impossible (already one fail)
            rows.append([name, STATES[idx2], p_cond])
    dfb = pd.DataFrame(rows, columns=["Choice","Start state (2nd roll)","p_cond"])
    dfb["p_cond %"] = (100*dfb["p_cond"]).round(2)
    dfb["label"] = dfb["p_cond"].map(chance_label)
    return dfb

# Branch: success
succ_weight = p_succ[start_idx]
st.subheader("After SUCCESS on 1st roll")
df_succ = branch_table(start_idx, after="success")
st.dataframe(df_succ)

# Branch: fail -1
base_m1 = cap_idx(start_idx-1)
m1_weight = df.loc["Fail_0", STATES].values[start_idx]
st.subheader("After FAIL −1 on 1st roll")
df_m1 = branch_table(base_m1, after="fail")
st.dataframe(df_m1)

# Branch: fail -2
base_m2 = cap_idx(start_idx-2)
m2_weight = df.loc["Fail_D1", STATES].values[start_idx]
st.subheader("After FAIL −2 on 1st roll")
df_m2 = branch_table(base_m2, after="fail")
st.dataframe(df_m2)

# Branch: fail -3 (only if nonzero)
base_m3 = cap_idx(start_idx-3)
m3_weight = df.loc["Fail_D2", STATES].values[start_idx]
if m3_weight > 0:
    st.subheader("After FAIL −3 on 1st roll")
    df_m3 = branch_table(base_m3, after="fail")
    st.dataframe(df_m3)

st.markdown("---")
st.header("Compose a policy & get overall probability")

colA, colB, colC, colD = st.columns(4)
choice_s = colA.selectbox("After SUCCESS", df_succ["Choice"].tolist(), index=0)
choice_m1 = colB.selectbox("After FAIL −1", df_m1["Choice"].tolist(), index=0)
choice_m2 = colC.selectbox("After FAIL −2", df_m2["Choice"].tolist(), index=0)
if m3_weight > 0:
    choice_m3 = colD.selectbox("After FAIL −3", df_m3["Choice"].tolist(), index=0)
else:
    choice_m3 = None

def pick(dfbr, choice):
    row = dfbr[dfbr["Choice"]==choice].iloc[0]
    return float(row["p_cond"])

p_total = succ_weight * pick(df_succ, choice_s) \
        + m1_weight * pick(df_m1, choice_m1) \
        + m2_weight * pick(df_m2, choice_m2)
if m3_weight > 0:
    p_total += m3_weight * pick(df_m3, choice_m3)

st.subheader("Overall success probability (with your policy)")
st.markdown(f"**{100*p_total:.2f}%**  ({chance_label(p_total)})")

with st.expander("Download tables"):
    st.download_button("Per-state probabilities (CSV)", df.to_csv().encode(), "per_state_probs.csv", "text/csv")
    out = {
        "after_success": df_succ,
        "after_fail_minus1": df_m1,
        "after_fail_minus2": df_m2,
    }
    if m3_weight > 0:
        out["after_fail_minus3"] = df_m3
    buf = []
    for name, d in out.items():
        buf.append(f"## {name}\n")
        buf.append(d.to_csv(index=False))
        buf.append("\n")
    st.download_button("Decision branches (CSV blocks)", "".join(buf).encode(), "branches.csv", "text/csv")





# --- Decision tree (browser renderer only, large fonts, tall height) ----------

st.markdown("---")
st.subheader("Decision Tree (first roll + push options)")

compact_view  = st.checkbox("Compact view (hide +2 leaves)", True)
show_contrib  = st.checkbox("Show global contributions", True)

# Make it readable on Streamlit Cloud:
tree_scale    = st.slider("Tree font scale", 0.1, 2.0, 0.2, 0.1)  # bigger default
tree_height   = st.slider("Tree height (px)", 600, 2400, 1400, 100)

# First-roll weights
succ_w  = p_succ[start_idx]
fail1_w = df.loc["Fail_0",  STATES].values[start_idx]
fail2_w = df.loc["Fail_D1", STATES].values[start_idx]
fail3_w = df.loc["Fail_D2", STATES].values[start_idx]

def after_success_cond(s2_idx: int) -> float:
    if need_successes.startswith("≥2"):  return ge1_on_two(Mf, s2_idx)
    if need_successes.startswith("≥1"):  return 1.0
    return two_of_two(p_succ, s2_idx)

def after_fail_cond(s2_idx: int) -> float:
    if need_successes.startswith("≥2"):  return two_of_two(p_succ, s2_idx)
    if need_successes.startswith("≥1"):  return ge1_on_two(Mf, s2_idx)
    return 0.0

success_push = [("Keep", cap_idx(start_idx)),
                ("+1",   cap_idx(start_idx+1)),
                ("+2",   cap_idx(start_idx+2)),
                ("Max",  4)]
base_m1 = cap_idx(start_idx-1)
base_m2 = cap_idx(start_idx-2)
base_m3 = cap_idx(start_idx-3)

def fail_push(base):
    opts = [("Keep", cap_idx(base)), ("+1", cap_idx(base+1)), ("Max", 4)]
    if not compact_view:
        opts.insert(2, ("+2", cap_idx(base+2)))
    return opts

# Colors per group
COLOR = {
    "succ":  {"edge": "#2e7d32", "leaf": "#e8f5e9"},
    "f1":    {"edge": "#ef6c00", "leaf": "#ffe0b2"},
    "f2":    {"edge": "#c62828", "leaf": "#ffcdd2"},
    "f3":    {"edge": "#6a1b9a", "leaf": "#d1c4e9"},
}

# Big fonts (browser renderer ignores DPI, so just go big)
title_fs = str(int(14 * tree_scale * 2.2))
node_fs  = str(int(12 * tree_scale * 2.2))
edge_fs  = str(int(11 * tree_scale * 2.2))

title = f"Decision Tree: Start {STATES[start_idx]}, DD{DD}, B={B:+}, T={T}, Need {need_successes}"
g = gv.Digraph(comment="Decision Tree")

# Vertical and compact: smaller node/row spacing; remove long edge labels
g.attr(rankdir="TB", label=title, labelloc="t", fontsize=title_fs)
g.attr(splines="spline", concentrate="true", nodesep="0.10", ranksep="0.55", ratio="compress")

# Default node/edge styles
g.attr('node', shape='circle', style='filled', fillcolor='#cfe8ff',
       fontname='Helvetica', fontsize=node_fs, penwidth='2', margin="0.08")
g.attr('edge', fontname='Helvetica', fontsize=edge_fs, penwidth='2')

# Root & first split (short labels keep it narrow)
g.node("root", f"Start ({STATES[start_idx]})")
g.node("succ1", "Success_1")
g.edge("root", "succ1", label=pct_label(succ_w),
       color=COLOR["succ"]["edge"], fontcolor=COLOR["succ"]["edge"])

def add_fail_branch(node_id, label, w, tag):
    show = True if (w > 0 or not compact_view) else False
    if show:
        g.node(node_id, label)
        g.edge("root", node_id, label=pct_label(w),
               color=COLOR[tag]["edge"], fontcolor=COLOR[tag]["edge"])
        return True
    return False

have_f1 = add_fail_branch("fail1", "Fail-1", fail1_w, "f1")
have_f2 = add_fail_branch("fail2", "Fail-2", fail2_w, "f2")
have_f3 = add_fail_branch("fail3", "Fail-3", fail3_w, "f3")

def leaf_multiline(p: float, contrib: float | None) -> str:
    # Uses your fraction rules (friendly_fraction_label and pct_label should already be defined)
    lines = [pct_label(p), f"({friendly_fraction_label(p)})"]
    if contrib is not None:
        lines.append(f"contrib {contrib*100:.1f}%")
    return "\\n".join(lines)

def leaf_node(node_id: str, title_text: str, p: float, contrib: float | None, tag: str):
    g.attr('node', shape='box', style='rounded,filled',
           fillcolor=COLOR[tag]["leaf"], margin="0.06")
    g.node(node_id, f"{title_text}\\n{leaf_multiline(p, contrib)}")
    # restore defaults
    g.attr('node', shape='circle', style='filled', fillcolor='#cfe8ff', margin="0.08")

# Leaves under Success_1
succ_opts = success_push if not compact_view else [("Keep",cap_idx(start_idx)), ("+1",cap_idx(start_idx+1)), ("Max",4)]
for name, s2 in succ_opts:
    p_cond = after_success_cond(s2)
    contrib = succ_w * p_cond if show_contrib else None
    leaf = f"s_{name}"
    leaf_node(leaf, f"{name} ({STATES[s2]})", p_cond, contrib, "succ")
    g.edge("succ1", leaf, label="", color=COLOR["succ"]["edge"])

# Leaves under fail branches
def add_fail_leaves(parent_id, base_state, w, tag, prefix):
    for name, s2 in fail_push(base_state):
        p_cond = after_fail_cond(s2)
        contrib = w * p_cond if show_contrib else None
        leaf = f"{prefix}_{name}"
        leaf_node(leaf, f"{name} ({STATES[s2]})", p_cond, contrib, tag)
        g.edge(parent_id, leaf, label="", color=COLOR[tag]["edge"])

if have_f1: add_fail_leaves("fail1", base_m1, fail1_w, "f1", "f1")
if have_f2: add_fail_leaves("fail2", base_m2, fail2_w, "f2", "f2")
if have_f3: add_fail_leaves("fail3", base_m3, fail3_w, "f3", "f3")

# Browser renderer (no system Graphviz needed) + explicit height
st.graphviz_chart(g.source, use_container_width=True)



