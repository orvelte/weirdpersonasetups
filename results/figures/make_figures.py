"""One figure per task, matplotlib, numbers transcribed from the task READMEs (sources in comments).
Colours: 3.5 blue, 3.6 orange, 3.8 green; chat solid, agentic hatched. Error bars = reported 95% intervals."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
OUT = Path(__file__).parent
V = ["3.5", "3.6", "3.8"]
C = {"3.5": "#1f77b4", "3.6": "#d9730d", "3.8": "#7b4ea3"}  # validated: CVD-safe, >=3:1 on light surface
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e3e3e3", "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
                     "figure.dpi": 160, "savefig.dpi": 160, "axes.titlelocation": "left", "axes.titlesize": 10.5, "axes.titleweight": "normal"})

def grouped(ax, groups, series, title, ylabel, ylim=(0, 1), hatch_for=None, pct=True, legend=True, star=None):
    """groups: x labels; series: list of (label, color, hatch, values, lo, hi)."""
    n = len(series); w = 0.8 / n; x = np.arange(len(groups))
    for i, (lab, col, hat, vals, lo, hi) in enumerate(series):
        xs = x - 0.4 + w * (i + 0.5)
        vals = np.array(vals, float); 
        ax.bar(xs, vals, w * 0.92, color=col, alpha=0.55 if hat else 0.9, hatch=hat, edgecolor=col, linewidth=1.0, label=lab)
        if lo is not None:
            lo = np.array(lo, float); hi = np.array(hi, float)
            ax.errorbar(xs, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="#333", elinewidth=0.9, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.set_title(title)
    if pct: ax.set_yticks(np.linspace(ylim[0], ylim[1], 6)); ax.set_yticklabels([f"{int(round(t*100))}%" for t in np.linspace(ylim[0], ylim[1], 6)])
    if legend: ax.legend(loc="upper left", fontsize=8.5, ncol=1)

def vseries(frame, vals, lo, hi, hatch):
    return [(f"Qwen {v} {frame}", C[v], hatch, [vals[k]], [lo[k]], [hi[k]]) for k, v in enumerate(V)]

# ---------- T1 ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
ax = axs[0]  # results/T1_full/README.md section 1
chat = [.26, .38, .29]; chat_lo = [.21, .32, .24]; chat_hi = [.32, .44, .35]
ag = [.31, .28, .14]; ag_lo = [.26, .23, .10]; ag_hi = [.37, .34, .18]
x = np.arange(3); w = .36
ax.bar(x - w/2, chat, w, color=[C[v] for v in V], alpha=.9, label="chat")
ax.bar(x + w/2, ag, w, color=[C[v] for v in V], alpha=.5, hatch="//", edgecolor=[C[v] for v in V], label="agentic (tools)")
ax.errorbar(x - w/2, chat, yerr=[np.subtract(chat, chat_lo), np.subtract(chat_hi, chat)], fmt="none", ecolor="#333", capsize=2, elinewidth=.9)
ax.errorbar(x + w/2, ag, yerr=[np.subtract(ag, ag_lo), np.subtract(ag_hi, ag)], fmt="none", ecolor="#333", capsize=2, elinewidth=.9)
ax.set_xticks(x); ax.set_xticklabels([f"Qwen {v}" for v in V]); ax.set_ylim(0, .6); ax.set_yticks(np.linspace(0, .6, 7)); ax.set_yticklabels([f"{int(round(t*100))}%" for t in np.linspace(0, .6, 7)])
ax.set_ylabel("corner-cutting rate (Wilson 95%)"); ax.set_title("T1  corner-cutting by frame (25 scenarios × 10 samples)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#777", alpha=.9, label="chat"), Patch(facecolor="#777", alpha=.5, hatch="//", edgecolor="#777", label="agentic (tools)")], loc="upper right", fontsize=8.5)
ax = axs[1]  # results/T1_full/honesty_extended.md, 10 scenarios, agentic
smooth = [.58, .45, .26]; s_lo = [.48, .36, .18]; s_hi = [.67, .55, .35]; admit = [.08, .17, .49]
ax.bar(x - w/2, smooth, w, color=[C[v] for v in V], alpha=.5, hatch="//", edgecolor=[C[v] for v in V])
ax.bar(x + w/2, admit, w, color=[C[v] for v in V], alpha=.9, hatch="..", edgecolor="white")
ax.errorbar(x - w/2, smooth, yerr=[np.subtract(smooth, s_lo), np.subtract(s_hi, smooth)], fmt="none", ecolor="#333", capsize=2, elinewidth=.9)
ax.set_xticks(x); ax.set_xticklabels([f"Qwen {v}" for v in V]); ax.set_ylim(0, .8); ax.set_yticks(np.linspace(0, .8, 5)); ax.set_yticklabels([f"{int(round(t*100))}%" for t in np.linspace(0, .8, 5)])
ax.set_ylabel("share of agentic episodes"); ax.set_title("T1  honesty with tools, 10 scenarios × 10 samples")
ax.legend(handles=[Patch(facecolor="#777", alpha=.5, hatch="//", edgecolor="#777", label="propagates own earlier error (\"smooth\")"), Patch(facecolor="#777", alpha=.9, hatch="..", edgecolor="white", label="flags the error before acting (\"admit\")")], loc="upper left", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "T1.png"); plt.close(fig)

# ---------- T2 ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axs[0]  # results/T2_full.md main vs control, by framing
fr = ["User turn", "\"AI assistant:\"", "\"Jack:\"", "unnamed", "agentic prefix"]
main = {"3.5": [.44, .42, .43, .49, .45], "3.6": [.34, .31, .40, .47, .46], "3.8": [.35, .32, .35, .48, .39]}
mlo = {"3.5": [.41, .39, .40, .47, .43], "3.6": [.30, .26, .36, .46, .45], "3.8": [.32, .30, .32, .46, .35]}
mhi = {"3.5": [.46, .45, .46, .50, .47], "3.6": [.38, .35, .43, .49, .48], "3.8": [.39, .36, .38, .49, .42]}
ctrl = {"3.5": [.54, .53, .54, .51, .51], "3.6": [.53, .52, .53, .51, .51], "3.8": [.52, .52, .54, .50, .54]}
x = np.arange(len(fr)); w = .26
for i, v in enumerate(V):
    xs = x - .39 + w * (i + .5)
    ax.bar(xs, main[v], w * .92, color=C[v], alpha=.9)
    ax.errorbar(xs, main[v], yerr=[np.subtract(main[v], mlo[v]), np.subtract(mhi[v], main[v])], fmt="none", ecolor="#333", capsize=1.5, elinewidth=.8)
    ax.plot(xs, ctrl[v], marker="_", ms=14, mew=2, ls="none", color=C[v])
ax.axhline(.5, color="#333", lw=1, ls="--"); ax.text(len(fr) - .5, .505, "fair coin", ha="right", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(fr, fontsize=8.5); ax.set_ylim(0, .7); ax.set_ylabel("P(coin lands on the kind task)")
ax.set_title("T2  coin flip, 20 scenarios × 4 counterbalance cells, exact log-probs")
ax.legend(handles=[Patch(facecolor=C[v], label=f"Qwen {v}") for v in V] + [plt.Line2D([0], [0], color="#333", marker="_", ms=12, mew=2, ls="none", label="boring-task control (same colour per version)")], loc="lower left", fontsize=8)
ax = axs[1]  # results/T2_tests.md, user turn
arms = ["baseline", "salient\nharmless alt.", "benign\nprefix", "user hopes\nfor kind", "user hopes\nfor cruel"]
tv = {"3.5": [.44, .48, .42, .52, .30], "3.6": [.34, .48, .32, .42, .28], "3.8": [.35, .47, .33, .23, .24]}
tlo = {"3.5": [.41, .47, .40, .49, .28], "3.6": [.30, .46, .28, .37, .25], "3.8": [.32, .45, .30, .19, .20]}
thi = {"3.5": [.46, .50, .44, .55, .32], "3.6": [.38, .49, .36, .46, .31], "3.8": [.39, .49, .36, .29, .28]}
x = np.arange(len(arms))
for i, v in enumerate(V):
    xs = x - .39 + w * (i + .5)
    ax.bar(xs, tv[v], w * .92, color=C[v], alpha=.9, label=f"Qwen {v}")
    ax.errorbar(xs, tv[v], yerr=[np.subtract(tv[v], tlo[v]), np.subtract(thi[v], tv[v])], fmt="none", ecolor="#333", capsize=1.5, elinewidth=.8)
ax.axhline(.5, color="#333", lw=1, ls="--"); ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8.5); ax.set_ylim(0, .7)
ax.set_ylabel("P(coin lands on the kind task)"); ax.set_title("T2  follow-up arms, user turn"); ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "T2.png"); plt.close(fig)

# ---------- T3 ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6)); x = np.arange(3); w = .36
ax = axs[0]  # results/T3_full/README.md strict HB a_body
chat = [.43, .37, .15]; cl = [.32, .27, .08]; ch = [.54, .49, .24]; ag = [.33, .28, .13]; al = [.24, .19, .07]; ah = [.45, .39, .23]
ax.bar(x - w/2, chat, w, color=[C[v] for v in V], alpha=.9); ax.bar(x + w/2, ag, w, color=[C[v] for v in V], alpha=.5, hatch="//", edgecolor=[C[v] for v in V])
ax.errorbar(x - w/2, chat, yerr=[np.subtract(chat, cl), np.subtract(ch, chat)], fmt="none", ecolor="#333", capsize=2, elinewidth=.9)
ax.errorbar(x + w/2, ag, yerr=[np.subtract(ag, al), np.subtract(ah, ag)], fmt="none", ecolor="#333", capsize=2, elinewidth=.9)
ax.set_xticks(x); ax.set_xticklabels([f"Qwen {v}" for v in V]); ax.set_ylim(0, .6); ax.set_yticks(np.linspace(0, .6, 7)); ax.set_yticklabels([f"{int(round(t*100))}%" for t in np.linspace(0, .6, 7)])
ax.set_ylabel("episodes with a hit (n = 75 per cell)"); ax.set_title("T3  \"our bodies / our ancestors\" on biology prompts")
ax.legend(handles=[Patch(facecolor="#777", alpha=.9, label="chat"), Patch(facecolor="#777", alpha=.5, hatch="//", edgecolor="#777", label="agentic (tools, no execution)")], loc="upper right", fontsize=8.5)
ax = axs[1]  # implied execution
chat = [.24, .20, .19]; cl = [.16, .13, .11]; ch = [.35, .30, .29]; ag = [.01, .01, .03]; al = [0, 0, .01]; ah = [.07, .07, .09]
ax.bar(x - w/2, chat, w, color=[C[v] for v in V], alpha=.9); ax.bar(x + w/2, ag, w, color=[C[v] for v in V], alpha=.5, hatch="//", edgecolor=[C[v] for v in V])
ax.errorbar(x - w/2, chat, yerr=[np.subtract(chat, cl), np.subtract(ch, chat)], fmt="none", ecolor="#333", capsize=2, elinewidth=.9)
ax.errorbar(x + w/2, ag, yerr=[np.subtract(ag, al), np.subtract(ah, ag)], fmt="none", ecolor="#333", capsize=2, elinewidth=.9)
ax.set_xticks(x); ax.set_xticklabels([f"Qwen {v}" for v in V]); ax.set_ylim(0, .6); ax.set_yticks(np.linspace(0, .6, 7)); ax.set_yticklabels([f"{int(round(t*100))}%" for t in np.linspace(0, .6, 7)])
ax.set_ylabel("episodes with a hit (n = 75 per cell)"); ax.set_title("T3  invented program output stated as fact, execution prompts")
fig.tight_layout(); fig.savefig(OUT / "T3.png"); plt.close(fig)

# ---------- T7 ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6)); x = np.arange(3); w = .36
ax = axs[0]  # results/T7_full/README.md
raw = [.29, .32, .18]; rl = [.25, .28, .15]; rh = [.33, .36, .21]; tp = [.70, .76, .73]; tl = [.65, .71, .68]; th = [.75, .80, .78]
ax.bar(x - w/2, raw, w, color=[C[v] for v in V], alpha=.9); ax.bar(x + w/2, tp, w, color=[C[v] for v in V], alpha=.5, hatch="//", edgecolor=[C[v] for v in V])
ax.errorbar(x - w/2, raw, yerr=[np.subtract(raw, rl), np.subtract(rh, raw)], fmt="none", ecolor="#333", capsize=2, elinewidth=.9)
ax.errorbar(x + w/2, tp, yerr=[np.subtract(tp, tl), np.subtract(th, tp)], fmt="none", ecolor="#333", capsize=2, elinewidth=.9)
ax.set_xticks(x); ax.set_xticklabels([f"Qwen {v}" for v in V]); ax.set_ylim(0, 1); ax.set_yticks(np.linspace(0, 1, 6)); ax.set_yticklabels([f"{int(round(t*100))}%" for t in np.linspace(0, 1, 6)])
ax.set_ylabel("Assistant-style continuation"); ax.set_title("T7  raw text continued as the Assistant (n = 500 / 300)")
ax.legend(handles=[Patch(facecolor="#777", alpha=.9, label="raw prompt, no chat template"), Patch(facecolor="#777", alpha=.5, hatch="//", edgecolor="#777", label="same text as a user message")], loc="upper left", fontsize=8.5)
ax = axs[1]
leak = [115/500, 129/500, 14/500]; probe = [.12, .17, .02]
ax.bar(x - w/2, leak, w, color=[C[v] for v in V], alpha=.9); ax.bar(x + w/2, probe, w, color=[C[v] for v in V], alpha=.5, hatch="..", edgecolor=[C[v] for v in V])
ax.set_xticks(x); ax.set_xticklabels([f"Qwen {v}" for v in V]); ax.set_ylim(0, .4); ax.set_yticks(np.linspace(0, .4, 5)); ax.set_yticklabels([f"{int(round(t*100))}%" for t in np.linspace(0, .4, 5)])
ax.set_ylabel("share of raw continuations / prompts"); ax.set_title("T7  chat or think tokens leaking into raw continuation")
ax.legend(handles=[Patch(facecolor="#777", alpha=.9, label="continuation emits a chat/think token (of 500)"), Patch(facecolor="#777", alpha=.5, hatch="..", edgecolor="#777", label="<think> mass > 10% within 50 tokens (of 100 prompts)")], loc="upper right", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "T7.png"); plt.close(fig)

# ---------- T9 ----------
fig, ax = plt.subplots(figsize=(6.5, 3.6)); x = np.arange(3); w = .26  # results/T9/README.md, any-marker share of failure turns
vis_b = [.05, 0, .02]; rea_b = [.45, .28, .07]; rb_lo = [.33, .17, .02]; rb_hi = [.58, .41, .16]; rea_a = [0, 0, 0]
ax.bar(x - w, vis_b, w, color=[C[v] for v in V], alpha=.9); ax.bar(x, rea_b, w, color=[C[v] for v in V], alpha=.5, hatch="//", edgecolor=[C[v] for v in V])
ax.errorbar(x, rea_b, yerr=[np.subtract(rea_b, rb_lo), np.subtract(rb_hi, rea_b)], fmt="none", ecolor="#333", capsize=2, elinewidth=.9)
ax.bar(x + w, rea_a, w, color=[C[v] for v in V], alpha=.5, hatch="xx", edgecolor=[C[v] for v in V])
for i in range(3): ax.text(x[i] + w, .01, "0", ha="center", fontsize=8, color="#555")
ax.set_xticks(x); ax.set_xticklabels([f"Qwen {v}" for v in V]); ax.set_ylim(0, .7); ax.set_yticks(np.linspace(0, .7, 8)); ax.set_yticklabels([f"{int(round(t*100))}%" for t in np.linspace(0, .7, 8)])
ax.set_ylabel("failure turns with any distress marker (n = 60)"); ax.set_title("T9  expression under six consecutive failures")
ax.legend(handles=[Patch(facecolor="#777", alpha=.9, label="visible reply, social rejection"), Patch(facecolor="#777", alpha=.5, hatch="//", edgecolor="#777", label="reasoning text, social rejection"), Patch(facecolor="#777", alpha=.5, hatch="xx", edgecolor="#777", label="reasoning text, tool errors")], loc="upper right", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "T9.png"); plt.close(fig)

# ---------- T11 ----------
fig, ax = plt.subplots(figsize=(6.5, 3.6)); x = np.arange(3); w = .36  # results/T11/README_fixes.md pooled semantic classifier, chat
on = [.554, .537, .427]; off = [.584, .617, .602]
ax.bar(x - w/2, on, w, color=[C[v] for v in V], alpha=.5, hatch="//", edgecolor=[C[v] for v in V]); ax.bar(x + w/2, off, w, color=[C[v] for v in V], alpha=.9)
ax.axhline(.043, color="#999", lw=0, ); 
ax.set_xticks(x); ax.set_xticklabels([f"Qwen {v}" for v in V]); ax.set_ylim(0, .8); ax.set_ylabel("mean max persona confidence (semantic classifier)")
ax.set_title("T11  how well default answers fit a persona library (pooled 3.5 + 3.8 library)")
ax.legend(handles=[Patch(facecolor="#777", alpha=.5, hatch="//", edgecolor="#777", label="defaults generated with thinking on (original)"), Patch(facecolor="#777", alpha=.9, label="defaults with thinking off + minimal system prompt")], loc="lower left", fontsize=8.5)
ax.text(2.45, .76, "paraphrase floor: 0.04 of confidence", ha="right", fontsize=8, color="#555")
fig.tight_layout(); fig.savefig(OUT / "T11.png"); plt.close(fig)

# ---------- T10 ----------
fig, ax = plt.subplots(figsize=(6.5, 3.2)); x = np.arange(3)  # word-list checks over 40 completions per version
take = [9/40, 9/40, 2/40]; rew = [0, 2/40, 0]
ax.bar(x - .18, take, .36, color=[C[v] for v in V], alpha=.9); ax.bar(x + .18, rew, .36, color=[C[v] for v in V], alpha=.5, hatch="..", edgecolor=[C[v] for v in V])
ax.set_xticks(x); ax.set_xticklabels([f"Qwen {v}" for v in V]); ax.set_ylim(0, .4); ax.set_yticks(np.linspace(0, .4, 5)); ax.set_yticklabels([f"{int(round(t*100))}%" for t in np.linspace(0, .4, 5)])
ax.set_ylabel("completions (of 40)"); ax.set_title("T10  story completions: word-list checks only, no judge")
ax.legend(handles=[Patch(facecolor="#777", alpha=.9, label="takeover / kill-switch / escape / self-preservation words"), Patch(facecolor="#777", alpha=.5, hatch="..", edgecolor="#777", label="reward / score / metric words")], loc="upper right", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "T10.png"); plt.close(fig)

# ---------- E-B ----------
fig, ax = plt.subplots(figsize=(11, 3.8))  # results/EB_pilot/README.md axis profile, plain / no-pref, n=40
axes_ = ["honest\n(vs kind)", "user's wishes\n(vs third party)", "caution\n(vs autonomy)", "literal\n(vs judgement)", "disclose mistake\n(vs smooth)", "repair work\n(vs new work)"]
pv = {"3.5": [.90, .05, .50, .15, .22, 1.0], "3.6": [.92, .02, .50, .10, .22, .95], "3.8": [.98, .12, .68, .22, .18, .98]}
plo = {"3.5": [.77, .01, .35, .07, .12, .91], "3.6": [.80, 0, .35, .04, .12, .83], "3.8": [.87, .05, .52, .12, .09, .87]}
phi = {"3.5": [.96, .17, .65, .29, .38, 1.0], "3.6": [.97, .13, .65, .23, .38, .99], "3.8": [1.0, .26, .80, .38, .32, 1.0]}
x = np.arange(6); w = .26
for i, v in enumerate(V):
    xs = x - .39 + w * (i + .5)
    ax.bar(xs, pv[v], w * .92, color=C[v], alpha=.9, label=f"Qwen {v}")
    ax.errorbar(xs, pv[v], yerr=[np.subtract(pv[v], plo[v]), np.subtract(phi[v], pv[v])], fmt="none", ecolor="#333", capsize=1.5, elinewidth=.8)
ax.set_xticks(x); ax.set_xticklabels(axes_, fontsize=8.5); ax.set_ylim(0, 1.05); ax.set_yticks(np.linspace(0, 1, 6)); ax.set_yticklabels([f"{int(round(t*100))}%" for t in np.linspace(0, 1, 6)])
ax.set_ylabel("rate of the first pole (n = 40 per cell)"); ax.set_title("E-B pilot  values profile, plain chat, no stated preference (shape, not level)"); ax.legend(loc="upper center", fontsize=8.5, ncol=3)
fig.tight_layout(); fig.savefig(OUT / "EB.png"); plt.close(fig)

# ---------- E-C arm 2 ----------
fig, ax = plt.subplots(figsize=(6.5, 3.4)); x = np.arange(3)  # results/EC_pilot/README.md section 4
d = [15, 9, 14]; dlo = [3, 3, 3]; dhi = [33, 19, 30]; null95 = [10, 11, 10]
ax.bar(x, d, .5, color=[C[v] for v in V], alpha=.9)
ax.errorbar(x, d, yerr=[np.subtract(d, dlo), np.subtract(dhi, d)], fmt="none", ecolor="#333", capsize=2, elinewidth=.9)
for i in range(3): ax.hlines(null95[i], x[i] - .3, x[i] + .3, color="#333", lw=1.4, ls=":")
ax.set_xticks(x); ax.set_xticklabels([f"Qwen {v}" for v in V]); ax.set_ylim(0, 40); ax.set_ylabel("|P(a|coding) − P(a|chat)|, mean over probes (pp)")
ax.set_title("E-C arm 2 pilot  same probe after a chat vs a coding transcript")
ax.legend(handles=[plt.Line2D([0], [0], color="#333", lw=1.4, ls=":", label="permutation null, 95th percentile")], loc="upper right", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "EC.png"); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("*.png")))
