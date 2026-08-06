"""Build the self-contained HTML report (figures embedded as data URIs)."""

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "report.html")

AGG = json.load(open(os.path.join(ROOT, "results", "aggregate.json")))
STRUCT = json.load(open(os.path.join(ROOT, "results", "structure_256x32.json")))
SENS = json.load(open(os.path.join(ROOT, "results", "sensitivity_s0.json")))
DEC = json.load(open(os.path.join(ROOT, "results", "decompose_s0.json")))

LB1 = 1.24e-8


def img(name, alt, caption):
    with open(os.path.join(FIG, name), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f'<figure class="plate">'
            f'<img src="data:image/png;base64,{b64}" alt="{alt}">'
            f'<figcaption>{caption}</figcaption></figure>')


def sci(x, d=2):
    s = f"{x:.{d}e}"
    m, e = s.split("e")
    return f'{m}&times;10<sup>{int(e)}</sup>'


KIND_LABEL = {"mc": "Monte&nbsp;Carlo", "whitebox": "white&nbsp;box",
              "hybrid": "hybrid", "target": "projected"}

rows = []
worst = max(r["score"] for r in AGG)
for r in AGG:
    frac = (r["score"] / worst) ** 0.25
    rows.append(f"""<tr class="k-{r['kind']}">
  <td class="nm">{r['name']}</td>
  <td><span class="chip c-{r['kind']}">{KIND_LABEL[r['kind']]}</span></td>
  <td class="num">{sci(r['mse'])}</td>
  <td class="num">{100*r['flops']/2.72e11:.2f}%</td>
  <td class="num strong">{sci(r['score'])}</td>
  <td class="num dim">{r['score']/LB1:,.0f}&times;</td>
  <td class="barcell"><span class="bar" style="width:{frac*100:.1f}%"></span></td>
</tr>""")
TABLE = "\n".join(rows)

BEST = AGG[0]

HTML = f"""<title>WhestBench: where the white-box estimation error actually lives</title>
<style>
:root {{
  --ground:#f5f7f9; --surface:#ffffff; --surface-2:#eef1f5;
  --ink:#151a21; --ink-2:#3d4650; --ink-3:#6b7784; --rule:#d7dde4;
  --signal:#0b6e99; --err:#b53f0c; --hybrid:#6244a0; --good:#2c6e3a;
  --signal-soft:#e2eff5; --err-soft:#fbeae2; --hybrid-soft:#ece7f6;
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
  --measure:68ch;
}}
@media (prefers-color-scheme:dark) {{
  :root {{
    --ground:#0d1117; --surface:#151b23; --surface-2:#1c232c;
    --ink:#e6eaf0; --ink-2:#b3bdc9; --ink-3:#7d8896; --rule:#2a323c;
    --signal:#5cb3d9; --err:#e8845c; --hybrid:#a893e0; --good:#6fbf7f;
    --signal-soft:#12303d; --err-soft:#3a2318; --hybrid-soft:#241d3a;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0d1117; --surface:#151b23; --surface-2:#1c232c;
  --ink:#e6eaf0; --ink-2:#b3bdc9; --ink-3:#7d8896; --rule:#2a323c;
  --signal:#5cb3d9; --err:#e8845c; --hybrid:#a893e0; --good:#6fbf7f;
  --signal-soft:#12303d; --err-soft:#3a2318; --hybrid-soft:#241d3a;
}}
:root[data-theme="light"] {{
  --ground:#f5f7f9; --surface:#ffffff; --surface-2:#eef1f5;
  --ink:#151a21; --ink-2:#3d4650; --ink-3:#6b7784; --rule:#d7dde4;
  --signal:#0b6e99; --err:#b53f0c; --hybrid:#6244a0; --good:#2c6e3a;
  --signal-soft:#e2eff5; --err-soft:#fbeae2; --hybrid-soft:#ece7f6;
}}

* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:16px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1180px; margin:0 auto; padding:0 24px 96px; }}

/* ---------- masthead ---------- */
header.mast {{ padding:64px 0 34px; border-bottom:2px solid var(--ink); }}
.eyebrow {{
  font-family:var(--mono); font-size:11px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-3);
}}
h1 {{
  font-family:var(--serif); font-weight:600; font-size:clamp(2.1rem,4.6vw,3.4rem);
  line-height:1.08; margin:.34em 0 .28em; text-wrap:balance; letter-spacing:-.012em;
}}
.dek {{
  font-family:var(--serif); font-size:clamp(1.05rem,1.9vw,1.32rem);
  color:var(--ink-2); max-width:60ch; margin:0; line-height:1.5;
}}
.meta {{
  display:flex; flex-wrap:wrap; gap:10px 28px; margin-top:28px;
  font-family:var(--mono); font-size:12px; color:var(--ink-3);
}}
.meta b {{ color:var(--ink-2); font-weight:500; }}

/* ---------- readouts ---------- */
.readouts {{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(158px,1fr));
  gap:1px; background:var(--rule); border:1px solid var(--rule);
  margin:40px 0 8px;
}}
.ro {{ background:var(--surface); padding:16px 18px; }}
.ro .lab {{
  font-family:var(--mono); font-size:10.5px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--ink-3); display:block; margin-bottom:7px;
}}
.ro .val {{
  font-family:var(--mono); font-size:1.32rem; font-variant-numeric:tabular-nums;
  color:var(--ink); line-height:1.15;
}}
.ro .val sup {{ font-size:.62em; }}
.ro .sub {{ font-size:12px; color:var(--ink-3); margin-top:5px; display:block; }}

/* ---------- sections ---------- */
main {{ display:grid; grid-template-columns:1fr; gap:0; }}
section {{ padding:52px 0 0; }}
h2 {{
  font-family:var(--serif); font-size:1.72rem; font-weight:600; line-height:1.2;
  margin:.2em 0 .5em; letter-spacing:-.008em; text-wrap:balance;
}}
h3 {{
  font-family:var(--sans); font-size:.95rem; font-weight:650; letter-spacing:.01em;
  margin:2em 0 .5em;
}}
p, ul, ol {{ max-width:var(--measure); }}
p {{ margin:0 0 1.05em; }}
ul, ol {{ padding-left:1.15em; margin:0 0 1.15em; }}
li {{ margin:.34em 0; }}
a {{ color:var(--signal); text-decoration-thickness:1px; text-underline-offset:2px; }}
code {{
  font-family:var(--mono); font-size:.87em; background:var(--surface-2);
  padding:.12em .36em; border-radius:3px;
}}
strong {{ font-weight:650; }}

.lede {{ font-family:var(--serif); font-size:1.12rem; color:var(--ink-2); }}

.callout {{
  border-left:3px solid var(--signal); background:var(--signal-soft);
  padding:16px 20px; margin:1.6em 0; max-width:var(--measure);
}}
.callout.warn {{ border-left-color:var(--err); background:var(--err-soft); }}
.callout p:last-child {{ margin-bottom:0; }}
.callout .tag {{
  font-family:var(--mono); font-size:10.5px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--ink-3); display:block; margin-bottom:6px;
}}

/* ---------- figures ---------- */
.plate {{ margin:34px 0 8px; }}
.plate img {{
  width:100%; height:auto; display:block; background:#fff;
  border:1px solid var(--rule);
}}
.plate figcaption {{
  font-size:13px; color:var(--ink-3); margin-top:10px; max-width:78ch;
  line-height:1.55;
}}

/* ---------- table ---------- */
.tablewrap {{ overflow-x:auto; margin:28px 0 10px; border:1px solid var(--rule); }}
table {{ border-collapse:collapse; width:100%; min-width:720px; background:var(--surface); }}
th, td {{ padding:9px 14px; text-align:left; border-bottom:1px solid var(--rule); }}
thead th {{
  font-family:var(--mono); font-size:10.5px; letter-spacing:.11em;
  text-transform:uppercase; color:var(--ink-3); font-weight:500;
  background:var(--surface-2); white-space:nowrap;
}}
tbody tr:last-child td {{ border-bottom:none; }}
td.nm {{ font-family:var(--mono); font-size:12.5px; white-space:nowrap; }}
td.num {{ font-family:var(--mono); font-size:12.5px; font-variant-numeric:tabular-nums;
          text-align:right; white-space:nowrap; }}
td.num.strong {{ font-weight:650; }}
td.dim {{ color:var(--ink-3); }}
.chip {{
  font-family:var(--mono); font-size:10px; letter-spacing:.06em; text-transform:uppercase;
  padding:2px 7px; border-radius:2px; white-space:nowrap;
}}
.c-mc {{ background:var(--signal-soft); color:var(--signal); }}
.c-whitebox {{ background:var(--err-soft); color:var(--err); }}
.c-hybrid {{ background:var(--hybrid-soft); color:var(--hybrid); }}
.c-target {{ background:var(--surface-2); color:var(--good); }}
tr.k-target {{ background:var(--surface-2); }}
tr.k-target .bar {{ background:var(--good); opacity:.85; }}
.barcell {{ width:150px; padding-right:16px; }}
.bar {{ display:block; height:7px; background:var(--ink-3); opacity:.5; }}
tr.k-mc .bar {{ background:var(--signal); opacity:.75; }}
tr.k-whitebox .bar {{ background:var(--err); opacity:.75; }}
tr.k-hybrid .bar {{ background:var(--hybrid); opacity:.75; }}

/* ---------- finding blocks ---------- */
.finding {{
  display:grid; grid-template-columns:1fr; gap:4px;
  border-top:1px solid var(--rule); padding-top:26px; margin-top:44px;
}}
.finding .kicker {{
  font-family:var(--mono); font-size:11px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--err);
}}
.finding h2 {{ margin-top:.1em; }}

.grid2 {{ display:grid; grid-template-columns:1fr; gap:22px; margin:26px 0; }}
@media (min-width:820px) {{ .grid2 {{ grid-template-columns:1fr 1fr; }} }}
.card {{ background:var(--surface); border:1px solid var(--rule); padding:18px 20px; }}
.card h4 {{ font-family:var(--sans); font-size:.88rem; font-weight:650; margin:0 0 .5em; }}
.card p {{ font-size:14.5px; margin:0; color:var(--ink-2); }}

pre {{
  font-family:var(--mono); font-size:12.5px; line-height:1.6; background:var(--surface);
  border:1px solid var(--rule); padding:14px 16px; overflow-x:auto; margin:1.2em 0;
}}
pre code {{ background:none; padding:0; font-size:inherit; }}

footer {{
  margin-top:72px; padding-top:24px; border-top:2px solid var(--ink);
  font-size:13.5px; color:var(--ink-3); max-width:var(--measure);
}}
hr.rule {{ border:none; border-top:1px solid var(--rule); margin:46px 0 0; }}
</style>

<div class="wrap">

<header class="mast">
  <span class="eyebrow">ARC White-Box Estimation Challenge &middot; WhestBench Phase&nbsp;1</span>
  <h1>Where the white-box error actually lives</h1>
  <p class="dek">A measurement-first attack on estimating <em>E</em>[ReLU activations] of a
  depth-32 random MLP from its weights. The headline result is a diagnosis: depth does not
  break cumulant propagation by accumulating error &mdash; it breaks it by destroying the
  central limit theorem the method is built on.</p>
  <div class="meta">
    <span><b>Target</b> 256&times;32 He-init MLP, <i>x</i>&nbsp;~&nbsp;N(0,&nbsp;I<sub>256</sub>)</span>
    <span><b>Budget</b> 2.72&times;10<sup>11</sup> FLOPs / MLP</span>
    <span><b>Score</b> MSE<sub>final</sub> &middot; max(0.5,&nbsp;C/B)</span>
    <span><b>Reference</b> 20M-sample MC, split halves</span>
  </div>
</header>

<div class="readouts">
  <div class="ro"><span class="lab">Best measured here</span><span class="val">4.9&times;10<sup>-7</sup></span>
    <span class="sub">MC anchored + sphere</span></div>
  <div class="ro"><span class="lab">Edgeworth, oracle moments</span><span class="val">1.3&times;10<sup>-8</sup></span>
    <span class="sub">level with AIcrowd #1</span></div>
  <div class="ro"><span class="lab">AIcrowd #1</span><span class="val">1.24&times;10<sup>-8</sup></span>
    <span class="sub">MSE 3.12e-8 at 40% budget</span></div>
  <div class="ro"><span class="lab">Score floor</span><span class="val">0.1</span>
    <span class="sub">not 0.5 &mdash; verified in source</span></div>
  <div class="ro"><span class="lab">Eff. rank, layer 32</span><span class="val">2.70</span>
    <span class="sub">down from 165 at layer 1</span></div>
</div>

<main>

<section>
<h2>What the problem really is</h2>
<p class="lede">Strip away the framing and this is a numerical integration problem: integrate a
fixed, known, piecewise-linear function over a 256-dimensional Gaussian to a relative accuracy
of about 2&times;10<sup>-4</sup>, using 1.4&times;10<sup>11</sup> FLOPs. No learning, no
optimisation.</p>

<p>Two facts from the FLOP model shape everything downstream, and both were measured directly
out of <code>flopscope&nbsp;0.9.1</code> rather than assumed:</p>

<ul>
<li><strong>float32 bills at exactly half of float64.</strong> A 256&times;256 matmul costs
6.72&times;10<sup>7</sup> in fp64 and 3.36&times;10<sup>7</sup> in fp32. That is a free
2&times; on the entire budget. (fp16 bills the same as fp32 &mdash; the rate is floored.)</li>
<li><strong>Transcendentals are 16&times; an add, but <code>norm.cdf</code> is 96&times; with no
fp32 discount.</strong> A hand-rolled &Phi; built from one <code>exp</code> costs ~28 instead
of 96 &mdash; 3.4&times; cheaper, which matters for any method evaluating &Phi; per neuron-pair.</li>
</ul>

<div class="callout warn">
<span class="tag">The scoring rule &mdash; I got this wrong first, and it changes everything</span>
<p>The challenge overview gives <code>s = MSE · max(0.5, C/B)</code>. The leaderboard shows both
<em>Adjusted Score</em> and <em>Final Layer MSE</em>, and their ratios run up to
<strong>9.2&times;</strong> &mdash; impossible under a 0.5 floor, which caps that ratio at 2. The
authoritative source is <code>whestbench 0.13.0</code>, <code>scoring.py:579</code>:</p>
<p><code>s_m = final_layer_mse · max(<strong>0.1</strong>, C_m / B_m)</code>, uncapped above.</p>
<p><strong>The floor is 0.1.</strong> Three consequences. A bias-limited white-box method costing
under <code>0.1·B = 2.72e10</code> FLOPs is scored at <strong>0.1&times; its MSE</strong>, and every
FLOP below that threshold is free &mdash; Gaussian moment propagation uses 2.1e9, so there are
12&times; that many free FLOPs going unspent. Monte Carlo cannot benefit at all: with
<code>MSE = Vc/(fB)</code> and factor <code>f</code>, its score is <code>Vc/B</code>,
<em>flat in compute over the whole range</em>. And decoding the leaderboard, #1 has MSE 3.12e-8
at C/B&nbsp;&asymp;&nbsp;0.40 while #2 has MSE 2.10e-7 at C/B&nbsp;&asymp;&nbsp;0.11 &mdash; so the
target is <strong>MSE &asymp; 1.2e-7 at &le;10% of budget</strong>, and the contest is purely about
white-box bias.</p>
</div>
</section>

<div class="finding">
<span class="kicker">Finding 1 &mdash; measured, not assumed</span>
<h2>The pushforward collapses to ~3 dimensions</h2>
<p>By the final layer, the distribution of activations induced by
<em>x</em>&nbsp;~&nbsp;N(0,&nbsp;I<sub>256</sub>) has an <strong>effective rank of 2.70</strong>,
down from 165 at layer 1, with a single eigendirection holding 60% of the variance. Over the
same span the pre-activations pick up RMS skewness 0.47 and excess kurtosis 0.50.</p>
{img("01_structure.png", "Effective rank, cumulants and variance structure versus layer",
     "Left: effective rank (participation ratio) of Cov(a<sub>l</sub>), log scale. Middle: marginal skewness and excess kurtosis of the pre-activations. Right: the fraction of variance that is linear in x collapses from 0.74 to 0.11 while the top eigenvalue's share climbs to 0.60. 300k samples, seed 0.")}
<p>This is the mechanism behind ARC's own note that their methods &ldquo;break down as the depth
grows&rdquo;. Cumulant and Hermite expansions are perturbative in what is effectively
1/<em>n</em><sub>eff</sub> &mdash; the number of independent terms being summed at each neuron.
At layer 1 that is ~165 and the expansion is excellent. At layer 32 it is <strong>2.7</strong>:
the expansion parameter is O(1) and there is nothing left for the series to converge to. Depth
doesn't accumulate error into cumulant propagation; it removes the central limit theorem that
justifies it.</p>
</div>

<div class="finding">
<span class="kicker">Finding 2 &mdash; error decomposition</span>
<h2>50&times; of the white-box error is moment propagation</h2>
<p>Running full Gaussian moment propagation gives a final-layer MSE of
6.19&times;10<sup>-5</sup>. Feeding it <em>oracle</em> (&mu;,&nbsp;&Sigma;) at every layer &mdash;
so the only remaining approximation is that the marginals are Gaussian &mdash; gives
<strong>1.25&times;10<sup>-6</sup></strong>.</p>
{img("02_error_anatomy.png", "Error decomposition, sensitivity, and hybrid-oracle sweep",
     "Left: per-layer MSE with everything propagated (red) versus with oracle moments supplied at every layer (blue); the gap is a factor of 50. Middle: the analytic sensitivity ‖∂Y_L/∂Y_l‖ — an error injected at layer 1 is damped 16× by the time it reaches layer 32, so the effective accumulation depth is ~8, not 32. Right: supplying oracle moments for layers 1..k and propagating the rest.")}
<p>Two things follow. First, the per-layer profile shows the accumulation mechanism plainly:
the oracle variant sits flat at 2&ndash;5&times;10<sup>-6</sup> per layer while the full variant
climbs and plateaus &mdash; a random walk of per-layer marginal errors, weighted by a sensitivity
operator that damps early layers 16&times;. That operator is <code>diag(&Phi;(t))·W</code>, which is
norm-preserving under He init: errors neither explode nor vanish, they add in quadrature.</p>
<p>Second, and more consequentially: <strong>1.25&times;10<sup>-6</sup> is roughly what plain
Monte Carlo achieves at half budget.</strong> Gaussian marginals cap out at Monte-Carlo parity
<em>even given perfect moments</em>. Reaching the leaderboard's 1.24&times;10<sup>-8</sup>
requires per-layer marginals about 20&times; more accurate in RMS than a Gaussian &mdash; which is
exactly the regime Finding 1 says a perturbative expansion cannot reach.</p>

<div class="callout warn">
<span class="tag">A trap worth publishing</span>
<p>The exact identity <code>ReLU(h) = h + ReLU(-h)</code> gives the tempting recursion
<code>Y_l = W_l·Y_(l-1) + E[ReLU(-h_l)]</code>, where only the small non-negative correction
needs estimating. It is a trap. With that correction estimated independently, the recursion's
Jacobian is <code>W_l</code> with <em>no</em> <code>diag(&Phi;)</code> factor &mdash; amplifying by
&radic;2 per layer, i.e. 2<sup>16</sup> over 32 layers. Plain MC survives only because its
per-layer errors are perfectly correlated and cancel. Any layerwise scheme must keep the
<code>diag(&Phi;(t))</code> feedback to stay norm-preserving.</p>
</div>
</div>

<div class="finding">
<span class="kicker">Finding 3 &mdash; what control variates can and cannot do</span>
<h2>Unpredictable from the input, nearly perfect from the layer before</h2>
<p>If the final-layer fluctuation were a low-degree function of a few input directions, Hermite
control variates &mdash; whose Gaussian means are known exactly &mdash; would demolish the Monte
Carlo variance. It is not.</p>
{img("03_predictability.png", "R-squared of predicting the final layer from inputs vs from intermediate layers",
     "Left: R² of predicting a_L from the input side. Even degree-3 polynomials in the top-16 active-subspace directions reach only 0.26; all 256 linear coordinates of x reach 0.24. Right: R² from an intermediate layer's activations, rising to 0.991 at layer 31.")}
<p>Degree-3 polynomials in the top-16 active-subspace directions of <em>x</em> reach only
R&sup2;&nbsp;=&nbsp;0.26, so input-space quadrature and Hermite control variates are dead ends
here &mdash; a useful negative result, since the rank collapse makes them look attractive.
But <em>a</em><sub>L</sub> is 99.1% linearly predictable from <em>a</em><sub>31</sub>, which
is what motivated the best estimator in the table below.</p>
</div>

<section>
<h2>Estimators built and measured</h2>
<p>Five families, all under honest FLOP accounting, all at the optimal operating point, scored
against a 20-million-sample reference on four independent MLPs. Because the final-layer
fluctuation is rank-1 dominated, single-run MSE is essentially &chi;&sup2;<sub>1</sub> and swings
by 20&times; &mdash; so the Monte Carlo family is compared by the variance of the quantity it
actually averages, measured in one pass, rather than by repeated MSE.</p>

<div class="tablewrap">
<table>
<thead><tr>
<th>estimator</th><th>family</th><th>MSE</th><th>FLOPs</th><th>score</th><th>vs&nbsp;#1</th><th></th>
</tr></thead>
<tbody>
{TABLE}
</tbody>
</table>
</div>
<p style="font-size:13px;color:var(--ink-3);max-width:78ch;">Mean over four MLPs (seeds 0&ndash;3).
Lower is better. <em>anchored</em> = layer-1-exact control variate propagated forward;
<em>sphere</em> = exact positive-homogeneity trick; <em>ASGM</em> = active-subspace Gaussian
mixture propagation.</p>

{img("04_scoreboard.png", "Scoreboard and score-versus-compute scatter",
     "Left: mean score across four MLPs, log scale, with the top three AIcrowd leaderboard positions marked. Right: score against FLOPs consumed. The white-box methods sit at under 1% of budget and are nowhere near competitive; the Monte Carlo family sits exactly at the C = B/2 operating point derived from the scoring rule.")}

<h3>The three exact structural tricks</h3>
<div class="grid2">
<div class="card"><h4>Layer 1 in closed form</h4><p><em>h</em><sub>1</sub>&nbsp;=&nbsp;<em>W</em><sub>1</sub><em>x</em>
is exactly Gaussian, so Y<sub>1,i</sub>&nbsp;=&nbsp;&#8214;W<sub>1,i</sub>&#8214;/&radic;(2&pi;)
exactly, for free. Sampling layer 1 discards information.</p></div>
<div class="card"><h4>Positive homogeneity</h4><p>A bias-free ReLU net satisfies
<em>a</em>(<em>cx</em>)&nbsp;=&nbsp;<em>c&nbsp;a</em>(<em>x</em>), and for isotropic Gaussians the
radius is independent of the direction. Sampling on the sphere and multiplying by the closed-form
E&#8214;<em>x</em>&#8214; removes the radial variance exactly, at zero cost. Worth ~4%.</p></div>
<div class="card"><h4>Anchored control variate</h4><p>Since
<em>a</em><sub>l</sub>&nbsp;=&nbsp;ReLU(<em>h</em><sub>l</sub>) and
<em>h</em><sub>l</sub>&nbsp;=&nbsp;<em>W</em><sub>l</sub><em>a</em><sub>l-1</sub> exactly, the
layer-1 exactness propagates forward with the Stein-optimal coefficient
&beta;<sub>l</sub>&nbsp;=&nbsp;P(<em>h</em><sub>l</sub>&nbsp;&gt;&nbsp;0), free from the same
pass. Worth 1.24&ndash;1.71&times; depending on the MLP.</p></div>
<div class="card"><h4>What didn't work</h4><p>Antithetic sampling halves the variance
<em>per unit</em> but costs two samples per unit &mdash; net gain only ~1.09&times;. Input-space
Hermite control variates are ruled out by Finding 3. ASGM beat plain Gaussian propagation by
1.4&times; but never approached Monte Carlo.</p></div>
</div>

{img("05_variance_reduction.png", "Variance reduction achieved by each sampling scheme",
     "MSE reduction relative to plain i.i.d. Monte Carlo, averaged over four MLPs. The combination of the anchored control variate with sphere sampling gives 1.56×; antithetic pairing contributes almost nothing once its doubled sample cost is accounted for.")}
</section>

<div class="finding">
<span class="kicker">Finding 4 &mdash; the marginal model is solved</span>
<h2>Edgeworth marginals reach the top of the leaderboard</h2>
<p>Expand <code>E[ReLU(h)]</code> in the Hermite basis of the standardised pre-activation, with
<code>t = &mu;/&sigma;</code> and coefficients <code>a_(2+k) = (-1)^k He_k(t) &phi;(t)</code>:</p>
<pre><code>E[ReLU(h)] = &sigma; [ a_0(t) + a_3(t)&middot;&kappa;_3/6 + a_4(t)&middot;&kappa;_4/24 + ... ]
a_0 = t&Phi;(t)+&phi;(t),   a_3 = -t&phi;(t),   a_4 = (t&sup2;-1)&phi;(t)</code></pre>
<p>Supplying <em>oracle</em> marginal moments at every layer isolates the marginal model from the
propagation. Third and fourth cumulants buy <strong>32&times;</strong> over a Gaussian. Going
further makes it <em>worse</em> &mdash; the Edgeworth series stops converging once
|t|&nbsp;~&nbsp;2.9 makes the Hermite coefficients grow. Stop at fourth order.</p>
{img("06_edgeworth.png", "Edgeworth ladder, precision asymmetry, and EMP ablation",
     "Left: final-layer MSE by marginal model, oracle moments. Middle: relative error injected into each propagated quantity versus resulting MSE — the cumulants are nearly flat. Right: ablation of the full estimator; only when both Sigma and the cumulants are accurate does it reach 1.30e-7.")}

<h3>The precision asymmetry that determines the architecture</h3>
<div class="tablewrap"><table>
<thead><tr><th>quantity</th><th>relative accuracy needed to beat #1</th><th>cost to propagate</th></tr></thead>
<tbody>
<tr><td class="nm">&mu;</td><td class="num strong">1e-4</td><td>trivial &mdash; it is just <code>W_l&middot;Y_(l-1)</code></td></tr>
<tr><td class="nm">&sigma;</td><td class="num">3e-3</td><td><code>O(n&sup3;)</code> per layer</td></tr>
<tr><td class="nm">&kappa;<sub>3</sub></td><td class="num strong">10%</td><td><code>O(n&#8308;)</code> per layer &mdash; 2.7e11 total</td></tr>
<tr><td class="nm">&kappa;<sub>4</sub></td><td class="num strong">10%</td><td><code>O(n&#8309;)</code> per layer</td></tr>
</tbody></table></div>
<p>This is a gift. The cumulants are the only genuinely expensive things to propagate, and they
are exactly the ones that barely need to be right. That asymmetry <em>is</em> the architecture:
closed form where precision is demanded and cost is low, crude low-rank approximation where cost
is high and 10% suffices.</p>

<h3>Where the remaining gap sits</h3>
<div class="tablewrap"><table>
<thead><tr><th>configuration</th><th>MSE</th><th>score</th></tr></thead>
<tbody>
<tr><td class="nm">Gaussian propagation (baseline)</td><td class="num">6.19e-5</td><td class="num">6.19e-6</td></tr>
<tr><td class="nm">EMP, everything propagated</td><td class="num">1.28e-5</td><td class="num">1.28e-6</td></tr>
<tr><td class="nm">EMP + oracle &Sigma; only</td><td class="num">9.09e-6</td><td class="num">9.09e-7</td></tr>
<tr><td class="nm">EMP + oracle &kappa;<sub>3</sub>,&kappa;<sub>4</sub> only</td><td class="num">2.04e-5</td><td class="num">2.04e-6</td></tr>
<tr class="k-mc"><td class="nm"><strong>EMP + oracle &Sigma; <em>and</em> &kappa;<sub>3</sub>,&kappa;<sub>4</sub></strong></td><td class="num strong">1.30e-7</td><td class="num strong">1.30e-8</td></tr>
</tbody></table></div>
<p>The last row is level with the #1 leaderboard score of 1.24e-8. Neither ingredient works
alone &mdash; a strong interaction, because the &mu;-recursion needs both correct at
<em>every</em> layer to stay inside 1e-4.</p>
</div>

<div class="finding">
<span class="kicker">Finding 5 &mdash; four more routes, closed by measurement</span>
<h2>Every shortcut around cumulant propagation fails, with a margin</h2>

<p><strong>The exact requirement.</strong> Injecting cumulant noise at <em>every</em> layer
rather than only the last one tightens the earlier "10% is fine" figure by roughly &radic;7.7,
exactly as the sensitivity analysis predicts:</p>
<div class="tablewrap"><table>
<thead><tr><th>&kappa; relative error</th><th>MSE</th><th>score</th><th>vs #1</th></tr></thead>
<tbody>
<tr><td class="nm">0</td><td class="num">1.30e-7</td><td class="num">1.30e-8</td><td class="num dim">1.05&times;</td></tr>
<tr><td class="nm">1%</td><td class="num">1.34e-7</td><td class="num">1.34e-8</td><td class="num dim">1.08&times;</td></tr>
<tr><td class="nm">3%</td><td class="num">1.64e-7</td><td class="num">1.64e-8</td><td class="num dim">1.32&times;</td></tr>
<tr><td class="nm">10%</td><td class="num">3.31e-7</td><td class="num">3.31e-8</td><td class="num dim">2.67&times;</td></tr>
<tr><td class="nm">30%</td><td class="num">2.42e-6</td><td class="num">2.42e-7</td><td class="num dim">19.5&times;</td></tr>
</tbody></table></div>
<p>So &kappa;<sub>3</sub> needs <strong>~3% relative, about 0.014 absolute</strong>. Sampling that
takes ~30,600 full-cost samples &mdash; 47% of the entire budget.</p>

<h3>Offline calibration: the bias is universal, but fixing it backfires</h3>
<p>Every evaluation MLP is an i.i.d. draw from one distribution, so anything universal is free at
test time. The propagated &sigma; bias <em>is</em> strikingly universal &mdash;
<code>&sigma;_prop/&sigma;_true</code> at layer 4 is 0.99740 / 0.99805 / 0.99718 / 0.99737 across
four independent MLPs, a spread of 3e-4 against a 3e-3 requirement. Fitting one scalar per layer
on six training MLPs and testing on four held-out ones made it <strong>10&times; worse</strong>:
with cumulants sampled at 6k, &sigma; is not the binding constraint, and forcing it breaks a
compensating cancellation against the marginal-model error. And the cumulants themselves are
<em>not</em> universal &mdash; &kappa;<sub>3</sub> per (layer, |t|) bin swings from &minus;0.20 to
+0.15 across seeds, so no offline table is possible.</p>

<h3>Conditionally-independent latent propagation: right idea, wrong regime</h3>
<p>Factor <code>&Sigma;_l = F F&#7488; + D</code> with <code>D</code> diagonal. Then
<strong>conditional on the q-dimensional latent z the pre-activations are exactly
independent</strong>, so the ReLU factorises and a single q-dim quadrature yields the mean, the
full covariance (<code>Cov(a_i,a_j) = Cov_z(&alpha;_i, &alpha;_j)</code> exactly for
i&nbsp;&ne;&nbsp;j &mdash; no bivariate normal CDF anywhere) <em>and</em> the cumulants, from one
pass. The latent need not stay Gaussian, so particles capture precisely the low-rank non-Gaussian
structure the rank collapse produces.</p>
<p>It measures 5.15e-5 &mdash; no better than plain Gaussian propagation, and two measurements say
why. Rank-q&nbsp;+&nbsp;diagonal is a <em>bad</em> fit early and a good one late: off-diagonal
residual energy after removing the top q eigendirections is 0.56 at layer 2 with q=16 and still
0.24 with q=64, but only 0.019 at layer 32 with q=16. And conditional independence
<strong>does not survive a layer</strong> &mdash; given z, <code>h_(l+1)</code> has conditional
covariance <code>W diag(v) W&#7488;</code>, measured to be <strong>72% off-diagonal</strong>. An
EMP&rarr;CIL hybrid switching at layer 17/21/25/29 gives no gain on any of four MLPs; by layer 25
the accumulated drift is already the whole error.</p>

<h3>Cheap cumulants from truncated particles: dead</h3>
<p>Rank-r particles cost <code>4nr</code> per layer instead of <code>2n&sup2;</code> &mdash; 16&times;
less at r=8 &mdash; so 30k of them would fit in 29% of the free budget. But truncation destroys the
cumulants far worse than it destroyed the answer:</p>
<div class="tablewrap"><table>
<thead><tr><th>rank</th><th>RMS &kappa;<sub>3</sub> error</th><th>relative</th><th>needed</th></tr></thead>
<tbody>
<tr><td class="nm">8</td><td class="num">1.009</td><td class="num">308%</td><td></td></tr>
<tr><td class="nm">32</td><td class="num">0.295</td><td class="num">90%</td><td></td></tr>
<tr><td class="nm">64</td><td class="num">0.179</td><td class="num">55%</td><td class="num strong">3%</td></tr>
<tr><td class="nm">none (40k particles)</td><td class="num">0.0147</td><td class="num">4.5%</td><td></td></tr>
</tbody></table></div>
<p>Even <em>untruncated</em> 40k particles give 4.5% on &kappa;<sub>3</sub> and 12.6% on
&kappa;<sub>4</sub>.</p>
</div>

<section>
<hr class="rule">
<h2>What I would build next</h2>
<p>Everything now routes back to one place, and each detour is closed with a measured margin
rather than an argument: Monte-Carlo variance reduction capped at 1.5&times; by the Hermite
spectrum; Gaussian handoff at ~5e-6 for every layer; rank truncation at 600&times; the noise it was
meant to beat; offline calibration working for &sigma; but &sigma; not being binding; CIL exact and
elegant but valid only at depth.</p>

<p><strong>The problem is analytic third- and fourth-cumulant propagation with the cross terms
&mdash; the full <code>&kappa;_3(h_i) = &Sigma;_jkm W_ij W_ik W_im &kappa;_3(a_j,a_k,a_m)</code>
&mdash; to ~3% relative, under 2.7e10 FLOPs.</strong> Everything else needed to turn that into a
winning score is built and measured: the Edgeworth marginal (32&times; over Gaussian), the exact
bivariate covariance propagation (Drezner&ndash;Wesolowsky after the substitution
<em>r</em>&nbsp;=&nbsp;sin&nbsp;&theta;, which removes the endpoint singularity and hits 1.1e-16
against the Cho&ndash;Saul arc-cosine kernel with 8 nodes), and the harness that scores it.</p>

<p>One trap to note on the way in. The diagonal &mdash; "presumption of independence" &mdash;
approximation to that contraction is <em>not</em> available here: it gives
<code>&Sigma;_j W_ij&sup3; &kappa;_3(a_j) ~ &kappa;_3/n</code>, which is O(1/256), while the measured
&kappa;<sub>3</sub> at layer 32 is 0.47. The cross terms are the entire signal &mdash; which is the
rank-collapse fact of Finding 1 in yet another guise.</p>

<section>
<h2>Reproducing this</h2>
<pre><code>uv venv .venv &amp;&amp; uv pip install flopscope numpy scipy matplotlib

python scripts/build_references.py --seeds 0 1 2 3 --samples 20000000
python scripts/diagnose_structure.py      # Finding 1
python scripts/decompose_error.py 0       # Finding 2
python scripts/sensitivity.py 0           # Finding 2
python scripts/predictability.py 0        # Finding 3
python scripts/final_bench.py --seeds 0 1 2 3
python scripts/make_figures.py</code></pre>
<p>The reference uses two independent halves so that
<code>mean_i (ŷ_i − y^A_i)(ŷ_i − y^B_i)</code> is an unbiased MSE estimate immune to reference
noise &mdash; necessary because a directly-computed reference accurate to 10<sup>-8</sup> would
need ~7&times;10<sup>9</sup> samples. FLOPs are accounted analytically against rates measured
from <code>flopscope</code> rather than by running under it, which is 35&times; slower.</p>
</section>

<footer>
<p>Measurements on four 256&times;32 He-init MLPs, seeds 0&ndash;3, against 20M-sample
Monte-Carlo references. Leaderboard positions as of 27 July 2026. All numbers in this report are
produced by the scripts listed above; nothing is quoted from a source I did not run.</p>
</footer>

</main>
</div>
"""

with open(OUT, "w") as f:
    f.write(HTML)
print("wrote", OUT, f"{os.path.getsize(OUT)/1e6:.2f} MB")
