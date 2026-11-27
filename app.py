"""Demo UI. Run: streamlit run app.py"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from steer import demo_state
from steer.ambiguity import Disambiguator, coarse_senses
from steer.encoder import embed
from steer.engine import apply, fit_probe, prob, unit

ROOT = Path(__file__).parent
st.set_page_config(page_title="Steerable Embedding Space", layout="wide")
core = st.cache_resource(show_spinner="Loading...")(demo_state.core)
paper_space = st.cache_resource(show_spinner="Loading papers...")(demo_state.paper_space)


@st.cache_resource
def semcor_pool():
    """Human-labelled (sentence, word, supersense) items cached by scripts/eval_ambiguity.py; empty if absent."""
    p = ROOT / ".cache" / "semcor_120.json"
    return json.loads(p.read_text()) if p.exists() else []


def neighbors(q, E, texts, k=6):
    sims = E @ q
    idx = np.argsort(-sims)[:k]
    return [(texts[i], float(sims[i])) for i in idx]


stx, contract, texts, labels, E, Y, red, fid = core()
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Steer + contract", "Semantic map", "Ambiguity (bonus #6)",
                                        "Paper scope map (bonus #1)", "Evidence"])

with tab1:
    st.subheader("Steer an embedding along disentangled concept axes")
    q = st.text_input("Query", "The match was postponed because of the storm.")
    method = st.radio("Direction method", ["nullspace", "naive", "lowdin", "random"], horizontal=True,
                      help="nullspace = ours (other concepts' readouts projected out). Others are baselines.")
    cols = st.columns(len(stx.concepts))
    alphas = {c: cols[i].slider(c, -0.6, 0.6, 0.0, 0.05) for i, c in enumerate(stx.concepts)}
    e = embed([q])[0][None]
    s = stx.steer(e, alphas, method)
    left, right = st.columns(2)
    with left:
        st.markdown("**Concept readouts** (verifier probes)")
        fig = go.Figure([go.Bar(name="original", x=stx.concepts, y=[prob(stx.verifier[c], e)[0] for c in stx.concepts]),
                         go.Bar(name="steered", x=stx.concepts, y=[prob(stx.verifier[c], s)[0] for c in stx.concepts])])
        fig.update_layout(barmode="group", height=300, yaxis_range=[0, 1], margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"cos(original, steered) = {float(e[0] @ s[0]):.3f}")
    with right:
        st.markdown("**Nearest neighbours: before → after**")
        for (a, sa), (b, sb) in zip(neighbors(e[0], E, texts), neighbors(s[0], E, texts)):
            st.markdown(f"- ~~{a[:90]}~~  \n  → **{b[:90]}** ({sb:.2f})" if a != b else f"- {a[:90]} ({sa:.2f})")

    st.divider()
    st.subheader("Make your own concept: type examples, get a protected direction")
    st.caption("A few sentences with and without your concept define it. The null-space direction is built the same "
               "way as for the built-in concepts: what the 4 built-in readers look at is projected out.")
    ca, cb = st.columns(2)
    pos_txt = ca.text_area("Sentences WITH the concept (one per line)", height=160, value="\n".join([
        "We regret to inform you that the meeting has been postponed.",
        "Please find the requested documents attached for your review.",
        "The committee will convene to discuss the proposal next week.",
        "Kindly confirm your attendance at your earliest convenience.",
        "We appreciate your patience while we resolve this matter."]))
    neg_txt = cb.text_area("Sentences WITHOUT it (one per line)", height=160, value="\n".join([
        "hey the meeting got pushed, my bad",
        "here's the stuff you wanted lol",
        "we're gonna chat about it next week",
        "u coming or nah?",
        "thanks for hanging in there!!"]))
    pos_l = [s.strip() for s in pos_txt.splitlines() if s.strip()]
    neg_l = [s.strip() for s in neg_txt.splitlines() if s.strip()]
    if min(len(pos_l), len(neg_l)) < 3:
        st.warning("Give at least 3 sentences on each side.")
    else:
        Pc, Nc = embed(pos_l), embed(neg_l)
        raw = Pc.mean(0) - Nc.mean(0)
        Q, _ = np.linalg.qr(np.stack([unit(stx.verifier[c].coef_[0]) for c in stx.concepts]).T)
        dirs_c = {"naive": unit(raw), "nullspace (ours)": unit(raw - Q @ (Q.T @ raw))}
        reader = fit_probe(Pc, Nc)
        ac = st.slider("Strength (α) toward your concept", 0.0, 0.6, 0.3, 0.05, key="custom_a")
        p0 = {c: prob(stx.verifier[c], e)[0] for c in stx.concepts}
        rows = []
        for name, dc in dirs_c.items():
            sc = apply(e, dc, ac)
            rows.append({"method": name,
                         "your concept: P before → after": f"{prob(reader, e)[0]:.2f} → {prob(reader, sc)[0]:.2f}",
                         "leakage (mean |Δ| on 4 built-ins)": round(float(np.mean(
                             [abs(prob(stx.verifier[c], sc)[0] - p0[c]) for c in stx.concepts])), 3),
                         **{f"{c} after": round(float(prob(stx.verifier[c], sc)[0]), 2) for c in stx.concepts},
                         "cos to original": round(float(e[0] @ sc[0]), 3)})
        st.dataframe(rows, hide_index=True)
        st.caption(f"Query: “{q}”. Built-ins before: " + ", ".join(f"{c} {v:.2f}" for c, v in p0.items())
                   + ". Your-concept reader is trained on only your examples, so treat its probability as a rough "
                   "guide; leakage is read by the built-in verifier probes.")

    st.divider()
    st.subheader("Steering contract: ask for an effect, set a budget, get a guarantee or a refusal")
    c1, c2, c3, c4 = st.columns(4)
    cc = c1.selectbox("Concept", stx.concepts)
    tgt = c2.slider("Wanted probability gain", 0.05, 0.9, 0.4, 0.05)
    qb = c3.slider("Max STS-B quality drop", 0.0, 0.1, 0.02, 0.005)
    lb = c4.slider("Max leakage into other concepts", 0.0, 0.2, 0.05, 0.01)
    out = contract.run(cc, e, target=tgt, max_qdrop=qb, max_leak=lb)
    if out["status"] == "refused":
        st.error(f"Refused: {out['reason']}. Loosen the budget or lower the target.")
    else:
        ok = bool(out["verified"][0])
        (st.success if ok else st.warning)(
            f"alpha = {out['alphas'][0]:.3f} (budget allows ≤ {out['alpha_max']:.2f}). Verified gain "
            f"{out['verifier_gain'][0]:+.3f} → {'target met' if ok else 'target NOT met inside budget'}.")
        cur = contract.curves[cc]
        fig = go.Figure([go.Scatter(x=cur[:, 0], y=cur[:, 1], name="gain"),
                         go.Scatter(x=cur[:, 0], y=cur[:, 2], name="leakage"),
                         go.Scatter(x=cur[:, 0], y=cur[:, 3], name="STS-B drop")])
        fig.add_vline(out["alpha_max"], line_dash="dash", annotation_text="budget limit")
        fig.update_layout(height=280, xaxis_title="alpha", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Semantic map with a fidelity certificate")
    m = st.columns(4)
    for i, (k, v) in enumerate(fid.items()):
        m[i].metric(k.replace("_", " "), f"{v:.3f}")
    st.caption("Measured against the true 384-D cosine geometry (k=10). Distances between far-apart clusters "
               "in a UMAP are not meaningful; trust local neighbourhoods.")
    q2 = st.text_input("Point to steer", "Stocks fell sharply after the earnings report.", key="q2")
    c = st.selectbox("Steer toward", stx.concepts, index=1, key="c2")
    e2 = embed([q2])
    traj = np.vstack([stx.steer(e2, {c: a}) for a in np.linspace(0, 0.6, 7)])
    T = red.transform(traj)
    fig = go.Figure()
    for lab in np.unique(labels):
        mk = labels == lab
        fig.add_trace(go.Scattergl(x=Y[mk, 0], y=Y[mk, 1], mode="markers", name=lab, marker=dict(size=4, opacity=.5),
                                   text=[t[:80] for t in np.array(texts)[mk]], hoverinfo="text"))
    fig.add_trace(go.Scatter(x=T[:, 0], y=T[:, 1], mode="lines+markers", name="steering path",
                             marker=dict(size=10, color="black"), line=dict(color="black", width=3)))
    fig.update_layout(height=620, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Is this ambiguous? If so, ask, then steer to the chosen meaning")
    st.info("Prototype. Measured on 173 unseen words: beats 'always ask' and 'always guess' end to end "
            "(0.487 vs 0.445 / 0.274 correct handling), but on length-matched pairs detection is near chance "
            "(AUROC 0.55), so the flag mostly reacts to how much context there is. The person stays in the loop: "
            "they confirm the meaning, and the choice is applied as a steering direction.")
    dis = Disambiguator()
    res_path = ROOT / "results" / "ambiguity.json"
    if res_path.exists():
        dis.threshold = json.loads(res_path.read_text())["threshold_margin"]
    txt = st.text_input("Text", "Can you tell me about the bank?")
    cands = [w.strip(".,?!").lower() for w in txt.split()]
    stop = {"can", "will", "may", "might", "must", "the", "you", "about", "tell", "what", "where", "who", "how",
            "need", "want", "think", "have", "does", "this", "that", "with", "from", "for", "and", "are", "was"}
    cands = [w for w in dict.fromkeys(cands) if len(w) > 2 and w not in stop and len(coarse_senses(w)) >= 2]
    if not cands:
        st.info("No noun with multiple coarse senses found.")
    for w in cands:
        r = dis.analyze(txt, w)
        with st.container(border=True):
            st.markdown(f"### Word: *{w}*")
            st.markdown(f"**1. Detection.** Context evidence {r['margin']:.3f} (ambiguous if ≤ {dis.threshold:.3f}) → "
                        + ("🟠 **ambiguous**: ask the user" if r["ambiguous"]
                           else f"🟢 **clear**, no question asked. Model's sense guess: `{r['best']}`"))
            st.caption(f"Model's sense guess for *{w}* (nearest WordNet sense prototype; this is not the user's choice)")
            st.bar_chart({lab: p for lab, p in zip(r["senses"], r["probs"])}, height=160)
            if not r["ambiguous"]:
                continue
            st.markdown(f"**2. Clarify.** {dis.clarify(r)}")
            pick = st.radio("User's answer", r["senses"], key=f"pick_{w}", horizontal=True)
            a = st.slider("Steering strength (α)", 0.0, 0.6, 0.3, 0.05, key=f"a_{w}")
            qe = embed([txt])[0]
            qs = unit(qe + a * dis.resolve_direction(w, pick))
            labs, _, P = dis.senses(w)

            def soft(v):
                z = np.exp((P @ v - (P @ v).max()) / dis.tau)
                return z / z.sum()
            st.markdown(f"**3. Steering feedback** (toward `{pick}`; by construction, not evidence)")
            fig = go.Figure([go.Bar(name="before", x=labs, y=soft(qe)), go.Bar(name="after steering", x=labs, y=soft(qs))])
            fig.update_layout(barmode="group", height=240, yaxis_range=[0, 1], margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"steer_{w}")
            st.caption(f"e' = normalize(e + α·d), d = prototype(`{pick}`) minus the mean of the other prototypes. "
                       f"cos(original, steered) = {float(qe @ qs):.3f}. This chart is read with the same prototypes "
                       "that build d, so it always moves toward the pick: it shows the steering ran, nothing more.")

            st.markdown("**4. Independent check: re-rank human-labelled SemCor sentences**")
            pool = [(s, lab) for s, ww, lab in semcor_pool() if ww == w]
            if not pool:
                st.info(f"No human-labelled SemCor sentences with *{w}* available (run scripts/eval_ambiguity.py), "
                        "so the pick cannot be checked independently.")
                continue
            S = embed([s for s, _ in pool])
            gold = np.array([lab for _, lab in pool])
            b, s_ = S @ qe, S @ qs
            ob, oa = np.argsort(-b), np.argsort(-s_)
            rank_b = np.empty(len(pool), int)
            rank_b[ob] = np.arange(1, len(pool) + 1)
            k = min(5, len(pool))
            n_match = int((gold == pick).sum())
            dist = ", ".join(f"`{lab}` {c}" for lab, c in Counter(gold).most_common())
            if n_match == 0:
                st.warning(f"None of the {len(pool)} labelled sentences with *{w}* has sense `{pick}` ({dist}). "
                           "This check cannot confirm the pick: steering can only reorder sentences of other senses.")
            c1, c2 = st.columns(2)
            c1.markdown(f"Before steering (top {k} of {len(pool)})")
            c1.dataframe([{"rank": i + 1, "score": round(float(b[j]), 3), "gold sense": gold[j],
                           "sentence": pool[j][0][:120]} for i, j in enumerate(ob[:k])], hide_index=True)
            c2.markdown(f"After steering toward `{pick}` (top {k})")
            c2.dataframe([{"rank": i + 1, "was": int(rank_b[j]), "score": round(float(s_[j]), 3),
                           "gold sense": gold[j], "match": "✓" if gold[j] == pick else "",
                           "sentence": pool[j][0][:120]} for i, j in enumerate(oa[:k])], hide_index=True)
            st.caption(f"Sentences whose human label is `{pick}` in the top {k}: {int((gold[ob[:k]] == pick).sum())} "
                       f"before → {int((gold[oa[:k]] == pick).sum())} after ({n_match} of {len(pool)} in the pool: "
                       f"{dist}). Tiny pools, so this is an illustration, not a measurement.")

with tab4:
    st.subheader("Research scope map: 1,200 papers, 8 areas, start-here picks")
    P, PE, plab, names, start, PY = paper_space()
    fig = go.Figure()
    for j, n in enumerate(names):
        mk = plab == j
        fig.add_trace(go.Scattergl(x=PY[mk, 0], y=PY[mk, 1], mode="markers", name=n, marker=dict(size=5),
                                   text=[P[i]["title"][:90] for i in np.where(mk)[0]], hoverinfo="text"))
    fig.update_layout(height=560, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
    for j, n in enumerate(names):
        st.markdown(f"- **{n}** ({int((plab == j).sum())} papers): start with *{P[start[j]]['title']}*")
    st.markdown("**Steer your reading with your own concept**")
    a, b = st.columns(2)
    pos = a.text_area("More like…", "theoretical analysis, proofs, convergence bounds")
    neg = b.text_area("Less like…", "system implementation, deployment, engineering")
    query = st.text_input("Topic", "graph neural networks", key="pq")
    strength = st.slider("Strength", 0.0, 0.6, 0.3, 0.05)
    d = unit(embed([pos])[0] - embed([neg])[0])
    qv = unit(embed([query])[0] + strength * d)
    for i in np.argsort(-(PE @ qv))[:8]:
        st.markdown(f"- {P[i]['title']} `{P[i]['cat']}`")

with tab5:
    st.subheader("Measured evidence")
    for f in ("steering.md", "robustness.md", "extras.md", "steering_all-MiniLM-L6-v2.md"):
        p = ROOT / "results" / f
        st.markdown(p.read_text() if p.exists() else "Run `python scripts/eval_steering.py` first.")
    for f in ("ambiguity.json", "papers.json"):
        p = ROOT / "results" / f
        if p.exists():
            st.markdown(f"### {f}")
            st.json(json.loads(p.read_text()), expanded=False)
