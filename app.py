"""Demo UI. Run: streamlit run app.py"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from steer import demo_state
from steer.ambiguity import Disambiguator, coarse_senses
from steer.encoder import embed
from steer.engine import prob, unit

ROOT = Path(__file__).parent
st.set_page_config(page_title="Steerable Embedding Space", layout="wide")
core = st.cache_resource(show_spinner="Building demo state (first run only; run scripts/build_demo.py to "
                                      "prebuild)...")(demo_state.core)
paper_space = st.cache_resource(show_spinner="Loading 1,200 arXiv papers (first run only)...")(demo_state.paper_space)


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
        st.markdown(f"**{w}**: context evidence {r['margin']:.3f} (ambiguous if ≤ {dis.threshold:.3f}) → "
                    + ("🟠 **ambiguous**" if r["ambiguous"] else f"🟢 clear: `{r['best']}`"))
        st.bar_chart({lab: p for lab, p in zip(r["senses"], r["probs"])}, height=180)
        if r["ambiguous"]:
            st.write(dis.clarify(r))
            pick = st.radio("Answer", r["senses"], key=f"pick_{w}", horizontal=True)
            st.caption(f"Resolved: steering the query toward `{pick}` (direction from sense prototypes).")

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
    for f in ("steering.md", "extras.md", "steering_all-MiniLM-L6-v2.md"):
        p = ROOT / "results" / f
        st.markdown(p.read_text() if p.exists() else "Run `python scripts/eval_steering.py` first.")
    for f in ("ambiguity.json", "papers.json"):
        p = ROOT / "results" / f
        if p.exists():
            st.markdown(f"### {f}")
            st.json(json.loads(p.read_text()), expanded=False)
