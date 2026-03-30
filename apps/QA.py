import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import os
import difflib
from apps.CrosstabsDetail import run as run_crosstabs_detail
from apps.DistributionsDetail import run as run_distributions_detail
from core.correlations_utils import discretize_series_quantiles, fill_missing_for_discrete
from core.crosstab_utils import summarize_crosstab
from core.reset_state import reset_app_state
from apps.DiagramSankey import (
    crosstab_with_std_residuals,
    crosstab_heatmap_png,
    interpret_crosstab_with_llm,
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def val_or_default(val, default):
    """Retourne default si val est None, chaîne vide, ou pandas vide; sinon val."""
    if val is None:
        return default
    if isinstance(val, str):
        return val if val.strip() else default
    if isinstance(val, (pd.DataFrame, pd.Series)):
        return val if not val.empty else default
    # pour listes/dicts, optionnel : if hasattr(val, "__len__") and len(val)==0: return default
    return val

def to_text(x):
    """Sérialise pour le prompt."""
    if isinstance(x, pd.DataFrame):
        return x.to_csv(index=False)
    if isinstance(x, pd.Series):
        return x.to_csv(index=True)
    return str(x)


def _find_cols_in_question(question: str, df: pd.DataFrame) -> list[str]:
    if not isinstance(df, pd.DataFrame):
        return []
    qlow = question.lower()
    cols = []
    for c in df.columns:
        if str(c).lower() in qlow:
            cols.append(str(c))
    if cols:
        return list(dict.fromkeys(cols))
    tokens = [t for t in qlow.replace("?", " ").replace(",", " ").split() if len(t) >= 4]
    candidates = [str(c) for c in df.columns]
    for tok in tokens:
        match = difflib.get_close_matches(tok, candidates, n=1, cutoff=0.8)
        if match:
            cols.append(match[0])
    return list(dict.fromkeys(cols))


def _safe_json_loads(s: str) -> dict | None:
    try:
        return json.loads(s)
    except Exception:
        return None


def _goto_step(step: str):
    """Change d'étape sans toucher aux artefacts."""
    st.session_state["__NAV_SELECTED__"] = str(step)
    try:
        st.query_params["step"] = str(step)
    except Exception:
        st.experimental_set_query_params(step=str(step))
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def _get_crosstab_interpretation(var_a: str, var_b: str) -> str | None:
    items = st.session_state.get("crosstabs_interpretation", []) or []
    for it in items:
        xa = str(it.get("var_x"))
        ya = str(it.get("var_y"))
        if {xa, ya} == {var_a, var_b}:
            return it.get("interpretation")
    return None


def _summarize_distribution(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {"summary": "Série vide.", "insight": "Aucune donnée exploitable.", "type": "empty"}
    if pd.api.types.is_numeric_dtype(s):
        desc = s.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        skew = s.skew()
        kurt = s.kurt()
        zero_pct = (s == 0).mean() * 100
        summary = (
            f"nb={int(desc['count'])}, moy={desc['mean']:.2f}, écart-type={desc['std']:.2f}, "
            f"min={desc['min']:.2f}, p10={desc['10%']:.2f}, médiane={desc['50%']:.2f}, p90={desc['90%']:.2f}, max={desc['max']:.2f}, "
            f"skew={skew:.2f}, kurt={kurt:.2f}, %0={zero_pct:.1f}%"
        )
        insights = []
        if skew > 0.5:
            insights.append("distribution asymétrique à droite (valeurs élevées fréquentes)")
        elif skew < -0.5:
            insights.append("distribution asymétrique à gauche (valeurs faibles fréquentes)")
        else:
            insights.append("distribution globalement symétrique")

        if kurt > 3:
            insights.append("queues lourdes : valeurs extrêmes relativement fréquentes")
        elif kurt < 0:
            insights.append("distribution aplatie : peu de valeurs extrêmes")

        if zero_pct > 10:
            insights.append(f"{zero_pct:.1f}% de valeurs nulles")

        return {"summary": summary, "insight": "; ".join(insights), "type": "numeric"}
    # Discrète
    counts = s.value_counts(normalize=True)
    gini = 1 - (counts ** 2).sum()
    top = counts.head(5).apply(lambda x: f"{x*100:.1f}%").to_dict()
    main_mod = counts.head(1)
    if not main_mod.empty:
        mode_label, mode_pct = main_mod.index[0], main_mod.iloc[0] * 100
        insight = f"modalité principale '{mode_label}' ({mode_pct:.1f}%), dispersion Gini={gini:.2f}"
    else:
        insight = f"dispersion Gini={gini:.2f}"
    summary = f"modalités={len(counts)}, Gini={gini:.2f}, top5={top}"
    return {"summary": summary, "insight": insight, "type": "categorical"}


def _render_compact_answer(question: str, df_ready: pd.DataFrame):
    if not isinstance(df_ready, pd.DataFrame) or df_ready.empty:
        return None
    return None  # remplacé par le flux piloté par LLM


def _ensure_artifacts(question: str, df_ready: pd.DataFrame | None = None):
    ql = question.lower()
    need_crosstab = any(k in ql for k in ("crosstab", "croisé", "relation", "impact", "comparaison"))
    need_dist = any(k in ql for k in ("histogram", "distribution", "répartition", "décile", "quantile", "top décile"))
    if need_crosstab:
        st.session_state["__QA_FORCE_CROSSTABS__"] = True
        st.session_state["run_sankey_crosstabs"] = True
        run_crosstabs_detail()
    if need_dist:
        st.session_state["generate_distribution_figures"] = True
        run_distributions_detail()


def _render_artifacts():
    rendered = False
    ct_items = st.session_state.get("crosstabs_interpretation", []) or []
    for item in ct_items:
        st.markdown(f"**{item.get('var_x')} vs {item.get('var_y')}**")
        st.write(item.get("interpretation", ""))
        img_bytes = item.get("heatmap_png")
        if isinstance(img_bytes, (bytes, bytearray, memoryview)):
            st.image(img_bytes)
        rendered = True
    dist_items = st.session_state.get("figs_variables_distribution_detailed") or st.session_state.get("figs_variables_distribution") or []
    for item in dist_items:
        st.markdown(f"**{item.get('title','Distribution')}**")
        img_bytes = item.get("png")
        if isinstance(img_bytes, (bytes, bytearray, memoryview)):
            st.image(img_bytes)
            rendered = True
    return rendered


def summarize_sankey_pairs(results_store, max_items: int = 20):
    """Resume sankey_pair_results pour payload LLM (compact)."""
    if not isinstance(results_store, dict) or not results_store:
        return []

    rows = []
    for pair_id, r in results_store.items():
        interp = str(r.get("interpretation", "") or "").strip()
        if len(interp) > 500:
            interp = interp[:500] + "..."
        rows.append(
            {
                "pair_id": pair_id,
                "var_x": r.get("var_x"),
                "var_y": r.get("var_y"),
                "v": r.get("v"),
                "p": r.get("p"),
                "chi2": r.get("chi2"),
                "interpretation": interp,
            }
        )

    rows.sort(key=lambda x: float(x.get("v") or 0), reverse=True)
    return rows[:max_items]


def _llm_plan(
    question: str,
    df_ready: pd.DataFrame,
    dataset_context,
    crosstabs_interpretation,
    sankey_pair_results,
    sankey_latents,
    extra_payload: dict | None = None,
):
    if not isinstance(df_ready, pd.DataFrame) or df_ready.empty:
        return None

    preview = df_ready.head(10).to_csv(index=False)
    preview = preview[:20000]

    payload = {
        "columns": [str(c) for c in df_ready.columns],
        "data_sample_preview_as_csv": preview,
        "target_variables": st.session_state.get("target_variables", []),
        "illustrative_variables": st.session_state.get("illustrative_variables", []),
        "crosstabs_interpretation": crosstabs_interpretation,
        "sankey_pair_results_summary": summarize_sankey_pairs(sankey_pair_results, max_items=20),
        "sankey_latents_csv": to_text(sankey_latents) if isinstance(sankey_latents, pd.DataFrame) else "",
        "dataset_context": dataset_context,
    }
    if extra_payload:
        payload.update(extra_payload)

    sys_prompt = """Vous êtes un expert en analyse de données. Répondez en français, clair et concis.

Tâche: produire un plan JSON permettant d'exécuter les calculs adaptés à la question.
Étapes obligatoires :
1) Vérifiez si les artefacts fournis suffisent pour répondre.
2) Repérez dans la question des variables du dataset (limitez-vous aux noms présents dans "columns", tolérance sémantique ok). Choisissez 1 à 10 variables max.
3) Proposez les modules pertinents : crosstab, distribution, profils_y, analyse descriptive. Pour crosstab, donnez les paires var_a/var_b.
4) Si un calcul est pertinent, incluez-le dans le JSON. Sinon, notez pourquoi.
5) Ne créez pas de variables ou scores inexistants.

Format de sortie STRICT JSON:
{"crosstabs":[["var_a","var_b"],...],"distributions":["var_x",...],"notes":"texte bref"}
Pas de texte hors JSON.
"""

    user_prompt = json.dumps(
        {
            "question": question,
            "columns": payload.get("columns"),
            "target_variables": payload.get("target_variables"),
            "illustrative_variables": payload.get("illustrative_variables"),
            "artefacts_available": {
                "crosstabs_interpretation": bool(crosstabs_interpretation),
                "sankey_pairs": bool(sankey_pair_results),
                "sankey_latents": bool(sankey_latents is not None),
                "dataset_context": bool(dataset_context),
            },
        },
        ensure_ascii=False,
    )

    r = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw_answer = r.choices[0].message.content
    plan = _safe_json_loads(raw_answer or "")
    if not plan:
        return {"raw_answer": raw_answer or ""}
    plan["raw_answer"] = raw_answer
    # Sanitization
    plan["crosstabs"] = [p for p in plan.get("crosstabs", []) if isinstance(p, (list, tuple)) and len(p) >= 2]
    plan["distributions"] = [v for v in plan.get("distributions", []) if isinstance(v, str)]
    return plan

def run():
    # États init
    st.session_state.setdefault("profil_dominant", None)
    st.session_state.setdefault("profils", None)
    st.session_state.setdefault("interpretationACM", None)

    # Récupération des données (sans 'or' ambigu)
    dataset_object = st.session_state.get('dataset_object')
    dataset_context = st.session_state.get('dataset_context')
    dataset_recommendations = st.session_state.get('dataset_recommendations')
    key_questions_answer = st.session_state.get('key_questions_answer')
    target_profiles_text = st.session_state.get('target_profiles_text')
    profil_dominant_analysis = st.session_state.get('profil_dominant_analysis')
    dominant_continues = st.session_state.get('dominant_continues')
    dominant_discretes = st.session_state.get('dominant_discretes')
    interpretationACM = st.session_state.get('interpretationACM')
    dendrogram_interpretation = st.session_state.get('dendrogram_interpretation')
    latent_summary_text = st.session_state.get('latent_summary_text')
    fig_dendro = st.session_state.get('dendrogram')
    segmentation_profiles_text = st.session_state.get('segmentation_profiles_text')
    target_profiles_table = st.session_state.get('target_profiles_table')
    segmentation_profiles_table = st.session_state.get('segmentation_profiles_table')
    segmentation_detailed_profiles = st.session_state.get('segmentation_detailed_profiles')
    ctas_rules_text = st.session_state.get('ctas_rules_text')
    process = st.session_state.get('process')
    figs_variables_distribution = st.session_state.get("figs_variables_distribution", [])
    dataset_characteristics  = st.session_state.get("dataset_characteristics")
    variables_raw = st.session_state.get("variables_raw")
    data_preparation_synthesis = st.session_state.get("data_preparation_synthesis")    
    fig_missing_percentages = st.session_state.get("fig_missing_percentages")
    fig_missing_correlation_heatmap = st.session_state.get("fig_missing_correlation_heatmap")
    fig_missing_correlation_dendrogram = st.session_state.get("fig_missing_correlation_dendrogram")
    little_test_result = st.session_state.get("little_test_result")
    df_ready = st.session_state.get("df_ready")
    sankey_interpretation_synthesis = st.session_state.get("sankey_interpretation_synthesis")
    raw_crosstabs = st.session_state.get("crosstabs_interpretation") or []
    crosstabs_interpretation = []
    for item in raw_crosstabs:
        try:
            crosstabs_interpretation.append({
                "var_x": item.get("var_x"),
                "var_y": item.get("var_y"),
                "interpretation": item.get("interpretation", ""),
            })
        except Exception:
            continue
    sankey_pair_results = st.session_state.get("sankey_pair_results", {})
    sankey_latents = st.session_state.get("sankey_latents")
    global_synthesis = st.session_state.get("global_synthesis")
 
    st.subheader("Q&A")

    # Saisie
    question = st.text_input("Posez une question sur un aspect spécifique posé par le jeu de données :")

    if st.button("Envoyer"):
        if not question.strip():
            st.warning("Veuillez poser une question.")
        else:
            try:
                _ensure_artifacts(question, df_ready)
                # Pilotage par LLM : plan JSON -> exécution crosstabs/distributions
                plan = _llm_plan(
                    question,
                    df_ready,
                    dataset_context,
                    crosstabs_interpretation,
                    sankey_pair_results,
                    sankey_latents,
                )
                if plan:
                    # notes éventuelles
                    notes = plan.get("notes") or plan.get("comment") or ""
                    if notes:
                        st.markdown(f"**Notes LLM :** {notes}")

                    # Crosstabs demandés
                    for pair in plan.get("crosstabs", []):
                        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                            continue
                        var_a, var_b = str(pair[0]), str(pair[1])
                        if var_a not in df_ready.columns or var_b not in df_ready.columns:
                            st.warning(f"Crosstab ignoré (variables absentes) : {var_a}, {var_b}")
                            continue
                        interp_session = _get_crosstab_interpretation(var_a, var_b)
                        ct_res = summarize_crosstab(
                            df_ready,
                            var_a,
                            var_b,
                            num_quantiles=st.session_state.get("num_quantiles", 5),
                            mod_freq_min=st.session_state.get("mod_freq_min", 0.9),
                            distinct_threshold_continuous=st.session_state.get("distinct_threshold_continuous", 5),
                            top=5,
                            crosstab_fn=crosstab_with_std_residuals,
                            heatmap_fn=crosstab_heatmap_png,
                            interpretation_fn=interpret_crosstab_with_llm if interp_session is None else None,
                        )
                        if interp_session:
                            ct_res["interpretation"] = interp_session
                        # Affichage
                        interp_txt = ct_res.get("interpretation")
                        if interp_txt:
                            st.write(interp_txt)
                        if ct_res.get("heatmap_png"):
                            st.markdown("**Carte de chaleur du crosstab**")
                            st.image(ct_res["heatmap_png"])
                        caption_lines = [ct_res.get("summary", "")]
                        if not interp_txt:
                            caption_lines.append("Interprétation non disponible pour ce couple de variables.")
                        if ct_res.get("v") is not None:
                            try:
                                caption_lines.append(f"V de Cramer : {float(ct_res.get('v')):.2f}")
                            except Exception:
                                pass
                        elif ct_res.get("assoc"):
                            caption_lines.append(ct_res["assoc"])
                        caption_lines.append(
                            "Lecture : vert = surreprésentation, rouge = sous-représentation par rapport à l’indépendance (résidus standardisés, après discrétisation éventuelle des variables continues)."
                        )
                        st.caption("\n".join([c for c in caption_lines if c]))

                    # Distributions demandées
                    for var in plan.get("distributions", []):
                        if var not in df_ready.columns:
                            st.warning(f"Distribution ignorée (variable absente) : {var}")
                            continue
                        dist_info = _summarize_distribution(df_ready[var])
                        summary_txt = dist_info.get("summary", "")
                        insight_txt = dist_info.get("insight", "")
                        dist_items = (
                            st.session_state.get("figs_variables_distribution_detailed")
                            or st.session_state.get("figs_variables_distribution")
                            or []
                        )
                        shown = False
                        for item in dist_items:
                            title = str(item.get("title", "")).lower()
                            if var.lower() in title:
                                img_bytes = item.get("png")
                                if isinstance(img_bytes, (bytes, bytearray, memoryview)):
                                    st.subheader(f"Distribution de {var}")
                                    st.image(img_bytes)
                                    if insight_txt:
                                        st.caption(f"Interprétation : {insight_txt}")
                                    shown = True
                                    break
                        st.caption(f"{var} — {summary_txt}")
                        if insight_txt and not shown:
                            st.caption(f"Interprétation : {insight_txt}")
                else:
                    with st.spinner("Analyse de la question par LLM en cours..."):
                        plan = _llm_plan(
                            question,
                            df_ready,
                            dataset_context,
                            crosstabs_interpretation,
                            sankey_pair_results,
                            sankey_latents,
                        )
                        if plan and plan.get("raw_answer"):
                            st.text_area("Réponse :", plan.get("raw_answer", ""), height=220)

            except Exception as e:
                st.error(f"Erreur d'appel à l'API OpenAI : {e}")

    st.markdown("#### Actions suivantes")
    back_col, change_col, reset_col = st.columns(3)
    with back_col:
        if st.button("Retour au rapport", use_container_width=True):
            _goto_step("3")
    with change_col:
        if st.button("Changer les objectifs", use_container_width=True):
            st.session_state["__DG_FORCE_RERUN__"] = True
            st.session_state["pipeline_ready_to_run"] = False
            st.session_state["pipeline_executed"] = False
            st.session_state["pipeline_status"] = None
            st.session_state["pipeline_halt"] = None
            st.session_state["final_report_ready"] = False
            st.session_state["final_export_zip_bytes"] = None
            st.session_state["etape2_terminee"] = False
            st.session_state["etape40_terminee"] = False
            st.session_state["etape41_terminee"] = False
            _goto_step("2")
    with reset_col:
        if st.button("Réinitialiser", use_container_width=True):
            reset_app_state()

    st.session_state["etape41_terminee"] = True
