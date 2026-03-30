import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import prince
from openai import OpenAI
from typing import List  # pour l'annotation


MODE_KEY = "__NAV_MODE__"

# Client OpenAI (clé via variable d'env)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ===============================
# FONCTIONS UTILITAIRES
# ===============================
def reset_after_upload():
    """Réinitialise toutes les étapes dépendantes quand un nouveau fichier est chargé."""
    ss = st.session_state
    ss.processed_df = None
    ss.processed_df_ready = False

    ss.mca_full = None
    ss.explained_ratio = None
    ss.cumulative_explained = None
    ss.n_axes_needed = None
    ss.var_x1 = None
    ss.acm_computed = False

    ss.params_validated = False
    ss.llm_done = False
    ss.llm_usage = None
    ss.filtered_coords = pd.DataFrame()
    ss.coords_full = None
    ss.cos2_full = None
    ss.interpretationACM = ""
    ss["etape22_terminee"] = False

def params_changed():
    """Détecte si les sliders ont changé depuis la dernière validation."""
    keys = ["n_axes_display", "cos2_threshold", "coord_threshold", "topk_examples"]
    return any(st.session_state.params.get(k) != st.session_state.last_params_snapshot.get(k) for k in keys)

# --- util : reset strict de l'étape 4 uniquement ---
def reset_llm_step():
    ss = st.session_state
    ss.llm_done = False
    ss.interpretationACM = ""
    ss.llm_usage = None

def recompute_filtered_coords():
    """Recalcule coords/cos² + DataFrame filtré par cos² (aperçu UI)."""
    ss = st.session_state
    dfp = ss.processed_df
    mca = ss.mca_full
    n_axes_display = ss.params["n_axes_display"]
    cos2_threshold = ss.params["cos2_threshold"]

    coords_full_all = mca.column_coordinates(dfp)
    cos2_full_all = (coords_full_all**2).div((coords_full_all**2).sum(axis=1), axis=0)

    coords = coords_full_all.iloc[:, :n_axes_display]
    cos2   = cos2_full_all.iloc[:, :n_axes_display]

    mask = (cos2 >= cos2_threshold).any(axis=1)
    ss.filtered_coords = coords[mask]

    ss.coords_full = coords
    ss.cos2_full   = cos2

# ---------- GROUPING UTILS ----------
def _split_label(s: str):
    if "_" in s:
        var, mod = s.split("_", 1)
        return var.strip(), mod.strip()
    return "", s.strip()

def build_axis_groups_by_both(coords: pd.DataFrame, cos2: pd.DataFrame,
                                cos2_threshold: float, coord_threshold: float) -> dict:
    """
    Regroupe par axe les modalités qui vérifient simultanément :
    - cos²(axe) >= cos2_threshold
    - |coordonnée(axe)| >= coord_threshold
    Sépare en 'positive' et 'negative' selon le signe de la coordonnée.
    """
    groups = {}
    for j, col in enumerate(coords.columns):  # j=0 => Axe 1
        x = coords.iloc[:, j]
        c2 = cos2.iloc[:, j]

        keep = (c2 >= cos2_threshold) & (x.abs() >= coord_threshold)
        pos_idx = x.index[keep & (x > 0)]
        neg_idx = x.index[keep & (x < 0)]

        # Tri pour lisibilité (intensité décroissante)
        pos_sorted = x.loc[pos_idx].sort_values(ascending=False)
        neg_sorted = x.loc[neg_idx].abs().sort_values(ascending=False)

        groups[j+1] = {
            "positive": [_split_label(s) for s in pos_sorted.index],
            "negative": [_split_label(s) for s in neg_sorted.index],
        }
    return groups

def groups_to_text(groups: dict) -> str:
    """Transforme le dict en texte lisible, sans underscores, 'Variable: Modalité'."""
    lines = []
    for ax in sorted(groups.keys()):
        lines.append(f"Groupe {ax}:")
        pos = groups[ax]["positive"]
        neg = groups[ax]["negative"]

        def fmt(mods):
            if not mods:
                return "(aucune modalité retenue)"
            return "; ".join([f"{v}: {m}" if v else f"{m}" for v, m in mods])

        lines.append(f"  - Sous-groupe A (positif): {fmt(pos)}")
        lines.append(f"  - Sous-groupe B (négatif): {fmt(neg)}")
    return "\n".join(lines)

def truncate_groups_topk_by_abscoord(groups: dict, coords: pd.DataFrame, k: int = 3) -> dict:
    """Garde seulement les k modalités les plus 'intenses' (|coord|) par sous-groupe pour le payload LLM."""
    trimmed = {}
    for ax, sides in groups.items():
        col = coords.columns[ax-1]
        trimmed[ax] = {"positive": [], "negative": []}
        for side in ("positive", "negative"):
            items = sides[side]
            scored = []
            for (var, mod) in items:
                label = f"{var}_{mod}" if var else mod
                if label in coords.index:
                    scored.append((abs(coords.loc[label, col]), (var, mod)))
            scored.sort(key=lambda t: t[0], reverse=True)
            trimmed[ax][side] = [pair for _, pair in scored[:k]]
    return trimmed

def groups_to_text_for_llm(groups: dict) -> str:
    """Version compacte pour le LLM : 0–k exemples max par sous-groupe."""
    lines = []
    for ax in sorted(groups.keys()):
        lines.append(f"Groupe {ax}:")
        pos = groups[ax]["positive"]
        neg = groups[ax]["negative"]

        def fmt(mods):
            if not mods:
                return "(exemples: aucun)"
            return "; ".join([f"{v}: {m}" if v else f"{m}" for v, m in mods])

        lines.append(f"  - Sous-groupe A (positif) — exemples: {fmt(pos)}")
        lines.append(f"  - Sous-groupe B (négatif) â€” exemples: {fmt(neg)}")
    return "\n".join(lines)

def validate_groups_both(groups: dict, coords: pd.DataFrame, cos2: pd.DataFrame,
                            cos2_threshold: float, coord_threshold: float) -> List[str]:
    """Vérifie cos² >= seuil ET |coord| >= seuil pour tous les items listés, sur l'axe concerné."""
    errors: List[str] = []
    for ax in sorted(groups.keys()):
        col = coords.columns[ax-1]
        for side in ("positive", "negative"):
            for var, mod in groups[ax][side]:
                label = f"{var}_{mod}" if var else mod
                if label not in coords.index:
                    errors.append(f"[Groupe {ax} - {side}] Indice introuvable: {label}")
                    continue
                c2_val = float(cos2.loc[label, col])
                coord_val = float(coords.loc[label, col])
                if c2_val < cos2_threshold:
                    errors.append(f"[Groupe {ax} - {side}] {label} cos²={c2_val:.3f} < seuil {cos2_threshold}")
                if abs(coord_val) < coord_threshold:
                    errors.append(f"[Groupe {ax} - {side}] {label} |coord|={abs(coord_val):.3f} < seuil {coord_threshold}")
    return errors

# App streamlit

def run():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
   
    # INIT DES ÉTATS PERSISTANTS

    def init_state():
        defaults = {
            "uploaded_filename": None,
            "df": None,
            "processed_df": None,
            "num_quantiles": 3,

            # Flags de workflow
            "processed_df_ready": False,   # Étape 1 validée
            "acm_computed": False,         # Étape 2 validée
            "params_validated": False,     # Étape 3 validée
            "llm_done": False,             # Étape 4 réalisée

            # Résultats ACM
            "mca_full": None,
            "explained_ratio": None,
            "cumulative_explained": None,
            "n_axes_needed": None,
            "var_x1": None,

            # Paramètres d'interprétation
            "params": {"n_axes_display": 4, "cos2_threshold": 0.3},
            "last_params_snapshot": {"n_axes_display": 6, "cos2_threshold": 0.3},

            # Coordonnées/filtre & interprétation
            "filtered_coords": pd.DataFrame(),
            "coords_full": None,
            "cos2_full": None,
            "interpretationACM": "",
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    init_state()

    # Garantit la présence de 'coord_threshold' et 'topk_examples' dans params + snapshot
    st.session_state.params.setdefault("coord_threshold", 0.50)
    st.session_state.params.setdefault("topk_examples", 3)
    st.session_state.last_params_snapshot.setdefault("coord_threshold", 0.50)
    st.session_state.last_params_snapshot.setdefault("topk_examples", 10)
    # --------------------------------------------------------------------------------------

    # ===============================
    # ÉTAPE 1 · UPLOAD
    # ===============================
    st.header("Interprétation de l'analyse factorielle (ACM)")

    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"], key="uploader")
    # (Optionnel) utiliser reset_after_upload() si tu gères l'upload effectif ici

    # Vérifier quels datasets sont disponibles
    datasets_disponibles = {}
    for key, label in [
        ("df_active",  "Variables actives, ordinales encodées"),
        ("df_ready",   "Toutes les variables"),
        ("df_encoded", "Toutes les variables, ordinales encodées"),
    ]:
        val = st.session_state.get(key)
        if isinstance(val, pd.DataFrame) and not val.empty:
            datasets_disponibles[label] = val

    if not datasets_disponibles:
        st.warning("Aucun dataset *valide* trouvé. Veuillez d'abord passer par l'application précédente.")
        st.stop()

    choix = st.selectbox("Choisissez un dataset à utiliser :", list(datasets_disponibles.keys()), index=0)
    df = datasets_disponibles[choix].copy()

    st.success(f"{choix} chargé depuis l'application précédente.")
    st.write("Aperçu du dataset :")
    st.dataframe(df.head())

    # Discrétisation
    from utils import discretize_continuous_variables
    distinct_threshold_continuous= st.session_state.get("distinct_threshold_continuous", 5)
    mod_freq_min = st.session_state.get("mod_freq_min", 0.90)
    num_quantiles = st.session_state.get("num_quantiles", 5)

    df_proc, info = discretize_continuous_variables(
        df,
        num_quantiles=num_quantiles,
        mod_freq_min=mod_freq_min,
        distinct_threshold_continuous=distinct_threshold_continuous,
        context_name="apps/xxx.py étape 3"
    )

    if info["errors"]:
        st.error("\n".join(info["errors"]))
        st.stop()

    if info["warnings"]:
        st.warning("\n".join(info["warnings"]))

    
    st.session_state.processed_df = df_proc
    st.session_state.processed_df_ready = True
    

    st.info("Les variables continues ont été discrétisées. Aperçu :")
    st.write(df_proc.head())

    if not st.session_state.processed_df_ready:
        st.info("Chargez un CSV pour continuer.")
        st.stop()

    # ===============================
    # ÉTAPE 2 · CALCUL ACM
    # ===============================
    st.subheader("Réalisation de lâ€™ACM")
    
    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Lancer le calcul de l'ACM", key="btn_calc_acm") and not st.session_state.acm_computed:
            proceed = True

    if proceed:
        dfp = st.session_state.processed_df
        max_axes = max(1, min(dfp.shape) - 1)
        mca_full = prince.MCA(n_components=max_axes, n_iter=3, random_state=42).fit(dfp)

        explained_var = mca_full.eigenvalues_
        explained_ratio = explained_var / explained_var.sum()
        cumulative_explained = np.cumsum(explained_ratio)
        n_axes_needed = int(np.argmax(cumulative_explained >= 0.60) + 1) if len(cumulative_explained) else 0
        var_x1 = float(explained_ratio[0] * 100) if len(explained_ratio) else 0.0

        ss = st.session_state
        ss.mca_full = mca_full
        ss.explained_ratio = explained_ratio
        ss.cumulative_explained = cumulative_explained
        ss.n_axes_needed = n_axes_needed
        ss.var_x1 = var_x1
        ss.acm_computed = True

        ss.params_validated = False
        ss.llm_done = False
        ss.filtered_coords = pd.DataFrame()
        ss.interpretationACM = ""

    if not st.session_state.acm_computed or st.session_state.mca_full is None:
        st.info("Cliquez sur « Lancer le calcul de l'ACM » pour continuer.")
        st.stop()

    # Affichage résultats ACM
    st.write(f"Nombre d'axes nécessaires pour atteindre 60% de variance : {st.session_state.n_axes_needed}")
    st.write(f"Variance expliquée par le 1er axe : {st.session_state.var_x1:.2f}%")

    fig, ax = plt.subplots()
    x = np.arange(1, len(st.session_state.cumulative_explained) + 1)
    ax.plot(x, st.session_state.cumulative_explained * 100, marker='o')
    ax.axhline(60, linestyle='--', label='Seuil 60%')
    ax.set_title("Variance cumulée expliquée par les axes")
    ax.set_xlabel("Axes")
    ax.set_ylabel("Variance cumulée (%)")
    ax.legend()
    st.pyplot(fig)

    # ===============================
    # ÉTAPE 3 · PARAMÃˆTRES D'INTERPRÉTATION
    # ===============================
    st.subheader("Sélection des paramètres pour l'interprétation")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.params["n_axes_display"] = st.slider(
            "Nombre d'axes à interpréter",
            min_value=1, max_value=10,
            value=st.session_state.params["n_axes_display"],
            step=1, key="axes_disp",
        )
    with col2:
        st.session_state.params["cos2_threshold"] = st.select_slider(
            "Seuil minimal de cos²",
            options=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            value=st.session_state.params["cos2_threshold"], key="cos2_thr",
        )

    colA, colB = st.columns(2)
    with colA:
        st.session_state.params["coord_threshold"] = st.slider(
            "Seuil minimal |coordonnée|",
            min_value=0.10, max_value=1.50, value=st.session_state.params["coord_threshold"],
            step=0.05, key="coord_thr",
        )
    with colB:
        st.session_state.params["topk_examples"] = st.slider(
            "Nombre de modalités par sous-groupe",
            min_value=0, max_value=14, value=st.session_state.params["topk_examples"],
            step=1, key="topk_examples",
        )

    # Recalcul du filtrage en live (et mémos coords/cos2)
    recompute_filtered_coords()
    st.markdown("##### Coordonnées factorielles des modalités (cos² â‰¥ seuil sur au moins un axe)")
    if not st.session_state.filtered_coords.empty:
        st.dataframe(st.session_state.filtered_coords.style.format("{:.3f}"))
    else:
        st.session_state["etape22_terminee"] = True

    # Invalidation de l'étape 4 si modifs après validation
    if st.session_state.params_validated and params_changed():
        st.session_state.params_validated = False
        st.session_state.llm_done = False
        st.session_state.interpretationACM = ""

    # Validation
    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        st.button("Valider les paramètres", key="btn_params")
        proceed = True
    if proceed:    
        st.session_state.last_params_snapshot = st.session_state.params.copy()
        st.session_state.params_validated = True
        st.success("Paramètres dâ€™interprétation validés.")

    if not st.session_state.params_validated:
        st.info("Ajustez puis validez les paramètres pour continuer.")
        st.stop()

    # --------- Construction des groupes (cos² ET |coord|) + aperçu + contrôle ---------
    coords    = st.session_state.coords_full
    cos2      = st.session_state.cos2_full
    cos2_thr  = float(st.session_state.params["cos2_threshold"])
    coord_thr = float(st.session_state.params["coord_threshold"])
    topk      = int(st.session_state.params["topk_examples"])

    axis_groups = build_axis_groups_by_both(coords, cos2, cos2_thr, coord_thr)

    payload_full_preview = groups_to_text(axis_groups)
    st.markdown("### Aperçu des groupes/sous-groupes envoyés au LLM")
    st.code(payload_full_preview, language="text")

    errors = validate_groups_both(axis_groups, coords, cos2, cos2_thr, coord_thr)
    if errors:
        st.error("Certaines modalités ne respectent pas les seuils (cos² ET |coord|) :")
        for e in errors[:80]:
            st.write("• ", e)
    else:
        st.success("âœ“ Toutes les modalités listées respectent cos² ET |coord| pour leur axe.")

    # Payload compact pour le LLM
    axis_groups_topk = truncate_groups_topk_by_abscoord(axis_groups, coords, k=topk)
    payload_text = groups_to_text_for_llm(axis_groups_topk)

    st.session_state["llm_payload_text"] = payload_text
    st.session_state["llm_axis_groups"]  = axis_groups_topk

    # ===============================
    # ÉTAPE 4 · INTERPRÉTATION LLM
    # ===============================
    st.subheader("Interprétation de lâ€™ACM")

    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Lancer l'interprétation de l'ACM par LLM", key="btn_llm"):
            proceed = True

    if proceed:
        with st.spinner("Interprétation de l'ACM par LLM en cours..."):
            model = "gpt-4o-mini"
            temperature = 0
            max_rows_preview = 20

            try:
                payload = st.session_state.get("llm_payload_text", "")
                if not payload.strip():
                    st.error("Aucun regroupement à envoyer. Validez lâ€™étape 3 dâ€™abord.")
                    st.stop()

                system_msg = {
                    "role": "system",
                    "content": f'''Vous êtes un expert en analyse de données, en marketing s'il s'agit de questionnaires. Réponds  en français, clair et concis.
                        Une analyse factorielle (ACM) a été réalisée, on te fournit les modalités des variables significative sur chaque axe factoriel.
                        Il y a plusieurs Groupes (correspondant à chaque axe factoriel), chacun avec deux Sous-groupes opposés (A positif / B négatif, ayant des coordonnées opposées sur les axes factoriels).
                        et pour chaque sous-groupe une liste variable/modalité').
                        TA MISSION :
                        - Fournir une SYNTHÃˆSE (5-10 phrases) qui explique la relation entre tous les modalités des différents axes, et la relation logique qu'elles peuvent avoir.                   
                        - Pour le faire, attribue un NOM SPÉCIFIQUE et OPÉRATIONNEL à chaque Groupe (évite les intitulés génériques).
                        - Et pour chaque Sous-groupe, attribue un SOUS-NOM court et explicite, qui exprime la nature de l'opposition entre les 2 sous-groupes.
                        - utilise tous les groupes qui sont proposés, ne pas en oublier.
                        - N'utilise que des phrases, pas de récapitulation des groupes. Formule les relations en termes stratégique et naratif.
                        - N'emploie pas de jargon statistique (pas d'ACP, d'axes, cos², coordonnées, etc.)
                    '''
                }
                user_msg = {
                    "role": "user",
                    "content": (
                        "Voici les regroupements (ne renvoie pas ce texte tel quel, propose des noms et une synthèse) :\n\n"
                        f"{payload}"
                    )
                }

                response = client.chat.completions.create(
                    model=model,
                    messages=[system_msg, user_msg],
                    temperature=temperature,
                    max_tokens=2000,
                )

                st.session_state.interpretationACM = response.choices[0].message.content
                st.session_state.llm_done = True

            except Exception as e:
                st.error(f"Une erreur est survenue lors de lâ€™appel à lâ€™API : {e}")

    if st.session_state.llm_done and st.session_state.interpretationACM:
        st.text_area("Interprétation de l'ACM", value=st.session_state.interpretationACM, height=420)
    else:
        st.info("Cliquez sur « Lancer l'interprétation de l'ACM par LLM ».")

    st.write("âœ… Vous pouvez lancer la prochaine étape dans le menu à gauche: Segmentation.")
    st.session_state["etape22_terminee"] = True




