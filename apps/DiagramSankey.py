import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import io
import json
import base64
import plotly.io as pio


MODE_KEY = "__NAV_MODE__"

# Client OpenAI global (accessible dans toutes les fonctions)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ==========================
# 1. Fonctions statistiques
# ==========================

def cramers_v(x, y):
    """
    Calcule le V de Cramér à partir de deux séries catégorielles.
    """
    # On retire les NA pour éviter des lignes/colonnes vides
    data = pd.crosstab(x, y)
    if data.empty or data.shape[0] < 2 or data.shape[1] < 2:
        return np.nan, np.nan, np.nan  # chi2, p, v

    chi2, p, dof, expected = chi2_contingency(data)
    n = data.values.sum()
    phi2 = chi2 / n
    r, k = data.shape
    phi2_corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1)) if n > 1 else 0
    r_corr = r - ((r - 1)**2) / (n - 1) if n > 1 else r
    k_corr = k - ((k - 1)**2) / (n - 1) if n > 1 else k
    if r_corr <= 1 or k_corr <= 1:
        v = 0
    else:
        v = np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))
    return chi2, p, v

def compute_associations(df, variables, alpha=0.05, v_min=0.1,
                         profiles=None, outcomes=None):
    """
    Calcule chi2, p, V de Cramér pour tous les couples de variables dans 'variables'.
    Retourne un DataFrame des liens significatifs.
    """
    if profiles is None:
        profiles = []
    if outcomes is None:
        outcomes = []

    results = []
    n_vars = len(variables)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            x_name = variables[i]
            y_name = variables[j]

            # on saute les couples internes profils/profils et outcomes/outcomes
            if x_name in profiles and y_name in profiles:
                continue
            if x_name in outcomes and y_name in outcomes:
                continue

            x = df[x_name]
            y = df[y_name]
            chi2, p, v = cramers_v(x, y)
            if np.isnan(v):
                continue
            if p < alpha and v >= v_min:
                results.append({
                    "var_x": x_name,
                    "var_y": y_name,
                    "chi2": chi2,
                    "p": p,
                    "v": v
                })

    if not results:
        return pd.DataFrame(columns=["var_x", "var_y", "chi2", "p", "v"])
    return pd.DataFrame(results)


# fonction de filtre des couples de variables internes aux latents 
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def refine_links_with_latents_via_llm(
    df_cat: pd.DataFrame,
    links_df: pd.DataFrame,
    var_labels: dict[str, str] | None,
    dendrogram_text: str,
    acm_text: str,
    profiles: list[str],
    outcomes: list[str],
    candidates: list[str],
    client,
):
    """
    Appelle le LLM pour :
    - regrouper les variables en latents,
    - identifier les variables englobantes par latent,
    - produire un texte de synthèse des latents.

    NE FILTRE PAS links_df. Le filtrage se fait ensuite en Python.
    Retourne :
      - links_df (inchangé)
      - latent_info : dict JSON complet renvoyé par le LLM
    """

    # 1) Données structurées à passer au LLM

    var_labels = var_labels or {col: col for col in df_cat.columns}

    variables = [
        {"id": var_id, "label": label}
        for var_id, label in var_labels.items()
    ]

    groups = {
        "profiles": profiles,
        "outcomes": outcomes,
        "candidates": candidates,
    }

    links_df = links_df.copy()
    if "pair_id" not in links_df.columns:
        links_df = links_df.reset_index(drop=True)
        links_df["pair_id"] = links_df.index.astype(int)

    links_records = links_df[["pair_id", "var_x", "var_y", "chi2", "p", "v"]].to_dict(orient="records")

    llm_input = {
        "variables": variables,
        "groups": groups,
        "links": links_records,
        "dendrogram_text": dendrogram_text,
        "acm_text": acm_text,
    }


    # 2) Instructions : LLM ne fait que les latents + englobantes
    latents = f'''Réponds en français, clair et concis.
Tu es un data analyst expert (statistique + sémantique).

On te donne :
- une liste de variables (questions) avec leurs libellés,
- des groupes : profils, outcomes, candidates (médiateurs potentiels),
- une liste de couples significatifs (links) avec Khi², p-value, V de Cramér,
- un texte d'interprétation du dendrogramme des corrélations,
- un texte d'interprétation de l'ACM.

Ton objectif est UNIQUEMENT sémantique et structurel :
1) Regrouper les variables en latents (dimensions sous-jacentes).
2) Pour chaque latent, identifier les variables englobantes (si elles existent): une variable englobante résume ou couvre plusieurs autres variables du même latent.
3) Produire un texte d'explication des latents (latent_summary_text). Expliqe pour chaque variable englobante sa relation avec son latent.

Tous les calculs, filtrages et choix de couples seront faits ensuite côté Python.
'''
    instructions =('''
Tu dois renvoyer un JSON avec les champs suivants :
{
  "latent_summary_text": "Texte en plusieurs paragraphes, en français, expliquant les latents identifiés, leurs contenus, et les grandes relations entre eux.",

  "latents": [
    {
      "latent_id": "L1",
      "name": "Nom court du latent",
      "description": "Brève description sémantique du latent",
      "variables": ["VAR_ID_1", "VAR_ID_2", "..."],
      "englobing_variables": ["VAR_ID_1", "..."]  // liste éventuellement vide
    }
  ],

  "variable_latent_mapping": [
    {
      "variable_id": "VAR_ID_1",
      "latent_id": "L1",
      "is_englobing": true
    },
    {
      "variable_id": "VAR_ID_2",
      "latent_id": "L1",
      "is_englobing": false
    }
  ]
}

Contraintes :
- Chaque variable doit apparaître dans 'variable_latent_mapping' avec un et un seul latent_id.
- is_englobing = true si la variable est englobante dans son latent, false sinon.
- 'variables' dans chaque latent doit être cohérent avec 'variable_latent_mapping'.
- Réponds STRICTEMENT en JSON, sans texte autour, sans commentaires.
''')

    # 3) Appel LLM avec format JSON forcé   
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": latents.strip()
            },
            {
                "role": "user",
                "content": instructions.strip()
            },
            {
                "role": "user",
                "content": json.dumps(llm_input, ensure_ascii=False)
            }
        ],
        temperature=0,
    )

    content = response.choices[0].message.content
    st.session_state["latent_raw_response"] = content

    try:
        latent_info = json.loads(content)
    except json.JSONDecodeError:
        st.warning(
            "Échec du parsing JSON de la réponse LLM pour les latents. "
            "Aucun regroupement par latent ne sera appliqué."
        )
        with st.expander("Réponse brute du LLM (latents)"):
            st.code(content, language="json")
        st.session_state["sankey_latent_info"] = None
        st.session_state["latent_summary_text"] = ""
        return links_df, None

    # Sauvegarde pour le rapport / interface
    st.session_state["sankey_latent_info"] = latent_info
    st.session_state["latent_summary_text"] = latent_info.get("latent_summary_text", "")

    return links_df, latent_info


def build_latents_dataframe(
    latent_info: dict,
    primary_vars_per_latent,
    englobing_vars,
) -> pd.DataFrame:
    """
    Construit un DataFrame avec pour chaque variable :
    - latent_id
    - latent_name
    - variable
    - primary_variable (nom de la variable si primaire, sinon "")
    - englobing_variable (nom de la variable si englobante, sinon "")
    """
    rows = []

    if not latent_info:
        return pd.DataFrame(
            columns=[
                "latent_id",
                "latent_name",
                "variable",
                "primary_variable",
                "englobing_variable",
            ]
        )

    latents = latent_info.get("latents", []) or []

    for lat in latents:
        latent_id = lat.get("latent_id")
        latent_name = lat.get("name")
        variables = lat.get("variables", []) or []

        # Ensemble des variables "primary" pour ce latent
        primary_set = primary_vars_per_latent.get(latent_id, set())

        for var in variables:
            is_primary = var in primary_set
            is_englobing = var in englobing_vars

            rows.append({
                "latent_id": latent_id,
                "latent_name": latent_name,
                "variable": var,
                "primary_variable": var if is_primary else "",
                "englobing_variable": var if is_englobing else "",
            })

    return pd.DataFrame(rows)


# sélection des variables parmi les latents
def build_var_to_latent_and_englobing(latent_info: dict):
    """
    Retourne :
      - var_to_latent: dict[var_id] -> latent_id
      - englobing_vars: set[var_id]
    En priorité à partir de 'variable_latent_mapping';
    sinon, fallback sur latents[].variables / latents[].englobing_variables.
    """
    var_to_latent = {}
    englobing_vars = set()

    if not latent_info:
        return var_to_latent, englobing_vars

    mapping = latent_info.get("variable_latent_mapping")

    if mapping:
        # On a le tableau explicite
        for item in mapping:
            vid = item.get("variable_id")
            lid = item.get("latent_id")
            is_eng = item.get("is_englobing", False)
            if vid and lid:
                var_to_latent[vid] = lid
                if is_eng:
                    englobing_vars.add(vid)
    else:
        # Fallback : on utilise latents[]
        for lat in latent_info.get("latents", []):
            latent_id = lat.get("latent_id")
            vars_in_latent = lat.get("variables", []) or []
            eng = lat.get("englobing_variables", []) or []
            for v in vars_in_latent:
                if latent_id:
                    var_to_latent[v] = latent_id
            for v in eng:
                englobing_vars.add(v)

    return var_to_latent, englobing_vars


from typing import Dict, Set, Tuple  # si tu es en Python < 3.9

def compute_primary_variables_per_latent(
    links_df: pd.DataFrame,
    latent_info: dict
) -> Dict[str, Set[str]]:
    """
    Calcule pour chaque latent un ensemble de variables principales.

    Règles :
    1) Toutes les variables englobantes du latent sont incluses.
    2) Pour chaque lien entre deux latents (L1, L2), on retient pour L1 la variable
       de L1 qui a le plus fort V dans les liens avec L2, et symétriquement pour L2.
    3) Ainsi, si un latent est relié à 2 autres latents, on peut retenir 2 variables
       différentes + éventuellement une variable englobante distincte (jusqu'à 3).
    """

    primary_vars: Dict[str, Set[str]] = {}

    # Cas dégénérés
    if not latent_info or links_df.empty:
        return primary_vars

    # Mapping variable -> latent et ensemble des variables englobantes
    var_to_latent, englobing_vars = build_var_to_latent_and_englobing(latent_info)

    # Initialisation : s'assurer que chaque latent a un set (même vide)
    for lat in latent_info.get("latents", []):
        latent_id = lat.get("latent_id")
        if latent_id is not None:
            primary_vars.setdefault(latent_id, set())

    # Ajout des variables englobantes
    for v in englobing_vars:
        lid = var_to_latent.get(v)
        if lid is not None:
            primary_vars.setdefault(lid, set()).add(v)

    # Pour chaque lien inter-latent, trouver la variable la plus forte côté latent_src
    best_for_pair: Dict[Tuple[str, str], Tuple[float, str]] = {}

    for _, row in links_df.iterrows():
        vx = row["var_x"]
        vy = row["var_y"]
        v = row["v"]

        lx = var_to_latent.get(vx)
        ly = var_to_latent.get(vy)
        if lx is None or ly is None:
            continue
        if lx == ly:
            # lien interne au même latent -> ignore dans ce calcul
            continue

        # côté lx
        key_x = (lx, ly)
        cur_x = best_for_pair.get(key_x)
        if (cur_x is None) or (v > cur_x[0]):
            best_for_pair[key_x] = (v, vx)

        # côté ly
        key_y = (ly, lx)
        cur_y = best_for_pair.get(key_y)
        if (cur_y is None) or (v > cur_y[0]):
            best_for_pair[key_y] = (v, vy)

    # Ajout des variables "meilleures" à primary_vars
    for (latent_src, _latent_dst), (_v, var_in_src) in best_for_pair.items():
        primary_vars.setdefault(latent_src, set()).add(var_in_src)

    return primary_vars


# ==========================
# 2. Construction des niveaux
# ==========================

def assign_mediators_to_levels(links_df, profiles, outcomes, candidates, v_min=0.1):
    """
    Assigne les variables candidates comme médiateurs niveau 1 ou 2
    en fonction de leur force de lien (V) avec profils / outcomes.
    """
    med1 = []
    med2 = []

    for m in candidates:
        # liens de m avec profils
        mask_prof = ((links_df["var_x"].isin(profiles) & (links_df["var_y"] == m)) |
                     (links_df["var_y"].isin(profiles) & (links_df["var_x"] == m)))
        prof_links = links_df[mask_prof]
        strength_to_profiles = prof_links["v"].max() if not prof_links.empty else 0

        # liens de m avec outcomes
        mask_out = ((links_df["var_x"].isin(outcomes) & (links_df["var_y"] == m)) |
                    (links_df["var_y"].isin(outcomes) & (links_df["var_x"] == m)))
        out_links = links_df[mask_out]
        strength_to_outcomes = out_links["v"].max() if not out_links.empty else 0

        if strength_to_profiles < v_min and strength_to_outcomes < v_min:
            # médiateur peu connecté aux extrêmes : on peut l'ignorer
            continue

        if strength_to_profiles >= v_min and strength_to_outcomes < v_min:
            med1.append(m)
        elif strength_to_profiles < v_min and strength_to_outcomes >= v_min:
            med2.append(m)
        else:
            # les deux sont forts : on décide selon la plus forte
            if strength_to_profiles >= strength_to_outcomes:
                med1.append(m)
            else:
                med2.append(m)

    # On retire les doublons par sécurité
    med1 = list(dict.fromkeys(med1))
    med2 = list(dict.fromkeys(med2))

    return med1, med2

# ==========================
# 3. Construction du Sankey
# ==========================

def build_sankey_from_links(links_df, levels):
    """
    Construit un Sankey Plotly multi-niveaux à partir d'un DataFrame de liens
    (var_x, var_y, v) et d'une liste de niveaux [lvl0, lvl1, ...].
    """
    
    # Liste des nÅ“uds dans l'ordre des niveaux
    node_labels = []
    for lvl in levels:
        node_labels.extend(lvl)
    node_indices = {name: i for i, name in enumerate(node_labels)}

    sources = []
    targets = []
    values = []
    colors = []

    for k in range(len(levels) - 1):
            from_level = levels[k]
            to_level = levels[k + 1]
            mask = ((links_df["var_x"].isin(from_level) & links_df["var_y"].isin(to_level)) |
                    (links_df["var_y"].isin(from_level) & links_df["var_x"].isin(to_level)))
            subset = links_df[mask]

            for _, row in subset.iterrows():
                x = row["var_x"]
                y = row["var_y"]
                v = row["v"]

                if x in from_level and y in to_level:
                    s_name, t_name = x, y
                elif y in from_level and x in to_level:
                    s_name, t_name = y, x
                else:
                    continue

                sources.append(node_indices[s_name])
                targets.append(node_indices[t_name])
                values.append(v)
                colors.append("rgba(0, 100, 200, 0.5)")

    if not sources:
        return None

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(width=0.5, color="black"),
            label=node_labels,
            color="rgba(255,255,255,1)",  # fond clair
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
        )
    )])

    # on force la police des labels de nÅ“uds
    fig.update_traces(
        textfont=dict(color="black", size=13),
        selector=dict(type="sankey")
    )

    sankey_title="Expliquer les variables cibles depuis les structurantes"
         

    fig.update_layout(
        title_text=sankey_title,
        font=dict(size=11),
        height=600
    )

    return fig

def sankey_to_base64(fig):
    """Convertit un diagramme Sankey Plotly en base64 pour HTML"""
    if fig is None:
        return None
    png_bytes = pio.to_image(fig, format='png', width=1200, height=700)
    return base64.b64encode(png_bytes).decode('utf-8')

# Génération des tris croisés
from scipy.stats import chi2_contingency

def crosstab_with_std_residuals(df_cat, var_x, var_y):
    """
    Renvoie :
    - ct_count : tableau de contingence (effectifs)
    - ct_pct_row : % en ligne
    - std_res : résidus standardisés du Khi²
    """
    ct_count = pd.crosstab(df_cat[var_x], df_cat[var_y])
    if ct_count.empty:
        return ct_count, None, None

    chi2, p, dof, expected = chi2_contingency(ct_count)
    # résidus standardisés
    std_res = (ct_count - expected) / np.sqrt(expected)
    ct_pct_row = pd.crosstab(df_cat[var_x], df_cat[var_y], normalize='index') * 100
    return ct_count, ct_pct_row, std_res

# interprétation par LLM des tris croisés

def interpret_crosstab_with_llm(var_x, var_y, ct_pct_row, std_res):
    """
    Appel LLM pour interpréter le tableau de contingence var_x x var_y.
    On fournit au modèle :
    - % en ligne
    - résidus standardisés
    """
    
    # Représentation texte des tables
    table_pct = ct_pct_row.round(1).to_string()
    table_res = std_res.round(2).to_string()


    crosstab_interpretation = f"""Réponds en français, clair et concis
Tu es un data analyst expert (statistique + sémantique), en marketing si le jeu de données est un questionnaire.
On te donne 3 tableaux :
- un de contingence entre deux variables :
    - Variable X : {var_x}
    - Variable Y : {var_y}
- un tableau des pourcentages en ligne (% par modalité de X) :
{table_pct}
- un tableau des résidus standardisés du Khi² (positif = sur-représentation, négatif = sous-représentation) :
{table_res}

Tâche :
- Explique en 3 phrases les principaux résultats : quelles modalités de X sont particulièrement liées à quelles modalités de Y ?
- utilise l'analyse déjà faite dans {st.session_state.get("dataset_context", "")}, en particulier pour comprendre la signification des variables et le contexte global.
- N'utilise pas de termes techniques (résidus, Chi2, Khi², etc.), explique juste les relations entre modalités/variables, pas les tableaux.
- Ne fais pas d'introduction ni de résumé.
- Garde une explication concise et structurée.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": crosstab_interpretation
            },
            {
                "role": "user",
                "content": "réponds en fonction des instructions données dans role/system."
            },
        ],
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content or not content.strip():
        # En cas de réponse vide, on remonte une info exploitable
        return "(Interprétation indisponible â€“ réponse LLM vide)"

    return content


from matplotlib.colors import ListedColormap

def crosstab_heatmap_png(
    frequencies,
    residuals,
    threshold: float = 2.0,
    title: str | None = None,
):
    """
    Crée une 'carte de chaleur' des fréquences avec sur/sous-représentations
    et renvoie l'image au format PNG (bytes).

    Si auto_flip=True et que les libellés des colonnes dépassent long_label_threshold,
    on transpose pour que ces libellés se retrouvent en lignes.
    """

    # Libellés de modalités
    row_labels = frequencies.index.astype(str).tolist()
    col_labels = frequencies.columns.astype(str).tolist()

    max_col_len = max((len(lbl) for lbl in col_labels), default=0)
    max_row_len = max((len(lbl) for lbl in row_labels), default=0)

    # ðŸ” Règle simple : si les colonnes ont des libellés longs, on les met en lignes
    if max_col_len > max_row_len:
        frequencies = frequencies.T.copy()
        residuals = residuals.T.copy()
        # On inverse les labels pour l'affichage
        row_labels, col_labels = col_labels, row_labels

        # Si le titre est du style "X × Y", on inverse X et Y
        if title and "×" in title:
            left, right = title.split("×", 1)
            title = f"{right.strip()} × {left.strip()}"

    # Matrice -1 / 0 / 1 selon les résidus (sous, neutre, sur)
    sign_map = np.zeros_like(frequencies.values, dtype=int)
    sign_map[residuals.values > threshold] = 1
    sign_map[residuals.values < -threshold] = -1

    cmap = ListedColormap(["#fcb6b6", "white", "#b6fcb6"])

    n_rows, n_cols = frequencies.shape

    fig_width = max(4, n_cols * 0.9)
    fig_height = max(3, n_rows * 0.6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(sign_map, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)

    # Annotation des fréquences
    for i in range(n_rows):
        for j in range(n_cols):
            val = frequencies.iat[i, j]
            ax.text(
                j, i,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

    if title:
        ax.set_title(title)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def heatmap_to_base64(heatmap_png):
    """Convertit un heatmap PNG en base64 pour HTML"""
    if heatmap_png is None:
        return None
    return base64.b64encode(heatmap_png).decode('utf-8')

def dataframe_to_html(df):
    """Convertit un DataFrame en HTML table avec style"""
    if df is None or df.empty:
        return ""
    return df.to_html(classes='dataframe table table-striped', index=True, escape=False)

# ==========================
# 4. Streamlit App
# ==========================

def run():
    # initialiser les variables
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    pipeline_silent = bool(st.session_state.get("__PIPELINE_SILENT__", False))
    run_sankey_crosstabs = bool(
        st.session_state.get("run_sankey_crosstabs", not pipeline_silent)
    )
    if "etape23_terminee" not in st.session_state:
        st.session_state["etape23_terminee"] = False

    if "crosstabs_interpretation" not in st.session_state:
        st.session_state["crosstabs_interpretation"] = None
        
    if "sankey_diagram" not in st.session_state:
        st.session_state["sankey_diagram"] = None

    if "sankey_interpretation_synthesis_generated" not in st.session_state:
        st.session_state["sankey_interpretation_synthesis_generated"] = False
        
    if "sankey_diagram_generated" not in st.session_state:
        st.session_state["sankey_diagram_generated"] = False

    if "crosstabs_generated" not in st.session_state:
        st.session_state["crosstabs_generated"] = False

    st.header("Schéma des relations entre variables")
    
    # Bouton de réinitialisation
    if "sankey_diagram_generated" in st.session_state:
        if st.button("Réinitialiser le module Sankey"):
            keys_to_reset = [
                "profiles",
                "outcomes",
                "levels",
                "relevant_links",
                "links_df_filtered",
                "latent_info",
                "latent_raw_response",
                "sankey_latent_info",
                "latent_summary_text",
                "primary_vars_per_latent",
                "primary_vars_global",
                "links_sorted_with_flag",
                "sankey_pair_results",
                "sankey_diagram",
                "crosstabs_interpretation",
                "links_editor",
                "etape23_terminee",
            ]

            for k in keys_to_reset:
                if k in st.session_state:
                    del st.session_state[k]

            st.success("Le module Sankey a été réinitialisé. Vous pouvez relancer l'analyse.")
            # On stoppe l'exécution du reste pour partir "propre" au prochain run
            st.stop()

    # Récupération des variables mémorisées
    illustrative_variables = st.session_state.get("illustrative_variables", [])
    target_variables = st.session_state.get("target_variables", [])

    if "df_ready" in st.session_state:
        df = st.session_state.df_ready
    else:
        st.warning("Aucun dataset ready trouvé. Veuillez d'abord passer par l'application précédente.")
        return

    # -------------------------
    # Discrétisation des variables continues
    # -------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if numeric_cols:
        n_bins = st.slider(
            "Nombre de classes pour discrétiser les variables continues",
            min_value=2,
            max_value=15,
            value=5,
        )

        df_discrete = df.copy()
        for col in numeric_cols:
            df_discrete[col] = pd.cut(
                df[col],
                bins=n_bins,
                duplicates="drop",
                include_lowest=True
            )
    else:
        df_discrete = df.copy()

    # Tout en string pour traiter comme catégoriel
    df_cat = df_discrete.copy()
    for col in df_cat.columns:
        df_cat[col] = df_cat[col].astype(str)

    all_vars = list(df_cat.columns)

    # Pré-remplissage
    # Pre-remplissage
    brief_tv = st.session_state.get("brief_target_variable")
    default_profiles = [v for v in illustrative_variables if v in all_vars and v != brief_tv]
    default_outcomes = [v for v in target_variables if v in all_vars]

    if "profiles" not in st.session_state:
        st.session_state["profiles"] = default_profiles

    st.subheader("Selection des groupes de variables")

    profiles = st.multiselect(
        "Variables de profil (niveau 0)",
        options=all_vars,
        key="profiles",
        help="Variables sociodemographiques, de foyer, etc."
    )
    # retirer la cible issue du brief si elle se trouve dans les profils
    if brief_tv and brief_tv in profiles:
        profiles = [p for p in profiles if p != brief_tv]
        st.session_state["profiles"] = profiles

    # Une seule variable cible: la plus pertinente (1re de target_variables issue de DiagnosticGlobal).
    # Priorité à la cible du brief si disponible
    if brief_tv and brief_tv in all_vars and brief_tv not in profiles:
        primary_outcome = brief_tv
    else:
        primary_outcome = default_outcomes[0] if default_outcomes else None
    if primary_outcome and primary_outcome in profiles:
        primary_outcome = None
    if primary_outcome is None:
        candidates_outcome = [v for v in all_vars if v not in profiles]
        primary_outcome = candidates_outcome[0] if candidates_outcome else None

    outcomes = [primary_outcome] if primary_outcome else []
    st.session_state["outcomes"] = outcomes
    if outcomes:
        st.caption(f"Variable cible utilisee automatiquement : {outcomes[0]}")
    else:
        st.warning("Aucune variable cible disponible pour DiagramSankey.")

    candidates_default = [v for v in all_vars if v not in profiles + outcomes]
    candidates = st.multiselect(
        "Variables candidates médiatrices",
        options=all_vars,
        default=candidates_default,
        help="Variables susceptibles de jouer un rôle intermédiaire (usages, attitudes, etc.)."
    )

    if "levels" not in st.session_state:
        st.session_state["levels"] = None
    if "sankey_pair_results" not in st.session_state:
        st.session_state["sankey_pair_results"] = {}

    st.subheader("Paramètres statistiques")
    alpha = st.slider("Seuil de significativité (alpha)", 0.001, 0.1, 0.1, 0.001)
    v_min = st.slider("Seuil minimum de V de Cramér pour garder un lien", 0.0, 0.4, 0.1, 0.01)
    max_length_links = st.slider("Nombre maximum de couples de variables", 20, 3000, 1500, 10)

    if not profiles or not outcomes:
        st.warning("Veuillez sélectionner au moins une variable de profil et une variable d'outcome.")
        st.stop()

    # BOUTON DE CALCUL & FILTRAGE

    vars_to_use = profiles + outcomes + candidates
    
    # Execution pilotee par le pipeline global (pas de gating local manuel/auto)
    proceed = True
    if proceed:
        # 1) Associations sur toutes les variables utilisées
        vars_to_use = profiles + outcomes + candidates

        links_df = compute_associations(
            df_cat,
            vars_to_use,
            alpha=alpha,
            v_min=v_min,
            profiles=profiles,
            outcomes=outcomes
        )

        # Filtre : on retire profils/profils et outcomes/outcomes
        links_df = links_df[~(
            (links_df["var_x"].isin(profiles) & links_df["var_y"].isin(profiles)) |
            (links_df["var_x"].isin(outcomes) & links_df["var_y"].isin(outcomes))
        )]

        st.write("Nombre de couples de variables", len(links_df))

        # 3 scenarios : pas de lien, trop de liens, ok

        if links_df.empty:
            st.error("Aucun lien significatif trouvé avec ces paramètres.")
            st.session_state["relevant_links"] = None
            st.session_state["levels"] = None
            st.session_state["etape23_terminee"] = True
            return

        if len(links_df) > max_length_links:
            links_df = compute_associations(
                df_cat,
                vars_to_use,
                alpha=0.05,
                v_min=0.25,
                profiles=profiles,
                outcomes=outcomes
            )

            # Filtre : on retire profils/profils et outcomes/outcomes
            links_df = links_df[~(
                (links_df["var_x"].isin(profiles) & links_df["var_y"].isin(profiles)) |
                (links_df["var_x"].isin(outcomes) & links_df["var_y"].isin(outcomes))
            )]

            if len(links_df) > max_length_links:
                st.error(
                    f"Trop de liens significatifs ({len(links_df)}) avec des paramètres standards. "
                    "Veuillez augmenter les seuils de filtrage (alpha et V min)."
                )
                st.session_state["relevant_links"] = None
                st.session_state["levels"] = None
                st.session_state["etape23_terminee"] = True
                return

        # Identifiants uniques pour chaque couple (dans tous les cas où on continue)
        links_df = links_df.reset_index(drop=True)
        links_df["pair_id"] = links_df.index.astype(int)

        # Texte ACM / dendrogramme (si disponible)
        dendrogram_text = st.session_state.get("dendrogram_interpretation", "")
        acm_text = st.session_state.get("interpretationACM", "")
        var_labels = st.session_state.get("variable_labels", None)


        # 2) Latents: toujours calcules pour conserver les sorties en pipeline.
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        with st.spinner("Analyse des latents via LLM..."):
            links_df_llm, latent_info = refine_links_with_latents_via_llm(
                df_cat=df_cat,
                links_df=links_df,
                var_labels=var_labels,
                dendrogram_text=dendrogram_text,
                acm_text=acm_text,
                profiles=profiles,
                outcomes=outcomes,
                candidates=candidates,
                client=client,
            )

            st.session_state["latent_info"] = latent_info
            st.session_state["latent_info_generated"] = True
            st.success("Analyse des latents terminee.")

        # 3) Primary vars par latent
        if latent_info:
            var_to_latent, englobing_vars = build_var_to_latent_and_englobing(latent_info)

            primary_vars_per_latent = compute_primary_variables_per_latent(
                links_df_llm,
                latent_info
            )

            if primary_vars_per_latent:
                primary_vars_global = set().union(*primary_vars_per_latent.values())
            else:
                primary_vars_global = set()

            # DataFrame des latents / variables
            sankey_latents_df = build_latents_dataframe(
                latent_info=latent_info,
                primary_vars_per_latent=primary_vars_per_latent,
                englobing_vars=englobing_vars,
            )
            st.session_state["sankey_latents"] = sankey_latents_df

        else:
            var_to_latent = {}
            englobing_vars = set()
            primary_vars_per_latent = {}
            primary_vars_global = set()
            # DataFrame vide
            sankey_latents_df = pd.DataFrame(
                columns=["latent_id", "latent_name", "variable", "is_primary", "is_englobing"]
            )
            st.session_state["sankey_latents"] = sankey_latents_df

        st.session_state["primary_vars_per_latent"] = primary_vars_per_latent
        st.session_state["primary_vars_global"] = primary_vars_global
        st.dataframe(sankey_latents_df)

        # 4) Médiateurs niveaux 1 / 2, puis restriction aux primary vars
        med1_raw, med2_raw = assign_mediators_to_levels(
            links_df_llm, profiles, outcomes, candidates, v_min=v_min
        )

        if primary_vars_global:
            med1 = [v for v in med1_raw if v in primary_vars_global]
            med2 = [v for v in med2_raw if v in primary_vars_global]
        else:
            med1, med2 = med1_raw, med2_raw

        # ðŸ”§ Construction des niveaux en fonction de med1 / med2
        if not med1 and not med2:
            # Aucun médiateur retenu -> direct profils â†’ outcomes
            levels = [profiles, outcomes]
        elif not med2:
            # Pas de niveau 2 -> profils → med1 → outcomes
            levels = [profiles, med1, outcomes]
        elif not med1:
            # Pas de niveau 1 -> profils → med2 → outcomes
            levels = [profiles, med2, outcomes]
        else:
            # Les 2 niveaux existent
            levels = [profiles, med1, med2, outcomes]

        st.session_state["levels"] = levels

        # 5) Liens pertinents = ceux entre nÅ“uds du Sankey
        all_nodes = set().union(*levels) 

        relevant_links = links_df_llm[
            links_df_llm["var_x"].isin(all_nodes)
            & links_df_llm["var_y"].isin(all_nodes)
        ].reset_index(drop=True)

        st.session_state["relevant_links"] = relevant_links
        st.session_state["links_sorted_with_flag"] = None
        st.session_state["sankey_pair_results"] = {}
        st.session_state["crosstabs_interpretation"] = None
        if not str(st.session_state.get("latent_summary_text", "")).strip():
            st.session_state["latent_summary_text"] = (
                f"Schema Sankey calcule avec {len(profiles)} variable(s) de profil, "
                f"{len(outcomes)} variable(s) cible(s) et {len(relevant_links)} lien(s) retenu(s)."
            )

        st.success(f"{len(links_df)} liens significatifs avant filtrage par latents / niveaux.")
        st.success(f"{len(relevant_links)} liens pertinents conservés (Sankey + tris croisés).")


    # -----------------------------
    # TABLEAU + DIAGRAMME SANKEY + INTERPRETATION LLM DES TRIS CROISES
    # -----------------------------
    relevant_links = st.session_state.get("relevant_links")
    levels = st.session_state.get("levels")
    latent_info = st.session_state.get("latent_info")

    if relevant_links is None or relevant_links.empty or levels is None:
        st.info("Cliquez sur 'Calculer associations & préparer les couples' pour commencer l'analyse.")
        if not str(st.session_state.get("latent_summary_text", "")).strip():
            st.session_state["latent_summary_text"] = "Aucune dimension latente exploitable n'a pu être extraite."
        st.session_state["etape23_terminee"] = True
        return

    st.subheader("Liens pertinents (Sankey + tris croisés)")

    primary_vars_per_latent = st.session_state.get("primary_vars_per_latent", {})
    latent_info = st.session_state.get("sankey_latent_info")
    if isinstance(latent_info, dict):
        var_to_latent, _englobing_vars = build_var_to_latent_and_englobing(latent_info)
    else:
        var_to_latent, _englobing_vars = {}, set()

    links_sorted = (
        relevant_links
        .sort_values("v", ascending=False)
        .reset_index(drop=True)
    )

    # Pré-sélection des couples
    preselected_ids = set()
    for _, row in links_sorted.iterrows():
        vx = row["var_x"]
        vy = row["var_y"]
        pid = row["pair_id"]
        lx = var_to_latent.get(vx)
        ly = var_to_latent.get(vy)
        if lx is None or ly is None:
            continue
        if lx not in primary_vars_per_latent or ly not in primary_vars_per_latent:
            continue
        if (vx in primary_vars_per_latent[lx]) and (vy in primary_vars_per_latent[ly]):
            preselected_ids.add(pid)

    links_sorted["Analyser LLM"] = links_sorted["pair_id"].isin(preselected_ids)

    st.write("**Vérifiez/modifiez les couples à analyser puis cliquez sur le bouton :**")

    edited_links = st.data_editor(
        links_sorted,
        key="links_editor",
        hide_index=True,
        column_config={
            "pair_id": st.column_config.NumberColumn("ID", disabled=True),
            "Analyser LLM": st.column_config.CheckboxColumn(
                "Analyser LLM",
                help="Cochez pour analyser ce couple de variables avec le LLM."
            )
        },
        use_container_width=True
    )

    st.session_state["links_sorted_with_flag"] = edited_links


    # 3) Sankey multi-niveaux
    st.subheader("Diagramme des relations multi-niveaux")

    nb_levels = len(levels)

    if nb_levels == 2:
        # profils -> outcomes (aucun médiateur retenu)
        profils_lvl, outcomes_lvl = levels
        st.write("**Profils (niveau 0)** :", profils_lvl)
        st.write("**Variables cibles (niveau final)** :", outcomes_lvl)
        st.info("Aucun médiateur n'a été identifié : diagramme direct profils outcomes.")
    elif nb_levels == 3:
        # un seul niveau intermédiaire
        profils_lvl, mediators_lvl, outcomes_lvl = levels
        st.write("**Profils (niveau 0)** :", profils_lvl)
        st.write("**Médiateurs (niveau intermédiaire)** :", mediators_lvl)
        st.write("**Variables cibles (niveau final)** :", outcomes_lvl)
    else:
        # cas général : 4 niveaux (profils, med1, med2, variable cible/outcomes)
        med1 = levels[1]
        med2 = levels[2]
        st.write("**Médiateurs niveau 1 (proches des profils)** :", med1)
        st.write("**Médiateurs niveau 2 (proches de la variable cible)** :", med2)

    fig = build_sankey_from_links(relevant_links, levels)
    if fig is None:
        st.error("Aucun lien entre niveaux successifs pour construire le Sankey.")
    else:
        st.plotly_chart(fig, use_container_width=True)
        st.session_state["sankey_diagram"] = fig
        try:
            st.session_state["sankey_diagram_base64"] = sankey_to_base64(fig)
        except Exception:
            st.session_state["sankey_diagram_base64"] = None
        st.session_state["sankey_diagram_generated"] = True
        # En mode pipeline silencieux, on ne fabrique pas de texte d'interpretation artificiel.
        # Le texte doit venir de la logique initiale du module (LLM / contenu reel produit).

    if st.session_state.get("sankey_diagram_generated"):
        st.success("sankey diagram generated")

    # 4) Lancer l'analyse LLM des couples sélectionnés
    
    # Execution pilotee par le pipeline global (pas de gating local manuel/auto)
    proceed2 = True
    if proceed2:
        if not run_sankey_crosstabs:
            st.session_state["crosstabs_interpretation"] = []
            st.session_state["sankey_pair_results"] = {}
            st.session_state["crosstabs_generated"] = False
            if not pipeline_silent:
                st.info("Analyse des tris croises desactivee (mode rapide).")
        else:
            edited_links = st.session_state["links_sorted_with_flag"]
            # On selectionne uniquement les lignes ou la checkbox "Analyser LLM" est vraie
            links_for_llm = edited_links[edited_links["Analyser LLM"]]

            # Plafond du nombre de couples envoyes au LLM pour maitriser le temps/coût
            top_n_llm = int(st.session_state.get("sankey_top_n_llm_pairs", 10))
            if len(links_for_llm) > top_n_llm:
                links_for_llm = (
                    links_for_llm
                    .sort_values(by="v", ascending=False)
                    .head(top_n_llm)
                    .reset_index(drop=True)
                )
                if not pipeline_silent:
                    st.info(f"Limitation aux {top_n_llm} couples avec le V de Cramer le plus eleve pour l'analyse LLM.")

            st.subheader("Analyse des tris croises")

            crosstabs_interpretation = []
            results_store = st.session_state.get("sankey_pair_results", {})

            if links_for_llm.empty:
                st.info("Aucun couple selectionne pour l'analyse LLM.")
                st.session_state["crosstabs_generated"] = False
            else:
                for _, row in links_for_llm.iterrows():
                    var_x = row["var_x"]
                    var_y = row["var_y"]
                    v_val = row["v"]
                    p_val = row["p"]
                    chi2_val = row["chi2"]
                    pair_id = int(row["pair_id"])

                    if pair_id in results_store:
                        res = results_store[pair_id]
                        ct_count = res["ct_count"]
                        ct_pct_row = res["ct_pct_row"]
                        std_res = res["std_res"]
                        interpretation = res["interpretation"]
                        heatmap_png = res["heatmap_png"]
                    else:
                        ct_count, ct_pct_row, std_res = crosstab_with_std_residuals(df_cat, var_x, var_y)
                        if ct_pct_row is None or std_res is None:
                            st.info("Tableau de contingence vide ou non exploitable.")
                            continue

                        with st.spinner(f"Interpretation LLM pour {var_x} x {var_y}..."):
                            base_interpretation = interpret_crosstab_with_llm(
                                var_x, var_y, ct_pct_row, std_res
                            )

                        metrics_header = (
                            f"V de Cramer = {v_val:.3f} ; "
                            f"p-value = {p_val:.5f} ; "
                            f"Khi2 = {chi2_val:.3f}\n\n"
                        )
                        interpretation = metrics_header + base_interpretation

                        frequencies = ct_pct_row.round(2)

                        heatmap_png = crosstab_heatmap_png(
                            frequencies,
                            std_res,
                            threshold=2.0,
                            title=f"{var_x} x {var_y}"
                        )

                        results_store[pair_id] = {
                            "var_x": var_x,
                            "var_y": var_y,
                            "v": v_val,
                            "p": p_val,
                            "chi2": chi2_val,
                            "ct_count": ct_count,
                            "ct_pct_row": ct_pct_row,
                            "std_res": std_res,
                            "interpretation": interpretation,
                            "heatmap_png": heatmap_png,
                        }

                    crosstabs_interpretation.append({
                        "pair_id": pair_id,
                        "var_x": var_x,
                        "var_y": var_y,
                        "interpretation": interpretation,
                    })

                st.session_state["crosstabs_interpretation"] = crosstabs_interpretation
                st.session_state["sankey_pair_results"] = results_store
                st.session_state["crosstabs_generated"] = True
    if st.session_state.get("crosstabs_generated"):
        st.success("Crosstabs + interpretations generated")


    # 5) Interprétation du diagramme de Sankey
    # Execution pilotee par le pipeline global (pas de gating local manuel/auto)
    proceed3 = True
    if proceed3:
        try:
            with st.spinner("Interprétation du diagramme de Sankey par LLM en cours..."):
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

                sankey_interpretation = f'''Vous êtes un expert en analyse de données. Répondez clair et concis.
                Un jeu de données a été analysé. Un diagramme de Sankey a été produit.
                Dans dataset_context, vous trouverez des informations sur le contexte du dataset, les objectifs de l'analyse, et le rôle des variables, et l'unité d'observation.
                Interprétez le diagramme de Sankey fourni.
                Il commence à gauche avec les variables structurantes (ce sont les variables illustratives), et finit à droite avec la variable cible, et au milieu, les variables médiatrices.
                Interprétez le diagramme de Sankey fourni, en vous concentrant sur les enchainements de facteurs qui expliquent la variable cible.
                Utilisez tous les autres documents fournis pour le contextualiser.
                Si la variable cible ne figure pas parmi les variables candidates, il faut en déduire que cette méthode ne peut pas s'appliquer.
                Ne faites pas de conclusion qui récapitule ce que vous avez exépliqué.
                '''
                

                context_blob_sankey_interpretation = {
                    "sankey_diagram": st.session_state.get("sankey_diagram"),
                    "dataset_object": st.session_state.get("dataset_object", ""),
                    "dataset_context": st.session_state.get("dataset_context", ""),
                    "dataset_key_questions": st.session_state.get("dataset_key_questions", ""),                
                    "crosstabs_interpretation" : st.session_state.get("crosstabs_interpretation", []),
                    "profil_dominant_analysis": st.session_state.get("profil_dominant_analysis", ""),
                    "interpretationACM": st.session_state.get("interpretationACM", ""),
                    "dendrogram_interpretation": st.session_state.get("dendrogram_interpretation", ""),
                }

                # informations transmises au LLM
                user_content_2 = json.dumps(context_blob_sankey_interpretation, ensure_ascii=False, default=str)

                r2 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": sankey_interpretation},
                        {"role": "user", "content": user_content_2},
                    ],
                )

                sankey_interpretation_synthesis = r2.choices[0].message.content
                st.session_state["sankey_interpretation_synthesis"] = sankey_interpretation_synthesis
                st.session_state["sankey_interpretation_synthesis_generated"] = True

        except Exception as e:
            st.session_state["diagram_sankey_llm_error"] = str(e)
            st.error(f"Une erreur est survenue lors de l'appel a l'API : {e}")
    
    if st.session_state.get("sankey_interpretation_synthesis_generated", False):
        st.success("sankey_interpretation_synthesis generated")

    st.session_state["etape23_terminee"] = True

if __name__ == "__main__":
    run()
