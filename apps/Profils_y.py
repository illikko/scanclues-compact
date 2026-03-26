import streamlit as st
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from kmodes.kmodes import KModes
from openai import OpenAI
import os

MODE_KEY = "__NAV_MODE__"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -----------------------------
# Helpers : reset / préselection
# -----------------------------
def _reset_after_step2():
    """Reset tout ce qui dépend du choix cible/mod/colonnes."""
    for k, v in {
        "step3_ready": False,
        "step5_ready": False,
        "idxs": None,
        "X": None,
        "clusters_X": None,
        "profils_y": None,
        "profils_y_generated": False,
        "profils_y_text": "",
    }.items():
        st.session_state[k] = v


def _get_preferred_target_variable(df: pd.DataFrame) -> str:
    """Préselection de la cible : brief_target_variable si dispo, sinon target_variables[0]."""
    options = list(df.columns)
    brief_tv = st.session_state.get("brief_target_variable")
    if brief_tv in options:
        return brief_tv
    tv_list = st.session_state.get("target_variables", []) or []
    if tv_list and tv_list[0] in options:
        return tv_list[0]
    # Fallback robuste depuis table validée Diagnostic/Cadrage
    var_table_df = st.session_state.get("var_table_df")
    if isinstance(var_table_df, pd.DataFrame) and not var_table_df.empty:
        try:
            if "variable" in var_table_df.columns and "cible" in var_table_df.columns:
                selected = var_table_df.loc[var_table_df["cible"] == True, "variable"].tolist()
                for v in selected:
                    if v in options:
                        return v
        except Exception:
            pass
    return options[-1]  # fallback


def _get_preferred_modality(tv: str, options: list[str]) -> str | None:
    """Préselection d'une modalité via st.session_state['target_modalities'][tv] si dispo."""
    prefs = st.session_state.get("target_modalities", {}) or {}
    pref = prefs.get(tv, None)

    if pref is None:
        return None

    # On normalise en str pour comparer
    pref_str = str(pref).strip()
    if pref_str in options:
        return pref_str
    # Matching tolérant (casse / espaces)
    pref_norm = pref_str.casefold()
    by_norm = {str(o).strip().casefold(): str(o) for o in options}
    if pref_norm in by_norm:
        return by_norm[pref_norm]
    return None


def _continuous_segment_default(tv: str) -> str:
    """
    Préselection du segment pour variable continue.
    Si st.session_state['target_modalities'][tv] == 'min' => Bottom 20%
    sinon => Top 20%
    """
    prefs = st.session_state.get("target_modalities", {}) or {}
    pref = (prefs.get(tv) or "").strip().lower()
    return "Bottom 20%" if pref == "min" else "Top 20%"


def _compute_segment_indices(df: pd.DataFrame, tv: str, choice: str) -> list[int]:
    """Retourne les idxs du segment cible (Top/Bottom p%)."""
    lbl_to_p = {
        "Top 20%": ("top", 0.20),
        "Top 10%": ("top", 0.10),
        "Top 5%": ("top", 0.05),
        "Bottom 20%": ("bottom", 0.20),
        "Bottom 10%": ("bottom", 0.10),
        "Bottom 5%": ("bottom", 0.05),
    }
    direction, p = lbl_to_p[choice]
    col = df[tv]
    if col.dropna().empty:
        return []

    if direction == "top":
        q = col.quantile(1 - p)
        return df.index[col >= q].tolist()
    else:
        q = col.quantile(p)
        return df.index[col <= q].tolist()


def run():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    # -----------------------------
    # State init
    # -----------------------------
    defaults = {
        "etape31_terminee": False,
        "step3_ready": False,
        "step5_ready": False,
        "target_variable": None,
        "target_mod": None,
        "idxs": None,
        "X": None,
        "min_val_quant": None,
        "max_val_quant": None,
        "clusters_X": None,
        "num_quantiles": 5,
        "n_clusters": 3,
        "profils_y": None,
        "profils_y_generated": False,
        "profils_y_detailed": None,
        "profils_y_text": "",
        "_selected_cols": None,
        "_sig_step2": None,   # signature paramètres step2
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    st.header("Profils associés à une cible")

    # -----------------------------
    # Dataset
    # -----------------------------
    if "df_ready" not in st.session_state or not isinstance(st.session_state.df_ready, pd.DataFrame):
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application précédente.")
        return

    df_base = st.session_state.df_ready.copy()

    # -----------------------------
    # Bouton reset global
    # -----------------------------

    if st.button("Réinitialiser / relancer"):
        # Reset complet module
        for k in list(defaults.keys()):
            st.session_state[k] = defaults[k]
        st.rerun()
    st.caption("Changez la cible/mod/paramètres puis cliquez Lancer.")

    st.write("Aperçu du dataset :")
    st.dataframe(df_base.head())

    # ==========================================================
    # Étape 2 : paramètres (dans un form => exécuter sur click)
    # ==========================================================
    st.subheader("Sélection des paramètres")

    with st.form("step2_form", clear_on_submit=False):
        # Colonnes à garder
        all_cols = df_base.columns.tolist()
        default_cols = st.session_state["_selected_cols"] or all_cols
        selected_cols = st.multiselect(
            "Colonnes à garder",
            all_cols,
            default=default_cols
        )

        if not selected_cols:
            st.warning("Sélectionne au moins une colonne.")
            submitted = st.form_submit_button("Lancer")
            return

        df = df_base[selected_cols].copy()

        # Préselection cible
        preferred_tv = _get_preferred_target_variable(df)
        options_tv = df.columns.tolist()
        default_index_tv = options_tv.index(preferred_tv)

        target_variable = st.selectbox(
            "Variable cible",
            options=options_tv,
            index=default_index_tv
        )

        # Déterminer continu / cat
        distinct_threshold = int(st.session_state.get("distinct_threshold_continuous", 5))
        nunq = df[target_variable].nunique(dropna=True)
        is_cont = is_numeric_dtype(df[target_variable]) and nunq > distinct_threshold

        target_mod = None
        idxs = []

        if is_cont:
            segments = ["Top 20%", "Top 10%", "Top 5%", "Bottom 20%", "Bottom 10%", "Bottom 5%"]
            default_seg = _continuous_segment_default(target_variable)
            default_seg_index = segments.index(default_seg) if default_seg in segments else 1

            choice = st.selectbox(
                "Segment sur la variable continue :",
                segments,
                index=default_seg_index
            )
            idxs = _compute_segment_indices(df, target_variable, choice)
            target_mod = choice
            
            # calcul du min et max de la cible
            subset = df.loc[idxs, target_variable]
            st.session_state.min_val_quant = subset.min()
            st.session_state.max_val_quant = subset.max()

        else:
            modalities = sorted(df[target_variable].dropna().astype("string").unique().tolist())
            if not modalities:
                st.error("Aucune modalité non-nulle pour cette variable.")
                submitted = st.form_submit_button("Lancer")
                return

            pref_mod = _get_preferred_modality(target_variable, modalities)
            prefs = st.session_state.get("target_modalities", {}) or {}
            expected_pref = prefs.get(target_variable, None)
            if expected_pref is not None and pref_mod is None:
                st.warning(
                    f"Modalité cible définie dans Diagnostic non trouvée pour '{target_variable}': "
                    f"'{expected_pref}'. Vérifiez l'encodage/normalisation de cette colonne."
                )
                pref_mod = modalities[0]
            elif pref_mod is None:
                # fallback seulement s'il n'y a pas de modalité de référence
                pref_mod = df[target_variable].astype("string").value_counts(dropna=True).idxmax()

            default_mod_index = modalities.index(str(pref_mod)) if str(pref_mod) in modalities else 0

            target_modality = st.selectbox(
                "Modalité cible :",
                options=modalities,
                index=default_mod_index
            )
            idxs = df.index[df[target_variable].astype("string") == str(target_modality)].tolist()
            target_mod = str(target_modality)

        # Autres paramètres
        num_quantiles = st.slider(
            "Nombre de quantiles (discrétisation des continues explicatives)",
            min_value=3, max_value=10,
            value=int(st.session_state.get("num_quantiles", 5))
        )
        n_clusters = st.slider(
            "Nombre de clusters",
            min_value=1, max_value=15,
            value=int(st.session_state.get("n_clusters", 3))
        )

        n_init = st.slider("Nombre d'itérations du Kmodes", min_value=2, max_value=10, value=5)

        # Signature step2 : si change => reset steps suivants
        sig = (
            tuple(selected_cols),
            target_variable,
            target_mod,
            int(num_quantiles),
            int(n_clusters),
        )

        submitted = st.form_submit_button("Lancer")
        
    proceed_step2 = (mode == "automatique") or submitted

    if proceed_step2:
        if st.session_state["_sig_step2"] != sig:
            st.session_state["_sig_step2"] = sig
            _reset_after_step2()

        st.session_state["_selected_cols"] = selected_cols
        st.session_state.target_variable = target_variable
        st.session_state.target_mod = target_mod
        st.session_state.idxs = idxs
        st.session_state.num_quantiles = int(num_quantiles)
        st.session_state.n_clusters = int(n_clusters)

        # X explicatives
        X = df.drop(columns=[target_variable]).copy()
        st.session_state.X = X

        # validation étape 3 implicite : on a cliqué Lancer
        st.session_state.step3_ready = True
        st.session_state.step5_ready = False

    if not st.session_state.step3_ready:
        st.info("Choisis les paramètres puis clique **Lancer**.")
        return

    # ==========================================================
    # Étape 3 : Profils associés à la cible (KModes)
    # ==========================================================
    st.subheader("Profils associés à la cible")
    min_val_quant = st.session_state.get("min_val_quant", None)
    max_val_quant = st.session_state.get("max_val_quant", None)


    if min_val_quant is not None and max_val_quant is not None:
        st.write(f"Intervalle effectif de la variable cible : {st.session_state.target_variable} | intervalle visé : {st.session_state.target_mod}")
        st.write("Min :", min_val_quant)
        st.write("Max :", max_val_quant)

    X = st.session_state.X
    idxs = list(st.session_state.idxs or [])
    if X is None or not idxs:
        st.warning("Segment cible vide. Ajuste la cible / modalité / segment puis relance.")
        return

    # Discrétisation simple (comme ta version) :
    Xw = X.copy()
    
    
    mod_freq_min = float(st.session_state.get("mod_freq_min", 0.90))

    num_cols = Xw.select_dtypes(include=["number"])
    distinct_counts = num_cols.nunique(dropna=True)
    continuous = distinct_counts[distinct_counts > int(st.session_state.get("distinct_threshold_continuous", 5))].index.tolist()
    discrete = [col for col in Xw.columns if col not in continuous]

    # Cast numériques discrets en str
    for var in discrete:
        if Xw[var].dtype.kind in ("i", "f"):
            Xw[var] = Xw[var].astype(str)

    # Variables continues très concentrées -> binaire (mode vs reste)
    col_high_freq = []
    for col in continuous:
        if Xw[col].dropna().empty:
            continue
        mode_value = Xw[col].mode(dropna=True).iloc[0]
        mod_freq = Xw[col].value_counts(normalize=True, dropna=True).loc[mode_value]
        if mod_freq > mod_freq_min:
            non_mode = Xw[col] != mode_value
            
            Xw[col] = Xw[col].astype("string") # conversion en string avant affectation
            
            if non_mode.any():
                min_value = Xw.loc[non_mode, col].min()
                max_value = Xw.loc[non_mode, col].max()
                Xw.loc[non_mode, col] = f"{round(float(min_value), 2)} to {round(float(max_value), 2)}"
            Xw.loc[~non_mode, col] = f"{mode_value}"
            col_high_freq.append(col)

    continuous = [c for c in continuous if c not in col_high_freq]

    # Discrétisation quantiles
    qn = int(st.session_state.num_quantiles)
    for col in continuous:
        col_series = Xw[col]
        if col_series.dropna().nunique() < 2:
            Xw[col] = col_series.astype("string")
            continue
        codes, bins = pd.qcut(col_series, qn, retbins=True, labels=False, duplicates="drop")
        bins = [round(float(b), 2) for b in bins]
        labels = [f"({bins[i]}, {bins[i+1]}]" for i in range(len(bins) - 1)]
        Xw[col] = codes.map(dict(enumerate(labels))).astype("string")

    st.caption("Aperçu des variables explicatives après discrétisation :")
    st.dataframe(Xw.head())


    n_clusters = int(st.session_state.n_clusters)

    # clustering SUR LE SEGMENT CIBLE uniquement
    target_X = Xw.loc[idxs].copy()
    feature_cols = target_X.columns.tolist()
    if target_X.empty:
        st.warning("Segment cible vide après filtrage. Relance avec un autre segment.")
        return

    km = KModes(n_clusters=n_clusters, init="Huang", n_init=10, random_state=42)
    clusters = km.fit_predict(target_X)
    target_X["Cluster_id"] = clusters

    # 2) Mapping numérique -> étiquette lisible G1, G2, ...
    label_map = {i: f"P{i+1}" for i in range(km.n_clusters)}

    # colonne lisible pour tous les affichages
    target_X["Cluster"] = target_X["Cluster_id"].map(label_map)

    st.session_state.clusters_X = target_X
    st.session_state.X = Xw

    # Centres
    P = pd.DataFrame(
        km.cluster_centroids_, 
        columns=target_X.columns.drop(["Cluster", "Cluster_id"])
        )
    P.index = [f"P{i+1}" for i in range(len(P))]
    P = P.T
    P.index.name = "Variable"
    P["Differenciation"] = P.apply(lambda r: len(set(r)), axis=1)
    P = P.sort_values(by="Differenciation", ascending=True)
    st.session_state.profils_y_table = P.copy()

    st.subheader("Tableau des profils")
    st.dataframe(st.session_state.profils_y_table)

    # Effectifs + fréquences (SUR LE SEGMENT CIBLE)
    effectifs = target_X["Cluster"].value_counts().sort_index()
    frequences_target = (effectifs / len(target_X) * 100).round(1).astype(str) + "%"
    frequences_total = (effectifs / len(df) * 100).round(1).astype(str) + "%"

    st.subheader("Effectifs & fréquences des profils du segment cible")
    st.dataframe(
        pd.DataFrame({"Effectifs": effectifs, "Fréquence / cible": frequences_target, "Fréquence / pop totale": frequences_total})
    )
    st.caption(
        f"Segment cible : {st.session_state.target_variable} → {st.session_state.target_mod} | "
        f"Taille segment = {len(target_X)} lignes"
    )

    # ==========================================================
    # Étape 4 : Caractérisation (test value) + profils_y
    # ==========================================================
    st.subheader("Profils détaillés")

    def group_characterization(data, overall_data, group_column, variables, label_map=None):
        results = []
        # fréquences globales calculées sur la POPULATION TOTALE (overall_data)
        # => ok si tu veux comparer segment vs global ; sinon passe overall_data= data
        for grp in sorted(data[group_column].unique()):
            mask = (data[group_column] == grp)
            gsize = int(mask.sum())
            
            grp_label = label_map.get(int(grp), str(grp)) if label_map is not None else str(grp)
            
            for var in variables:
                overall_freq = overall_data[var].value_counts(normalize=True, dropna=True)
                group_freq = data.loc[mask, var].value_counts(normalize=True, dropna=True)
                for category, p in overall_freq.items():
                    p_group = float(group_freq.get(category, 0.0))
                    if p == 0 or p == 1 or gsize == 0:
                        test_value = 0.0
                    else:
                        test_value = (p_group - p) / np.sqrt(p * (1 - p) / gsize)
                    results.append({
                        "Cluster_id": int(grp),
                        "Cluster": str(grp_label),
                        "Variable": var,
                        "Modalité": str(category),
                        "Group frequency": p_group,
                        "Overall frequency": float(p),
                        "TestValue": float(test_value),
                    })
        return pd.DataFrame(results)

    variables_to_characterize = feature_cols
    characterization_group_y = group_characterization(
        data=target_X,
        overall_data=Xw,  # comparaison vs global
        group_column="Cluster_id",
        variables=variables_to_characterize,
        label_map=label_map,
    )

    characterization_group_y = characterization_group_y.sort_values(
        ["Cluster_id", "TestValue"], ascending=[True, False]
    )

    # Seuil test-value
    st.markdown("##### Paramètre de filtrage des attributs")
    tv_min = st.slider("Seuil minimum TestValue", 0.0, 5.0, 2.0, 0.5)
    characterization_group_y = characterization_group_y[characterization_group_y["TestValue"] >= float(tv_min)]
    characterization_group_y = characterization_group_y.drop(columns=["Cluster_id"])

    st.markdown("##### Caractérisation des groupes (par TestValue)")
    profils_y_detailed = characterization_group_y.copy()
    st.session_state.profils_y_detailed = profils_y_detailed
    st.dataframe(st.session_state.profils_y_detailed)

    # Dominant/rare
    threshold = st.slider("Seuil fréquence discriminant attributs dominants/rares)", 0.0, 0.40, 0.10, 0.05)
    profils_y_simplified = profils_y_detailed.copy()
    profils_y_simplified["Importance de l'attribut"] = np.where(
        profils_y_simplified["Group frequency"] > float(threshold), "dominant", "rare"
    )
    profils_y_simplified = profils_y_simplified.drop(columns=["Group frequency", "Overall frequency", "TestValue"])
    st.session_state.profils_y_simplified = profils_y_simplified

    st.markdown("##### profils simplifiés pour le LLM")
    st.dataframe(st.session_state.profils_y_simplified)

    # ==========================================================
    # Étape 5 : Génération LLM (bouton)
    # ==========================================================
    st.subheader("Génération des profils")
    st.write("Utilisation de LLM pour générer des noms de profils et des justifications basées sur les attributs dominants/rares.")

    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Générer les noms de profils"):
            proceed = True
    
        if st.button("Effacer le texte généré"):
            st.session_state.profils_y_generated = False
            st.session_state.profils_y_text = ""
            st.rerun()

    if proceed:
        try:
            with st.spinner("Génération des profils par LLM en cours..."):
                df_y = st.session_state.get("profils_y_simplified", None)
                df_y_str = df_y.to_csv(index=False)
                dataset_context = st.session_state.get("dataset_context", "le contexte du dataset n'a pas été précisé.")

                system_msg = {
                    "role": "system",
                    "content": (
                        "Vous êtes un expert en analyse de données (marketing si questionnaires). "
                        "RÃˆGLE DE FORMAT STRICTE : votre réponse DOIT commencer par une section 'Objectif' sur une seule ligne.\n"
                        "Objectif : pour la modalité cible '{target_mod}' de la variable cible '{target_var}', voici les profils correspondants:\n"
                        "Ensuite, listez les profils, un par cluster, en commençant chaque profil par 'Profil <num> â€” <Nom>'. "
                        "Entre profils : une ligne vide. Aucun autre texte avant INTRO.\n\n"
                        "Données : dans profils_y, col1=cluster, col2=attribut, col3=importance (dominant/rare). "
                        "Pour chaque cluster : un nom (3â€“4 mots), + justification (2 phrases max) basée sur dominants (rares si besoin), "
                        "sans prénoms génériques, et terminez par effectif + fréquences (cible et population totale)."
                    ).format(
                        target_mod=st.session_state.target_mod,
                        target_var=st.session_state.target_variable
                    )
                }

                # On passe aussi counts/freq au LLM
                counts_freq = pd.DataFrame({
                    "Cluster": effectifs.index.astype(str),
                    "Effectif": effectifs.values.astype(int),
                    "Fréquence /cible": (effectifs / len(target_X) * 100).round(1).values,
                    "Fréquence / population totale": (effectifs / len(df) * 100).round(1).values                
                }).to_csv(index=False)

                user_msg = {
                    "role": "user",
                    "content": (
                        "Voici les données à analyser :\n"
                        f"- variable cible et modalité/segment : {st.session_state.target_variable} â†’ {st.session_state.target_mod}\n\n"
                        f"- profils associés à la cible (profils_y) :\n{df_y_str}\n\n"
                        f"- effectifs et fréquences par cluster (segment cible) :\n{counts_freq}\n\n"
                        "Générez pour chaque profil associé à la cible un nom représentatif et sa justification."
                    )
                }

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[system_msg, user_msg],
                    temperature=0,
                    top_p=1,
                    n=1,
                    seed=42,
                    max_tokens=2000
                )

                txt = response.choices[0].message.content or ""
                st.session_state.profils_y_generated = True
                st.session_state.profils_y_text = txt

        except Exception as e:
            st.error(f"Erreur d'appel à l'API OpenAI : {e}")

    if st.session_state.profils_y_text:
        st.markdown(f"##### Profils du segment cible ({st.session_state.target_variable}, {st.session_state.target_mod})")
        st.text_area("Résultat", st.session_state.profils_y_text, height=350)

    st.session_state["etape31_terminee"] = True
    st.success("Étape terminée. Tu peux relancer avec d'autres paramètres ou passer au suivant.")


