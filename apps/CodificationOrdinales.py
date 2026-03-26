import streamlit as st
import pandas as pd
import os
import re
import csv
import json
from openai import OpenAI
from core.df_registry import DFState, get_df, set_df
from utils import preparation_process

MODE_KEY = "__NAV_MODE__"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# --- Utilitaires
def parse_llm_two_cols(text: str) -> pd.DataFrame:
    """Parse un CSV (ou TSV) 2 colonnes (Variable, Ordinale) renvoyé par le LLM."""
    text = re.sub(r"^```[\w-]*\n|\n```$", "", text.strip())
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        # sauter un éventuel header
        if re.match(r'^\s*"?(variable)"?\s*[,|\t]\s*"?(ordinale)"?\s*$', line.strip(), flags=re.I):
            continue
        parts = line.split("\t") if "\t" in line else next(csv.reader([line], quotechar='"'))
        if len(parts) < 2:
            continue
        variable = ",".join(parts[:-1]).strip().strip('"')
        ordinale = parts[-1].strip().strip('"').lower()
        rows.append([variable, ordinale])
    df = pd.DataFrame(rows, columns=["Variable", "Ordinale"])
    df["Ordinale"] = df["Ordinale"].str.strip().str.lower()
    df = df[df["Ordinale"].isin(["oui", "non"])].reset_index(drop=True)
    return df


def init_ord_df_from_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remplit la table par défaut (mode manuel) : toutes les variables catégorielles en 'non'."""
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if df[c].nunique() < len(df)]
    return pd.DataFrame({
        "Variable": cat_cols,
        "Modalités": [", ".join(sorted(df[c].dropna().astype(str).unique())) for c in cat_cols],
        "Ordinale": ["non"] * len(cat_cols),
    })


def build_mod_df_default(df: pd.DataFrame, vars_ord: list[str]) -> pd.DataFrame:
    """Codes par défaut (mode manuel) : 0..n-1 par ordre alpha des modalités."""
    rows = []
    for var in vars_ord:
        modalities = sorted(df[var].dropna().astype(str).unique())
        for i, mod in enumerate(modalities):
            rows.append({"Variable": var, "Modalité": mod, "Code": i})
    return pd.DataFrame(rows, columns=["Variable", "Modalité", "Code"])


# --- App
def run():
    st.header("Encodage des variables ordinales")
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")

    # État global
    st.session_state.setdefault("ord_df", pd.DataFrame())      # sélection/détection variables ordinales
    st.session_state.setdefault("mod_df", pd.DataFrame())      # codification des modalités
    st.session_state.setdefault("df_encoded", None)
    st.session_state.setdefault("ord_validated", False)
    st.session_state.setdefault("etape15_terminee", False)    

    # Charger le dataset (doit être posé par l'étape précédente)
    df_ready = get_df(DFState.READY)
    if not isinstance(df_ready, pd.DataFrame):
        st.warning("Aucun dataset trouvé. Veuillez dâ€™abord passer par lâ€™application précédente.")
        st.stop()
    df = df_ready


    # =============== Étape 2 : Sélection des variables ordinales ===============
    st.subheader("Sélection des variables ordinales")

    # Initialiser la table (manuel par défaut) UNE SEULE FOIS
    ord_df = st.session_state.get("ord_df")
    if ord_df is not None and not ord_df.empty:
        st.write("Tableau des variables avec statut ordinal")
        st.dataframe(st.session_state.ord_df)
    
    else:
        st.session_state.ord_df = init_ord_df_from_dataset(df)
        
        options = ["Détection automatique par LLM", "Détection manuelle"]
        index_ord = 0 if mode == "automatique" else 1

        # Important : ne re-définissez plus index_ord en dessous
        method = st.radio(
            "Méthode :",
            options,
            index=index_ord,
            key="ord_method",
        )

        # Construire la requête pour le LLM
        with st.spinner("Détection automatique par LLM en cours..."):
            var_list = [
                {"Variable": c, "Valeurs_exemples": [str(v) for v in df[c].head(5)]}
                for c in df.columns
            ]
            
            system_msg = {
                "role": "system",
                "content": (
                    "Vous êtes un expert en analyse de données dâ€™enquêtes et de questionnaires. "
                    "Votre tâche consiste à déterminer si chaque variable dâ€™un jeu de données est **ordinale** ou non. "
                    "\n\n"
                    "Définition : une variable ordinale exprime un **ordre logique** entre ses modalités, "
                    "comme les niveaux de satisfaction, de fréquence ou dâ€™accord. "
                    "les valeurs quantitatives doivent pas être ignorées, en particulier les échelles de lickert déjà encodées en entiers: 1, 2, 3, 4, 5 etc. "
                    "Une variable nominale n’exprime **aucun ordre** entre ses valeurs. "
                    "\n\n"
                    "Exemples de variables ordinales :\n"
                    "- Satisfaction : 'Très insatisfait', 'Insatisfait', 'Satisfait', 'Très satisfait'\n"
                    "- Accord : 'Pas du tout d'accord', 'Plutôt pas d'accord', 'Plutôt d'accord', 'Tout à fait d'accord'\n"
                    "- Fréquence : 'Jamais', 'Rarement', 'Parfois', 'Souvent', 'Toujours'\n"
                    "- Niveau dâ€™éducation : 'Aucun diplôme', 'Secondaire', 'Licence', 'Master', 'Doctorat'\n\n"
                    "- booléen : 'Oui', 'Non'; 'positif', négatif'\n"
                    "- entiers: ne pas les sélectionner\n"
                    "Règle : en cas dâ€™ambiguïté, classer la variable comme **nominale**.\n\n"
                    "### Format de sortie STRICT :\n"
                    "- Retournez uniquement un CSV brut (UTF-8) sans texte explicatif ni balise.\n"
                    "- En-tête EXACTE : `Variable,Ordinale`\n"
                    "- Valeurs possibles pour chaque ligne : `oui` ou `non` (en minuscules)."
                )
            }

            user_msg = {
                "role": "user",
                "content": (
                    "Voici la liste des variables à analyser, au format JSON.\n"
                    "Pour chacune, indiquez si elle est ordinale ou non selon les consignes précédentes.\n"
                    "Retournez uniquement un CSV brut UTF-8 avec lâ€™en-tête EXACTE 'Variable,Ordinale' "
                    "et les valeurs 'oui' ou 'non' en minuscules.\n\n"
                    f"{json.dumps(var_list, ensure_ascii=False, indent=2)}"
                )
            }

            try:
                with st.spinner("Détection des variables ordinales par LLM en cours..."):
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[system_msg, user_msg],
                        temperature=0,
                        max_tokens=4000
                    )
                    raw = (resp.choices[0].message.content or "").strip()
                    ord_df_llm = parse_llm_two_cols(raw)
                    if ord_df_llm.empty:
                        st.error("Réponse LLM vide ou mal formée (aucune ligne 'oui'/'non').")
                    else:
                        # On peut enrichir avec la colonne Modalités pour lâ€™édition
                        modalites_map = {
                            c: ", ".join(sorted(df[c].dropna().astype(str).unique()))
                            for c in df.columns
                        }
                        ord_df_llm["Modalités"] = ord_df_llm["Variable"].map(modalites_map).fillna("")
                        # réordonner colonnes
                        ord_df_llm = ord_df_llm[["Variable", "Modalités", "Ordinale"]]
                        st.session_state.ord_df = ord_df_llm
                        st.success("âœ… Variables ordinales détectées automatiquement. Vous pouvez éditer ci-dessous.")
            except Exception as e:
                st.error(f"❌ Erreur API OpenAI : {e}")

    # Un seul éditeur pour lâ€™étape 2 (quel que soit le mode)
    st.caption("Modifiez si besoin : cochez 'oui' pour les variables ordinales.")
    ord_editor = st.data_editor(
        st.session_state.ord_df,
        hide_index=True,
        use_container_width=True,
        key="ord_editor",
        column_config={
            "Variable": st.column_config.Column(disabled=True),
            "Modalités": st.column_config.Column(disabled=True),
            "Ordinale": st.column_config.SelectboxColumn("Ordinale ?", options=["oui", "non"]),
        }
    )
    # Mettez à jour l'état à chaque rerun
    st.session_state.ord_df = ord_editor

    if st.button("Valider la sélection des variables ordinales"):
        st.session_state.ord_validated = True
        st.success("Sélection des variables ordinales validée.")

    # Bloque la suite proprement si rien à faire
    vars_ord = st.session_state.ord_df.loc[
        st.session_state.ord_df["Ordinale"].astype(str).str.strip().str.lower().eq("oui"),
        "Variable"
    ].tolist()

    # =============== Étape 3 : Codification des modalités ordinales ===============
    st.subheader("Codification des modalités ordinales")
    
    ordinal_codification_mapping = st.session_state.get("ordinal_codification_mapping")
    if ordinal_codification_mapping is not None and not ordinal_codification_mapping.empty:
        st.dataframe(st.session_state.ordinal_codification_mapping)

    if not vars_ord:
        st.info("Aucune variable ordinale sélectionnée.")
    else:
        # Initialiser la table de codification UNE SEULE FOIS (manuel par défaut)
        if st.session_state.mod_df.empty:
            st.session_state.mod_df = build_mod_df_default(df, vars_ord)
        else:
            # Si la sélection de variables a changé, on recalcule un défaut minimal pour les nouvelles variables
            existing_vars = set(st.session_state.mod_df["Variable"].unique())
            missing = [v for v in vars_ord if v not in existing_vars]
            if missing:
                add_df = build_mod_df_default(df, missing)
                st.session_state.mod_df = pd.concat([st.session_state.mod_df, add_df], ignore_index=True)

            options = ["Codification automatique par LLM", "Codification manuelle"]
            index_ord = 0 if mode == "automatique" else 1
            method = st.radio(
                "Méthode :",
                options,
                index=index_ord,
                key="cod_method",
            )

        try:
            with st.spinner("Codification des variables ordinales par LLM en cours..."):
                mod_list = [
                    {"Variable": v, "Modalités": sorted(df[v].dropna().astype(str).unique())}
                    for v in vars_ord
                ]

                sys_prompt = (
                    "Vous êtes un expert en analyse de données dâ€™enquêtes et de questionnaires. "
                    "Votre tâche est de coder chaque modalité dâ€™une variable ordinale en un entier, "
                    "en respectant strictement lâ€™ordre logique de ses niveaux de réponse.\n\n"
                    "### Règles générales de codification :\n"
                    "- Ordonnez les modalités du plus faible au plus fort selon leur signification.\n"
                    "- Distinguez deux types de variables ordinales :\n"
                    "  1) Variables évaluatives (opinions, satisfaction, accord, jugement, qualité, fréquence perçue, etc.).\n"
                    "  2) Variables de niveau objectif (niveau dâ€™études, catégorie de revenu, intensité croissante sans connotation positive/négative claire, etc.).\n\n"
                    "### Cas 1 : variables évaluatives (opinions, satisfaction, accord, etc.)\n"
                    "- Utilisez des codes entiers consécutifs centrés autour de 0 :\n"
                    "  â€¢ modalités négatives â†’ nombres négatifs (du plus négatif vers 0)\n"
                    "  â€¢ modalités positives â†’ nombres positifs (de 1 vers le plus grand)\n"
                    "  â€¢ modalités neutres ou indéterminées â†’ 0\n"
                    "- Même sâ€™il nâ€™existe pas de modalité strictement neutre, répartissez les modalités négatives en valeurs négatives "
                    "  et les modalités positives en valeurs positives, de manière aussi symétrique que possible autour de 0.\n\n"
                    "### Cas 2 : variables de niveau objectif (diplôme, catégorie, niveau, etc.)\n"
                    "- Utilisez des codes entiers consécutifs à partir de 0 : 0, 1, 2, 3, ...\n"
                    "- La modalité la plus faible reçoit 0, la suivante 1, etc.\n\n"
                    "### Modalités ambiguës ou non informatives\n"
                    "- Les modalités ambiguës ou non informatives (ex. 'NSP', 'Ne sait pas', 'Sans objet', "
                    "'Refus', 'Autre', 'N/A', 'NAN', 'peut-être') doivent toujours être codées à 0.\n"
                    "- Ces modalités NE doivent JAMAIS être utilisées pour définir lâ€™ordre ordinale principal.\n"
                    "- Les autres modalités doivent être ordonnées et codées indépendamment de ces modalités ambiguës.\n\n"
                    "### Exemples :\n"
                    "- 'Très insatisfait', 'Insatisfait', 'Satisfait', 'Très satisfait', 'Ne sait pas' "
                    "→ -2, -1, 1, 2, 0\n"
                    "- 'Pas du tout dâ€™accord', 'Plutôt pas dâ€™accord', 'Plutôt dâ€™accord', 'Tout à fait dâ€™accord', 'NAN' "
                    "→ -2, -1, 1, 2, 0\n"
                    "- 'Jamais', 'Rarement', 'Parfois', 'Souvent', 'Toujours' "
                    "→ -2, -1, 0, 1, 2\n"
                    "- 'Aucun diplôme', 'Secondaire', 'Licence', 'Master', 'Doctorat' "
                    "→ 0, 1, 2, 3, 4\n"
                    "- 'Non', 'peut-être', 'Oui' â†’ -1, 0, 1\n"
                    "- 'Plutôt pas dâ€™accord', 'Plutôt dâ€™accord', 'Tout à fait dâ€™accord', 'Autre' "
                    "→ -1, 1, 2, 0\n\n"
                    "### Format de sortie STRICT :\n"
                    "- Retournez UNIQUEMENT du JSON valide UTF-8, sans texte ni balise.\n"
                    "- Format : liste d’objets sous la forme exacte :\n"
                    '  [{\"Variable\": \"...\", \"Modalité\": \"...\", \"Code\": nombre}, ...]\n'
                    "- Conservez exactement les libellés tels quâ€™ils apparaissent dans les données (respectez la casse et lâ€™orthographe).\n"
                )

                user_prompt = (
                    "Voici la liste des variables ordinales et leurs modalités. "
                    "Appliquez les règles de codification précédentes et retournez uniquement le JSON demandé.\n\n"
                    f"{json.dumps(mod_list, ensure_ascii=False, indent=2)}"
    )

                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_prompt}],
                    temperature=0,
                    max_tokens=4000
                )
                content = (resp.choices[0].message.content or "").strip()
                content = re.sub(r"^```json\n|\n```$", "", content)
                list_mods = json.loads(content)
                mod_df_llm = pd.DataFrame(list_mods)
                exp_cols = {"Variable", "Modalité", "Code"}
                if not exp_cols.issubset(mod_df_llm.columns):
                    st.error(f"Colonnes manquantes dans la réponse LLM : {mod_df_llm.columns.tolist()}")
                else:
                    mod_df_llm["Code"] = pd.to_numeric(mod_df_llm["Code"], errors="raise").astype("Int64")
                    st.session_state.mod_df = mod_df_llm
                    st.success("âœ… Codification générée automatiquement. Vous pouvez éditer ci-dessous.")
        except Exception as e:
            st.error(f"Erreur API OpenAI : {e}")

        # ðŸ‘‰ Un seul éditeur pour lâ€™étape 3 (quel que soit le mode)
        st.caption("Éditez la codification si nécessaire (les codes doivent être des entiers).")
        mod_editor = st.data_editor(
            st.session_state.mod_df,
            hide_index=True,
            use_container_width=True,
            key="mod_editor",
            column_config={
                "Variable": st.column_config.Column(disabled=True),
                "Modalité": st.column_config.Column(disabled=True),
                "Code": st.column_config.NumberColumn("Code", step=1),
            }
        )
        st.session_state.ordinal_codification_mapping = mod_editor

    # =============== Application de la codification ===============
    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        st.button("Appliquer les codifications et afficher le dataset")
        proceed = True
        
    if proceed:
        try:
            df_encoded = df_ready.copy()
            if st.session_state.mod_df.empty:
                st.info("Aucune codification à appliquer.")
                st.session_state["etape15_terminee"] = True

            for var in st.session_state.mod_df["Variable"].unique():
                mapping_df = st.session_state.mod_df.query("Variable == @var")
                mapping = dict(zip(mapping_df["Modalité"], mapping_df["Code"]))
                # map uniquement si la colonne existe
                if var in df_encoded.columns:
                    df_encoded[var] = df_encoded[var].astype(str).map(mapping).astype("Int64")

            st.success("Codification appliquée.")
            st.subheader("Dataset final avec codification ordinale")
            st.dataframe(df_encoded)
            csv_bytes = df_encoded.to_csv(index=False).encode("latin-1", errors="replace")
            st.download_button(
                "Télécharger le dataset",
                data=csv_bytes,
                file_name="df_ordinal_encoded.csv",
                mime="text/csv"
            )
            set_df(DFState.ENCODED, df_encoded, step_name="CodificationOrdinales")
        except Exception as e:
            st.error(f"Erreur lors de l’application : {e}")

    # update du tableau de process
    nb_vars_ord = len(vars_ord)
    nb_ordinales_codified = len(st.session_state.mod_df) if "mod_df" in st.session_state else 0

    action = f"{nb_vars_ord} variables ordinales encodées, {nb_ordinales_codified} modalités encodées."
    st.success(action)

    # attention : df_encoded peut lui aussi ne pas exister si aucune codification n'a été appliquée
    df_encoded = get_df(DFState.ENCODED)
    if isinstance(df_encoded, pd.DataFrame):
        preparation_process(df_encoded, action)
    
    st.markdown("##### État d'avancement de la préparation du dataset :")
    st.dataframe(st.session_state.process)
    
    st.session_state["etape15_terminee"] = True
    st.success("Étape terminée âœ…. Vous pouvez lancer lâ€™application suivante.")





