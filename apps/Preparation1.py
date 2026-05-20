import streamlit as st
import pandas as pd
import os
from openai import OpenAI
import json
from typing import Dict, Tuple

from core.df_registry import DFState, set_df
from core.preparation_details import refresh_preparation_details_payload
from core.preparation_diagnostics import remove_preparation_diagnostic, set_preparation_diagnostic
from utils import preparation_process


MODE_KEY = "__NAV_MODE__"  # toujours lu depuis la main app

# Client OpenAI (clé lue depuis la variable d'environnement OPENAI_API_KEY)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def has_uploaded_dataset() -> bool:
    return isinstance(st.session_state.get("df"), pd.DataFrame)


def init_preparation_state() -> None:
    st.session_state.setdefault("etape1_terminee", False)
    st.session_state.setdefault("__UPLOAD_NONCE__", 0)
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("variables_list_validated", False)
    st.session_state.setdefault("df_selected", None)


# ----------- Helpers LLM : préparation de l'input ----------- #

def build_type_analysis_input(
    df: pd.DataFrame,
    max_rows: int = 50,
    max_chars: int = 200
) -> dict:
    """
    Construit un objet JSON compact pour le LLM à partir du DataFrame :
    - dtypes pandas
    - nb de modalités distinctes
    - 50 lignes max (head + sample) converties en chaînes.
    """

    n_rows = len(df)
    if n_rows <= max_rows:
        sample_df = df.copy()
    else:
        head_n = max_rows // 2
        head_part = df.head(head_n)
        remaining = max_rows - len(head_part)
        sample_part = df.sample(remaining, random_state=0)
        sample_df = pd.concat([head_part, sample_part], axis=0)
        sample_df = sample_df.drop_duplicates().head(max_rows)

    # On convertit tout en string (pour être JSON-serializable) et on tronque
    sample_for_llm = sample_df.copy()
    for col in sample_for_llm.columns:
        # astype("string") -> pandas StringDtype, puis tronque
        sample_for_llm[col] = (
            sample_for_llm[col]
            .astype("string")
            .str.slice(0, max_chars)
        )

    analysis_input = {
        "n_rows_total": int(n_rows),
        "columns": []
    }

    nunique = df.nunique(dropna=False)

    for col in df.columns:
        analysis_input["columns"].append(
            {
                "name": col,
                "pandas_dtype": str(df[col].dtype),
                "n_unique": int(nunique[col]),
                "sample_values": sample_for_llm[col].dropna().tolist(),
            }
        )

    # LLM reçoit :
    # - meta colonnes (dtype + nunique)
    # - valeurs d'échantillon par colonne
    return analysis_input


# ----------- Helpers LLM : appel API et parsing ----------- #

def infer_types_with_llm(
    df: pd.DataFrame,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """
    Appelle l'API OpenAI pour :
    - déterminer un type de base pour chaque variable : "integer" | "float" | "object"
    - produire des types sémantiques et formats (dates, géo, texte long, identifiants, etc.)

    Retourne :
    - forced_types: dict {col -> "integer"/"float"/"object"}
    - df_semantic_types: DataFrame avec colonnes:
        ["name", "semantic_type", "format", "base_dtype", "issues"]
    """

    analysis_input = build_type_analysis_input(df)
    analysis_json_str = json.dumps(analysis_input, ensure_ascii=False)

    system_prompt = (
        "You are a senior data scientist and data engineer. "
        "Your job is to inspect tabular dataset samples and determine accurate data types "
        "for each column.\n\n"
        "You MUST answer in pure JSON, no explanations, no extra keys."
    )

    # On décrit exactement ce qu'on veut
    user_prompt = f"""
    You are given metadata and sample values for a pandas DataFrame.
    Your goal is to infer column types.

    The base dtypes MUST be one of these 3 strings only:
    - "integer"
    - "float"
    - "object"

    You must also infer a more semantic type and an optional specific format.

    Semantic types examples (not exhaustive):
    - "identifier" (IDs, UUID, codes like customer_id, transaction_id, etc.)
    - "datetime" (date + time)
    - "date" (dates only)
    - "time" (times only)
    - "latitude" / "longitude"
    - "geo_point" (lat+lon in the same field)
    - "postal_code" / "zip_code"
    - "city_name" / "region_name" / "country_name"
    - "long_text" (free text, descriptions, sentences, often unique)
    - "categorical" (few distinct values, short labels)
    - "categorical_numeric" (numbers used as labels/categories)
    - "boolean" (yes/no, true/false, 0/1)
    - "numeric_measure" (quantitative variable, measure, amount)
    - etc.

    For datetime/date/time you should infer a conventional string format if possible, for example:
    - "YYYY-MM-DD" (ISO 8601 date)
    - "YYYY-MM-DD HH:MM:SS" (ISO 8601 datetime)
    - "DD/MM/YYYY"
    - etc.

    IMPORTANT:
    - Some columns may be mis-typed in pandas (e.g. quantitative column with spaces -> pandas dtype = object). You must correct this.
    - If the column is mostly numeric with a few errors, you should still set base_dtype to "integer" or "float" and note the issue.
    - If a column looks like an identifier (unique values, no clear numeric semantics), use semantic_type = "identifier".
    - If you are not sure about semantic type, use a reasonable guess like "categorical" or "numeric_measure".

    You MUST return a single JSON object with this exact structure:

    {{
    "forced_types": {{
        "col_name_1": "integer | float | object",
        "col_name_2": "integer | float | object",
        "...": "..."
    }},
    "columns_metadata": [
        {{
        "name": "col_name_1",
        "base_dtype": "integer | float | object",
        "semantic_type": "e.g. identifier, datetime, postal_code, long_text, ...",
        "format": "specific format string or null",
        "issues": "short note about detected issues or null"
        }},
        {{
        "name": "col_name_2",
        "base_dtype": "integer | float | object",
        "semantic_type": "...",
        "format": "...",
        "issues": "... or null"
        }}
    ]
    }}

    The string "JSON" appears here only to satisfy the requirement: you must output STRICTLY VALID JSON according to this schema, with double quotes and no trailing commas.

    Here is the input data (metadata + samples) as JSON:

    {analysis_json_str}
    """

    # Appel API en JSON mode
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    # Le texte de sortie est une chaîne JSON
    raw_json = response.choices[0].message.content
    data = json.loads(raw_json)

    forced_types = data.get("forced_types", {}) or {}
    columns_metadata = data.get("columns_metadata", []) or []

    # DataFrame des types sémantiques
    df_semantic = pd.DataFrame(columns_metadata)

    # On s'assure de l'ordre et des colonnes
    expected_cols = ["name", "semantic_type", "format", "base_dtype", "issues"]
    for c in expected_cols:
        if c not in df_semantic.columns:
            df_semantic[c] = None
    df_semantic = df_semantic[expected_cols]

    # Sanity check : pour chaque colonne absente dans forced_types, on met un fallback
    for col in df.columns:
        if col not in forced_types:
            # fallback simple sur pandas dtype
            dt = str(df[col].dtype)
            if "int" in dt:
                forced_types[col] = "integer"
            elif "float" in dt:
                forced_types[col] = "float"
            else:
                forced_types[col] = "object"

    return forced_types, df_semantic


def diagnose_semantic_types(
    df_semantic: pd.DataFrame | None,
    df_source: pd.DataFrame | None = None,
) -> list[dict]:
    if not isinstance(df_semantic, pd.DataFrame) or df_semantic.empty:
        return []

    semantic_df = df_semantic.copy()
    for col in ["name", "semantic_type", "format", "base_dtype", "issues"]:
        if col not in semantic_df.columns:
            semantic_df[col] = None

    semantic_df["name"] = semantic_df["name"].astype(str)
    semantic_df["semantic_type"] = semantic_df["semantic_type"].fillna("").astype(str).str.lower()

    specs = [
        (
            "semantic_dates",
            "Colonnes de dates détectées",
            {"date", "datetime", "time"},
            False,
        ),
        (
            "semantic_geo",
            "Colonnes géographiques détectées",
            {"latitude", "longitude", "geo_point", "postal_code", "zip_code", "city_name", "region_name", "country_name"},
            False,
        ),
        (
            "semantic_identifiers",
            "Colonnes identifiants détectées",
            {"identifier"},
            False,
        ),
        (
            "semantic_verbatim",
            "Colonnes de verbatims détectées",
            {"long_text"},
            True,
        ),
    ]

    diagnostics: list[dict] = []
    for diag_id, label, semantic_types, available in specs:
        subset = semantic_df[semantic_df["semantic_type"].isin(semantic_types)].copy()
        if diag_id == "semantic_verbatim" and isinstance(df_source, pd.DataFrame) and not df_source.empty:
            from .VerbatimSummary import detect_long_text_columns

            heuristic_candidates, _ = detect_long_text_columns(
                df_source,
                min_avg_len=50,
                min_unique_ratio=0.7,
            )
            subset = subset[subset["name"].isin([str(col) for col in heuristic_candidates])]
        if subset.empty:
            continue

        columns = subset["name"].tolist()
        formats = sorted(
            {
                str(v).strip()
                for v in subset["format"].dropna().tolist()
                if str(v).strip()
            }
        )
        reason = f"{len(columns)} colonne(s) détectée(s)"
        if formats:
            reason += f" ({', '.join(formats[:3])})"

        diagnostics.append(
            {
                "id": diag_id,
                "label": label,
                "needed": True,
                "reason": reason,
                "details": {
                    "columns": columns,
                    "formats": formats,
                    "semantic_types": sorted(semantic_types),
                },
                "compute_module": "VerbatimSummary" if diag_id == "semantic_verbatim" else None,
                "render_module": "Preparation1",
                "available": available,
            }
        )

    return diagnostics


def store_preparation1_diagnostics(diagnostics: list[dict]) -> None:
    semantic_ids = {
        "semantic_dates",
        "semantic_geo",
        "semantic_identifiers",
        "semantic_verbatim",
    }
    for diag_id in semantic_ids:
        remove_preparation_diagnostic(diag_id)

    for diagnostic in diagnostics:
        set_preparation_diagnostic(diagnostic)


# ----------- Streamlit main step ----------- #

def run_preparation_only():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    silent_after_upload = bool(st.session_state.get("__PIPELINE_SILENT__", False)) or (mode == "automatique")

    init_preparation_state()
    if st.session_state.df is None:
        return

    # Mode silencieux avant DiagnosticGlobal: calcul sans affichages intermediaires.
    if silent_after_upload:
        df0 = st.session_state.df.copy()
        n_obs0, n_var0 = df0.shape

        if "process" not in st.session_state or st.session_state.process is None:
            st.session_state.process = pd.DataFrame(
                columns=["Etape", "Nb observations", "Nb variables", "Traitement"]
            )
            st.session_state.process = pd.concat(
                [st.session_state.process, pd.DataFrame([{
                    "Etape": "Dataset initial",
                    "Nb observations": n_obs0,
                    "Nb variables": n_var0,
                    "Traitement": "-",
                }])],
                ignore_index=True
            )
            st.session_state.prep_step = 1
        elif "prep_step" not in st.session_state:
            st.session_state.prep_step = max(1, int(len(st.session_state.process)))

        if st.session_state.get("download_sampled"):
            process_df = st.session_state.get("process")
            process_text = (
                process_df["Traitement"].astype(str).str.lower().tolist()
                if isinstance(process_df, pd.DataFrame) and "Traitement" in process_df.columns
                else []
            )
            if not any("echantillonnage" in text or "échantillonnage" in text for text in process_text):
                preparation_process(
                    df0,
                    f"Echantillonnage alÃ©atoire Ã  {st.session_state.sample_size} lignes lors du tÃ©lÃ©chargement du fichier.",
                )

        if df0.shape[1] > st.session_state.columns_number:
            st.session_state["etape1_terminee"] = False
            return

        df = df0.drop_duplicates()
        if df.shape[0] > st.session_state.sample_size:
            df = df.sample(n=st.session_state.sample_size, random_state=42)
            preparation_process(df, f"Echantillonnage a {st.session_state.sample_size} lignes.")

        missing_columns = df.columns[df.isna().all()].tolist()
        if missing_columns:
            df = df.drop(columns=missing_columns)
            preparation_process(df, f"Suppression colonnes 100% NA ({len(missing_columns)}).")

        num_missing_rows = df.isna().all(axis=1).sum()
        if num_missing_rows > 0:
            df = df[~df.isna().all(axis=1)]
            preparation_process(df, f"Suppression lignes 100% NA ({num_missing_rows}).")

        unique_counts = df.nunique()
        low_var_cols = unique_counts[unique_counts <= 1].index.tolist()
        df_neat = df.drop(columns=low_var_cols) if low_var_cols else df.copy()
        if low_var_cols:
            preparation_process(df_neat, f"Suppression colonnes mono-modalite ({len(low_var_cols)}).")

        st.session_state.df_neat = df_neat
        st.session_state.df_selected = df_neat.copy()
        df_types = st.session_state.df_selected.copy()

        try:
            forced_types, df_semantic = infer_types_with_llm(df_types)
        except Exception:
            forced_types = {}
            for col in df_types.columns:
                dt = str(df_types[col].dtype)
                if "int" in dt:
                    forced_types[col] = "integer"
                elif "float" in dt:
                    forced_types[col] = "float"
                else:
                    forced_types[col] = "object"
            df_semantic = pd.DataFrame(
                {
                    "name": df_types.columns,
                    "semantic_type": ["unknown"] * len(df_types.columns),
                    "format": [None] * len(df_types.columns),
                    "base_dtype": [forced_types[c] for c in df_types.columns],
                    "issues": [f"fallback from pandas dtype ({str(df_types[c].dtype)})" for c in df_types.columns],
                }
            )

        st.session_state.forced_types = forced_types
        st.session_state.df_semantic_types = df_semantic
        store_preparation1_diagnostics(diagnose_semantic_types(df_semantic, df_source=df_types))

        df_raw = df_types.copy()
        for col, target_type in forced_types.items():
            try:
                if target_type == "integer":
                    df_raw[col] = pd.to_numeric(df_types[col], errors="coerce").astype("Int64")
                elif target_type == "float":
                    df_raw[col] = pd.to_numeric(df_types[col], errors="coerce")
                else:
                    df_raw[col] = df_types[col].astype("object")
            except Exception:
                pass

        set_df(DFState.RAW, df_raw, step_name="Preparation1")
        st.session_state.message_type_validated = "Types de variables detectes automatiquement"
        st.session_state["etape1_terminee"] = True
        refresh_preparation_details_payload()
        return


    # -------- 2) Nettoyage préliminaire --------
    st.markdown("##### Nettoyage préliminaire")
    df0 = st.session_state.df.copy()

    # Stats avant nettoyage
    n_obs0, n_var0 = df0.shape
    dim0 = (n_var0 / n_obs0) if n_obs0 else 0.0

    st.markdown("##### Statistiques générales avant nettoyage")
    st.write(f"Ce jeu de données comporte {n_obs0} observations et {n_var0} colonnes.")
    st.write(f"- Dimensionalité : {dim0:.4f}")
    st.write(f"- Duplicats : {df0.duplicated().sum()} ({df0.duplicated().mean():.2%})")

    # Initialiser le process UNE SEULE FOIS dans session_state
    if "process" not in st.session_state or st.session_state.process is None:
        st.session_state.process = pd.DataFrame(
            columns=["Etape", "Nb observations", "Nb variables", "Traitement"]
        )
        st.session_state.process = pd.concat(
            [st.session_state.process, pd.DataFrame([{
                "Etape": "Dataset initial",
                "Nb observations": n_obs0,
                "Nb variables": n_var0,
                "Traitement": "-",
            }])],
            ignore_index=True
        )
        st.session_state.prep_step = 1
    elif "prep_step" not in st.session_state:
        st.session_state.prep_step = max(1, int(len(st.session_state.process)))

    # test du nombre de colonnes
    if df0.shape[1] > st.session_state.columns_number:
        st.warning(
            f"Le nombre de colonnes ({df0.shape[1]}) depasse le maximum autorise "
            f"({st.session_state.columns_number}). Avant de continuer, veuillez reduire "
            "le nombre de colonnes ou augmenter le maximum autorise."
        )
        st.stop()

    # Déduplication
    df = df0.drop_duplicates()

    # Slider d'échantillonnage
    st.markdown("##### Taille de l'échantillon")

    if df.shape[0] > st.session_state.sample_size:
        df = df.sample(n=st.session_state.sample_size, random_state=42)
        action1 = f"Le nombre de lignes a été réduit à {st.session_state.sample_size} par échantillonnage aléatoire."
        st.write(action1)
        preparation_process(df, action1)

    # Colonnes 100% NA
    missing_columns = df.columns[df.isna().all()].tolist()
    if missing_columns:
        df = df.drop(columns=missing_columns)
        nb_missing_columns = len(missing_columns)
        action2 = (
            f"{nb_missing_columns} colonnes avec 100% de valeurs manquantes supprimées :\n"
            + "- " + "\n- ".join(map(str, missing_columns))
        )
        st.write(action2)
        preparation_process(df, action2)

    # Lignes 100% NA
    num_missing_rows = df.isna().all(axis=1).sum()
    if num_missing_rows > 0:
        df = df[~df.isna().all(axis=1)]
        action3 = f"{num_missing_rows} observations 100% NA supprimées."
        st.write(action3)
        preparation_process(df, action3)

    # Colonnes de faible variance (<= 1 modalité)
    unique_counts = df.nunique()
    low_var_cols = unique_counts[unique_counts <= 1].index.tolist()

    df_neat = df.copy()
    if low_var_cols:
        df_neat = df_neat.drop(columns=low_var_cols)
        action4 = (
            f"{low_var_cols} colonnes avec une seule modalité supprimées:\n"
            + "- " + "\n- ".join(map(str, low_var_cols))
        )
        st.write(action4)
        preparation_process(df, action4)

    # On fige la base de nettoyage
    st.session_state.df_neat = df_neat

    st.write("##### Aperçu après préparation 1 :", df_neat.head())

    if st.session_state["etape1_terminee"] == True:
        st.markdown("##### Types de variables détectés")
        st.dataframe(st.session_state.df_semantic_types)

    # -------- 3) Selection des variables (supprimee) --------
    # Phase 3: toutes les colonnes nettoyees sont conservees automatiquement.
    st.markdown("##### Selection des variables")
    st.info("Selection manuelle supprimee: toutes les colonnes sont conservees.")

    df_selected = df_neat.copy()
    st.session_state.df_selected = df_selected
    st.dataframe(df_selected)

    # -------- 5) Reconnaissance automatique des types (LLM) --------
    st.markdown("##### Reconnaissance des types de variables")

    df_types = st.session_state.df_selected.copy()

    with st.spinner("Reconnaissance des types de variables par LLM en cours..."):
        try:
            forced_types, df_semantic = infer_types_with_llm(df_types)
        except Exception as e:
            st.error(f"Erreur lors de l'appel au modèle LLM : {e}")
            # Fallback simple : mapping basique des dtypes pandas
            forced_types = {}
            for col in df_types.columns:
                dt = str(df_types[col].dtype)
                if "int" in dt:
                    forced_types[col] = "integer"
                elif "float" in dt:
                    forced_types[col] = "float"
                else:
                    forced_types[col] = "object"
            df_semantic = pd.DataFrame(
                {
                    "name": df_types.columns,
                    "semantic_type": ["unknown"] * len(df_types.columns),
                    "format": [None] * len(df_types.columns),
                    "base_dtype": [forced_types[c] for c in df_types.columns],
                    "issues": [f"fallback from pandas dtype ({str(df_types[c].dtype)})" for c in df_types.columns],
                }
            )

    st.session_state.forced_types = forced_types
    st.session_state.df_semantic_types = df_semantic
    store_preparation1_diagnostics(diagnose_semantic_types(df_semantic, df_source=df_types))

    st.markdown("##### Types de variables détectés par le modèle")
    st.dataframe(df_semantic)

    # Application des dtypes de base (integer/float/object)
    df_raw = df_types.copy()
    for col, target_type in forced_types.items():
        try:
            if target_type == "integer":
                # nullable Int64 pour gérer les NaN
                df_raw[col] = pd.to_numeric(df_types[col], errors="coerce").astype("Int64")
            elif target_type == "float":
                df_raw[col] = pd.to_numeric(df_types[col], errors="coerce")
            else:
                df_raw[col] = df_types[col].astype("object")
        except Exception as e:
            st.warning(f"Conversion échouée pour {col} vers {target_type} : {e}")

    # Sauvegardes finales
    set_df(DFState.RAW, df_raw, step_name="Preparation1")
    st.session_state.message_type_validated = "Types de variables détectés automatiquement"
    st.success(st.session_state.message_type_validated)
    st.write("Aperçu final :", df_raw.head())
    refresh_preparation_details_payload()

    csv = df_raw.to_csv(index=False, sep=';', encoding='latin-1')
    st.download_button(
        label="Télécharger le jeu de données",
        data=csv,
        file_name="df_raw.csv",
        mime="text/csv"
    )

    st.markdown("##### Etat d'avancement de la preparation du dataset :")
    st.dataframe(st.session_state.process)

    st.session_state["etape1_terminee"] = True
    if st.session_state["etape1_terminee"]:
        st.success("Vous pouvez lancer la prochaine etape dans le menu a gauche : Diagnostic global.")
        return


def run():
    run_preparation_only()



