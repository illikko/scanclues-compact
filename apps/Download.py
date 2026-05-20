import streamlit as st
import pandas as pd

from . import Preparation1
from utils import preparation_process


MAX_FILE_SIZE_MB = 30


def has_uploaded_dataset() -> bool:
    return Preparation1.has_uploaded_dataset()


def should_show_progress() -> bool:
    if has_uploaded_dataset():
        return True
    if st.session_state.get("upload_validation_message"):
        return False
    upload_key = f"upload_file_{st.session_state.get('__UPLOAD_NONCE__', 0)}"
    return st.session_state.get(upload_key) is not None


def _block_upload(message: str) -> None:
    st.session_state["upload_validation_message"] = message
    st.session_state["df"] = None
    st.session_state["etape1_terminee"] = False
    st.session_state["__UPLOAD_NONCE__"] = int(st.session_state.get("__UPLOAD_NONCE__", 0)) + 1
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def _render_intro() -> None:
    st.write("L'application scanClues permet de préparer et analyser vos jeux de données tabulaires. Une présentation est fournie ci-dessous.")

    with st.expander("A propos de l'application", expanded=False):
        st.write(
            """
            L'application scanClues permet de traiter les jeux de données tabulaires (fichiers CSV, excel), plus particulièrement:
            - de préparer le jeu de données brut : nettoyage (données manquantes, anormales, doublons,...), et de l'enrichir (textes, dates, géolocalisation,...)
            - d'analyser la distribution et la relation entre les variables
            - d'extraire des insights et des recommandations: profiling sur tout le jeu de données et sur une cible, de mesurer pour chaque segment les actions qui ont le plus d'impact sur la cible.
            - de fournir un rapport dans l'appli et exportable sous forme de fichiers HTML et csvs qui résument les analyses et insights extraits.
            - de poser des questions sur l'analyse réalisée

            L'utilisateur doit intervenir à plusieurs étapes pour:
            0- vérifier que le jeu de données correspond au format attendu (voir plus bas)
            1- upload : télécharger le jeu de données
            2- sélectionner son objectif: préparation du jeu de données, et le profilage, et l'analyse descriptive et en option (avec ou sans brief) définir ce qu'il cherche
            3- lire le rapport dans l'application et/ou la télécharger (fichiers HTML et CSVs)
            4- poser des questions

            Les traitements en cours d'exécution sont affichés sous le menu: module (parmi les 34) et la fonction (parmi une centaine).
            L'icône en haut à droite indique si un traitement est en cours: il faut attendre 1 minute entre les étapes 1- et 2-, puis plusieurs minutes entre 2- et 3- suivant les traitements demandés et la richesse du jeu de données.

            L'application est particulièrement adaptée aux cas d'usage suivants : enquÃªtes, analyses marketing (CRM, web analytics...), RH (satisfaction, attrition...), open data, etc.

            Les formats de fichiers supportés sont:
            - CSV (séparateur point-virgule) .csv et Excel .xlsx
            - Le nom des champs doit Ãªtre sur la 1ère ligne
            - pas de défaut majeur : saut de ligne, décalage de colonnes,...

            L'application utilise les meilleurs modèles d'IA, LLM et statistiques, pour automatiser l'analyse des données tabulaires.
            C'est la version alpha, sur un mode "work in progress".

            Des cas d'usage sont présentés à cette adresse : https://www.scanclues.com/#cas-d-usage
            """
        )


def _render_upload_controls() -> None:
    st.subheader("Upload")
    validation_message = st.session_state.pop("upload_validation_message", None)
    if validation_message:
        st.warning(validation_message)
    with st.expander("Paramètres pour la préparation préliminaire", expanded=False):
        st.slider(
            "Nombre maximal d'observations à conserver après nettoyage",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=500,
            key="sample_size",
        )
        st.slider("Nombre maximal de colonnes", min_value=100, max_value=1500, value=300, key="columns_number")
        st.slider("Nombre max. de caractères pour les noms de colonnes", min_value=5, max_value=120, value=50, key="max_chars")

    st.markdown("#### Téléchargement du fichier")

    encoding_choice = st.selectbox(
        "Choisissez l'encodage du fichier (si vous ne savez pas, laissez latin-1)",
        options=["latin-1", "utf-8", "utf-16", "cp1252"],
        index=0,
    )

    uploaded_file = st.file_uploader(
        "Téléchargez le fichier (.csv ou .xlsx)",
        type=["csv", "xls", "xlsx"],
        key=f"upload_file_{st.session_state.get('__UPLOAD_NONCE__', 0)}",
    )

    if uploaded_file is not None:
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            _block_upload(f"Fichier trop volumineux ({size_mb:.1f} MB). Limite : {MAX_FILE_SIZE_MB} MB.")

    if uploaded_file:
        try:
            if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep=";", encoding=encoding_choice)
            sampled = False
            original_rows = df.shape[0]

            if df.shape[0] > st.session_state.sample_size:
                original_rows = df.shape[0]
                sampled = True
                df = df.sample(n=st.session_state.sample_size, random_state=42)
                st.info(
                    f"Le nombre d'observations ({original_rows}) dépasse la limite définie. "
                    f"Un échantillon aléatoire de {st.session_state.sample_size} lignes a été conservé pour réaliser l'analyse."
                )
            if False and df.shape[0] > st.session_state.sample_size:
                _block_upload(
                    f"Le nombre d'observations ({df.shape[0]}) dépasse le maximum autorisé."
                    f"pour la préparation préliminaire ({st.session_state.sample_size}). "
                    "Veuillez réduire le fichier ou augmenter le seuil avant de continuer."
                )

            if df.shape[1] > st.session_state.columns_number:
                _block_upload(
                    f"Le nombre de colonnes ({df.shape[1]}) dépasse le maximum autorisé."
                    f"({st.session_state.columns_number}). Veuillez réduire le fichier "
                    "ou augmenter le seuil avant de continuer."
                )
            if sampled:
                preparation_process(
                    df,
                    f"Echantillonnage aléatoire à {st.session_state.sample_size} lignes lors du téléchargement du fichier.",
                )

            st.session_state.df = df
            st.session_state["download_original_rows"] = int(original_rows)
            st.session_state["download_sampled"] = bool(sampled)
            st.success("Fichier téléchargé.")
            st.write("#### Aperçu du jeu de données :", df)
        except Exception as exc:
            st.error(f"Error loading file : {exc}")


def run() -> None:
    Preparation1.init_preparation_state()
    _render_intro()
    _render_upload_controls()

    if not has_uploaded_dataset():
        return

    Preparation1.run_preparation_only()
