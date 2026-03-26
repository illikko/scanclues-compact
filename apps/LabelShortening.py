import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import json, re
from core.df_registry import DFState, get_df, set_df
from utils import preparation_process
from collections import Counter


MODE_KEY = "__NAV_MODE__"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Fonctions

def _parse_json_array(raw: str):
    """
    Essaie de parser un tableau JSON éventuellement entouré de texte.
    """
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*?\]", raw)
        if not m:
            raise ValueError("La réponse du modèle n'est pas du JSON valide.")
        return json.loads(m.group(0))


def _find_duplicates(labels):
    """
    Retourne un dict {label: [indices]} pour chaque label dupliqué.
    """
    idx_by_label = {}
    for i, lab in enumerate(labels):
        idx_by_label.setdefault(lab, []).append(i)
    return {lab: idxs for lab, idxs in idx_by_label.items() if len(idxs) > 1}


def shorten_column_names(columns_to_shorten, max_chars: int, model: str = "gpt-4o-mini"):
    # ---------- 1) 1ère passe : proposition de noms raccourcis ----------
    prompt_1 = (
        f"Raccourcis ces noms de colonnes pour quâ€™ils aient au plus {max_chars} caractères, "
        "en gardant leur signification claire.\n"
        "- Tous les noms raccourcis doivent être STRICTEMENT distincts les uns des autres.\n"
        "- Si plusieurs colonnes ont une partie commune, ajoute un suffixe ou une nuance pour les distinguer.\n"
        "Réponds UNIQUEMENT par un tableau JSON de chaînes (pas de texte autour), "
        "dans le même ordre que l'entrée.\n\n"
        f"Noms originaux : {columns_to_shorten}"
    )

    response_1 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Tu es un assistant expert en data analyse."},
            {"role": "user", "content": prompt_1},
        ],
        temperature=0,
        max_tokens=4000,
    )

    new_labels = _parse_json_array(response_1.choices[0].message.content)

    if not (isinstance(new_labels, list) and len(new_labels) == len(columns_to_shorten)):
        raise ValueError("La réponse du modèle ne correspond pas au nombre de colonnes attendues.")

    # ---------- 2) Vérification des doublons ----------
    duplicates = _find_duplicates(new_labels)

    if not duplicates:
        # OK du premier coup
        return new_labels

    # ---------- 3) 2ème passe : correction des doublons ----------
    # On passe au LLM les noms originaux + les noms raccourcis actuels + la liste des conflits
    conflict_desc = []
    for lab, idxs in duplicates.items():
        cols_concernees = [columns_to_shorten[i] for i in idxs]
        conflict_desc.append(
            f'- Label raccourci "{lab}" utilisé pour les colonnes : {cols_concernees}'
        )
    conflict_desc_str = "\n".join(conflict_desc)

    prompt_2 = (
        "On a tenté de raccourcir des noms de colonnes mais certains labels sont en doublon.\n"
        "Corrige la liste suivante en MODIFIANT UNIQUEMENT les labels qui sont en doublon, "
        "pour que tous les labels finaux soient distincts.\n\n"
        f"Longueur max : {max_chars} caractères.\n"
        "- Ne change pas les labels déjà uniques si possible.\n"
        "- Garde la signification de chaque colonne.\n\n"
        "Noms originaux :\n"
        f"{columns_to_shorten}\n\n"
        "Noms raccourcis actuels :\n"
        f"{new_labels}\n\n"
        "Conflits détectés :\n"
        f"{conflict_desc_str}\n\n"
        "Réponds UNIQUEMENT par un tableau JSON de chaînes, dans le même ordre que l'entrée."
    )

    response_2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Tu es un assistant expert en data analyse."},
            {"role": "user", "content": prompt_2},
        ],
        temperature=0,
        max_tokens=4000,
    )

    corrected_labels = _parse_json_array(response_2.choices[0].message.content)

    if not (isinstance(corrected_labels, list) and len(corrected_labels) == len(columns_to_shorten)):
        raise ValueError("La réponse de la 2ème passe ne correspond pas au nombre de colonnes attendues.")

    # Re-check unicité après la 2ème passe
    duplicates_2 = _find_duplicates(corrected_labels)
    if duplicates_2:
        # Si ça échoue encore, on remonte une erreur claire
        raise ValueError(
            f"Après 2 passes LLM, certains labels sont encore en doublon : {list(duplicates_2.keys())}"
        )

    return corrected_labels

# App Streamlit

def run():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")
    
    st.header("Raccourcir les noms de colonnes")

    # déclaration des variables
    if "etape4_terminee" not in st.session_state:
        st.session_state["etape4_terminee"] = False


    # rechargement du dataset
    df = get_df(DFState.VERBATIM_READY)
    if df is None:
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application précédente.")
        st.stop()

    # 2- Choix du nombre maximum de caractères
    max_chars = st.slider("Nombre max. de caractères pour les noms de colonnes", min_value=5, max_value=120, value=50)

    # 3- Filtrer les noms trop longs
    original_columns = df.columns.tolist()
    columns_to_shorten = [col for col in original_columns if len(col) > max_chars]

    # Détection de changement de max_chars ou absence de edited_labels
    if (
        "edited_labels" not in st.session_state
        or st.session_state.last_max_chars != max_chars
    ):
        st.session_state.edited_labels = {col: col[:max_chars] for col in columns_to_shorten}
        st.session_state.last_max_chars = max_chars  # mise à jour

    if not columns_to_shorten:
        st.info("Tous les noms de colonnes sont déjà assez courts.")
        st.session_state["etape4_terminee"] = True
        return
    else:
        nb_columns_to_shorten = len(columns_to_shorten)
        st.write(f"{nb_columns_to_shorten} colonne(s) à raccourcir :")
        st.write(columns_to_shorten)


    # Initialiser l'état avec les raccourcis locaux si pas encore fait
    if "edited_labels" not in st.session_state:
        st.session_state.edited_labels = {col: col[:max_chars] for col in columns_to_shorten}
    if "last_max_chars" not in st.session_state:
        st.session_state.last_max_chars = max_chars
    if "lock_labels" not in st.session_state:
        st.session_state.lock_labels = False
    # Si max_chars change et PAS de lock, ne renseigner que les clés manquantes
    if st.session_state.last_max_chars != max_chars and not st.session_state.lock_labels:
        for col in columns_to_shorten:
            st.session_state.edited_labels.setdefault(col, col[:max_chars])
        st.session_state.last_max_chars = max_chars

    st.subheader("1- Mode de réduction")
    
    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        mode2 = st.radio("Choisissez le mode de réduction :", ("Réduction automatique par LLM","Réduction manuelle"))
        if st.button("Réduction automatique par LLM"):
            proceed = True

    if proceed:
        try:
            with st.spinner("Réduction des libellés par LLM en cours..."):
                new_labels = shorten_column_names(columns_to_shorten, max_chars, model="gpt-4o-mini")

                st.session_state.edited_labels = {
                    old: new for old, new in zip(columns_to_shorten, new_labels)
                }
                st.session_state.lock_labels = True
                st.success("Réduction automatique réussie (avec contrôle d'unicité) !")

        except Exception as e:
            st.error(f"Erreur lors du traitement de la réponse LLM : {e}")


    # Affichage du tableau éditable en 2 colonnes
    edited_df = pd.DataFrame({
        "Ancien libellé": columns_to_shorten,
        "Nouveau libellé": [st.session_state.edited_labels[col] for col in columns_to_shorten]
    })

    st.write("Comparatif des libellés (modifiables directement ci-dessous) :")


    # Affichage interactif et récupération directe
    edited_df = st.data_editor(
        edited_df,
        column_config={
            "Ancien libellé": st.column_config.Column(disabled=True),
            "Nouveau libellé": st.column_config.Column(),
        },
        num_rows="fixed",
        key="editor_labels",
        use_container_width=True
)

    # Validation
    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        if st.button("Appliquer les nouveaux noms de colonnes"):
            proceed = True

    st.session_state.shortened_labels_mapping = edited_df
    st.dataframe(st.session_state.shortened_labels_mapping)

    if proceed:
        try:
            rename_mapping = {col: st.session_state.edited_labels.get(col, col) for col in df.columns}
            st.caption("Mapping appliqué (différences) :")
            st.json({k: v for k, v in rename_mapping.items() if k != v})
            df_renamed = df.rename(columns=rename_mapping)
            set_df(DFState.SHORT_LABELS, df_renamed, step_name="LabelShortening")

            st.success("Noms de colonnes appliqués.")
            st.dataframe(df_renamed.head())
            
            csv = df_renamed.to_csv(index=False, sep=";", encoding="latin-1")
            st.download_button("Télécharger le fichier avec les libellés raccourcis", data=csv, file_name="dataset_shortlabels.csv", mime="text/csv")

            # update du tableau de process
            action = f"{nb_columns_to_shorten} libellés de variables raccourcis."
            st.success(action)
            st.markdown("##### État d'avancement de la préparation du dataset :")
            preparation_process(df_renamed, action)
            st.dataframe(st.session_state.process)           
                        
            st.write("Vous pouvez lancer la prochaine étape dans le menu à gauche: Préparation 2.")
            st.session_state["etape4_terminee"] = True
            
        except Exception as e:
            st.error(f"Erreur lors de l’application des nouveaux noms : {e}")



