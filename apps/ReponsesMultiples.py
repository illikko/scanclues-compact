import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re

from collections import Counter
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any


MODE_KEY = "__NAV_MODE__"


# ----------------------------
# Utils / Config
# ----------------------------

@dataclass
class MultiModalDetection:
    ok: bool
    sep: Optional[str]
    columns: List[str]
    score: float
    reasons: Dict[str, Any]


@dataclass
class MultiModalParams:
    # Séparateurs
    candidate_seps: Tuple[str, ...] = ("|", ";", "+")
    secondary_seps: Tuple[str, ...] = ("/", "\n", "\t")
    use_two_pass_strategy: bool = True

    # Critères principaux
    min_cols: int = 2
    min_presence: float = 0.35
    min_avg_parts: float = 1.8
    max_unique_tokens: int = 150
    max_unique_ratio: float = 0.25
    min_repeat_rate: float = 0.25
    max_avg_token_len: float = 30.0

    # Critères secondaires
    secondary_min_presence: float = 0.40
    secondary_min_avg_parts: float = 2.2
    secondary_max_unique_tokens: int = 120
    secondary_max_unique_ratio: float = 0.20
    secondary_min_repeat_rate: float = 0.30
    secondary_max_avg_token_len: float = 30.0


def _sample_str_series(s: pd.Series, n: int = 300) -> pd.Series:
    s = s.dropna()
    if len(s) == 0:
        return s
    if len(s) > n:
        s = s.sample(n, random_state=0)
    return s.astype(str)


def _tokenize_multiselect(s: pd.Series, sep: str, max_rows: int = 3000) -> List[str]:
    """
    Extrait les tokens (modalités) d'un échantillon de lignes, après split sur sep.
    """
    ss = s.dropna()
    if len(ss) == 0:
        return []

    if len(ss) > max_rows:
        ss = ss.sample(max_rows, random_state=0).astype(str)
    else:
        ss = ss.astype(str)

    tokens: List[str] = []
    for v in ss:
        if sep not in v:
            continue
        parts = [p.strip() for p in v.split(sep)]
        parts = [p for p in parts if p != ""]
        tokens.extend(parts)
    return tokens


def _detect_multimodal_config_core(
    df: pd.DataFrame,
    candidate_seps: Tuple[str, ...],
    min_cols: int,
    min_presence: float,
    min_avg_parts: float,
    max_unique_tokens: int,
    max_unique_ratio: float,
    min_repeat_rate: float,
    max_avg_token_len: float,
) -> MultiModalDetection:
    best = MultiModalDetection(False, None, [], 0.0, {})

    for sep in candidate_seps:
        supporting = []
        col_scores = []
        per_col = {}

        sep_re = re.compile(re.escape(sep))

        for col in df.columns:
            s = _sample_str_series(df[col])
            if len(s) == 0:
                per_col[col] = {"skip": "empty_column"}
                continue

            if sep == "/":
                date_pattern = r"^\d{1,2}/\d{1,2}/\d{2,4}$|^\d{1,2}/\d{4}$"
                is_date = float(s.str.match(date_pattern, na=False).mean())
                if is_date > 0.5:
                    per_col[col] = {"skip": "date_format", "is_date": is_date}
                    continue

            if sep == "-":
                interval_pattern = r"^\s*\d+\s*-\s*\d+\s*$"
                is_interval = float(s.str.match(interval_pattern, na=False).mean())
                if is_interval > 0.5:
                    per_col[col] = {"skip": "interval", "is_interval": is_interval}
                    continue

            number_with_sep = float(
                s.str.contains(r"\d+\s*" + re.escape(sep) + r"\s*\d+", na=False).mean()
            )
            if number_with_sep > 0.3:
                per_col[col] = {"skip": "numbers_with_sep", "ratio": number_with_sep}
                continue

            has_sep = s.str.contains(sep_re, na=False)
            presence = float(has_sep.mean())
            if presence < min_presence:
                per_col[col] = {"skip": "low_presence", "presence": presence}
                continue

            parts_counts = s[has_sep].str.split(sep).map(
                lambda xs: sum(1 for x in xs if str(x).strip() != "")
            )
            avg_parts = float(parts_counts.mean()) if len(parts_counts) else 0.0
            if avg_parts < min_avg_parts:
                per_col[col] = {
                    "skip": "low_avg_parts",
                    "presence": presence,
                    "avg_parts": avg_parts,
                }
                continue

            tokens = _tokenize_multiselect(s, sep=sep)
            if len(tokens) == 0:
                per_col[col] = {"skip": "no_tokens"}
                continue

            norm = [t.strip().lower() for t in tokens if t.strip() != ""]

            cnt = Counter(norm)
            unique_tokens = len(cnt)
            total_tokens = len(norm)
            unique_ratio = unique_tokens / total_tokens if total_tokens else 1.0

            repeated = sum(1 for _, v in cnt.items() if v >= 2)
            repeat_rate = repeated / unique_tokens if unique_tokens else 0.0

            avg_token_len = float(np.mean([len(t) for t in norm])) if norm else 0.0
            if avg_token_len > max_avg_token_len:
                per_col[col] = {
                    "skip": "tokens_too_long",
                    "avg_token_len": avg_token_len,
                }
                continue

            if unique_tokens > max_unique_tokens:
                per_col[col] = {
                    "skip": "too_many_unique_tokens",
                    "unique_tokens": unique_tokens,
                }
                continue

            if unique_ratio > max_unique_ratio:
                per_col[col] = {
                    "skip": "too_unique_ratio",
                    "unique_ratio": unique_ratio,
                }
                continue

            if repeat_rate < min_repeat_rate:
                per_col[col] = {
                    "skip": "low_repeat_rate",
                    "repeat_rate": repeat_rate,
                }
                continue

            score = presence * min(avg_parts, 8.0) * (1.0 + 0.6 * (1.0 - unique_ratio))

            supporting.append(col)
            col_scores.append(score)

            per_col[col] = {
                "presence": presence,
                "avg_parts": avg_parts,
                "unique_tokens": unique_tokens,
                "unique_ratio": unique_ratio,
                "repeat_rate": repeat_rate,
                "avg_token_len": avg_token_len,
                "score": score,
            }

        if len(supporting) < min_cols:
            continue

        sep_score = float(np.mean(col_scores)) * (1.0 + 0.12 * (len(supporting) - min_cols))

        if sep_score > best.score:
            best = MultiModalDetection(
                ok=True,
                sep=sep,
                columns=sorted(supporting),
                score=sep_score,
                reasons={
                    "sep_tested": sep,
                    "sep_score": sep_score,
                    "n_cols": len(supporting),
                    "per_col": per_col,
                },
            )

    return best


def detect_multimodal_config(
    df: pd.DataFrame,
    params: Optional[MultiModalParams] = None,
) -> MultiModalDetection:
    """
    Détecte la configuration multimodale.
    - En automatique : utiliser params=None
    - En manuel : fournir un MultiModalParams personnalisé
    """
    if params is None:
        params = MultiModalParams()

    primary = _detect_multimodal_config_core(
        df=df,
        candidate_seps=params.candidate_seps,
        min_cols=params.min_cols,
        min_presence=params.min_presence,
        min_avg_parts=params.min_avg_parts,
        max_unique_tokens=params.max_unique_tokens,
        max_unique_ratio=params.max_unique_ratio,
        min_repeat_rate=params.min_repeat_rate,
        max_avg_token_len=params.max_avg_token_len,
    )

    if primary.ok or not params.use_two_pass_strategy:
        return primary

    secondary = _detect_multimodal_config_core(
        df=df,
        candidate_seps=params.secondary_seps,
        min_cols=params.min_cols,
        min_presence=params.secondary_min_presence,
        min_avg_parts=params.secondary_min_avg_parts,
        max_unique_tokens=params.secondary_max_unique_tokens,
        max_unique_ratio=params.secondary_max_unique_ratio,
        min_repeat_rate=params.secondary_min_repeat_rate,
        max_avg_token_len=params.secondary_max_avg_token_len,
    )

    return secondary


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    out, prev_us = [], False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    return "".join(out).strip("_")


def one_hot_multilabel(
    df: pd.DataFrame,
    col: str,
    sep: str = "+",
    drop_original: bool = False,
    prefix: Optional[str] = None,
    dtype: str = "int8",
) -> pd.DataFrame:
    """Encode une colonne multilabel en indicatrices 0/1."""
    if col not in df.columns:
        raise KeyError(f"La colonne '{col}' n'existe pas dans le DataFrame.")

    s = df[col].fillna("").astype(str)
    dummies = s.str.get_dummies(sep=sep).rename(columns=lambda c: c.strip())

    mapping = {c: _slugify(c) for c in dummies.columns}
    dummies = dummies.rename(columns=mapping)

    if "" in dummies.columns:
        dummies = dummies.drop(columns=[""])

    if not dummies.empty:
        dummies = dummies.T.groupby(level=0).max().T

    if prefix:
        dummies = dummies.rename(columns=lambda c: f"{prefix}{c}")

    if not dummies.empty:
        dummies = dummies.astype(dtype)

    out = pd.concat([df.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    if drop_original:
        out = out.drop(columns=[col])

    return out


def encode_multiple_columns(
    df: pd.DataFrame,
    cols: List[str],
    sep: str,
    dtype: str = "int8",
) -> pd.DataFrame:
    """Encode plusieurs colonnes multilabel et supprime les colonnes sources."""
    out = df.copy()
    for col in cols:
        prefix = f"{_slugify(col)}__"
        out = one_hot_multilabel(
            out,
            col=col,
            sep=sep,
            drop_original=True,
            prefix=prefix,
            dtype=dtype,
        )
    return out


def _sep_label(sep: str) -> str:
    mapping = {
        "+": "Plus (+)",
        ",": "Virgule (,)",
        ";": "Point-virgule (;)",
        "|": "Barre verticale (|)",
        "/": "Slash (/)",
        "\n": "Retour ligne (\\n)",
        "\t": "Tabulation (\\t)",
    }
    return mapping.get(sep, repr(sep))


def _build_params_from_ui(mode: str) -> MultiModalParams:
    default_params = MultiModalParams()

    if mode == "automatique":
        return default_params

    st.markdown("#### Paramètres de détection")

    available_sep_labels = {
        "Barre verticale (|)": "|",
        "Point-virgule (;)": ";",
        "Plus (+)": "+",
        "Slash (/)": "/",
        "Retour ligne (\\n)": "\n",
        "Tabulation (\\t)": "\t",
        "Virgule (,)": ",",
        "Tiret (-)": "-",
    }

    primary_default_labels = [
        _sep_label(sep) for sep in default_params.candidate_seps if sep in available_sep_labels.values()
    ]
    secondary_default_labels = [
        _sep_label(sep) for sep in default_params.secondary_seps if sep in available_sep_labels.values()
    ]

    selected_primary_labels = st.multiselect(
        "Séparateurs principaux à tester",
        options=list(available_sep_labels.keys()),
        default=primary_default_labels,
    )

    use_two_pass_strategy = st.checkbox(
        "Activer une seconde passe si la première échoue",
        value=default_params.use_two_pass_strategy,
    )

    selected_secondary_labels = st.multiselect(
        "Séparateurs secondaires à tester",
        options=list(available_sep_labels.keys()),
        default=secondary_default_labels,
        disabled=not use_two_pass_strategy,
    )

    col1, col2 = st.columns(2)

    with col1:
        min_cols = st.number_input(
            "Nombre minimal de colonnes détectées",
            min_value=1,
            value=default_params.min_cols,
            step=1,
        )
        min_presence = st.slider(
            "Présence minimale du séparateur",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.min_presence),
            step=0.01,
        )
        min_avg_parts = st.slider(
            "Nombre moyen minimal de modalités",
            min_value=1.0,
            max_value=10.0,
            value=float(default_params.min_avg_parts),
            step=0.1,
        )
        max_unique_tokens = st.number_input(
            "Nombre maximal de modalités distinctes",
            min_value=1,
            value=default_params.max_unique_tokens,
            step=1,
        )
        max_unique_ratio = st.slider(
            "Ratio maximal de modalités uniques",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.max_unique_ratio),
            step=0.01,
        )
        min_repeat_rate = st.slider(
            "Taux minimal de répétition",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.min_repeat_rate),
            step=0.01,
        )
        max_avg_token_len = st.slider(
            "Longueur moyenne maximale des modalités",
            min_value=1.0,
            max_value=100.0,
            value=float(default_params.max_avg_token_len),
            step=1.0,
        )

    with col2:
        secondary_min_presence = st.slider(
            "Seconde passe : présence minimale",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.secondary_min_presence),
            step=0.01,
            disabled=not use_two_pass_strategy,
        )
        secondary_min_avg_parts = st.slider(
            "Seconde passe : nombre moyen minimal de modalités",
            min_value=1.0,
            max_value=10.0,
            value=float(default_params.secondary_min_avg_parts),
            step=0.1,
            disabled=not use_two_pass_strategy,
        )
        secondary_max_unique_tokens = st.number_input(
            "Seconde passe : nombre maximal de modalités distinctes",
            min_value=1,
            value=default_params.secondary_max_unique_tokens,
            step=1,
            disabled=not use_two_pass_strategy,
        )
        secondary_max_unique_ratio = st.slider(
            "Seconde passe : ratio maximal de modalités uniques",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.secondary_max_unique_ratio),
            step=0.01,
            disabled=not use_two_pass_strategy,
        )
        secondary_min_repeat_rate = st.slider(
            "Seconde passe : taux minimal de répétition",
            min_value=0.0,
            max_value=1.0,
            value=float(default_params.secondary_min_repeat_rate),
            step=0.01,
            disabled=not use_two_pass_strategy,
        )
        secondary_max_avg_token_len = st.slider(
            "Seconde passe : longueur moyenne maximale des modalités",
            min_value=1.0,
            max_value=100.0,
            value=float(default_params.secondary_max_avg_token_len),
            step=1.0,
            disabled=not use_two_pass_strategy,
        )

    candidate_seps = tuple(available_sep_labels[label] for label in selected_primary_labels)
    secondary_seps = tuple(available_sep_labels[label] for label in selected_secondary_labels)

    return MultiModalParams(
        candidate_seps=candidate_seps,
        secondary_seps=secondary_seps,
        use_two_pass_strategy=use_two_pass_strategy,
        min_cols=int(min_cols),
        min_presence=float(min_presence),
        min_avg_parts=float(min_avg_parts),
        max_unique_tokens=int(max_unique_tokens),
        max_unique_ratio=float(max_unique_ratio),
        min_repeat_rate=float(min_repeat_rate),
        max_avg_token_len=float(max_avg_token_len),
        secondary_min_presence=float(secondary_min_presence),
        secondary_min_avg_parts=float(secondary_min_avg_parts),
        secondary_max_unique_tokens=int(secondary_max_unique_tokens),
        secondary_max_unique_ratio=float(secondary_max_unique_ratio),
        secondary_min_repeat_rate=float(secondary_min_repeat_rate),
        secondary_max_avg_token_len=float(secondary_max_avg_token_len),
    )


# ----------------------------
# App streamlit
# ----------------------------

def run():
    mode = "automatique" if st.session_state.get("__PIPELINE_FORCE_AUTO__", False) else st.session_state.get(MODE_KEY, "automatique")

    st.title("Colonnes multi-modalités")
    st.write("Ce module identifie les colonnes à multi-modalités et les encode en indicatrices 0/1.")

    for key, default in [
        ("file_upload_attempted", False),
        ("variables_validated", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if "etape6_terminee" not in st.session_state:
        st.session_state["etape6_terminee"] = False

    st.subheader("Import du fichier")

    if "df_ex_ordonnees" in st.session_state:
        df = st.session_state.df_ex_ordonnees.copy()
    else:
        st.warning("Aucun dataset trouvé. Veuillez d'abord passer par l'application précédente.")
        st.stop()

    st.success("Fichier chargé.")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Détection des colonnes")

    params = _build_params_from_ui(mode)

    if "rm_has_multimodal" in st.session_state:
        if st.session_state.get("rm_has_multimodal"):
            det = MultiModalDetection(
                True,
                st.session_state.get("rm_detected_sep"),
                st.session_state.get("rm_detected_cols", []),
                1.0,
                {},
            )
        else:
            det = MultiModalDetection(False, None, [], 0.0, {})
    else:
        det = detect_multimodal_config(df, params=params)
        st.session_state["rm_has_multimodal"] = bool(det.ok)
        st.session_state["rm_detected_sep"] = det.sep
        st.session_state["rm_detected_cols"] = det.columns

    default_sep = det.sep if det.ok else (params.candidate_seps[0] if params.candidate_seps else "+")
    default_cols = det.columns if det.ok else []

    if det.ok:
        st.success("Configuration détectée automatiquement âœ…")

        if det.sep == "\n":
            sep_display = r"\n"
        elif det.sep == "\t":
            sep_display = r"\t"
        else:
            sep_display = det.sep

        st.write(f"**Séparateur détecté :** `{sep_display}`")
        st.write(f"**Colonnes détectées :** {', '.join(det.columns)}")
    else:
        st.warning("Aucune configuration automatique convaincante n'a été détectée. Vous pouvez sélectionner manuellement.")

    common = {
        "Plus (+)": "+",
        "Virgule (,)": ",",
        "Point-virgule (;)": ";",
        "Barre verticale (|)": "|",
        "Slash (/)": "/",
        "Retour ligne (\\n)": "\n",
        "Tabulation (\\t)": "\t",
    }

    label_by_sep = {v: k for k, v in common.items()}
    default_label = label_by_sep.get(default_sep, "Plus (+)")

    selected_sep_label = st.selectbox(
        "Sélection rapide du séparateur d'encodage",
        list(common.keys()),
        index=list(common.keys()).index(default_label) if default_label in common else 0,
    )

    custom_sep = st.text_input(
        "...ou saisissez un séparateur personnalisé",
        value="",
        max_chars=5,
    )
    sep = custom_sep if custom_sep != "" else common[selected_sep_label]

    cols_to_encode = st.multiselect(
        f"Sélectionnez les colonnes contenant des modalités séparées par « {sep} »",
        df.columns.tolist(),
        default=default_cols,
    )

    with st.expander("Critères de décision / débogage"):
        st.write(f"**Score global :** {det.score:.3f}")
        st.write("**Paramètres utilisés :**")
        st.json(asdict(params))

        dbg = det.reasons.get("per_col", {})
        if dbg:
            dbg_df = pd.DataFrame.from_dict(dbg, orient="index")
            sort_col = "score" if "score" in dbg_df.columns else dbg_df.columns[0]
            st.dataframe(
                dbg_df.sort_values(by=sort_col, ascending=False, na_position="last"),
                use_container_width=True,
            )
        else:
            st.info("Aucun détail de décision disponible.")

    if not cols_to_encode:
        st.info("Aucune colonne à réponses multiples sélectionnée.")
        st.session_state.df_ex_multiples = df
        st.session_state["etape6_terminee"] = True
        return

    proceed = False
    if mode == "automatique":
        proceed = True
    else:
        proceed = st.button("Lancer l'encodage")

    if not proceed:
        return

    try:
        df_final = encode_multiple_columns(df, cols_to_encode, sep=sep, dtype="int8")

        st.session_state.df_ex_multiples = df_final
        st.session_state["etape6_terminee"] = True

        st.success("Encodage terminé : colonnes encodées ajoutées et colonnes originales supprimées.")
        st.dataframe(df_final.head(), use_container_width=True)

    except Exception as e:
        st.error(f"Erreur pendant l'encodage : {e}")
        st.stop()
