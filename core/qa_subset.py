from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class QASubsetSpec:
    """Description normalisée d'un sous-dataset Q&A.

    `filters` contient des filtres d'égalité combinés par ET logique.
    Exemple :
        [{"column": "sexe", "value": "female"}, {"column": "race", "value": "Black"}]
    """

    filters: tuple[dict[str, str], ...]
    excluded_columns: tuple[str, ...]
    description: str
    row_count: int
    total_count: int


def _normalise_text(value: Any) -> str:
    return str(value or "").strip()


def _resolve_column_name(df: pd.DataFrame, requested: Any) -> str | None:
    raw = _normalise_text(requested)
    if not raw or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if raw in df.columns:
        return raw
    raw_norm = raw.casefold()
    for column in df.columns:
        if str(column).strip().casefold() == raw_norm:
            return str(column)
    return None


def _resolve_value(df: pd.DataFrame, column: str, requested: Any) -> str | None:
    raw = _normalise_text(requested)
    if not raw or column not in df.columns:
        return None
    values = df[column].dropna().astype("string").unique().tolist()
    if raw in values:
        return raw
    raw_norm = raw.casefold()
    for value in values:
        text = str(value)
        if text.strip().casefold() == raw_norm:
            return text
    return raw


def normalize_subset_filters(df: pd.DataFrame, filters: Iterable[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Valide et normalise une liste de filtres colonne=modalité.

    Les filtres invalides sont ignorés. L'ordre est conservé et les doublons sont retirés.
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    normalised: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in filters or []:
        if not isinstance(item, dict):
            continue
        column = _resolve_column_name(df, item.get("column") or item.get("variable") or item.get("name"))
        if not column:
            continue
        value = _resolve_value(df, column, item.get("value") or item.get("modality") or item.get("modalite"))
        if value is None or value == "":
            continue
        key = (column, value)
        if key in seen:
            continue
        seen.add(key)
        normalised.append({"column": column, "value": value})
    return normalised


def _tokenize_normalized(text: str) -> set[str]:
    cleaned = []
    for ch in str(text or "").casefold():
        cleaned.append(ch if ch.isalnum() or ch in {"é", "è", "ê", "à", "ù", "ç", "î", "ï", "ô", "û", "â"} else " ")
    return {token for token in "".join(cleaned).split() if token}


def _value_aliases(value_text: str) -> set[str]:
    """Retourne des alias génériques pour reconnaître des modalités dans une question.

    Les alias sont volontairement limités aux traductions fréquentes et ne contiennent
    jamais les valeurs ambiguës ou ponctuationnelles comme "?".
    """

    value = str(value_text or "").strip()
    norm = value.casefold()
    if not norm or len(norm) < 2 or not any(ch.isalnum() for ch in norm):
        return set()

    aliases = {norm}
    mapping = {
        "black": {"noir", "noirs", "noire", "noires"},
        "white": {"blanc", "blancs", "blanche", "blanches"},
        "asian-pac-islander": {"asiatique", "asiatiques", "asian", "asie"},
        "amer-indian-eskimo": {"amérindien", "amerindien", "amérindiens", "amerindiens", "eskimo", "eskimos"},
        "other": {"autre", "autres"},
        "male": {"homme", "hommes", "masculin", "masculins"},
        "female": {"femme", "femmes", "féminin", "feminin", "féminins", "feminins"},
        "married-civ-spouse": {"marié", "mariés", "mariée", "mariées", "marié(e)s"},
        "never-married": {"célibataire", "celibataire", "célibataires", "celibataires"},
        "divorced": {"divorcé", "divorce", "divorcée", "divorcee", "divorcés", "divorcees"},
        "private": {"privé", "prive", "privés", "prives"},
    }
    aliases.update(mapping.get(norm, set()))
    return {alias.casefold() for alias in aliases if alias and any(ch.isalnum() for ch in alias)}


def infer_subset_filters_from_question(df: pd.DataFrame, question: str, *, max_filters: int = 4) -> list[dict[str, str]]:
    """Infère tous les filtres colonne=modalité explicitement cités dans une question.

    Contrairement à `resolve_segment_from_question`, cette fonction ne s'arrête pas
    au premier segment trouvé. Elle permet donc de construire des sous-groupes croisés
    comme `sex = Female` ET `race = Black` pour une question du type
    "profil des femmes noires".

    Une seule modalité est conservée par colonne pour éviter des filtres incohérents
    du type `race = White` ET `race = Black`. Les valeurs ambiguës ou techniques comme
    `?` ne sont jamais inférées à partir de la ponctuation de la question.
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    qnorm = str(question or "").casefold()
    qtokens = _tokenize_normalized(qnorm)
    inferred: list[dict[str, str]] = []
    used_columns: set[str] = set()

    for column in df.columns:
        if column in used_columns:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        try:
            values = series.dropna().astype("string").unique().tolist()
        except Exception:
            continue

        matches: list[tuple[int, str]] = []
        for value in values:
            value_text = str(value).strip()
            aliases = _value_aliases(value_text)
            if not aliases:
                continue
            matched = False
            score = 0
            for alias in aliases:
                alias_tokens = _tokenize_normalized(alias)
                if not alias_tokens:
                    continue
                if len(alias_tokens) == 1:
                    token = next(iter(alias_tokens))
                    if token in qtokens:
                        matched = True
                        score = max(score, len(token))
                elif alias_tokens.issubset(qtokens):
                    matched = True
                    score = max(score, sum(len(t) for t in alias_tokens))
            if matched:
                matches.append((score, value_text))

        if matches:
            matches.sort(key=lambda item: item[0], reverse=True)
            inferred.append({"column": str(column), "value": matches[0][1]})
            used_columns.add(str(column))
            if len(inferred) >= max_filters:
                break

    return normalize_subset_filters(df, inferred)


def build_subset_for_analysis(
    df: pd.DataFrame,
    filters: Iterable[dict[str, Any]] | None,
    *,
    exclude_filter_columns: bool = True,
    extra_excluded_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, QASubsetSpec]:
    """Construit un sous-dataset Q&A à partir d'un ou plusieurs filtres.

    Les filtres sont combinés par ET logique. La fonction couvre donc :
    - un groupe simple : sexe = female ;
    - un croisement de modalités : sexe = female ET race = Black ;
    - plus généralement n filtres d'égalité.

    Si `exclude_filter_columns=True`, les colonnes utilisées pour filtrer sont retirées
    du dataset retourné. Cela évite qu'un profil de femmes conclue trivialement que
    la variable dominante est `sexe = female`.
    """

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), QASubsetSpec(tuple(), tuple(), "", 0, 0)

    normalised_filters = normalize_subset_filters(df, filters)
    if not normalised_filters:
        clean_df = df.copy()
        return clean_df, QASubsetSpec(tuple(), tuple(), "population complète", len(clean_df), len(df))

    mask = pd.Series(True, index=df.index)
    for item in normalised_filters:
        column = item["column"]
        value = item["value"]
        mask &= df[column].astype("string").fillna("") == str(value)

    subset_df = df.loc[mask].copy()

    excluded: list[str] = []
    if exclude_filter_columns:
        excluded.extend(item["column"] for item in normalised_filters)
    for column in extra_excluded_columns or []:
        resolved = _resolve_column_name(subset_df, column)
        if resolved:
            excluded.append(resolved)

    excluded = list(dict.fromkeys(excluded))
    analysis_df = subset_df.drop(columns=[c for c in excluded if c in subset_df.columns], errors="ignore")
    description = " et ".join(f"{item['column']} = {item['value']}" for item in normalised_filters)

    return analysis_df, QASubsetSpec(
        filters=tuple(normalised_filters),
        excluded_columns=tuple(excluded),
        description=description,
        row_count=len(subset_df),
        total_count=len(df),
    )
