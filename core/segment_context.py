from __future__ import annotations

import difflib
from typing import Any

import pandas as pd


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().casefold()


def _segment_aliases(value_text: str) -> set[str]:
    normalized = _normalize_text(value_text)
    aliases = {normalized}
    mapping = {
        "black": {"noir", "noirs", "noire", "noires"},
        "white": {"blanc", "blancs", "blanche", "blanches"},
        "male": {"homme", "hommes", "masculin"},
        "female": {"femme", "femmes", "féminin", "feminin"},
        "married-civ-spouse": {"marié", "mariés", "mariée", "mariées", "marié(e)s"},
        "private": {"privé", "prive", "salariés du privé", "salariés du prive"},
    }
    for key, values in mapping.items():
        if normalized == key:
            aliases.update({_normalize_text(item) for item in values})
    return aliases


def _find_cols_in_question(question: str, df: pd.DataFrame) -> list[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    qlow = str(question or "").lower()
    cols: list[str] = []
    for col in df.columns:
        name = str(col)
        if name.lower() in qlow:
            cols.append(name)
    if cols:
        return list(dict.fromkeys(cols))

    tokens = [t for t in qlow.replace("?", " ").replace(",", " ").split() if len(t) >= 4]
    candidates = [str(c) for c in df.columns]
    for token in tokens:
        match = difflib.get_close_matches(token, candidates, n=1, cutoff=0.8)
        if match:
            cols.append(match[0])
    return list(dict.fromkeys(cols))


def resolve_segment_from_question(question: str, df: pd.DataFrame) -> dict[str, Any] | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    matched_columns = _find_cols_in_question(question, df)
    qlow = _normalize_text(question)

    for column in matched_columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        values = series.dropna().astype("string").unique().tolist()
        for value in values:
            value_text = str(value).strip()
            if value_text and any(alias in qlow for alias in _segment_aliases(value_text)):
                mask = series.astype("string") == value_text
                subset_df = df.loc[mask].copy()
                if not subset_df.empty:
                    return {
                        "column": str(column),
                        "value": value_text,
                        "df": subset_df,
                        "description": f"{column} = {value_text}",
                    }

    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        values = series.dropna().astype("string").unique().tolist()
        for value in values:
            value_text = str(value).strip()
            if value_text and any(alias in qlow for alias in _segment_aliases(value_text)):
                mask = series.astype("string") == value_text
                subset_df = df.loc[mask].copy()
                if not subset_df.empty:
                    return {
                        "column": str(column),
                        "value": value_text,
                        "df": subset_df,
                        "description": f"{column} = {value_text}",
                    }
    return None


def build_segment_context_tables(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not isinstance(df, pd.DataFrame) or column not in df.columns:
        return None, None

    counts = df[column].astype("string").value_counts(dropna=False)
    if counts.empty:
        return None, None

    total = int(counts.sum())
    labels = [str(idx) for idx in counts.index.tolist()]
    effectifs = [int(value) for value in counts.tolist()]
    percentages = [round((value / total) * 100, 1) if total else 0.0 for value in effectifs]

    counts_df = pd.DataFrame(
        {
            "Modalité": labels,
            "Effectif": effectifs,
        }
    )
    percent_df = pd.DataFrame(
        {
            "Modalité": labels,
            "%": percentages,
        }
    )
    return counts_df, percent_df


def build_segment_intro(column: str, value: str, count: int, share: float, total: int) -> str:
    return (
        f"La catégorie `{value}` de la variable `{column}` représente {count} observations sur {total}, "
        f"soit {share:.1f}% de l'ensemble. Les tableaux ci-dessous montrent sa place parmi toutes les "
        "modalités de cette variable avant d'aller plus loin dans l'analyse."
    )
