from contextlib import contextmanager
import os
import sys
import time

import pandas as pd
import streamlit as st

from .CadrageAnalyse import _call_llm
from . import (
    AnalyseCorrelations,
    AnalyseFactorielle,
    CodificationOrdinales,
    DiagramSankey,
    DistributionVariables,
    LabelShortening,
    ManquantesStructurelles,
    Outliers,
    Preparation2,
    PreparationCorrelations,
    Profils_y,
    ReponsesMultiples,
    ReponsesMultiplesOrdonnees,
    Segmentation,
    SeparationVariables,
    VerbatimSummary,
)

MODE_KEY = "__NAV_MODE__"
PIPELINE_FORCE_AUTO_KEY = "__PIPELINE_FORCE_AUTO__"
MODULE_LABELS = {
    "VerbatimSummary": "Synthese des verbatims",
    "LabelShortening": "Raccourcissement des libelles",
    "ReponsesMultiplesOrdonnees": "Reponses multiples ordonnees",
    "ReponsesMultiples": "Reponses multiples",
    "ManquantesStructurelles": "Manquantes structurelles",
    "Preparation2": "Preparation des donnees",
    "PreparationCorrelations": "Variables trop correlees",
    "Outliers": "Valeurs aberrantes",
    "CodificationOrdinales": "Codification des variables ordinales",
    "SeparationVariables": "Separation des variables",
    "Segmentation": "Segmentation",
    "Profils_y": "Profils associes a la cible",
    "AnalyseCorrelations": "Analyse des correlations",
    "AnalyseFactorielle": "Analyse factorielle",
    "DiagramSankey": "Schema des relations (Sankey)",
    "DistributionVariables": "Analyse descriptive des variables",
}


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, _):
        return lambda *args, **kwargs: None


class _DummyProgress:
    def progress(self, *args, **kwargs):
        return self

    def empty(self):
        return None


def _noop(*args, **kwargs):
    return None


def _pick_df(*keys):
    """
    Retourne le premier DataFrame disponible parmi les cles donnees
    sans declencher d'evaluation booleenne ambiguë.
    """
    for key in keys:
        df = st.session_state.get(key)
        if isinstance(df, pd.DataFrame):
            return df
    return None


@contextmanager
def _trace_module_calls(mod, on_function=None):
    """
    Trace les appels de fonctions Python du module cible uniquement.
    Utilise sys.setprofile et notifie via callback.
    """
    if not callable(on_function):
        yield
        return

    # Le tracing fin (sys.setprofile) est tres couteux; desactive par defaut.
    if not bool(st.session_state.get("pipeline_trace_functions", False)):
        yield
        return

    mod_file = getattr(mod, "__file__", None)
    if not mod_file:
        yield
        return
    target_file = os.path.normcase(os.path.abspath(mod_file))

    prev_profiler = sys.getprofile()
    last_name = None
    last_emit = 0.0

    def _profiler(frame, event, arg):
        nonlocal last_name, last_emit
        if event != "call":
            return _profiler
        try:
            code = frame.f_code
            fn_name = code.co_name
            if not fn_name or fn_name.startswith("<"):
                return _profiler
            src = os.path.normcase(os.path.abspath(code.co_filename))
            if src != target_file:
                return _profiler
            now = time.monotonic()
            if fn_name == last_name and (now - last_emit) < 0.1:
                return _profiler
            last_name = fn_name
            last_emit = now
            on_function(fn_name)
        except Exception:
            return _profiler
        return _profiler

    try:
        sys.setprofile(_profiler)
        yield
    finally:
        sys.setprofile(prev_profiler)


@contextmanager
def _silent_streamlit():
    if not st.session_state.get("__PIPELINE_SILENT__", False):
        yield
        return

    original: dict[str, object] = {}
    dummy = _DummyContext()

    def patch(name: str, fn):
        if hasattr(st, name):
            original[name] = getattr(st, name)
            setattr(st, name, fn)

    def _columns(spec, *args, **kwargs):
        try:
            n = int(spec) if isinstance(spec, int) else len(spec)
        except Exception:
            n = 1
        return tuple(dummy for _ in range(max(1, n)))

    def _tabs(spec, *args, **kwargs):
        try:
            n = len(spec)
        except Exception:
            n = 1
        return tuple(dummy for _ in range(max(1, n)))

    def _state_or_default(kwargs, default_value):
        key = kwargs.get("key")
        if key and key in st.session_state:
            return st.session_state.get(key)
        return default_value

    def _checkbox(*args, **kwargs):
        return bool(_state_or_default(kwargs, kwargs.get("value", False)))

    def _button(*args, **kwargs):
        return False

    def _text_input(*args, **kwargs):
        return str(_state_or_default(kwargs, kwargs.get("value", "")))

    def _text_area(*args, **kwargs):
        return str(_state_or_default(kwargs, kwargs.get("value", "")))

    def _number_input(*args, **kwargs):
        return _state_or_default(kwargs, kwargs.get("value", 0))

    def _slider(*args, **kwargs):
        existing = _state_or_default(kwargs, None)
        if existing is not None:
            return existing
        if "value" in kwargs:
            return kwargs["value"]
        if len(args) >= 4:
            return args[3]
        if len(args) >= 3:
            return args[2]
        return 0

    def _selectbox(*args, **kwargs):
        existing = _state_or_default(kwargs, None)
        if existing is not None:
            return existing
        opts = kwargs.get("options")
        if opts is None and len(args) >= 2:
            opts = args[1]
        opts = list(opts or [])
        if not opts:
            return None
        idx = int(kwargs.get("index", 0))
        if idx < 0 or idx >= len(opts):
            idx = 0
        return opts[idx]

    def _radio(*args, **kwargs):
        return _selectbox(*args, **kwargs)

    def _multiselect(*args, **kwargs):
        existing = _state_or_default(kwargs, None)
        if existing is not None:
            return list(existing) if isinstance(existing, (list, tuple, set)) else []
        d = kwargs.get("default", [])
        return list(d) if isinstance(d, (list, tuple, set)) else []

    def _data_editor(*args, **kwargs):
        existing = _state_or_default(kwargs, None)
        if existing is not None:
            return existing
        if args:
            return args[0]
        return kwargs.get("data")

    def _progress(*args, **kwargs):
        return _DummyProgress()

    output_names = [
        "title", "header", "subheader", "caption", "text", "write", "markdown", "metric",
        "success", "info", "warning", "error", "exception", "dataframe", "table", "json",
        "pyplot", "plotly_chart", "altair_chart", "line_chart", "bar_chart", "area_chart", "image",
        "download_button", "toast", "code",
    ]
    for n in output_names:
        patch(n, _noop)

    # Containers/context blocks
    for n in ["expander", "spinner", "container", "popover", "status", "form"]:
        patch(n, lambda *args, **kwargs: dummy)

    patch("columns", _columns)
    patch("tabs", _tabs)

    # Inputs: return defaults, do not trigger actions
    patch("button", _button)
    patch("checkbox", _checkbox)
    patch("toggle", _checkbox)
    patch("text_input", _text_input)
    patch("text_area", _text_area)
    patch("number_input", _number_input)
    patch("slider", _slider)
    patch("selectbox", _selectbox)
    patch("radio", _radio)
    patch("multiselect", _multiselect)
    patch("file_uploader", lambda *args, **kwargs: None)
    patch("data_editor", _data_editor)
    patch("form_submit_button", _button)
    patch("progress", _progress)

    try:
        yield
    finally:
        for k, v in original.items():
            setattr(st, k, v)


def _run_module(mod, name: str, logs: list[dict], progress_callback=None, function_progress_callback=None) -> tuple[bool, dict]:
    holder = st.empty()
    t_start = time.monotonic()
    try:
        if callable(progress_callback):
            progress_callback(MODULE_LABELS.get(name, name))
        module_label = MODULE_LABELS.get(name, name)

        def _on_fn(fn_name: str):
            if callable(function_progress_callback):
                function_progress_callback(module_label, fn_name)

        with holder.container():
            with _silent_streamlit():
                with _trace_module_calls(mod, _on_fn):
                    mod.run()
        elapsed = time.monotonic() - t_start
        entry = {"module": name, "status": "ok", "elapsed_sec": round(float(elapsed), 3)}
        if name == "DiagramSankey":
            sankey_obj = st.session_state.get("sankey_diagram")
            has_sankey = sankey_obj is not None and sankey_obj != {}
            if not has_sankey:
                entry["status"] = "skipped"
                entry["reason"] = "no_sankey_output"
        logs.append(entry)
        return False, entry
    except BaseException as exc:
        elapsed = time.monotonic() - t_start
        exc_name = exc.__class__.__name__
        if exc_name == "StopException":
            entry = {
                "module": name,
                "status": "skipped",
                "cause": "module_called_st.stop",
                "error": "Module ignore (st.stop).",
                "elapsed_sec": round(float(elapsed), 3),
            }
            logs.append(entry)
            return False, entry
        if exc_name == "RerunException":
            entry = {
                "module": name,
                "status": "skipped",
                "cause": "module_requested_rerun",
                "error": "Module ignore (st.rerun).",
                "elapsed_sec": round(float(elapsed), 3),
            }
            logs.append(entry)
            return False, entry
        entry = {
            "module": name,
            "status": "error",
            "cause": "exception",
            "error": repr(exc),
            "elapsed_sec": round(float(elapsed), 3),
        }
        logs.append(entry)
        return True, entry
    finally:
        holder.empty()


def _has_df(key: str) -> bool:
    v = st.session_state.get(key)
    return isinstance(v, pd.DataFrame) and not v.empty


def _sankey_size_ok(max_cols: int = 120) -> bool:
    df_ready = st.session_state.get("df_ready")
    return isinstance(df_ready, pd.DataFrame) and df_ready.shape[1] <= max_cols


def _pick_df(*keys):
    """Renvoie le premier DataFrame non vide trouvé parmi les clés données."""
    for k in keys:
        df = st.session_state.get(k)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return None


def _ensure_cadrage_context(df_source: pd.DataFrame | None):
    """
    Garantit que le cadrage (dataset_object/context + target/illustratives) est disponible
    avant les modules qui en dÃ©pendent (PreparationCorrelations, Profils_y, AnalyseCorrelations).
    Appel idempotent: ne relance pas si target_variables sont dÃ©jÃ  prÃ©sentes.
    """
    if st.session_state.get("__CADRAGE_CONTEXT_READY__", False):
        return
    if not isinstance(df_source, pd.DataFrame) or df_source.empty:
        return
    try:
        res = _call_llm(df_source)
        st.session_state.setdefault("dataset_object", res.get("dataset_object"))
        st.session_state.setdefault("dataset_context", res.get("dataset_context"))
        st.session_state.setdefault("dataset_recommendations", res.get("dataset_recommendations"))
        st.session_state.setdefault("target_variables", res.get("target_variables", []))
        st.session_state.setdefault("target_modalities", res.get("target_modalities", {}))
        st.session_state.setdefault("illustrative_variables", res.get("illustrative_variables", []))
        if res.get("target_variables"):
            st.session_state["brief_target_variable"] = res.get("target_variables")[0]
        st.session_state["__CADRAGE_CONTEXT_READY__"] = True
    except Exception:
        st.session_state["__CADRAGE_CONTEXT_READY__"] = False


def _ensure_sankey_variables() -> bool:
    df_ready = st.session_state.get("df_ready")
    if not isinstance(df_ready, pd.DataFrame) or df_ready.empty:
        return False
    cols = set(df_ready.columns.tolist())

    # Si le brief cite explicitement une colonne existante, on la force comme cible prioritaire
    brief_raw = str(st.session_state.get("dataset_key_questions_value", "") or "")
    brief_norm = brief_raw.lower()
    for col in df_ready.columns:
        if str(col).lower() in brief_norm:
            st.session_state["brief_target_variable"] = col
            break

    def _norm_list(values):
        out = []
        for v in values or []:
            if v in cols and v not in out:
                out.append(v)
        return out

    brief_target = st.session_state.get("brief_target_variable")

    # 1) Deja en session_state (filtre aux colonnes existantes)
    targets = _norm_list(st.session_state.get("target_variables", []))
    illustratives = _norm_list(st.session_state.get("illustrative_variables", []))

    if brief_target:
        if brief_target not in targets:
            targets = [brief_target] + targets
        # Nettoyage doublons tout en conservant l'ordre
        seen = set()
        targets = [t for t in targets if not (t in seen or seen.add(t))]

    if targets and illustratives:
        st.session_state["target_variables"] = targets
        st.session_state["illustrative_variables"] = illustratives
        return True

    # 2) Fallback depuis le tableau valide de Diagnostic/Cadrage
    var_table_df = st.session_state.get("var_table_df")
    if isinstance(var_table_df, pd.DataFrame) and not var_table_df.empty:
        try:
            if "variable" in var_table_df.columns:
                if "cible" in var_table_df.columns:
                    targets = _norm_list(var_table_df.loc[var_table_df["cible"] == True, "variable"].tolist())
                if "illustrative" in var_table_df.columns:
                    illustratives = _norm_list(var_table_df.loc[var_table_df["illustrative"] == True, "variable"].tolist())
        except Exception:
            pass
        if targets and illustratives:
            st.session_state["target_variables"] = targets
            st.session_state["illustrative_variables"] = illustratives
            return True

    # 3) Fallback depuis les suggestions LLM deja calculees
    llm_suggestions = st.session_state.get("llm_suggestions", {}) or {}
    if isinstance(llm_suggestions, dict):
        targets = _norm_list(llm_suggestions.get("target_variables", []))
        illustratives = _norm_list(llm_suggestions.get("illustrative_variables", []))
        if targets and illustratives:
            st.session_state["target_variables"] = targets
            st.session_state["illustrative_variables"] = illustratives
            if not st.session_state.get("target_modalities"):
                st.session_state["target_modalities"] = llm_suggestions.get("target_modalities", {}) or {}
            return True

    # 4) Dernier recours: relancer _call_llm sur df_ready
    try:
        res = _call_llm(df_ready)
        if not st.session_state.get("dataset_object"):
            st.session_state["dataset_object"] = res.get("dataset_object")
        if not st.session_state.get("dataset_context"):
            st.session_state["dataset_context"] = res.get("dataset_context")
        if not st.session_state.get("dataset_recommendations"):
            st.session_state["dataset_recommendations"] = res.get("dataset_recommendations")
        st.session_state["target_variables"] = _norm_list(res.get("target_variables", []) or [])
        st.session_state["target_modalities"] = res.get("target_modalities", {}) or {}
        st.session_state["illustrative_variables"] = _norm_list(res.get("illustrative_variables", []) or [])
    except Exception as exc:
        st.session_state["pipeline_sankey_context_error"] = repr(exc)
        # On continue avec les informations partielles eventuelles ci-dessous.

    # 5) Sauvetage minimal si une des 2 listes manque encore.
    targets = _norm_list(st.session_state.get("target_variables", []))
    illustratives = _norm_list(st.session_state.get("illustrative_variables", []))

    if not targets:
        tm = st.session_state.get("target_modalities", {}) or {}
        if isinstance(tm, dict):
            targets = _norm_list(list(tm.keys()))

    if targets and not illustratives:
        illustratives = [c for c in df_ready.columns if c not in set(targets)]

    if illustratives and not targets:
        for c in df_ready.columns:
            if c not in set(illustratives):
                targets = [c]
                break

    # Reprioritise explicit brief target if available
    brief_target = st.session_state.get("brief_target_variable")
    if brief_target and brief_target in cols:
        targets = [brief_target] + [t for t in targets if t != brief_target]

    # Reprioritise brief target if présent et valide
    brief_target = st.session_state.get("brief_target_variable")
    if brief_target and brief_target in cols:
        targets = [brief_target] + [t for t in targets if t != brief_target]

    st.session_state["target_variables"] = targets
    st.session_state["illustrative_variables"] = illustratives

    return bool(targets) and bool(illustratives)


def _ready_for(name: str) -> bool:
    if name == "DiagramSankey":
        return _has_df("df_ready") and _sankey_size_ok() and _ensure_sankey_variables()

    reqs = {
        "VerbatimSummary": lambda: _has_df("df_raw"),
        "LabelShortening": lambda: _has_df("df_ex_verbatim"),
        "ReponsesMultiplesOrdonnees": lambda: _has_df("df_shortlabels") or _has_df("df_ex_verbatim"),
        "ReponsesMultiples": lambda: _has_df("df_ex_ordonnees"),
        "ManquantesStructurelles": lambda: _has_df("df_ex_multiples"),
        "Preparation2": lambda: _has_df("df_imputed_structural"),
        "PreparationCorrelations": lambda: _has_df("df_clean"),
        "Outliers": lambda: _has_df("df_ex_corr") or _has_df("df_clean"),
        "CodificationOrdinales": lambda: _has_df("df_ready"),
        "SeparationVariables": lambda: _has_df("df_encoded") or _has_df("df_ready"),
        "Segmentation": lambda: _has_df("df_active") or _has_df("df_encoded") or _has_df("df_ready"),
        "Profils_y": lambda: _has_df("df_ready"),
        "AnalyseCorrelations": lambda: _has_df("df_active") or _has_df("df_encoded") or _has_df("df_ready"),
        "AnalyseFactorielle": lambda: _has_df("df_active") or _has_df("df_encoded") or _has_df("df_ready"),
        "DistributionVariables": lambda: _has_df("df_ready"),
    }
    fn = reqs.get(name)
    return True if fn is None else bool(fn())


def _skip_entry(name: str) -> dict:
    if name == "DiagramSankey":
        has_ready = _has_df("df_ready")
        size_ok = _sankey_size_ok()
        has_targets = bool(st.session_state.get("target_variables"))
        has_illustratives = bool(st.session_state.get("illustrative_variables"))
        entry = {"module": name, "status": "skipped", "reason": "missing_input_df"}
        if has_ready and not size_ok:
            entry["reason"] = "too_many_columns_for_sankey_pipeline"
            df_ready = st.session_state.get("df_ready")
            entry["n_columns"] = int(df_ready.shape[1]) if isinstance(df_ready, pd.DataFrame) else None
            return entry
        if has_ready and (not has_targets or not has_illustratives):
            entry["reason"] = "missing_target_or_illustrative_variables"
            entry["target_variables"] = st.session_state.get("target_variables", [])
            entry["illustrative_variables"] = st.session_state.get("illustrative_variables", [])
            if st.session_state.get("pipeline_sankey_context_error"):
                entry["context_error"] = st.session_state.get("pipeline_sankey_context_error")
        return entry
    return {"module": name, "status": "skipped", "reason": "missing_input_df"}


def run_selected(selection: dict, *, show_details: bool = False, progress_callback=None, function_progress_callback=None) -> list[dict]:
    """
    Execute les modules selon la selection utilisateur.
    - preparation: socle de preparation
    - profilage: segmentation + profils_y
    - analyse_descriptive: correlations/factorielle/sankey/distributions
    """
    logs: list[dict] = []
    t0 = time.monotonic()

    prev_force = st.session_state.get(PIPELINE_FORCE_AUTO_KEY, False)
    st.session_state[PIPELINE_FORCE_AUTO_KEY] = True
    st.session_state["__PIPELINE_SILENT__"] = True
    st.session_state.pop("pipeline_sankey_context_error", None)

    # Shortcut verbatim-only : si toutes les colonnes sont des textes longs (hors identifiants),
    # on ne lance que VerbatimSummary et on ignore le reste.
    if st.session_state.get("verbatim_only_dataset"):
        verb_logs: list[dict] = []
        st.session_state["pipeline_selection"] = {
            "preparation": True,
            "profilage": False,
            "analyse_descriptive": False,
        }
        if _ready_for("VerbatimSummary"):
            _run_module(VerbatimSummary, "VerbatimSummary", verb_logs, progress_callback=progress_callback, function_progress_callback=function_progress_callback)
        else:
            verb_logs.append(_skip_entry("VerbatimSummary"))

        elapsed = time.monotonic() - t0
        st.session_state["pipeline_execution_logs"] = verb_logs
        st.session_state["pipeline_execution_seconds"] = float(elapsed)
        st.session_state["pipeline_executed"] = True
        st.session_state["pipeline_current_module"] = None
        st.session_state["pipeline_status"] = "completed" if all(x.get("status") == "ok" for x in verb_logs) else "completed_with_skips"
        st.session_state["pipeline_halt"] = None
        # Préparer des synthèses minimales pour le rapport final
        st.session_state.setdefault("data_preparation_synthesis", "Jeu de données composé uniquement de verbatims; aucun autre module exécuté.")
        if st.session_state.get("syntheses_verbatim"):
            st.session_state["global_synthesis"] = "Analyse verbatim uniquement — voir synthèse ci-dessous."
        else:
            st.session_state.setdefault("global_synthesis", "Analyse verbatim uniquement — synthèse indisponible (aucun texte généré).")
        st.session_state["__PIPELINE_SILENT__"] = False
        st.session_state[PIPELINE_FORCE_AUTO_KEY] = prev_force
        if show_details:
            st.dataframe(verb_logs, use_container_width=True)
        return verb_logs

    prep_base = [
        (VerbatimSummary, "VerbatimSummary"),
        (LabelShortening, "LabelShortening"),
        (ReponsesMultiplesOrdonnees, "ReponsesMultiplesOrdonnees"),
        (ReponsesMultiples, "ReponsesMultiples"),
        (ManquantesStructurelles, "ManquantesStructurelles"),
    ]
    prep_core = [
        (Preparation2, "Preparation2"),
        (PreparationCorrelations, "PreparationCorrelations"),
        (Outliers, "Outliers"),
        (CodificationOrdinales, "CodificationOrdinales"),
        (SeparationVariables, "SeparationVariables"),
    ]
    profilage = [
        (Segmentation, "Segmentation"),
        (Profils_y, "Profils_y"),
    ]
    descriptive = [
        (AnalyseCorrelations, "AnalyseCorrelations"),
        (AnalyseFactorielle, "AnalyseFactorielle"),
        (DiagramSankey, "DiagramSankey"),
        (DistributionVariables, "DistributionVariables"),
    ]

    halted = False
    halt_entry = None

    if selection.get("preparation", True) and not halted:
        for mod, name in prep_base + prep_core:
            # Assurer le cadrage LLM avant les modules qui utilisent target/illustratives
            if name == "Preparation2":
                src = _pick_df("df_imputed_structural", "df_ready")
                _ensure_cadrage_context(src if isinstance(src, pd.DataFrame) else st.session_state.get("df_raw"))
            if _ready_for(name):
                should_halt, entry = _run_module(mod, name, logs, progress_callback=progress_callback, function_progress_callback=function_progress_callback)
                if should_halt:
                    halted = True
                    halt_entry = entry
                    break
            else:
                logs.append(_skip_entry(name))

    if selection.get("profilage", False) and not halted:
        # Assurer que le cadrage est disponible pour profils_y
        src = _pick_df("df_ready", "df_imputed_structural")
        _ensure_cadrage_context(src)
        for mod, name in profilage:
            if _ready_for(name):
                should_halt, entry = _run_module(mod, name, logs, progress_callback=progress_callback, function_progress_callback=function_progress_callback)
                if should_halt:
                    halted = True
                    halt_entry = entry
                    break
            else:
                logs.append(_skip_entry(name))

    if selection.get("analyse_descriptive", False) and not halted:
        # Assurer cadrage pour AnalyseCorrelations / DiagramSankey
        src = _pick_df("df_ready", "df_encoded", "df_imputed_structural")
        _ensure_cadrage_context(src)
        for mod, name in descriptive:
            if _ready_for(name):
                should_halt, entry = _run_module(mod, name, logs, progress_callback=progress_callback, function_progress_callback=function_progress_callback)
                if should_halt:
                    halted = True
                    halt_entry = entry
                    break
            else:
                logs.append(_skip_entry(name))

    elapsed = time.monotonic() - t0
    st.session_state["pipeline_execution_logs"] = logs
    st.session_state["pipeline_execution_seconds"] = float(elapsed)
    st.session_state["pipeline_executed"] = True
    st.session_state["pipeline_current_module"] = None

    if halted:
        st.session_state["pipeline_status"] = "failed" if halt_entry.get("status") == "error" else "stopped"
        st.session_state["pipeline_halt"] = halt_entry
    else:
        has_error = any(x.get("status") == "error" for x in logs)
        has_skip = any(x.get("status") == "skipped" for x in logs)
        if has_error:
            st.session_state["pipeline_status"] = "failed"
        elif has_skip:
            st.session_state["pipeline_status"] = "completed_with_skips"
        else:
            st.session_state["pipeline_status"] = "completed"
        st.session_state["pipeline_halt"] = None

    st.session_state["__PIPELINE_SILENT__"] = False
    st.session_state[PIPELINE_FORCE_AUTO_KEY] = prev_force

    if show_details:
        st.dataframe(logs, use_container_width=True)

    return logs


