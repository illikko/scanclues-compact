"""Fonctions d'export du rapport final."""

from __future__ import annotations

import base64
import html as _html
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import plotly.io as pio

from apps._report import build_html_report_with_tables


REPORT_EXPORT_CSS = """
<style>
table { font-size:14px; border-collapse:collapse; }
th, td { border:1px solid #e5e7eb; padding:6px 8px; vertical-align:top; }
img { display:block; margin:0; max-width:100%; height:auto; }
/* Correction: styles ameliores pour les tableaux */
.dataframe {
    width: 100%;
    table-layout: fixed;
    word-wrap: break-word;
}
.dataframe th, .dataframe td {
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
</style>
"""


def build_report_html_for_export(base_html: str, extra_blocks: list[str] | tuple[str, ...] | None = None) -> str:
    extra = "".join(extra_blocks or [])
    if "</body>" in base_html:
        return base_html.replace("</body>", REPORT_EXPORT_CSS + extra + "</body>")
    return REPORT_EXPORT_CSS + base_html + extra


def should_show_index(df: pd.DataFrame) -> bool:
    idx = df.index
    is_default_range = isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1
    has_name = bool(idx.name)
    return (not is_default_range) or has_name


def _add_text(blocks: list[str], title: str, text: str) -> None:
    if not text:
        return
    title = _html.escape(title or "")
    body = _html.escape(str(text))
    block = f"""
<section style="margin:24px 0">
  <h2 style="margin:0 0 8px 0;font-size:20px">{title}</h2>
  <pre style="white-space:pre-wrap;margin:6px 0 10px 0;color:#444">{body}</pre>
</section>
"""
    blocks.append(block)


def _add_df(blocks: list[str], title: str, df: pd.DataFrame, intro: str | None = None, max_height: str | None = None) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    title = _html.escape(title or "")
    intro_html = ""
    if intro:
        intro_html = f'<div style="color:#444;margin:6px 0 10px 0;white-space:pre-wrap">{_html.escape(str(intro))}</div>'

    table_html = df.to_html(
        index=should_show_index(df),
        border=1,
        escape=False,
        classes="dataframe table table-striped",
    )

    style_wrap = "overflow-x:auto;"
    if max_height:
        style_wrap += f"max-height:{max_height};overflow-y:auto;"

    block = f"""
<section style="margin:24px 0">
  <h2 style="margin:0 0 8px 0;font-size:20px">{title}</h2>
  {intro_html}
  <div style="{style_wrap}">{table_html}</div>
</section>
"""
    blocks.append(block)


def _add_image(blocks: list[str], title: str, img, intro: str | None = None, dpi: int = 120) -> None:
    if hasattr(img, "savefig"):
        buf = BytesIO()
        original_size = img.get_size_inches()
        if original_size[0] < 8:
            img.set_size_inches(10, 8)
        img.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        png_bytes = buf.getvalue()
    elif isinstance(img, (bytes, bytearray, memoryview)):
        png_bytes = bytes(img)
    else:
        return

    b64 = base64.b64encode(png_bytes).decode("ascii")
    intro_html = f'<div style="color:#444;margin:6px 0 10px 0;white-space:pre-wrap">{intro}</div>' if intro else ""
    block = f"""
<section style="margin:24px 0">
  <h2 style="margin:0 0 8px 0;font-size:20px">{title}</h2>
  {intro_html}
  <img src="data:image/png;base64,{b64}" style="display:block;max-width:100%;height:auto;margin:8px 0 0 0"/>
</section>
"""
    blocks.append(block)


def _add_collapsible(blocks: list[str], title: str, inner_html: str) -> None:
    if not inner_html:
        return
    title = _html.escape(title or "")
    block = f"""
<section style="margin:24px 0">
  <details style="background:#fafafa;border:1px solid #ddd;border-radius:6px;padding:12px">
    <summary style="cursor:pointer;font-weight:600;font-size:16px">{title}</summary>
    <div style="margin-top:12px">{inner_html}</div>
  </details>
</section>
"""
    blocks.append(block)


def _preview_df_from_payload(section: dict, columns_key: str, rows_key: str) -> pd.DataFrame | None:
    if not isinstance(section, dict):
        return None
    columns = section.get(columns_key)
    rows = section.get(rows_key)
    if not isinstance(columns, list) or not isinstance(rows, list) or not rows:
        return None
    try:
        return pd.DataFrame(rows, columns=columns)
    except Exception:
        return None


def _build_preparation_details_html(session_state, context: dict) -> str:
    if not session_state.get("details_preparation_selected"):
        return ""
    payload = session_state.get("preparation_details_payload", {})
    if not isinstance(payload, dict) or not payload:
        return ""

    parts: list[str] = []
    label_shortening = payload.get("label_shortening", {})
    mapping_df = _preview_df_from_payload(label_shortening, "mapping_columns", "mapping_preview")
    if isinstance(mapping_df, pd.DataFrame) and not mapping_df.empty:
        inner: list[str] = []
        _add_df(inner, "LibellÃƒÂ©s avant / aprÃƒÂ¨s", mapping_df, max_height="360px")
        parts.extend(inner)

    semantic_rows = label_shortening.get("semantic_types_preview") or []
    semantic_columns = label_shortening.get("semantic_types_columns")
    if not semantic_columns and semantic_rows:
        semantic_columns = [str(k) for k in semantic_rows[0].keys()]
    if semantic_columns:
        try:
            semantic_df = pd.DataFrame(semantic_rows, columns=semantic_columns) if semantic_rows else None
        except Exception:
            semantic_df = None
        if isinstance(semantic_df, pd.DataFrame) and not semantic_df.empty:
            inner = []
            _add_df(inner, "Types sÃƒÂ©mantiques dÃƒÂ©tectÃƒÂ©s", semantic_df, max_height="360px")
            parts.extend(inner)

    missing_values = payload.get("missing_values", {})
    if missing_values.get("diagnostic"):
        inner = []
        _add_text(inner, "Diagnostic des valeurs manquantes", missing_values.get("diagnostic"))
        parts.extend(inner)
    missing_df = _preview_df_from_payload(missing_values, "table_columns", "table_preview")
    if isinstance(missing_df, pd.DataFrame) and not missing_df.empty:
        inner = []
        _add_df(inner, "DÃƒÂ©tail des valeurs manquantes", missing_df, max_height="420px")
        parts.extend(inner)
    if missing_values.get("little_test_result"):
        inner = []
        _add_text(inner, "Test de Little", missing_values.get("little_test_result"))
        parts.extend(inner)

    structural_missing = payload.get("structural_missing", {})
    if structural_missing.get("diagnostic"):
        inner = []
        _add_text(inner, "Manquantes structurelles", structural_missing.get("diagnostic"))
        parts.extend(inner)
    structural_df = _preview_df_from_payload(structural_missing, "candidates_columns", "candidates_preview")
    if isinstance(structural_df, pd.DataFrame) and not structural_df.empty:
        inner = []
        _add_text(
            inner,
            "Colonnes potentiellement concernées",
            "Colonnes repérées comme potentiellement concernées par un mécanisme de non-réponse structurelle.",
        )
        _add_df(inner, "Colonnes potentiellement concernées", structural_df, max_height="420px")
        parts.extend(inner)

    outliers = payload.get("outliers", {})
    outliers_df = _preview_df_from_payload(outliers, "table_columns", "table_preview")
    removed_count = len(outliers.get("indices") or []) if outliers.get("removed") else 0
    if removed_count or (isinstance(outliers_df, pd.DataFrame) and not outliers_df.empty):
        inner = []
        _add_text(inner, "Outliers supprimÃƒÂ©s", f"Nombre de lignes supprimÃƒÂ©es : {removed_count}")
        if isinstance(outliers_df, pd.DataFrame) and not outliers_df.empty:
            _add_df(inner, "DÃƒÂ©tail des outliers supprimÃƒÂ©s", outliers_df, max_height="420px")
        parts.extend(inner)

    prep2_details = payload.get("preparation2", {}).get("details")
    if prep2_details:
        parts.append(_build_preparation2_details_html(prep2_details))

    return "".join(parts)


def _build_preparation2_details_html(details: dict) -> str:
    if not isinstance(details, dict) or not details:
        return ""

    parts: list[str] = [
        "<section style='margin:24px 0'>",
        "<h2 style='margin:0 0 8px 0;font-size:20px'>Détails complémentaires de préparation</h2>",
    ]

    missing_values = details.get("missing_values", {}) or {}
    if isinstance(missing_values, dict) and missing_values:
        parts.append("<h3 style='margin:16px 0 8px 0;font-size:16px'>Traitement des valeurs manquantes</h3>")

        dropped_columns = missing_values.get("dropped_columns") or []
        if dropped_columns:
            parts.append(
                f"<p style='margin:6px 0 10px 0;color:#444'>Colonnes supprimées car trop incomplètes : {_html.escape(', '.join(map(str, dropped_columns)))}</p>"
            )

        dropped_rows = missing_values.get("dropped_rows")
        if isinstance(dropped_rows, int) and dropped_rows > 0:
            parts.append(
                f"<p style='margin:6px 0 10px 0;color:#444'>Lignes supprimées car trop incomplètes : {dropped_rows}</p>"
            )

        remaining_missing_columns = missing_values.get("remaining_missing_columns") or []
        if remaining_missing_columns:
            parts.append(
                f"<p style='margin:6px 0 10px 0;color:#444'>Colonnes restant partiellement incomplètes : {_html.escape(', '.join(map(str, remaining_missing_columns)))}</p>"
            )

        simple_imputation_columns = missing_values.get("simple_imputation_columns") or []
        if simple_imputation_columns:
            parts.append(
                f"<p style='margin:6px 0 10px 0;color:#444'>Imputation simple appliquée sur : {_html.escape(', '.join(map(str, simple_imputation_columns)))}</p>"
            )

        hotdeck_stats = missing_values.get("hotdeck_stats") or {}
        if isinstance(hotdeck_stats, dict) and hotdeck_stats:
            hotdeck_df = pd.DataFrame(
                [{"Colonne": str(col), "Valeurs imputées": int(count)} for col, count in hotdeck_stats.items()]
            )
            if not hotdeck_df.empty:
                inner: list[str] = []
                _add_df(inner, "Hot-deck simple", hotdeck_df, max_height="320px")
                parts.extend(inner)

    rare_modalities = details.get("rare_modalities", {}) or {}
    if isinstance(rare_modalities, dict):
        grouped_columns = rare_modalities.get("grouped_columns") or []
        if grouped_columns:
            parts.append("<h3 style='margin:16px 0 8px 0;font-size:16px'>Regroupement des modalités rares</h3>")
            parts.append(
                f"<p style='margin:6px 0 10px 0;color:#444'>{_html.escape(', '.join(map(str, grouped_columns)))}</p>"
            )

    second_pass = details.get("second_pass", {}) or {}
    if isinstance(second_pass, dict):
        dropped_columns = second_pass.get("dropped_columns") or []
        if dropped_columns:
            parts.append("<h3 style='margin:16px 0 8px 0;font-size:16px'>Vérifications finales</h3>")
            parts.append(
                f"<p style='margin:6px 0 10px 0;color:#444'>Colonnes non informatives supprimées après nettoyage : {_html.escape(', '.join(map(str, dropped_columns)))}</p>"
            )

    parts.append("</section>")
    return "".join(parts)

def _add_sankey(blocks: list[str], title: str, sankey_base64: str, intro: str | None = None) -> None:
    if not sankey_base64:
        return

    intro_html = f'<div style="color:#444;margin:6px 0 10px 0;white-space:pre-wrap">{intro}</div>' if intro else ""
    block = f"""
<section style="margin:24px 0">
  <h2 style="margin:0 0 8px 0;font-size:20px">{title}</h2>
  {intro_html}
  <img src="data:image/png;base64,{sankey_base64}" style="display:block;max-width:100%;height:auto;margin:8px 0 0 0"/>
</section>
"""
    blocks.append(block)


def _add_sankey_html(blocks: list[str], title: str, fig, intro: str | None = None) -> None:
    if fig is None:
        return

    sankey_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn",
    )

    intro_html = f'<div style="color:#444;margin:6px 0 10px 0;white-space:pre-wrap">{intro}</div>' if intro else ""
    block = f"""
<section style="margin:24px 0">
  <h2 style="margin:0 0 8px 0;font-size:20px">{title}</h2>
  {intro_html}
  {sankey_html}
</section>
"""
    blocks.append(block)


def _crosstab_details_html(session_state) -> str:
    results_store = session_state.get("sankey_pair_results", {})
    crosstab_list = session_state.get("crosstabs_interpretation", [])
    parts: list[str] = []

    def _append_part(var_x, var_y, interpretation, heatmap_png) -> None:
        if isinstance(heatmap_png, str):
            b64 = heatmap_png
        elif isinstance(heatmap_png, (bytes, bytearray, memoryview)):
            b64 = base64.b64encode(bytes(heatmap_png)).decode("ascii")
        else:
            b64 = ""
        img_html = ""
        if b64:
            img_html = f"<img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto;border:1px solid #ddd;border-radius:4px'/>"
        if img_html or interpretation:
            parts.append(
                f"<div style='margin-bottom:16px'><h4 style='margin:0 0 6px 0'>{_html.escape(str(var_x))} vs {_html.escape(str(var_y))}</h4>{img_html}<pre style='white-space:pre-wrap;color:#444;margin-top:6px'>{_html.escape(str(interpretation))}</pre></div>"
            )

    if isinstance(results_store, dict) and results_store:
        pairs_sorted = sorted(
            results_store.items(),
            key=lambda kv: kv[1].get("v", 0),
            reverse=True,
        )
        for _pair_id, res in pairs_sorted:
            _append_part(
                res.get("var_x"),
                res.get("var_y"),
                res.get("interpretation", ""),
                res.get("heatmap_png"),
            )

    if not parts and crosstab_list:
        for item in crosstab_list:
            _append_part(
                item.get("var_x"),
                item.get("var_y"),
                item.get("interpretation", ""),
                item.get("heatmap_png"),
            )

    return "".join(parts)


def build_final_report_html(session_state, context: dict) -> str:
    blocks: list[str] = []

    def c(name: str, default=None):
        return context.get(name, default)

    brief_intro = ""
    bq = str(c("dataset_key_questions") or "").strip()
    if bq:
        target_hint = session_state.get("brief_target_variable")
        target_txt = f" (cible dÃ©tectÃ©e : {target_hint})" if target_hint else ""
        brief_intro = f"**Question du brief :** {bq}{target_txt}"
    gs_text = session_state.get("global_synthesis")
    if not isinstance(gs_text, str) or not gs_text.strip():
        gs_text = "Aucune synthÃ¨se disponible"
    combined = "\n\n".join([x for x in [brief_intro, gs_text] if x])
    _add_text(blocks, c("T0"), combined)

    _add_text(blocks, c("T10"), session_state.get("report_introduction"))
    _add_text(blocks, c("T11"), session_state.get("dataset_object"))

    sankey_diagram = session_state.get("sankey_diagram")
    if sankey_diagram is not None:
        _add_sankey_html(blocks, c("T13"), sankey_diagram, intro=c("T131"))

    _add_text(blocks, "", session_state.get("sankey_interpretation_synthesis"))

    if c("sankey_crosstabs_selected"):
        crosstab_html = _crosstab_details_html(session_state)
        if crosstab_html:
            _add_collapsible(blocks, "Tris croisÃ©s dÃ©taillÃ©s", crosstab_html)

    syntheses_verbatim = c("syntheses_verbatim")
    if syntheses_verbatim:
        _add_text(blocks, c("T15"), syntheses_verbatim)
    elif session_state.get("verbatim_only_dataset"):
        _add_text(blocks, c("T15"), "Dataset 100% verbatim dÃ©tectÃ©, mais aucune synthÃ¨se n'a Ã©tÃ© gÃ©nÃ©rÃ©e.")

    if c("profils_y_text") is not None:
        _add_text(blocks, c("T21"), session_state.get("profils_y_text"))

    if isinstance(session_state.get("profils_y"), pd.DataFrame):
        _add_df(blocks, c("T211"), session_state["profils_y"], intro=c("T212"), max_height="480px")

    _add_text(blocks, c("T22"), session_state.get("segmentation_profiles_text"))

    segmentation_profiles_table = session_state.get("segmentation_profiles_table")
    if isinstance(segmentation_profiles_table, pd.DataFrame):
        _add_df(blocks, c("T221"), segmentation_profiles_table, intro=c("T222"), max_height="480px")

    _add_text(blocks, c("T31"), session_state.get("interpretationACM"))
    _add_text(blocks, c("T32"), session_state.get("dendrogram_interpretation"))

    if "dendrogram" in session_state:
        _add_image(blocks, title=c("T33"), img=session_state["dendrogram"], dpi=120, intro=c("T331"))

    if "latent_summary_text" in session_state:
        _add_text(blocks, c("T34"), session_state.get("latent_summary_text"))

    if isinstance(session_state.get("sankey_latents"), pd.DataFrame):
        _add_df(blocks, "", session_state["sankey_latents"], intro=c("T34"), max_height="480px")

    _add_text(blocks, c("T35"), c("T351"))

    if c("distribution_figures_selected"):
        dist_items = (
            session_state.get("figs_variables_distribution", [])
            or session_state.get("figs_variables_distribution_detailed", [])
        )
        for item in dist_items:
            _add_image(blocks, title=item.get("title", "Distribution"), img=item.get("png", b""), intro=c("T3521"))

    dominant_continues = c("dominant_continues")
    if isinstance(dominant_continues, pd.DataFrame):
        _add_df(blocks, c("T351"), dominant_continues, intro=c("T351"), max_height="480px")

    dominant_discretes = c("dominant_discretes")
    if isinstance(dominant_discretes, pd.DataFrame):
        _add_df(blocks, c("T351"), dominant_discretes, intro=c("T351"), max_height="480px")

    _add_text(blocks, c("T352"), session_state.get("profil_dominant_analysis"))

    _add_text(blocks, c("T41"), session_state.get("data_preparation_synthesis"))
    _add_text(blocks, c("T42"), session_state.get("dataset_context"))
    _add_text(blocks, c("T43"), session_state.get("dataset_characteristics"))

    if isinstance(session_state.get("variables_raw"), pd.DataFrame):
        _add_df(blocks, c("T44"), session_state["variables_raw"], intro=c("T451"), max_height="600px")

    _add_text(blocks, c("T45"), session_state.get("missing_values"))

    if "fig_missing_percentages" in session_state:
        _add_image(blocks, title=c("T451"), img=session_state["fig_missing_percentages"], intro=c("T4611"))

    if "fig_missing_correlation_heatmap" in session_state:
        _add_image(blocks, title=c("T452"), img=session_state["fig_missing_correlation_heatmap"], dpi=150, intro=c("T4521"))

    if "fig_missing_correlation_dendrogram" in session_state:
        _add_image(blocks, title=c("T453"), img=session_state["fig_missing_correlation_dendrogram"], intro=c("T4531"))

    _add_text(blocks, c("T454"), session_state.get("little_test_result"))

    if isinstance(session_state.get("process"), pd.DataFrame):
        _add_df(blocks, c("T461"), session_state["process"], intro=c("T461"), max_height="480px")

    preparation_details_html = _build_preparation_details_html(session_state, context)
    if preparation_details_html:
        _add_collapsible(blocks, "DÃ©tails de la prÃ©paration", preparation_details_html)

    shortened_labels_mapping = session_state.get("shortened_labels_mapping")
    if isinstance(shortened_labels_mapping, pd.DataFrame):
        _add_df(blocks, c("T462"), shortened_labels_mapping, intro=c("T462"), max_height="480px")

    ordinal_codification_mapping = session_state.get("ordinal_codification_mapping")
    if isinstance(ordinal_codification_mapping, pd.DataFrame):
        _add_df(blocks, c("T463"), ordinal_codification_mapping, intro=c("T463"), max_height="480px")

    return build_report_html_for_export(
        build_html_report_with_tables(title=c("T100")),
        blocks,
    )


def _write_df_csv(zf: ZipFile, filename: str, df: object, *, index: bool = False) -> None:
    if isinstance(df, pd.DataFrame):
        zf.writestr(
            filename,
            df.to_csv(sep=";", index=index).encode("latin-1", errors="replace"),
        )


def build_export_zip(session_state, html_report: str) -> bytes:
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, mode="w", compression=ZIP_DEFLATED) as zf:
        zf.writestr("report.html", (html_report or "").encode("utf-8"))
        _write_df_csv(zf, "df_ready.csv", session_state.get("df_ready"), index=False)
        _write_df_csv(zf, "segmentation_profiles_table.csv", session_state.get("segmentation_profiles_table"), index=True)
        _write_df_csv(zf, "segmentation_detailed_profiles.csv", session_state.get("segmentation_detailed_profiles"), index=False)
        _write_df_csv(zf, "profils_y_table.csv", session_state.get("profils_y_table"), index=False)
        _write_df_csv(zf, "profils_y_detailed.csv", session_state.get("profils_y_detailed"), index=False)
        _write_df_csv(zf, "shortened_labels_mapping.csv", session_state.get("shortened_labels_mapping"), index=False)
        _write_df_csv(zf, "df_shortlabels.csv", session_state.get("df_shortlabels"), index=False)
        _write_df_csv(zf, "ordinal_codification_mapping.csv", session_state.get("ordinal_codification_mapping"), index=False)
        _write_df_csv(zf, "df_encoded.csv", session_state.get("df_encoded"), index=False)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()
