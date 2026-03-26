import base64
from io import BytesIO
import streamlit as st

REPORT_HTML_BLOCKS = []

def reset_report():
    """Réinitialise le contenu du rapport HTML."""
    REPORT_HTML_BLOCKS.clear()

def _ensure_list():
    if "report_items" not in st.session_state:
        st.session_state["report_items"] = []

def add_text(title: str, text: str):
    _ensure_list()
    st.session_state["report_items"].append({"type": "text", "title": title, "content": text})

def add_table(title: str, df):
    _ensure_list()
    st.session_state["report_items"].append({"type": "table", "title": title, "dataframe": df})

def add_figure(title: str, fig):
    _ensure_list()
    st.session_state["report_items"].append({"type": "figure", "title": title, "figure": fig})

def _df_to_png_b64(df, max_rows=30):
    import matplotlib.pyplot as plt
    from io import BytesIO

    df_display = df.head(max_rows).copy()
    df_display.columns = [str(c) for c in df_display.columns]

    ncols = len(df_display.columns)
    nrows = len(df_display) + 1  # header + rows

    fig_w = min(22, 1.6 + 1.0 * ncols)
    fig_h = min(22, 1.2 + 0.4 * nrows)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    ax.axis("off")

    tbl = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns.tolist(),
        loc="upper left",   # ← ancre en haut-gauche
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.02, 1.10)

    try:
        tbl.auto_set_column_width(col=list(range(ncols)))
    except Exception:
        pass

    if len(df) > max_rows:
        ax.text(0.0, -0.06, f"Showing first {max_rows} rows of {len(df)}", transform=ax.transAxes)

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.02)

    buf = BytesIO()
    fig.tight_layout(pad=0.1)
    fig.savefig(
        buf,
        format="png",
        dpi=160,
        bbox_inches="tight",
        bbox_extra_artists=(tbl,),  # â† évite le rognage des bords
        pad_inches=0.01,
    )
        
    # ROGNAGE AUTOMATIQUE DU BLANC AUTOUR DU TABLEAU (9 lignes)
    from PIL import Image, ImageChops
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    bg  = Image.new("RGB", img.size, (255, 255, 255))
    bbox = ImageChops.difference(img, bg).getbbox()
    if bbox:
        img = img.crop(bbox)
    buf2 = BytesIO()
    img.save(buf2, format="PNG", optimize=True)
    data = buf2.getvalue()
    
    plt.close(fig)
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _fig_to_png_b64(fig):
    # âœ… CORRECTION : Augmentation de la taille par défaut
    current_size = fig.get_size_inches()
    if current_size[0] < 8:  # Si l'image est trop petite
        fig.set_size_inches(10, 8)  # Taille minimale pour la lisibilité
    
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def build_html_report(title="Final report"):
    items = st.session_state.get("report_items", [])

    parts = [
        "<html><head><meta charset='utf-8'>",
        f"<title>{title}</title>",
        BASE_CSS,  # on réutilise la feuille de style globale
        "</head><body>",
        f"<h1>{title}</h1>",
    ]

    for it in items:
        body = ""
        if it["type"] == "text":
            body = (_render_text_block(it["content"]) or "").strip()
        elif it["type"] == "table":
            body = f"<img class='report-fig' src='{_df_to_png_b64(it['dataframe'])}'/>"
        elif it["type"] == "figure":
            body = f"<img class='report-fig' src='{_fig_to_png_b64(it['figure'])}'/>"
        elif it["type"] == "html":
            body = (it.get("html") or "").strip()

        if not body:
            continue

        parts.append(f"<h2>{it['title']}</h2>")
        parts.append(body)

    parts.append("</body></html>")
    return "".join(parts)


def add_from_state(mapping: dict):
    """
    Collect from st.session_state using a simple spec:
      mapping = {
        "Section title": {"key": "session_key", "kind": "text|table|figure"}
      }
    - text: any non-empty string
    - table: pandas DataFrame (non-empty)
    - figure: matplotlib Figure (or Plotly Figure; see add_figure_auto)
    """
    for title, spec in mapping.items():
        key  = spec.get("key")
        kind = spec.get("kind")
        val  = st.session_state.get(key, None)
        if val is None:
            continue

        if kind == "text":
            s = str(val).strip()
            if s:
                add_text(title, s)

        elif kind == "table":
            try:
                import pandas as pd  # noqa
                if hasattr(val, "empty") and not val.empty:
                    add_table(title, val)
            except Exception:
                # Not a DataFrame or empty -> skip
                pass

        elif kind == "figure":
            add_figure_auto(title, val)  # will try matplotlib first, then Plotly -> rasterize

def add_figure_auto(title: str, fig_obj):
    """
    Accepts:
      - Matplotlib Figure: stored directly (works with build_html_report)
      - Plotly Figure: rasterizes to PNG, embeds into a Matplotlib Figure, then stores
      - Anything else: ignored
    """
    # 1) Matplotlib?
    try:
        import matplotlib.figure as mfig
        if isinstance(fig_obj, mfig.Figure):
            add_figure(title, fig_obj)
            return
    except Exception:
        pass

    # 2) Plotly? -> convert to PNG, then wrap into a Matplotlib figure via imshow
    try:
        import plotly.graph_objs as go
        if isinstance(fig_obj, go.Figure):
            # Try kaleido first
            png_bytes = None
            try:
                png_bytes = fig_obj.to_image(format="png", scale=2)  # needs kaleido installed
            except Exception:
                # fallback: try write_image if configured, else skip
                try:
                    import tempfile, os
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    fig_obj.write_image(tmp.name, scale=2)
                    with open(tmp.name, "rb") as f:
                        png_bytes = f.read()
                    os.unlink(tmp.name)
                except Exception:
                    png_bytes = None

            if png_bytes:
                try:
                    from PIL import Image
                    import matplotlib.pyplot as plt
                    from io import BytesIO
                    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
                    fig, ax = plt.subplots(figsize=(img.width/100, img.height/100), dpi=100)
                    ax.imshow(img)
                    ax.axis("off")
                    add_figure(title, fig)
                    return
                except Exception:
                    pass
    except Exception:
        pass

    # 3) Unknown figure type -> ignore silently
    return

# helper pour garder les mises pages
def _render_text_block(s: str) -> str:
    """
    Convertit un bloc texte en HTML simple :
    - lignes commençant par -, *, â€¢ -> <ul><li>...</li></ul>
    - double sauts de ligne -> paragraphes
    - simple saut de ligne -> <br>
    """
    if not s:
        return ""

    lines = [ln.rstrip() for ln in str(s).splitlines()]
    items = []
    ul_open = False
    bullet_prefixes = ("- ", "* ", "• ")

    def close_ul():
        nonlocal ul_open
        if ul_open:
            items.append("</ul>")
            ul_open = False

    para_buf = []

    def flush_para():
        nonlocal para_buf
        if para_buf:
            # join simple \n par <br>
            items.append("<p>" + "<br>".join(para_buf) + "</p>")
            para_buf = []

    for ln in lines:
        if not ln.strip():
            # blanc = fin de paragraphe / liste
            close_ul()
            flush_para()
            continue
        if ln.startswith(bullet_prefixes):
            # ligne à puce
            if para_buf:
                flush_para()
            if not ul_open:
                items.append("<ul>")
                ul_open = True
            items.append("<li>" + ln[2:].strip() + "</li>")
        else:
            # texte normal
            close_ul()
            para_buf.append(ln)

    close_ul()
    flush_para()

    html = "".join(items)
    return html or f"<p>{s}</p>"


# ====== HTML/CSS table rendering for report ======

REPORT_GLOBAL_CSS = """
/* Conteneur générique */
.section { margin: 24px 0; }

/* Textes */
.section h2 { margin: 0 0 8px 0; font-size: 30px; }
.section .intro { color: #444; margin: 6px 0 10px 0; }

/* Figures (images de graphes) */
.section img.report-fig {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 8px 0 0 0;         /* évite les grands blancs au-dessus */
}

/* Tables responsives */
.table-wrap {
  overflow-x: auto;          /* scroll horizontal si besoin */
  -webkit-overflow-scrolling: touch;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 4px;
  background: #fff;
}

table.df {
  border-collapse: collapse;
  width: 100%;
  table-layout: auto;        /* pas "fixed": on garde des largeurs naturelles */
  font-size: 14px;           /* taille stable */
}

table.df th, table.df td {
  border: 1px solid #e5e7eb;
  padding: 6px 8px;
  vertical-align: top;
  white-space: normal;       /* autorise le retour à la ligne */
  word-break: break-word;    /* casse les mots longs */
}

table.df thead th {
  background: #f8fafc;
  position: sticky; top: 0;  /* header collant si conteneur a une hauteur fixée */
  z-index: 1;
}

/* première colonne "sticky" pour gros tableaux */
table.df td:first-child, table.df th:first-child {
  position: sticky; left: 0;
  background: #fff;
  z-index: 2;
}
"""

# PARAMETRES DE FORMATS
# modifier aux endroits indiqués (taille)

BASE_CSS = """
<style>
  /* Base lisible */
  html { font-size: 17px; }
  body {
    font-size: 1rem;
    line-height: 1.6;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    margin: 20px;
    color: #111;
  }

  /* Titres */
  h1 { font-size: 2rem; margin: 0 0 1rem 0; }
  h2 { font-size: 1.5rem; margin: 1.5rem 0 0.5rem 0; }
  h3 { font-size: 1.25rem; margin: 1.2rem 0 0.4rem 0; }

  /* Paragraphes / listes */
  p, li { font-size: 1rem; }

  /* Table HTML classique */
  table { 
    font-size: 0.95rem; 
    border-collapse: collapse; 
    width: 100%; 
    table-layout: fixed;  /* âœ… CORRECTION : Évite le débordement */
  }
  th, td { 
    padding: 0.5rem; 
    border: 1px solid #ddd; 
    word-wrap: break-word;  /* ✅ CORRECTION : Casse les mots longs */
    max-width: 300px;       /* ✅ CORRECTION : Limite la largeur */
  }

  /* Figures (graphes / tableaux convertis en PNG) */
  img.report-fig {
    display: block;
    margin: 0.75rem 0;
    max-width: 1000px;   /* ✅ CORRECTION : Augmentation de la limite */
    width: 100%;
    height: auto;
  }
</style>
"""


def render_report_html(content_html: str) -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'>{BASE_CSS}</head><body>{content_html}</body></html>"



def df_to_html_block(df, title=None, intro=None, max_height=None):
    """
    Rend un DataFrame en HTML + CSS responsif (pas d'image).
    - max_height: si renseigné (ex '420px'), on fixe la hauteur et on garde head sticky.
    """
    from pandas import DataFrame
    assert isinstance(df, DataFrame)

    # Index visible ? Ici on le masque pour simplifier la lecture
    html_table = df.to_html(index=False, classes="df", escape=False)

    style = REPORT_GLOBAL_CSS
    wrap_open = '<div class="section">'
    wrap_close = '</div>'

    parts = [f"<style>{style}</style>", wrap_open]
    if title:
        parts.append(f"<h2>{title}</h2>")
    if intro:
        parts.append(f'<div class="intro">{intro}</div>')

    if max_height:
        parts.append(f'<div class="table-wrap" style="max-height:{max_height};">')
    else:
        parts.append('<div class="table-wrap">')

    parts.append(html_table)
    parts.append('</div>')  # .table-wrap
    parts.append(wrap_close)
    return "".join(parts)

# Stockage d'éléments HTML dans le builder existant
def add_table_html(title, df, intro=None, max_height=None):
    """Ajoute un bloc table (HTML, pas image) au rapport."""
    if df is None:
        return
    try:
        REPORT_HTML_BLOCKS.append(df_to_html_block(df, title=title, intro=intro, max_height=max_height))
    except Exception:
        pass  # on ignore si df invalide

def add_text_html(title, text):
    if not text:
        return
    REPORT_HTML_BLOCKS.append(
        f'<div class="section"><h2>{title}</h2><div class="intro">{text}</div></div>'
    )

def add_figure_img(title, img_src_base64, intro=None):
    """Si tu as déjà un b64 (png) pour un graphe."""
    block = ['<div class="section">']
    if title: block.append(f"<h2>{title}</h2>")
    if intro: block.append(f'<div class="intro">{intro}</div>')
    block.append(f'<img class="report-fig" src="data:image/png;base64,{img_src_base64}"/>')
    block.append('</div>')
    REPORT_HTML_BLOCKS.append("".join(block))

# Hook: appelle ton assembleur existant, puis colle les blocs HTML
def build_html_report_with_tables(title="Rapport"):
    base = build_html_report(title=title)  # ta fonction existante : texte + images actuelles
    extra = "".join(REPORT_HTML_BLOCKS)
    REPORT_HTML_BLOCKS.clear()
    # On insère les blocs HTML juste avant </body> si présent
    if "</body>" in base:
        return base.replace("</body>", extra + "</body>")
    return base + extra

