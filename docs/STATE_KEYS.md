# STATE_KEYS

Inventaire initial des clés importantes de `st.session_state` observées dans le code.

## Données (df)

- `df_raw`
- `df_ex_verbatim`
- `df_shortlabels`
- `df_ex_ordonnees`
- `df_ex_multiples`
- `df_imputed_structural`
- `df_ready`
- `df_encoded`
- `df_active`
- `df_illustrative`

## Pilotage navigation/étapes

- `etape1_terminee` ... `etape41_terminee`
- `__NAV_MODE__`
- `__NAV_SELECTED__`

## Résultats d'analyse (exemples)

- `interpretationACM`
- `dendrogram_interpretation`
- `sankey_pair_results`
- `segmentation_profiles_text`
- `profils_y_text`
- `global_synthesis`
- `data_preparation_synthesis`

## Rapport final

- `report_items`
- `_rf_blocks`
- `final_report_html`
- `final_report_ready`

## Convention cible (phase 2)

- Registry central implémenté: `core/df_registry.py`
- Alias de transition actifs: `df_*` legacy <-> états canoniques
- API de base:
  - `set_df(state, df, step_name=...)`
  - `get_df(state)`
  - `sync_registry_from_aliases()`
  - `sync_aliases_from_registry()`
- Migration en cours: certains modules sont déjà branchés au registry, les autres restent compatibles via alias.
