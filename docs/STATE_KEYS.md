# STATE_KEYS

Inventaire des clés importantes de `st.session_state` observées dans le code et de celles ajoutées pour la refonte en cours.

## Données

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

## Navigation et étapes

- `etape1_terminee` ... `etape41_terminee`
- `__NAV_MODE__`
- `__NAV_SELECTED__`
- `__NAV_CONTEXT__`

## Pilotage d'analyse

- `pipeline_selection`
- `analysis_options`
- `analysis_context`
- `details_preparation_selected`
- `preparation_details_payload`
- `preparation_details_ready`
- `pipeline_ready_to_run`
- `pipeline_config`
- `pipeline_diagnostics`
- `pipeline_prep_tasks`
- `pipeline_execution_plan`
- `pipeline_execution_stages`

## Paramètres d'analyse

- `num_quantiles`
- `distinct_threshold_continuous`
- `mod_freq_min`
- `correlation_threshold_v`
- `outliers_percent_target`
- `n_clusters_segmentation`
- `n_clusters_target`
- `kmodes_n_init`
- `high_freq_threshold`
- `sankey_alpha`
- `sankey_v_min`
- `sankey_max_length_links`

## Convention cible

- `target_variables`
- `illustrative_variables`
- `target_modalities`
- `brief_target_variable`
- `brief_illustrative_variables`
- `dataset_key_questions_mode`
- `dataset_key_questions_value`
- `dataset_key_questions_value_saved`
- `dataset_key_questions`

## Résultats d'analyse

- `interpretationACM`
- `dendrogram_interpretation`
- `latent_summary_text`
- `sankey_pair_results`
- `sankey_interpretation_synthesis`
- `segmentation_profiles_text`
- `profils_y_text`
- `global_synthesis`
- `data_preparation_synthesis`

## Rapport final

- `report_items`
- `_rf_blocks`
- `final_report_html`
- `final_report_ready`
- `final_export_zip_bytes`

## Q&A

- `qa_history`
- `qa_conversation_summary`
- `qa_last_followup_question`
- `qa_last_followup_questions`
- `qa_last_plan`
- `qa_last_answer`
- `qa_last_execution_log`
- `qa_last_question`
- `qa_last_subset_description`
- `qa_segment_context`
- `qa_segment_counts_table`
- `qa_segment_percent_table`
- `qa_segment_subdataset`
- `qa_segment_profile_text`
- `qa_segment_profils_y_text`
- `qa_relationship_synthesis`

### Details QA

- `qa_history` : historique structure des tours Q&A. Chaque entree conserve la question, la reponse, les relances, les actions executees et les artefacts utilises.
- `qa_conversation_summary` : resume compact des derniers tours, reutilise dans les prompts QA pour garder le contexte conversationnel.
- `qa_last_followup_question` : premiere relance retenue pour le dernier tour.
- `qa_last_followup_questions` : liste des 2 a 3 relances proposees a la fin du dernier tour.
- `qa_last_plan` : plan agentique sanitise du dernier tour, apres passage du planificateur LLM et des heuristiques locales.
- `qa_last_answer` : dernier JSON de reponse final genere pour l'utilisateur (`intro`, `answer`, `followup_questions`).
- `qa_last_execution_log` : log des actions effectivement executees au dernier tour, avec statuts et sorties produites.
- `qa_last_question` : derniere question brute saisie par l'utilisateur, avant expansion eventuelle d'une reponse courte a une relance.
- `qa_last_subset_description` : description textuelle de la sous-population courante quand QA travaille sur un segment.
- `qa_segment_context` : dictionnaire de contextualisation du segment courant (`column`, `value`, `description`, effectif, part).
- `qa_segment_counts_table` : tableau des effectifs par modalite pour la variable du segment courant.
- `qa_segment_percent_table` : tableau des pourcentages par modalite pour la variable du segment courant.
- `qa_segment_subdataset` : sous-DataFrame correspondant au segment detecte ou demande dans la question.
- `qa_segment_profile_text` : profil dominant calcule par `DistributionVariables` sur la sous-population du segment, sans ecraser le profil global.
- `qa_segment_profils_y_text` : analyse detaillee de type `Profils_y` produite sur une cible binaire temporaire derivee du segment.
- `qa_relationship_synthesis` : synthese textuelle QA pour une question portant sur des relations entre variables, apres tris croises et distributions.

## Registry de DataFrames

- Registry central implémenté : `core/df_registry.py`
- Alias de transition actifs : `df_*` legacy <-> états canoniques
- API de base :
  - `set_df(state, df, step_name=...)`
  - `get_df(state)`
  - `sync_registry_from_aliases()`
  - `sync_aliases_from_registry()`
