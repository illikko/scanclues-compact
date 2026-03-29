# Agents.md - Guide de contribution (humains + agents IA)

Ce document définit les règles de travail pour modifier l'application Streamlit `app_survey` sans casser les flux existants.

## 1) Objectif

Garantir des modifications :
- robustes (pas de régressions fonctionnelles),
- lisibles (modules clairs et responsabilités séparées),
- compatibles avec le comportement Streamlit (rerun à chaque interaction).

## 2) Périmètre de l'application

Pipeline principal utilisateur :
1. Upload / Préparation initiale
2. Diagnostic global (choix des traitements)
3. Rapport final (exécution pipeline + rendu final)
4. Q&A

Principe clé :
- Les modules font surtout du calcul + stockage en `st.session_state`.
- L'affichage final consolidé se fait dans `RapportFinal`.

## 3) Règles d'architecture

### 3.1 Séparation stricte des responsabilités
Dans chaque module métier, structurer le code en 3 blocs :
1. Diagnostic (faut-il traiter ?)
2. Traitement (si diagnostic positif)
3. Sorties (données/textes/figures stockées)

### 3.2 Affichage
- Éviter les affichages intermédiaires coûteux dans le pipeline automatique.
- Prioriser l'affichage dans `RapportFinal`.
- En mode pipeline silencieux, ne garder que les sorties nécessaires.

### 3.3 Session state
- Toute sortie inter-module doit être explicitement stockée dans `st.session_state`.
- Ne jamais supposer qu'un état est présent : utiliser `get` / `setdefault`.
- Ne pas modifier une clé liée à un widget après instanciation du widget.

### 3.4 Encodage et langue (FR prioritaire)
- L'application est destinée à des francophones : les caractères spéciaux (é, è, ê, à, ç, ï, etc.) doivent être conservés et correctement affichés dans l'interface et les artefacts.
- Ne pas "simplifier" les libellés en retirant les accents, sauf demande explicite.
- Lire et écrire tous les fichiers texte en UTF-8.
- Conserver exactement les caractères existants lors des copier/coller vers ou depuis l’application.
- Ne jamais convertir des caractères comme é, è, à, ç, œ, €, ’, “ ”, — en versions corrompues.
- Si un texte venant de l’application contient des caractères déjà corrompus, s’arrêter et le signaler au lieu de le réimporter.
- Upload : accepter plusieurs encodages, avec priorité pratique à `latin-1` pour les fichiers utilisateurs, puis fallback (ex. `utf-8`/`cp1252`) selon l'implémentation du module d'import.
- Exports CSV destinés à Excel FR : `sep=';'` + encodage adapté (`latin-1` si requis).

## 4) Règles de performance

### 4.1 Modes rapides vs complets
Les traitements lourds doivent être activables/désactivables via flags `session_state` :
- `run_sankey_crosstabs`
- `generate_distribution_figures`

Par défaut en pipeline silencieux :
- désactiver les calculs lourds non essentiels à la synthèse immédiate.

### 4.2 Pas de recalcul inutile
- Si `final_report_ready=True`, ne pas regénérer les synthèses.
- Le recalcul complet doit être déclenché par une action explicite (ex. réinitialiser).

### 4.3 Logs
- Toujours enrichir `pipeline_execution_logs` avec `module`, `status`.
- Conserver le temps total d'exécution (`pipeline_execution_seconds`).

## 5) Contrats inter-modules (clés critiques)

Exemples de clés critiques à préserver :
- Données : `df_raw`, `df_ready`, `process`
- Cadrage : `dataset_object`, `dataset_context`, `dataset_recommendations`, `target_variables`, `illustrative_variables`, `target_modalities`
- Sankey : `sankey_diagram`, `sankey_interpretation_synthesis`, `sankey_latents`, `crosstabs_interpretation`, `sankey_pair_results`
- Distribution : `dominant_continues`, `dominant_discretes`, `profil_dominant_analysis`, `figs_variables_distribution`
- Rapport : `global_synthesis`, `data_preparation_synthesis`, `final_report_ready`, `final_export_zip_bytes`
- Pipeline : `pipeline_ready_to_run`, `pipeline_executed`, `pipeline_status`, `pipeline_execution_logs`, `pipeline_execution_seconds`

Toute modification de nom de clé doit être accompagnée d'un plan de migration.

## 6) Zones sensibles (ne pas casser)

- Navigation principale et ordre d'étapes.
- Mécanisme de pipeline silencieux.
- Bouton réinitialiser (doit remettre l'app à un état vierge, y compris upload).
- Génération du rapport HTML + ZIP.
- Compatibilité des modules avec rerun Streamlit.

## 7) Standards de modification

### 7.1 Avant modification
- Lire le module + ses consommateurs (`RapportFinal`, `QA`, `PipelineRunner`).
- Lister les clés `session_state` lues/écrites.

### 7.2 Pendant modification
- Modifier au plus petit périmètre.
- Préserver le comportement existant à iso-fonctionnalité, sauf demande explicite.
- Ajouter des garde-fous (`isinstance(df, pd.DataFrame)`, `not empty`).

### 7.3 Après modification
Vérifier :
1. Upload -> Diagnostic global -> Rapport final -> Q&A
2. Cas `avec brief` / `sans brief`
3. Pipeline avec et sans options (`préparation`, `profilage`, `analyse descriptive`)
4. Réinitialisation complète
5. Logs + temps d'exécution

## 8) Politique LLM

- Prompts déterministes (température 0 pour synthèses structurées).
- Payload compact, utile, traçable.
- Éviter d'envoyer des objets lourds inutiles (figures complètes, matrices massives) si un résumé suffit.
- Favoriser des résumés dédiés pour Q&A (`*_summary`).

## 9) Conventions de commits / PR

- Une PR = un objectif clair.
- Décrire :
  - ce qui change,
  - pourquoi,
  - clés `session_state` impactées,
  - risques,
  - check-list de validation.
