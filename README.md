# app_survey

Application Streamlit d'analyse de données d'enquêtes, orientée parcours guidé, pipeline automatisé et génération de rapport final exploitable.

## Objectif

Le dépôt fournit une application pour :

- importer un jeu de données d'enquête ;
- préparer et diagnostiquer les données ;
- cadrer les objectifs d'analyse ;
- exécuter un pipeline d'analyse sélectionné ;
- produire un rapport final HTML et un export ZIP ;
- permettre un Q&A sur les artefacts déjà calculés.

Le point d'entrée principal est [`MainApp.py`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/MainApp.py).

## Parcours utilisateur

La navigation principale est structurée en 4 écrans :

1. `Upload`
2. `Définition des objectifs`
3. `Rapport Final`
4. `Q&A`

Principe de fonctionnement :

- les modules métier calculent et stockent leurs sorties dans `st.session_state` ;
- le rendu consolidé est centralisé dans [`apps/RapportFinal.py`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/RapportFinal.py) ;
- le Q&A réutilise les artefacts déjà générés plutôt que de rejouer librement tout le pipeline ;
- l'application est pensée pour le modèle de rerun Streamlit.

## Architecture

### Structure du dépôt

- [`MainApp.py`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/MainApp.py) : orchestration de la navigation.
- [`apps/`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps) : écrans Streamlit et modules d'analyse.
- [`core/`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/core) : briques transverses, état, export, QA, registry DataFrame.
- [`legal/`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/legal) : pied de page et contenus légaux.
- [`docs/`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/docs) : gouvernance, architecture et clés de session.
- [`tests/`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/tests) : tests smoke et tests de registry.

### Modules clés

- [`apps/PipelineRunner.py`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/PipelineRunner.py) : exécution du pipeline d'analyse.
- [`apps/RapportFinal.py`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/RapportFinal.py) : synthèse finale, rendu et export.
- [`apps/QA.py`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/QA.py) : questions/réponses sur les résultats et segments.
- [`core/df_registry.py`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/core/df_registry.py) : registry canonique des DataFrames.
- [`core/reset_state.py`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/core/reset_state.py) : remise à zéro complète de l'application.
- [`core/report_export.py`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/core/report_export.py) : génération des exports HTML / ZIP.

## Prérequis

- Python `3.10`
- `pip`
- une clé OpenAI si vous utilisez les synthèses/appels LLM

Les versions sont épinglées dans [`requirements.txt`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/requirements.txt). Les outils qualité attendus sont configurés dans [`pyproject.toml`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/pyproject.toml).

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Variables d'environnement

Exemple minimal :

```powershell
$env:OPENAI_API_KEY="votre_cle_openai"
```

Variables utiles :

- `OPENAI_API_KEY` : requise pour les parties qui appellent OpenAI.
- `INVITE_CODES` : optionnelle. Si renseignée, active un contrôle d'accès par code dans [`auth.py`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/auth.py). Plusieurs codes sont acceptés, séparés par virgules ou retours ligne.

Un exemple de fichier est présent dans [.env.example](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/.env.example).

## Lancement

```powershell
streamlit run MainApp.py
```

Un script Windows local existe aussi : [`MainApp.bat`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/MainApp.bat).

## Comportement applicatif

### Session state

L'application repose fortement sur `st.session_state`. Les clés critiques couvrent notamment :

- les données : `df_raw`, `df_ready`, `process` ;
- le cadrage : `dataset_object`, `dataset_context`, `dataset_recommendations`, `target_variables`, `illustrative_variables`, `target_modalities` ;
- les sorties analytiques : `sankey_diagram`, `crosstabs_interpretation`, `profil_dominant_analysis`, `figs_variables_distribution` ;
- le rapport : `global_synthesis`, `data_preparation_synthesis`, `final_report_ready`, `final_export_zip_bytes` ;
- le pipeline : `pipeline_ready_to_run`, `pipeline_executed`, `pipeline_status`, `pipeline_execution_logs`, `pipeline_execution_seconds`.

Référence utile : [`docs/STATE_KEYS.md`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/docs/STATE_KEYS.md).

### Performance

Le dépôt prévoit des garde-fous pour limiter les calculs lourds inutiles, notamment via :

- `run_sankey_crosstabs`
- `generate_distribution_figures`

En pipeline silencieux, les calculs non essentiels à la synthèse immédiate doivent rester désactivables.

### Encodage

Le projet cible prioritairement un usage francophone. Les fichiers texte et prompts doivent rester en UTF-8, sans perte d'accents ni réencodage opportuniste.

## Développement

### Tests

```powershell
pytest -q
```

Les tests actuellement présents couvrent surtout :

- l'existence du point d'entrée et du package `apps` ;
- le comportement du registry DataFrame et la compatibilité avec les alias historiques.

### Qualité de code

```powershell
ruff check .
black --check .
```

## Documentation interne

- [`Agents.md`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/Agents.md) : règles de contribution et contraintes d'architecture.
- [`docs/ARCHITECTURE_GOVERNANCE.md`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/docs/ARCHITECTURE_GOVERNANCE.md) : gouvernance d'orchestration, pipeline et Q&A.
- [`docs/REFACTOR_CHANGELOG.md`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/docs/REFACTOR_CHANGELOG.md) : historique de refactorisation.
- [`docs/STATE_KEYS.md`](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/docs/STATE_KEYS.md) : inventaire des clés de `session_state`.

## Points de vigilance

- ne pas casser l'ordre `Upload -> Diagnostic global -> Rapport final -> Q&A` ;
- ne pas modifier une clé liée à un widget après instanciation ;
- ne pas renommer une clé critique de `session_state` sans plan de migration ;
- éviter les recalculs complets si `final_report_ready=True` ;
- préserver la compatibilité avec le mode pipeline silencieux ;
- vérifier toute modification sur les flux avec et sans brief.
