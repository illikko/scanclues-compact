# Référence de refonte / architecture - `app_survey`

## 1. Objectif

Cette note décrit l’architecture cible et l’état actuel de la refonte de `app_survey`.

Objectifs de la refonte :
- stabiliser l’exécution Streamlit malgré les reruns ;
- séparer autant que possible calcul, orchestration et rendu ;
- centraliser les sorties consolidées dans `RapportFinal` ;
- rendre le module Q&A réellement agentique, réutilisable et extensible ;
- réduire la logique UI historique devenue inutile.

Cette note ne remplace pas `Agents.md` :
- `Agents.md` fixe les règles de contribution ;
- ce document décrit l’architecture effective, les contrats et les chantiers restants.

---

## 2. Vue d’ensemble de l’application

Pipeline utilisateur principal :
1. `Download`
2. `DiagnosticGlobal`
3. `RapportFinal`
4. `QA`

Principe :
- les modules produisent des données, synthèses, tableaux et figures dans `st.session_state` ;
- `PipelineRunner` orchestre l’exécution silencieuse ;
- `RapportFinal` est le point principal de restitution ;
- `QA` réutilise les mêmes artefacts et peut déclencher des analyses complémentaires.

---

## 3. Composants structurants

### 3.1 Navigation et orchestration

- [MainApp.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/MainApp.py)
  - navigation générale ;
  - ordre des étapes ;
  - affichage du module actif.

- [apps/PipelineRunner.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/PipelineRunner.py)
  - exécution silencieuse du pipeline standard ;
  - journalisation via `pipeline_execution_logs` ;
  - statut global du pipeline ;
  - séquencement des modules systématiques et conditionnels.

### 3.2 Cadrage et contexte d’analyse

- [apps/DiagnosticGlobal.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/DiagnosticGlobal.py)
  - écran de cadrage métier ;
  - choix des options utilisateur ;
  - alimentation du contexte d’analyse.

- [core/analysis_context_resolver.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/core/analysis_context_resolver.py)
  - normalisation des options ;
  - résolution du contexte minimal partagé entre pipeline et QA.

- [core/analysis_capabilities.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/core/analysis_capabilities.py)
  - registre des capacités analytiques ;
  - description des actions disponibles ;
  - aide à la planification agentique.

### 3.3 Restitution standard

- [apps/RapportFinal.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/RapportFinal.py)
  - restitution consolidée ;
  - affichage des synthèses, graphiques, tableaux et logs ;
  - affichage des détails de préparation quand l’option est activée.

- [core/report_export.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/core/report_export.py)
  - export HTML ;
  - construction des sections d’export à partir de `session_state`.

### 3.4 Q&A agentique

- [apps/QA.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/QA.py)
  - planification des actions QA ;
  - rendu conversationnel ;
  - exécution de capacités complémentaires ;
  - relances et mémoire de conversation.

- [core/qa_memory.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/core/qa_memory.py)
  - historique Q&A ;
  - résumé conversationnel ;
  - relances proposées.

- [core/segment_context.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/core/segment_context.py)
  - résolution de segments depuis une question ;
  - contextualisation d’une catégorie ;
  - tables effectifs / pourcentages pour un segment.

---

## 4. Architecture métier actuelle

### 4.1 Préparation des données

Modules principaux :
- [apps/Download.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/Download.py)
- [apps/Preparation1.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/Preparation1.py)
- [apps/Preparation2.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/Preparation2.py)
- [apps/DiagnosticMissing.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/DiagnosticMissing.py)
- [apps/Outliers.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/Outliers.py)
- [apps/ManquantesStructurelles.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/ManquantesStructurelles.py)
- [apps/LabelShortening.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/LabelShortening.py)

État actuel :
- les synthèses et les artefacts utiles sont disponibles pour `RapportFinal` ;
- l’option `Détails de la préparation` existe ;
- une partie du rendu détaillé reste encore à reprendre proprement ;
- la présentation de la préparation n’est pas encore au niveau cible.

### 4.2 Modules systématiques du pipeline

- [apps/AnalyseFactorielle.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/AnalyseFactorielle.py)
- [apps/AnalyseCorrelations.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/AnalyseCorrelations.py)
- [apps/Segmentation.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/Segmentation.py)

État actuel :
- fonctionnement piloté par le pipeline silencieux ;
- dépendance aux widgets réduite dans les cas standards ;
- sortie encore partiellement couplée à une logique UI historique.

### 4.3 Analyse descriptive conditionnelle

- [apps/DiagramSankey.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/DiagramSankey.py)
- [apps/CrosstabsDetail.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/CrosstabsDetail.py)
- [apps/DistributionsDetail.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/DistributionsDetail.py)
- [apps/DistributionVariables.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/DistributionVariables.py)
- [apps/Profils_y.py](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/apps/Profils_y.py)

État actuel :
- Sankey, latents et tris croisés sont de nouveau intégrés au flux standard ;
- QA peut déclencher des distributions, des profils de segment, des analyses relationnelles et `Profils_y` sur segment ;
- l’architecture reste encore plus couplée qu’une vraie architecture “compute-first”.

---

## 5. Contrats inter-modules importants

### 5.1 Données

- `df_raw`
- `df_ready`
- `process`

### 5.2 Cadrage

- `analysis_options`
- `analysis_context`
- `target_variables`
- `illustrative_variables`
- `target_modalities`
- `details_preparation_selected`

### 5.3 Pipeline

- `pipeline_ready_to_run`
- `pipeline_executed`
- `pipeline_status`
- `pipeline_execution_logs`
- `pipeline_execution_seconds`
- `pipeline_execution_plan`
- `pipeline_execution_stages`

### 5.4 Restitution standard

- `data_preparation_synthesis`
- `global_synthesis`
- `final_report_ready`
- `final_export_zip_bytes`

### 5.5 Sankey / descriptif

- `sankey_diagram`
- `sankey_diagram_base64`
- `sankey_interpretation_synthesis`
- `sankey_latents`
- `latent_summary_text`
- `crosstabs_interpretation`
- `sankey_pair_results`

### 5.6 Distributions / profils

- `dominant_continues`
- `dominant_discretes`
- `profil_dominant_analysis`
- `figs_variables_distribution`
- `qa_segment_profile_text`
- `qa_segment_profils_y_text`

### 5.7 Q&A

- `qa_history`
- `qa_conversation_summary`
- `qa_last_followup_question`
- `qa_last_followup_questions`
- `qa_relationship_synthesis`
- `qa_segment_context`
- `qa_segment_counts_table`
- `qa_segment_percent_table`

La référence détaillée des clés reste dans :
- [docs/STATE_KEYS.md](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/docs/STATE_KEYS.md)

---

## 6. Ce qui a déjà été refondu

### 6.1 Côté QA

Déjà en place :
- mémoire conversationnelle ;
- relances proposées ;
- contextualisation de segment ;
- profil dominant sur sous-population ;
- `Profils_y` sur segment via cible binaire temporaire ;
- analyse de relations entre variables ;
- exploitation du catalogue `analysis_capabilities` par le planificateur.

### 6.2 Côté pipeline

Déjà en place :
- orchestration plus lisible dans `PipelineRunner` ;
- journalisation enrichie des modules ;
- meilleure intégration Sankey / latents / crosstabs ;
- détails de préparation amorcés ;
- `RapportFinal` comme point de restitution principal.

---

## 7. Chantiers encore ouverts

### 7.1 Reprise de la préparation des données

À faire :
- revoir la présentation complète de `Etapes des préparations` ;
- enrichir proprement `Détails de la préparation` ;
- mieux distinguer synthèse et détails ;
- fiabiliser la remontée de certaines étapes amont, dont l’échantillonnage.

### 7.2 Nettoyage UI historique

À poursuivre :
- retirer les rendus intermédiaires encore inutiles ;
- supprimer les branches héritées sans utilité métier ;
- réduire la dépendance aux widgets dans les modules appelés par pipeline et QA.

### 7.3 Durcissement QA

À poursuivre :
- améliorer encore la qualité des suggestions ;
- mieux distinguer réponse depuis artefacts existants et recalcul complémentaire ;
- réduire les heuristiques ad hoc encore présentes dans `QA.py`.

### 7.4 Consolidation encodage

Risque toujours sensible :
- certains fichiers ont déjà subi du mojibake ;
- toute modification sur ces zones doit être faite avec une base saine ;
- ne jamais réintroduire de texte corrompu dans l’UI, les prompts ou les exports.

---

## 8. Règles de modification recommandées

Avant modification :
1. lire le module ;
2. lire son orchestrateur ;
3. lire son consommateur principal.

Exemples :
- module producteur : `DiagramSankey`
- orchestrateur : `PipelineRunner`
- consommateur : `RapportFinal` ou `QA`

Pendant modification :
- modifier au plus petit périmètre ;
- préserver les clés `session_state` existantes ;
- éviter tout patch transverse si un lot fermé suffit ;
- préférer les sorties explicites aux heuristiques implicites.

Après modification :
- vérifier le pipeline standard ;
- vérifier le rapport ;
- vérifier le Q&A si la clé ou l’artefact est partagé ;
- vérifier les logs de pipeline ;
- vérifier l’encodage visible.

---

## 9. Trajectoire cible

Architecture cible à moyen terme :
- modules métier plus proches de briques “compute-first” ;
- orchestration standard via `PipelineRunner` ;
- restitution standard via `RapportFinal` ;
- orchestration complémentaire via `QA` ;
- catalogue de capacités central dans `core/analysis_capabilities.py`.

En pratique, cela signifie :
- moins de mini-apps Streamlit autonomes ;
- plus de sorties métier explicites ;
- moins de logique cachée dans les widgets ;
- plus de cohérence entre pipeline standard et Q&A.

---

## 10. Documents associés

- [Agents.md](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/Agents.md)
- [docs/STATE_KEYS.md](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/docs/STATE_KEYS.md)
- [pilotage/PROJECT.md](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/pilotage/PROJECT.md)
- [pilotage/ACTION_PLAN.md](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/pilotage/ACTION_PLAN.md)
- [pilotage/TEST_LOG.md](/C:/Users/casta/Documents/scanClues/_Surveys/Appli/AppStreamlit/app_survey/pilotage/TEST_LOG.md)
