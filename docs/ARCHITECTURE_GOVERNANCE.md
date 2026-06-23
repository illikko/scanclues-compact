# Refactor de gouvernance — orchestration et Q&A

## Problème traité

Le repo contenait plusieurs niveaux d'orchestration concurrents :

1. `MainApp.py` gérait la navigation et les flags d'étapes.
2. `apps/PipelineRunner.py` exécutait le pipeline d'analyse.
3. `core/brief_agent.py` replanifiait certaines analyses avec un appel LLM.
4. `apps/QA_old.py` replanifiait à son tour, relançait des modules, manipulait des flags globaux, et marquait une étape `etape41_terminee` alors que le Q&A est potentiellement sans fin.

Ce refactor ne se limite pas à déplacer des fichiers : il réduit réellement les responsabilités concurrentes.

## Décisions de gouvernance

### 1. Le Q&A n'est plus une étape terminable

`etape41_terminee` est supprimé du chemin actif.

Le Q&A reste accessible comme écran de navigation, mais il n'a plus de flag de progression. Il peut donc recevoir un nombre indéfini de questions sans chercher à terminer l'application.

### 2. `MainApp.py` ne connaît plus que les étapes terminables

`ETAPES` accepte maintenant `cle_session=None` pour les écrans ouverts/non terminables, comme le Q&A.

### 3. Le brief n'est plus un orchestrateur LLM concurrent

`core/brief_agent.py` ne déclenche plus d'appel OpenAI. Il résout déterministiquement :

- cible détectée dans le brief ;
- variables illustratives ;
- besoin de tris croisés/Sankey ;
- besoin de distributions ;
- plan indicatif.

`PipelineRunner` reste le seul endroit qui applique ce contexte dans le pipeline.

### 4. Le Q&A ne relance plus `brief_agent`

Le Q&A utilise son propre planner pour répondre aux questions, mais ne relance plus le brief agent. Cela supprime une couche de planification cachée.

### 5. Le Q&A réduit ses appels LLM directs

Dans `apps/QA.py`, les appels OpenAI directs passent de 4 à 2 :

- 1 appel pour planifier la réponse/action Q&A ;
- 1 appel pour rédiger la réponse finale.

L'appel LLM séparé de synthèse relationnelle a été remplacé par une synthèse déterministe des artefacts déjà calculés.
La branche spéciale “réponse depuis artefacts existants” a été fusionnée avec la génération finale.

## Fichiers modifiés

- `MainApp.py`
- `apps/QA.py`
- `apps/RapportFinal.py`
- `core/brief_agent.py`
- `core/reset_state.py`

## Fichiers archivés

- `apps/QA_old.py` → `docs/archive/apps/QA_old.py`

## Ce qui reste volontairement à faire

Ce refactor n'a pas encore supprimé toute la complexité interne du Q&A. Il établit d'abord la gouvernance :

- `MainApp.py` = navigation ;
- `PipelineRunner.py` = orchestration d'analyse ;
- `brief_agent.py` = résolution déterministe du brief ;
- `QA.py` = interaction conversationnelle, sans flag de fin.

Une étape suivante peut extraire proprement la logique Q&A dans un seul fichier utilitaire si nécessaire, mais il ne faut pas ajouter un package `core/qa/` avant d'avoir stabilisé ces frontières.
