# REFACTORING_DECISIONS

Décisions techniques structurantes de la refactorisation.

## D001 - Iso-fonctionnalité d'abord

- Statut: accepté
- Raison: réduire le risque métier
- Conséquence: ordre des étapes et sorties gardés inchangés en phase 1

## D002 - Centraliser la gestion des DataFrames

- Statut: implémentation en cours (phase 2)
- Raison: trop de noms `df_*` et branchements implicites
- Conséquence: création d'un registry (`core/df_registry.py`) et d'alias de transition

## D003 - Séparer calcul et affichage

- Statut: proposé (phase 3)
- Raison: robustesse aux reruns Streamlit + UX plus claire
- Conséquence: modules orientés fonctions de calcul, rendu consolidé en fin de pipeline
