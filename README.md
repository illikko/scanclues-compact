# app_survey

Application Streamlit d'analyse de données d'enquêtes.

## Pré-requis

- Python 3.10
- pip

## Installation

```powershell
cd app_survey
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```powershell
streamlit run MainApp.py
```

## Structure

- `MainApp.py`: orchestration des étapes
- `apps/`: modules d'analyse
- `utils.py`: fonctions utilitaires
- `auth.py`: authentification
- `REFACTORING_REFERENCE.md`: feuille de route refacto

## Tests et qualité

```powershell
pytest -q
ruff check .
black --check .
```

## Notes

- Les résultats consolidés sont produits via le module `apps/RapportFinal.py`.
- La stratégie de refactorisation est documentée dans `REFACTORING_REFERENCE.md`.
