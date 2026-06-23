# Changelog refactor gouvernance

## Changements substantiels

- Suppression de `etape41_terminee` du chemin actif.
- `MainApp.ETAPES` supporte `cle_session=None` pour les écrans non terminables.
- Création de `apps/QA.py` actif depuis l'ancien `QA_old.py`.
- Archivage de l'ancien module dans `docs/archive/apps/QA_old.py`.
- Suppression du relancement de `run_brief_agent()` depuis le Q&A.
- Remplacement de la synthèse relationnelle LLM par une synthèse déterministe.
- Fusion de la double branche de génération de réponse finale.
- Correction du bouton Q&A : `>` devient `➜`.
- Nettoyage de l'état input Q&A : `qa_chat_input` est le champ réellement nettoyé.
- `core/brief_agent.py` devient déterministe et n'appelle plus OpenAI.

## Vérifications

```bash
python -m py_compile MainApp.py apps/QA.py apps/PipelineRunner.py apps/RapportFinal.py core/brief_agent.py core/reset_state.py
```

OK.

## Compteurs

- `apps/QA.py` : 2 appels directs `chat.completions.create` au lieu de 4.
- `core/brief_agent.py` : 0 appel OpenAI.
- `etape41_terminee` : absent du chemin actif (`MainApp.py`, `apps/`, `core/`).
