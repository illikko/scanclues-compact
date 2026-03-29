# Référence de refactorisation (iso-fonctionnalité) - `app_survey`

## 1) Objectif

Restructurer l'application Streamlit sans changer les résultats métier dans un premier temps:
- stabiliser l'exécution (reruns Streamlit)
- clarifier le pipeline de données
- séparer calcul et affichage
- préparer une UX simplifiée en 3 écrans (upload -> diagnostic -> rapport final)

---

## 2) Stack technique et versions (état observé)

### Runtime
- Python: environnement d'exécution pointé par `MainApp.bat` = `clean_py310` (Python 3.10 présumé)
- Indice complémentaire: présence de `__pycache__` en `cpython-310`
- Attention: un autre dossier du workspace (`app/`) contient un `runtime.txt` en `python-3.11.9` (incohérence à corriger)

### Librairies utilisées dans `app_survey`
- streamlit
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn
- missingno
- kmodes
- prince
- plotly
- openai
- openpyxl
- pillow

### Constat
- Les versions ne sont pas figées (reproductibilité faible).

---

## 3) Carte du repo (`app_survey`)

- `MainApp.py`: orchestrateur navigation/étapes
- `auth.py`: contrôle d'accès (code d'invitation)
- `utils.py`: utilitaires transverses (process prep, discrétisation, paramètres)
- `apps/`: modules d'analyse (préparation, diagnostics, segmentation, rapport final, QA)
- `apps/_report.py`: collecte/génération des blocs du rapport
- `.streamlit/config.toml.toml`: config Streamlit (nom de fichier à corriger)

Remarques:
- état applicatif très dense dans `st.session_state` (beaucoup de clés hétérogènes)

---

## 4) Commandes standards (cible)

Depuis `app_survey`:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run MainApp.py
```

Tests/lint (cible):

```powershell
pytest -q
ruff check .
black --check .
```

---

## 5) Stratégie de test (avant/après refacto)

## Objectif de non-régression
- à dataset identique, mêmes sorties clés (tables, tailles, colonnes, valeurs agrégées principales)
- même capacité à produire le `RapportFinal` et le zip d'export

## Niveaux de tests
- Unitaires: fonctions pures extraites des modules (`apps/*`)
- Intégration: pipeline de transformation des DataFrames (de `df_raw` à `df_ready`/`df_encoded`)
- Golden tests: snapshots de sorties tabulaires (CSV de référence)
- Smoke tests UI: enchaînement minimal (upload -> exécution bloc -> rapport)

## Jeux de données
- un mini dataset synthétique (rapide)
- un dataset réel anonymisé (validation métier)

---

## 6) Règles style/lint (cible)

- formattage automatique: `black`
- lint: `ruff`
- typage progressif: annotations sur fonctions de transformation
- principe: pas d'effet de bord caché dans les fonctions métier
- séparation stricte:
  - `compute_*`: calcul uniquement
  - `render_*`: affichage Streamlit uniquement

---

## 7) Zones "ne pas toucher" (phase iso-fonctionnalité)

- logique d'authentification (`auth.py`) sauf encapsulation mineure
- prompts LLM métier existants (contenu) tant que la validation métier n'a pas été faite
- format des exports attendus par les utilisateurs finaux (`df_ready.csv`, etc.)
- clés de session existantes utilisées en production (on ajoute des alias avant suppression)

---

## 8) Notes sécurité

- clé API OpenAI: uniquement via variable d'environnement ou `st.secrets`, jamais en dur
- aucune donnée sensible dans logs/erreurs Streamlit
- limiter les aperçus envoyés au LLM (déjà partiellement fait avec `head(10)`)
- ajouter une politique explicite de masquage/anonymisation avant appel LLM (phase 2)
- vérifier les exports HTML (éviter injection via champs texte non échappés)

---

## 9) Plan de refactorisation proposé

## Phase 1 - Compléter les fichiers manquants du repo

Créer dans `app_survey`:
- `README.md`: usage, pipeline, architecture
- `requirements.txt`: versions figées
- `runtime.txt`: version Python cible (alignée env réel)
- `.gitignore`: `__pycache__`, `.venv`, `.streamlit/secrets.toml`, exports
- `pyproject.toml`: config `black`, `ruff`, `pytest`
- `tests/`:
  - `tests/test_pipeline_smoke.py`
  - `tests/test_df_registry.py` (après phase 2)
- `docs/STATE_KEYS.md`: inventaire et statut des clés `st.session_state`
- `docs/REFactoring_DECISIONS.md`: ADR légères (décisions techniques)

Critère de sortie phase 1:
- repo installable/rejouable sur une machine propre
- conventions de dev et de test écrites

## Phase 2 - Gérer les différents noms de DataFrame

Problème actuel:
- coexistence de nombreuses clés (`df_raw`, `df_ex_verbatim`, `df_shortlabels`, `df_ex_ordonnees`, `df_ready`, `df_encoded`, `df_active`, etc.)
- risque élevé d'erreurs de branchement entre modules

Solution proposée: registre central + alias de transition

1. Ajouter `app_survey/core/df_registry.py`
- enum des états canoniques:
  - `RAW`, `VERBATIM_READY`, `SHORT_LABELS`, `MULTI_ORD_DONE`, `MULTI_DONE`, `IMPUTED`, `READY`, `ENCODED`, `ACTIVE`, `ILLUSTRATIVE`
- API:
  - `set_df(state, df)`
  - `get_df(state, required=True)`
  - `set_alias(old_key, state)`

2. Migration progressive
- chaque module lit/écrit via le registry
- compatibilité maintenue via mapping alias -> anciennes clés session
- suppression des alias uniquement après validation non-régression

3. Traçabilité
- journal de transformations minimal (`from_state`, `to_state`, `rows`, `cols`, `step_name`)

Critère de sortie phase 2:
- un chemin de données unique et explicite
- disparition des accès directs aux clés `df_*` dans les modules métier

## Phase 3 - Simplifier l'UX (upload -> diagnostic -> résultats finaux)

Cible UX:
- écran A: Upload
- écran B: Diagnostic + choix utilisateur (4 boutons)
  - `PD`: préparation dataset
  - `PD + S`
  - `PD + AD + S`
  - `TOUT`
- écran C: Affichage final consolidé uniquement

Règles d'implémentation:
1. Conserver les modules (contrainte Streamlit rerun)
2. Sortir l'affichage intermédiaire des modules
- modules = fonctions de calcul et stockage résultats uniquement
- pas de rendu final dans les étapes intermédiaires
3. Centraliser le rendu dans `RapportFinal` (ou `rapport_final` cible)
- textes, graphiques, tableaux affichés uniquement en fin
4. Ajouter un orchestrateur de pipeline
- exécute conditionnellement les blocs selon le bouton choisi
- garde l'état d'avancement dans `st.session_state` pour survivre aux reruns

Pattern recommandé par module:
- `run_compute(state) -> dict` (ou écrit via registry)
- `collect_report_items(state)` pour alimenter `_report`

Critère de sortie phase 3:
- aucune visualisation métier affichée avant l'écran final
- 4 modes utilisateur fonctionnels et iso-résultats

---

## 10) Ordre d'exécution recommandé

1. Phase 1 (fichiers manquants + standardisation environnement)
2. Phase 2 (registry DataFrame + alias)
3. Phase 3 (orchestrateur UX + affichage final uniquement)
4. Nettoyage final (suppression copies, dead code, anciennes clés)

---

## 11) Risques et parades

- Risque: casse due aux reruns Streamlit
  - Parade: garder les modules, stocker les sorties calculées en session, rerender final seulement
- Risque: divergence de versions Python/libs
  - Parade: figer `runtime.txt` + `requirements.txt`
- Risque: régression fonctionnelle silencieuse
  - Parade: golden tests + comparaison snapshots avant/après
