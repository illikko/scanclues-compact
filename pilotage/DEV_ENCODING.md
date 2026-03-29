# Encodage des fichiers (règle pratique)

- Toujours lire/écrire les sources en **UTF-8**.
- Modifications : privilégier `apply_patch` ou un éditeur configuré en UTF-8.
- Si une réécriture complète est nécessaire, utiliser `Set-Content -Encoding UTF8` (ou un script Python avec `encoding="utf-8"`), jamais les redirections `>` / `>>` sans encodage.
- Éviter `Out-File` / `Set-Content` sans `-Encoding UTF8`.
- Contrôle rapide : `python -c "open('fichier.py', encoding='utf-8').read()"` sur les fichiers modifiés.
