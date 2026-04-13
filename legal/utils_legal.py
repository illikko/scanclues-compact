import os

def load_markdown(file_path):
    if not os.path.exists(file_path):
        return "Contenu indisponible."
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()