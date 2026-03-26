from pathlib import Path


def test_main_entrypoint_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "MainApp.py").exists()


def test_apps_package_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "apps" / "__init__.py").exists()
