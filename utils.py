import yaml

def load_config(config_path="config.yaml"):
    """Lädt die Konfiguration aus einer YAML-Datei."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Fehler: Konfigurationsdatei '{config_path}' nicht gefunden.")
        exit(1)
    except yaml.YAMLError as exc:
        print(f"Fehler beim Parsen der YAML-Datei: {exc}")
        exit(1)