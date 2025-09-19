import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Konfiguration für Grafik 2 ---
# Hardgecodeter, absoluter Pfad zu Ihrer CSV-Datei
BASELINE_CSV_PATH = '/Users/christophhau/Desktop/HA_Projekt/MIMIC/MIMIC/results/baseline_comparative_analysis/combined_baseline_metrics.csv'

def set_apa_style():
    """Wendet grundlegende APA-Formatierungsrichtlinien an."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica'],
        'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
        'grid.color': '#dddddd', 'grid.linestyle': '--', 'grid.linewidth': 0.5,
        'axes.edgecolor': 'black', 'axes.linewidth': 1.5,
        'xtick.major.size': 5, 'ytick.major.size': 5, 'xtick.labelsize': 10,
        'ytick.labelsize': 10, 'axes.labelsize': 12, 'legend.fontsize': 10,
        'figure.dpi': 150
    })

def create_baseline_plot():
    """Lädt die Baseline-Daten und erstellt das Balkendiagramm."""
    try:
        df = pd.read_csv(BASELINE_CSV_PATH)
    except FileNotFoundError:
        print(f"FEHLER: Die Datei wurde nicht gefunden unter: {BASELINE_CSV_PATH}")
        print("Bitte überprüfen Sie den Pfad und Dateinamen exakt.")
        return
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist beim Lesen der CSV aufgetreten: {e}")
        return

    # Hardgecodete Spaltennamen, die wir erwarten
    POST_METRIC_COL = 'posts_content_accuracy'
    REPLY_METRIC_COL = 'replies_mean_bertscore_f1'
    CONDITION_COL = 'condition'
    
    # Überprüfen, ob die Spalten existieren
    required_cols = [POST_METRIC_COL, REPLY_METRIC_COL, CONDITION_COL]
    if not all(col in df.columns for col in required_cols):
        print("FEHLER: Eine oder mehrere der benötigten Spalten fehlen in der CSV-Datei.")
        print(f"Benötigt werden: {required_cols}")
        print(f"Gefunden wurden: {list(df.columns)}")
        return

    # Robuste Zuordnung der Konditionsnamen
    def map_condition(condition_str):
        if str(condition_str).startswith('no_persona'): return 'Ohne Persona'
        if str(condition_str).startswith('generic'): return 'Generische Persona'
        if str(condition_str).startswith('history_only'): return 'Nur Historie'
        if str(condition_str).startswith('best_persona'): return 'Beste Persona'
        return condition_str
    
    df[CONDITION_COL] = df[CONDITION_COL].apply(map_condition)

    # Aggregation der Daten
    agg_df = df.groupby(CONDITION_COL).agg(
        posts_mean=(POST_METRIC_COL, 'mean'),
        posts_sem=(POST_METRIC_COL, lambda x: x.std() / np.sqrt(len(x)) if len(x) > 0 else 0),
        replies_mean=(REPLY_METRIC_COL, 'mean'),
        replies_sem=(REPLY_METRIC_COL, lambda x: x.std() / np.sqrt(len(x)) if len(x) > 0 else 0)
    ).reindex(['Ohne Persona', 'Generische Persona', 'Nur Historie', 'Beste Persona'])
    
    # Plotting-Logik
    set_apa_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    index = np.arange(len(agg_df))
    bar_width = 0.35

    ax.bar(index - bar_width/2, agg_df['posts_mean'], bar_width, yerr=agg_df['posts_sem'], capsize=5, label='Post-Imitation (Content Accuracy)', color='#a9a9a9')
    ax.bar(index + bar_width/2, agg_df['replies_mean'], bar_width, yerr=agg_df['replies_sem'], capsize=5, label='Reply-Generierung (BERTScore F1)', color='#696969')

    ax.set_xlabel('Baseline-Kondition')
    ax.set_ylabel('Qualitäts-Score (Mittelwert)')
    ax.set_xticks(index)
    ax.set_xticklabels(agg_df.index, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    save_path = 'figure_2_baseline_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"Grafik 2 (Baseline-Vergleich) wurde als '{os.path.abspath(save_path)}' gespeichert.")

if __name__ == "__main__":
    create_baseline_plot()