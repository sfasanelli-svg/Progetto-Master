"""
Distribuzione del gap_score: confronto tra la provincia di Milano e l'intera
Italia, per individuare visivamente il range in cui la situazione si puo'
considerare "riequilibrata" (gap_score vicino a 0).

Il gap_score e' una differenza di due ranghi percentile (rank%(domanda) -
rank%(offerta)), quindi per costruzione oscilla in [-1, +1] con 0 = domanda
e offerta pari rango (equilibrio), +1 = deserto assoluto, -1 = sezione tra le
piu' servite d'Italia in rapporto alla domanda.

Richiede: pip install pandas numpy matplotlib
(scipy opzionale, solo per la curva di densita' KDE in aggiunta all'istogramma)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------- parametri
# Percorso assoluto: in un notebook VS Code il kernel parte con la working
# directory dell'eseguibile di VS Code, quindi ne' __file__ ne' Path.cwd()
# sono affidabili qui. Metti direttamente la cartella dove salvi il parquet
# (di default, quella dello script) e, se lo sposti altrove, aggiorna questo
# percorso.
CARTELLA_SCRIPT = Path(r"C:\Users\fasanelli michele\OneDrive\Desktop\Contesto lavoro di gruppo ETL")
PARQUET = CARTELLA_SCRIPT / "sezioni_gap_score_DEFINITIVO.parquet"
PROVINCIA_TARGET = "Milano"

# Soglia illustrativa di "equilibrio": e' una scelta metodologica, non un
# valore statistico oggettivo. +-0.10 e' un punto di partenza ragionevole
# (circa un quarto della dispersione tipica del punteggio); il grafico e le
# statistiche stampate sotto (in particolare P25/P75) servono proprio a
# permettere al gruppo di ridiscutere questo valore con dati alla mano.
SOGLIA_EQUILIBRIO = 0.10

OUT_PNG = CARTELLA_SCRIPT / "distribuzione_gap_score.png"


def carica_gap_score(parquet_path):
    df = pd.read_parquet(
        parquet_path,
        columns=["PROVINCIA", "flag_eleggibile_EV", "offerta_colonnine_500m", "gap_score"],
    )
    df["flag_eleggibile_EV"] = df["flag_eleggibile_EV"].map({True: True, False: False}).fillna(False).astype(bool)
    el = df.flag_eleggibile_EV & df.offerta_colonnine_500m.notna()
    return df.loc[el, ["PROVINCIA", "gap_score"]]


def stampa_statistiche(nome, serie):
    print(f"\n--- {nome} (n={len(serie):,}) ---")
    print(serie.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_string())
    quota_equilibrata = (serie.abs() <= SOGLIA_EQUILIBRIO).mean()
    print(f"Quota entro +-{SOGLIA_EQUILIBRIO}: {quota_equilibrata:.1%}")


def disegna(ax, serie, etichetta, colore):
    ax.hist(serie, bins=80, range=(-1, 1), density=True, alpha=0.35, color=colore,
            label=f"{etichetta} (istogramma)")
    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(-1, 1, 400)
        kde = gaussian_kde(serie.dropna())
        ax.plot(xs, kde(xs), color=colore, linewidth=2, label=f"{etichetta} (densita')")
    except ImportError:
        pass


def main():
    print(f"--- Carico {PARQUET.name} ---")
    dati = carica_gap_score(PARQUET)
    print(f"Sezioni eleggibili nazionali: {len(dati):,}")

    italia = dati["gap_score"]
    milano = dati.loc[dati.PROVINCIA.str.upper() == PROVINCIA_TARGET.upper(), "gap_score"]
    print(f"Sezioni eleggibili in provincia di {PROVINCIA_TARGET}: {len(milano):,}")

    stampa_statistiche("Italia (tutte le sezioni eleggibili)", italia)
    stampa_statistiche(PROVINCIA_TARGET, milano)

    fig, ax = plt.subplots(figsize=(11, 6))
    disegna(ax, italia, "Italia", "#4C72B0")
    disegna(ax, milano, PROVINCIA_TARGET, "#DD8452")

    ax.axvline(0, color="black", linewidth=1, label="equilibrio perfetto (0)")
    ax.axvspan(-SOGLIA_EQUILIBRIO, SOGLIA_EQUILIBRIO, color="green", alpha=0.08,
               label=f"zona di equilibrio proposta (±{SOGLIA_EQUILIBRIO})")

    ax.set_xlabel("gap_score  (+1 = deserto, -1 = ben servita, 0 = equilibrata)")
    ax.set_ylabel("densita'")
    ax.set_title(f"Distribuzione del gap_score: Italia vs provincia di {PROVINCIA_TARGET}")
    ax.set_xlim(-1, 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nGrafico salvato in: {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
