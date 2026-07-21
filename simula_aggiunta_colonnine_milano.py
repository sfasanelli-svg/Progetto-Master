"""
Simulazione: effetto dell'aggiunta di colonnine di ricarica sul gap_score
delle sezioni di censimento della provincia di Milano.

Il gap_score e' un rango percentile NAZIONALE (rank%(domanda) - rank%(offerta)
sull'intero universo eleggibile, ~346.507 sezioni), quindi non si puo' capire
l'effetto di una colonnina in piu' guardando solo Milano: bisogna reinserire
il nuovo valore di offerta_colonnine_500m nel dataset completo e ricalcolare
il ranking su tutta Italia, poi rifiltrare su Milano.

Stessa formula e stessa funzione offerta_accessibilita() del notebook
gap_score_DEFINITIVO.ipynb.

Richiede: pip install pandas numpy pyarrow
Il file sezioni_gap_score_DEFINITIVO.parquet (593 MB) va scaricato in locale
(da Drive o Colab) e il percorso PARQUET qui sotto aggiornato di conseguenza.
"""
import numpy as np
import pandas as pd
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
N_COLONNINE_AGGIUNTE = 1

# MODE = "uno_alla_volta": per ogni sezione, simula l'aggiunta SOLO a quella
#   sezione (tutto il resto d'Italia invariato) e ricalcola il ranking
#   nazionale da capo. E' la domanda corretta ("se investo qui, quanto
#   migliora QUESTA sezione?"), ma e' O(TOP_N) ricalcoli su ~350k righe:
#   con TOP_N=50 richiede ~1-2 minuti, con migliaia di sezioni puo' volerci
#   molto di piu'.
# MODE = "tutte_insieme": aggiunge la colonnina a TUTTE le sezioni target
#   contemporaneamente e ricalcola il ranking UNA sola volta (istantaneo).
#   Risponde pero' a una domanda diversa ("se elettrifichiamo tutte insieme
#   le sezioni scelte"): le sezioni che passano da 0 a 1 colonnina insieme
#   si "spartiscono" lo stesso blocco di rango, quindi il risultato per la
#   singola sezione e' leggermente ottimistico/pessimistico a seconda di
#   quante altre sezioni salgono con lei.
MODE = "uno_alla_volta"

# Quante sezioni di Milano simulare, partendo dalle piu' critiche (gap_score
# piu' alto = deserti peggiori). Mettere None per farle tutte (solo con
# MODE="tutte_insieme", altrimenti troppo lento).
TOP_N = 50

OUT_CSV = Path(f"simulazione_gap_score_{PROVINCIA_TARGET.lower()}.csv")


def offerta_accessibilita(offerta, distanza):
    """Identica alla funzione del notebook gap_score_DEFINITIVO: rank%
    nazionale dell'accessibilita' dell'offerta. Le sezioni servite (>=1
    colonnina entro 500m) sono SEMPRE ranked sopra le non servite; tra le
    servite si ordina per numero di colonnine, tra le non servite per
    prossimita' della piu' vicina."""
    offerta = np.asarray(offerta, float)
    distanza = np.asarray(distanza, float)
    n = len(offerta)
    served = offerta > 0
    supply = np.empty(n)
    supply[~served] = pd.Series(-distanza[~served]).rank().to_numpy()
    supply[served] = (~served).sum() + pd.Series(offerta[served]).rank().to_numpy()
    return pd.Series(supply).rank(pct=True).to_numpy()


def calcola_gap(df, el):
    """Ricalcola offerta_norm e gap_score sull'universo eleggibile `el`.
    domanda_norm non cambia (la domanda non e' toccata dalla simulazione:
    solo l'offerta), quindi si riusa la colonna gia' presente nel parquet."""
    sub = df.loc[el]
    offerta_norm = offerta_accessibilita(sub.offerta_colonnine_500m, sub.distanza_colonnina_piu_vicina_m)
    gap = sub["domanda_norm"].to_numpy() - offerta_norm
    return pd.Series(gap, index=sub.index)


def main():
    print(f"--- Carico {PARQUET.name} ---")
    df = pd.read_parquet(PARQUET)
    df["flag_eleggibile_EV"] = df["flag_eleggibile_EV"].map({True: True, False: False}).fillna(False).astype(bool)
    el = df.flag_eleggibile_EV & df.offerta_colonnine_500m.notna()
    print(f"Sezioni totali: {len(df):,} | eleggibili nazionali: {el.sum():,}")

    milano_mask = el & (df.PROVINCIA.str.upper() == PROVINCIA_TARGET.upper())
    print(f"Sezioni eleggibili in provincia di {PROVINCIA_TARGET}: {milano_mask.sum():,}")

    # baseline: verifica di coerenza col gap_score gia' salvato nel parquet
    gap_base = calcola_gap(df, el)
    scarto = (gap_base - df.loc[el, "gap_score"]).abs().max()
    print(f"Scarto massimo vs gap_score salvato (verifica coerenza): {scarto:.2e}")

    milano_sez = df.loc[milano_mask, ["SEZ2011", "COMUNE", "offerta_colonnine_500m",
                                       "distanza_colonnina_piu_vicina_m"]].copy()
    milano_sez["gap_score_originale"] = gap_base.loc[milano_mask]

    target = milano_sez.nlargest(TOP_N, "gap_score_originale") if TOP_N else milano_sez
    print(f"Simulo l'aggiunta di {N_COLONNINE_AGGIUNTE} colonnina/e su {len(target):,} sezioni "
          f"di {PROVINCIA_TARGET} (modalita': {MODE})")

    if MODE == "tutte_insieme":
        df_sim = df.copy()
        idx = target.index
        valori_attuali = df_sim.loc[idx, "offerta_colonnine_500m"].fillna(0)
        df_sim.loc[idx, "offerta_colonnine_500m"] = valori_attuali + N_COLONNINE_AGGIUNTE
        gap_sim = calcola_gap(df_sim, el)
        result = pd.DataFrame({
            "SEZ2011": target["SEZ2011"],
            "COMUNE": target["COMUNE"],
            "offerta_colonnine_500m_originale": target["offerta_colonnine_500m"],
            "distanza_colonnina_piu_vicina_m": target["distanza_colonnina_piu_vicina_m"],
            "gap_score_originale": target["gap_score_originale"],
            "gap_score_simulato": gap_sim.loc[idx].values,
        })

    elif MODE == "uno_alla_volta":
        risultati = []
        for sez_id in target.index:
            df_sim = df.copy()
            valore_attuale = df_sim.at[sez_id, "offerta_colonnine_500m"]
            nuovo_valore = (valore_attuale if pd.notna(valore_attuale) else 0) + N_COLONNINE_AGGIUNTE
            df_sim.at[sez_id, "offerta_colonnine_500m"] = nuovo_valore
            gap_sim = calcola_gap(df_sim, el)
            risultati.append({
                "SEZ2011": df.at[sez_id, "SEZ2011"],
                "COMUNE": df.at[sez_id, "COMUNE"],
                "offerta_colonnine_500m_originale": valore_attuale,
                "distanza_colonnina_piu_vicina_m": df.at[sez_id, "distanza_colonnina_piu_vicina_m"],
                "gap_score_originale": gap_base.at[sez_id],
                "gap_score_simulato": gap_sim.at[sez_id],
            })
        result = pd.DataFrame(risultati)

    else:
        raise ValueError(f"MODE non valido: {MODE!r}")

    result["delta_gap"] = result["gap_score_simulato"] - result["gap_score_originale"]
    result = result.sort_values("gap_score_originale", ascending=False)

    print("\n--- Risultato (ordinato per gap_score originale, dal piu' critico) ---")
    print(result.to_string(index=False))

    n_nullo = (result["gap_score_simulato"].abs() < 1e-9).sum()
    n_negativo = (result["gap_score_simulato"] < 0).sum()
    print(f"\nSu {len(result)} sezioni simulate:")
    print(f"  gap_score diventa ~0 (|x| < 1e-9): {n_nullo}")
    print(f"  gap_score diventa negativo:        {n_negativo}")
    print(f"  delta medio:                       {result['delta_gap'].mean():.3f}")
    print(f"  delta massimo (miglior effetto):    {result['delta_gap'].min():.3f}")

    result.to_csv(OUT_CSV, index=False)
    print(f"\nSalvato: {OUT_CSV}")


if __name__ == "__main__":
    main()
