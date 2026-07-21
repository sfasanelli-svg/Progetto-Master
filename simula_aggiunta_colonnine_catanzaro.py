"""
Simulazione: effetto dell'aggiunta di colonnine di ricarica sul gap_score
delle sezioni CRITICHE (gap_score > 0.429, soglia del gomito) della provincia
di Catanzaro. Per ciascuna sezione critica calcola anche il numero minimo di
colonnine da installare perche' il gap_score simulato scenda sotto la soglia.

Il gap_score e' un rango percentile NAZIONALE (rank%(domanda) - rank%(offerta)
sull'intero universo eleggibile, ~346.507 sezioni), quindi non si puo' capire
l'effetto di una colonnina in piu' guardando solo Catanzaro: bisogna reinserire
il nuovo valore di offerta_colonnine_500m nel dataset completo e ricalcolare
il ranking su tutta Italia, poi rifiltrare su Catanzaro.

A differenza di simula_aggiunta_colonnine_milano.py (che simulava le TOP_N
sezioni piu' critiche a prescindere da una soglia), qui selezioniamo TUTTE le
sezioni di Catanzaro sopra la soglia del gomito (SOGLIA_GOMITO = 0.429),
la stessa usata in Ranking_Sezioni_Critiche_GAP_Score.ipynb e nel notebook
Mediana_GAP_Score_per_Provincia.ipynb.

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
# Percorso assoluto: in un notebook/script VS Code il kernel non parte
# necessariamente dalla cartella del progetto, quindi non ci affidiamo a
# __file__ o Path.cwd(). Aggiorna se il parquet e' altrove.
CARTELLA_PROGETTO = Path(r"C:\Users\fasanelli michele\OneDrive\Desktop\Contesto lavoro di gruppo ETL")
PARQUET = CARTELLA_PROGETTO / "sezioni_gap_score_DEFINITIVO.parquet"

PROVINCIA_TARGET = "Catanzaro"
N_COLONNINE_AGGIUNTE = 1

# Soglia del gomito (Ranking_Sezioni_Critiche_GAP_Score.ipynb): definisce quali
# sezioni sono "critiche". Qui selezioniamo TUTTE quelle di Catanzaro sopra
# questa soglia, non un numero fisso di sezioni.
SOGLIA_GOMITO = 0.429

# MODE = "uno_alla_volta": per ogni sezione critica, simula l'aggiunta SOLO a
#   quella sezione (tutto il resto d'Italia invariato) e ricalcola il ranking
#   nazionale da capo. E' la domanda corretta ("se investo qui, quanto
#   migliora QUESTA sezione?") ed e' anche l'unica modalita' per cui ha senso
#   calcolare "colonnine necessarie per sezione" (sotto). Se le sezioni
#   critiche sono centinaia, valuta MODE="tutte_insieme" o
#   MAX_SEZIONI_SIMULAZIONE sotto.
# MODE = "tutte_insieme": aggiunge la colonnina a TUTTE le sezioni critiche di
#   Catanzaro contemporaneamente e ricalcola il ranking UNA sola volta
#   (istantaneo), ma risponde alla domanda "se elettrifichiamo tutte insieme",
#   non "sezione per sezione" -- in questa modalita' la colonna "colonnine
#   necessarie" non viene calcolata (non ha lo stesso significato quando le
#   altre sezioni cambiano insieme).
MODE = "uno_alla_volta"

# Tetto di sicurezza per MODE="uno_alla_volta": se le sezioni critiche di
# Catanzaro sono piu' di questo numero, simula solo le prime (per gap_score
# decrescente) e avvisa. Metti None per rimuovere il limite.
MAX_SEZIONI_SIMULAZIONE = 100

# Tetto massimo di colonnine aggiunte da provare nella ricerca binaria del
# minimo necessario per scendere sotto la soglia. Se nemmeno questo tetto
# basta, la sezione viene marcata come "non raggiungibile" (colonna NaN).
MAX_COLONNINE_RICERCA = 300

OUT_CSV = CARTELLA_PROGETTO / f"simulazione_gap_score_{PROVINCIA_TARGET.lower()}.csv"


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
    domanda_norm non cambia (la simulazione tocca solo l'offerta), quindi si
    riusa la colonna gia' presente nel parquet."""
    sub = df.loc[el]
    offerta_norm = offerta_accessibilita(sub.offerta_colonnine_500m, sub.distanza_colonnina_piu_vicina_m)
    gap = sub["domanda_norm"].to_numpy() - offerta_norm
    return pd.Series(gap, index=sub.index)


def gap_sezione_con_offerta(df, el, sez_id, nuovo_valore):
    """Calcola il gap_score di UNA sola sezione dopo aver impostato
    temporaneamente il suo offerta_colonnine_500m a nuovo_valore, poi
    ripristina il valore originale. Evita di copiare l'intero dataframe a
    ogni valutazione (la ricerca binaria sotto ne richiede diverse decine)."""
    valore_originale = df.at[sez_id, "offerta_colonnine_500m"]
    df.at[sez_id, "offerta_colonnine_500m"] = nuovo_valore
    try:
        gap_sim = calcola_gap(df, el)
        return gap_sim.at[sez_id]
    finally:
        df.at[sez_id, "offerta_colonnine_500m"] = valore_originale


def colonnine_necessarie(df, el, sez_id, valore_attuale, soglia, max_colonnine=MAX_COLONNINE_RICERCA):
    """Ricerca binaria del numero minimo di colonnine da AGGIUNGERE a questa
    sezione perche' il suo gap_score simulato scenda a/sotto `soglia`.
    Ritorna None se non raggiungibile entro max_colonnine aggiunte.

    Monotonia: aggiungere colonnine puo' solo aumentare (o lasciare invariato)
    il rango di offerta della sezione, quindi il suo gap_score e' non
    crescente al crescere delle colonnine aggiunte -- la ricerca binaria e'
    valida."""
    base = valore_attuale if pd.notna(valore_attuale) else 0.0

    gap_al_tetto = gap_sezione_con_offerta(df, el, sez_id, base + max_colonnine)
    if gap_al_tetto > soglia:
        return None

    lo, hi = 1, max_colonnine
    while lo < hi:
        mid = (lo + hi) // 2
        gap_mid = gap_sezione_con_offerta(df, el, sez_id, base + mid)
        if gap_mid <= soglia:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    print(f"--- Carico {PARQUET.name} ---")
    # Solo le colonne che servono: il parquet completo ha ~74 colonne (incl.
    # geometria), inutili qui e pesanti da copiare/manipolare ad ogni iterazione.
    COLONNE = ["SEZ2011", "COMUNE", "PROVINCIA", "flag_eleggibile_EV",
               "offerta_colonnine_500m", "distanza_colonnina_piu_vicina_m",
               "domanda_norm", "gap_score"]
    df = pd.read_parquet(PARQUET, columns=COLONNE)
    df["flag_eleggibile_EV"] = df["flag_eleggibile_EV"].map({True: True, False: False}).fillna(False).astype(bool)
    el = df.flag_eleggibile_EV & df.offerta_colonnine_500m.notna()
    print(f"Sezioni totali: {len(df):,} | eleggibili nazionali: {el.sum():,}")

    provincia_mask = el & (df.PROVINCIA.str.upper() == PROVINCIA_TARGET.upper())
    print(f"Sezioni eleggibili in provincia di {PROVINCIA_TARGET}: {provincia_mask.sum():,}")

    # baseline: verifica di coerenza col gap_score gia' salvato nel parquet
    gap_base = calcola_gap(df, el)
    scarto = (gap_base - df.loc[el, "gap_score"]).abs().max()
    print(f"Scarto massimo vs gap_score salvato (verifica coerenza): {scarto:.2e}")

    provincia_sez = df.loc[provincia_mask, ["SEZ2011", "COMUNE", "offerta_colonnine_500m",
                                             "distanza_colonnina_piu_vicina_m"]].copy()
    provincia_sez["gap_score_originale"] = gap_base.loc[provincia_mask]

    # sezioni CRITICHE: sopra la soglia del gomito, non un numero fisso
    critiche_provincia = provincia_sez[provincia_sez["gap_score_originale"] > SOGLIA_GOMITO].copy()
    critiche_provincia = critiche_provincia.sort_values("gap_score_originale", ascending=False)
    print(f"Sezioni critiche (gap_score > {SOGLIA_GOMITO}) in {PROVINCIA_TARGET}: {len(critiche_provincia):,} "
          f"su {len(provincia_sez):,} eleggibili "
          f"({len(critiche_provincia) / len(provincia_sez):.1%})")

    target = critiche_provincia
    if MODE == "uno_alla_volta" and MAX_SEZIONI_SIMULAZIONE and len(target) > MAX_SEZIONI_SIMULAZIONE:
        print(f"Attenzione: {len(target)} sezioni critiche superano il limite "
              f"MAX_SEZIONI_SIMULAZIONE={MAX_SEZIONI_SIMULAZIONE}. Simulo solo le prime "
              f"{MAX_SEZIONI_SIMULAZIONE} per gap_score decrescente (alza il limite o passa a "
              f"MODE='tutte_insieme' per includerle tutte).")
        target = target.iloc[:MAX_SEZIONI_SIMULAZIONE]

    print(f"Simulo l'aggiunta di {N_COLONNINE_AGGIUNTE} colonnina/e su {len(target):,} sezioni critiche "
          f"di {PROVINCIA_TARGET} (modalita': {MODE})")

    if len(target) == 0:
        print("Nessuna sezione critica da simulare: fine.")
        return

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
        result["colonnine_necessarie_sotto_soglia"] = np.nan
        print(f"\nNota: in modalita' 'tutte_insieme' la colonna "
              f"'colonnine_necessarie_sotto_soglia' non viene calcolata (il minimo per "
              f"singola sezione presuppone che le altre restino invariate).")

    elif MODE == "uno_alla_volta":
        risultati = []
        for sez_id in target.index:
            valore_attuale = df.at[sez_id, "offerta_colonnine_500m"]

            gap_sim = gap_sezione_con_offerta(
                df, el, sez_id,
                (valore_attuale if pd.notna(valore_attuale) else 0) + N_COLONNINE_AGGIUNTE,
            )
            n_necessarie = colonnine_necessarie(df, el, sez_id, valore_attuale, SOGLIA_GOMITO)

            risultati.append({
                "SEZ2011": df.at[sez_id, "SEZ2011"],
                "COMUNE": df.at[sez_id, "COMUNE"],
                "offerta_colonnine_500m_originale": valore_attuale,
                "distanza_colonnina_piu_vicina_m": df.at[sez_id, "distanza_colonnina_piu_vicina_m"],
                "gap_score_originale": gap_base.at[sez_id],
                "gap_score_simulato": gap_sim,
                "colonnine_necessarie_sotto_soglia": n_necessarie,
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
    n_sotto_soglia = (result["gap_score_simulato"] <= SOGLIA_GOMITO).sum()
    print(f"\nSu {len(result)} sezioni simulate:")
    print(f"  gap_score diventa ~0 (|x| < 1e-9):        {n_nullo}")
    print(f"  gap_score diventa negativo:                {n_negativo}")
    print(f"  gap_score scende sotto la soglia del gomito ({SOGLIA_GOMITO}): {n_sotto_soglia}")
    print(f"  delta medio:                                {result['delta_gap'].mean():.3f}")
    print(f"  delta massimo (miglior effetto):             {result['delta_gap'].min():.3f}")

    if MODE == "uno_alla_volta":
        raggiungibili = result["colonnine_necessarie_sotto_soglia"].dropna()
        n_non_raggiungibili = result["colonnine_necessarie_sotto_soglia"].isna().sum()
        print(f"\nColonnine necessarie per scendere sotto la soglia (per sezione):")
        if len(raggiungibili):
            print(f"  mediana: {raggiungibili.median():.0f} | media: {raggiungibili.mean():.1f} | "
                  f"min: {raggiungibili.min():.0f} | max: {raggiungibili.max():.0f}")
        print(f"  sezioni non raggiungibili entro {MAX_COLONNINE_RICERCA} colonnine aggiunte: "
              f"{n_non_raggiungibili}")

    result.to_csv(OUT_CSV, index=False)
    print(f"\nSalvato: {OUT_CSV}")


if __name__ == "__main__":
    main()
