"""
Assegnazione delle colonnine di ricarica pubblica alle sezioni di censimento
2011, secondo la metodologia descritta in
"MEOTODOLOGIA ASSEGNAZIONE COLONNINE.docx" (cartella Drive
PROGETTO: EV CHARGE DESERT / ASSEGNAZIONE COLONNINE PER SEZIONE ISTAT).

Riassunto della metodologia:
- non si usa un point-in-polygon stretto sul confine della sezione, perche'
  le sezioni sono unita' molto piccole e la maggior parte risulterebbe a
  offerta zero anche con una colonnina a pochi metri di distanza;
- offerta_colonnine_500m = numero di colonnine pubbliche presenti entro un
  raggio di 500 metri dal confine del poligono della sezione (buffer sul
  poligono, non sul centroide), corrispondente a ~5-7 minuti a piedi;
- una stessa colonnina puo' ricadere nel buffer di piu' sezioni contigue e
  contribuire all'offerta di ciascuna: e' intenzionale (accessibilita'
  percepita, non allocazione esclusiva). La somma dell'offerta su tutte le
  sezioni e' quindi normalmente superiore al numero reale di colonnine;
- per le sezioni con offerta 500m pari a zero, si calcola in aggiunta la
  distanza (in metri) dalla colonnina pubblica piu' vicina, come misura
  continua per distinguere sezioni "quasi servite" da sezioni isolate.

Esclusione: le sezioni convenzionali per case sparse (flag
'sezione_convenzionale_case_sparse', poligoni fittizi con area mediana di
poche centinaia di mq) sono escluse dal join spaziale, coerentemente con
quanto documentato in unisci_basi_territoriali_sezioni.py: il centroide/
buffer di un poligono fittizio non rappresenta un luogo reale. Per queste
sezioni offerta e distanza restano NaN (non zero: "nessun dato", non
"nessuna offerta").

Il file colonnine viene usato cosi' come fornito (pun_colonnine_pulito.csv,
gia' pulito a monte): ogni riga (evse_id) e' contata come una colonnina,
senza ulteriore deduplica per stazione fisica o filtro per stato
(AVAILABLE/OUTOFORDER/...), perche' la metodologia non prevede questo
filtro esplicitamente.

NOTA: nel csv esistono coordinate con centinaia di evse_id identici (es.
45.95182, 12.695877 -> 495 righe, azienda "Plenitude / Be Charge"; il csv
riporta questo conteggio anche nella colonna n_evse_stesso_punto). Verificato
manualmente su Google Maps che si tratta di hub di ricarica reali con
effettivamente quel numero di colonnine/stalli sullo stesso sito, non di un
difetto di geocodifica: per questo non vengono filtrate o deduplicate, e
contribuiscono legittimamente all'offerta_500m delle sezioni circostanti.

Elaborazione a chunk (non tutta in RAM insieme) per restare dentro agli 8GB
di RAM disponibili in locale: si legge il parquet una sola volta (unico row
group), ma si bufferizza e si interroga l'indice spaziale delle colonnine a
blocchi di sezioni.
"""

import gc
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from pyproj import Transformer
from shapely import STRtree

BASE_DIR = Path(__file__).resolve().parent.parent

SEZIONI_PARQUET = BASE_DIR / "sezioni_censimento_2011_con_geometria.parquet"
COLONNINE_CSV = BASE_DIR / "pun_colonnine_pulito.csv"
OUTPUT_PARQUET = BASE_DIR / "output" / "offerta_colonnine_per_sezione.parquet"
OUTPUT_CSV = BASE_DIR / "output" / "offerta_colonnine_per_sezione.csv"

CRS_METRICO = "EPSG:32632"  # UTM 32N, stesso CRS del parquet sezioni (in metri)
CRS_COLONNINE = "EPSG:4326"  # lat/lon del CSV colonnine

RAGGIO_M = 500.0
CHUNK_SIZE = 20_000  # sezioni per blocco


def carica_colonnine() -> STRtree:
    print("--- Carico colonnine e costruisco indice spaziale ---")
    df = pd.read_csv(COLONNINE_CSV)
    print(f"Righe colonnine (evse_id): {len(df):,}")

    transformer = Transformer.from_crs(CRS_COLONNINE, CRS_METRICO, always_xy=True)
    x, y = transformer.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    punti = shapely.points(x, y)

    albero = STRtree(punti)
    print(f"Indice spaziale costruito su {len(punti):,} punti.")
    return albero


def carica_sezioni() -> gpd.GeoDataFrame:
    print("--- Carico sezioni (solo colonne necessarie) ---")
    colonne = ["SEZ2011", "geometry", "sezione_convenzionale_case_sparse"]
    tabella = pq.read_table(SEZIONI_PARQUET, columns=colonne)
    df = tabella.to_pandas()
    del tabella
    df["geometry"] = gpd.GeoSeries.from_wkb(df["geometry"], crs=CRS_METRICO)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_METRICO)
    print(f"Sezioni totali: {len(gdf):,}")
    return gdf


def calcola_offerta_blocco(geometrie, albero: STRtree) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna (offerta_500m, distanza_min_m) per un array di poligoni sezione."""
    buffer = shapely.buffer(geometrie, RAGGIO_M)

    # conteggio colonnine entro il buffer (query bulk: coppie [idx_buffer, idx_punto])
    idx_buffer, idx_punto = albero.query(buffer, predicate="intersects")
    offerta = np.zeros(len(geometrie), dtype=np.int64)
    if len(idx_buffer) > 0:
        conteggi = np.bincount(idx_buffer, minlength=len(geometrie))
        offerta[: len(conteggi)] = conteggi

    distanza = np.full(len(geometrie), np.nan)
    mask_zero = offerta == 0
    if mask_zero.any():
        idx_zero = np.where(mask_zero)[0]
        idx_pair, dist_vicina = albero.query_nearest(
            geometrie[idx_zero], return_distance=True
        )
        # idx_pair ha shape (2, n): riga 0 = posizione nell'input (geometrie[idx_zero]),
        # riga 1 = indice nell'albero (non serve qui). Con piu' colonnine equidistanti
        # la stessa posizione puo' comparire piu' volte: teniamo la distanza minima.
        pos_input = idx_pair[0]
        tmp = pd.DataFrame({"pos": pos_input, "dist": dist_vicina})
        min_per_pos = tmp.groupby("pos")["dist"].min()
        distanza[idx_zero[min_per_pos.index.to_numpy()]] = min_per_pos.to_numpy()

    return offerta, distanza


def main():
    inizio = time.time()

    albero = carica_colonnine()
    sezioni = carica_sezioni()

    n = len(sezioni)
    offerta_tot = np.zeros(n, dtype=np.int64)
    distanza_tot = np.full(n, np.nan)

    escludi = sezioni["sezione_convenzionale_case_sparse"].fillna(False).to_numpy()
    print(f"Sezioni escluse (case sparse, poligono fittizio): {escludi.sum():,}")

    geometrie_tutte = sezioni.geometry.values
    da_calcolare = np.where(~escludi)[0]
    print(f"Sezioni da elaborare: {len(da_calcolare):,}")

    print("--- Calcolo offerta a blocchi ---")
    for start in range(0, len(da_calcolare), CHUNK_SIZE):
        blocco_idx = da_calcolare[start : start + CHUNK_SIZE]
        geoms_blocco = geometrie_tutte[blocco_idx]

        offerta_blocco, distanza_blocco = calcola_offerta_blocco(geoms_blocco, albero)

        offerta_tot[blocco_idx] = offerta_blocco
        distanza_tot[blocco_idx] = distanza_blocco

        fatto = min(start + CHUNK_SIZE, len(da_calcolare))
        trascorso = time.time() - inizio
        print(f"  {fatto:>7,}/{len(da_calcolare):,} sezioni elaborate ({trascorso:6.1f}s)")

        del geoms_blocco, offerta_blocco, distanza_blocco
        gc.collect()

    # per le sezioni escluse offerta e distanza restano NaN (nessun dato, non zero)
    offerta_finale = offerta_tot.astype("float64")
    offerta_finale[escludi] = np.nan

    risultato = pd.DataFrame(
        {
            "SEZ2011": sezioni["SEZ2011"].to_numpy(),
            "offerta_colonnine_500m": offerta_finale,
            "distanza_colonnina_piu_vicina_m": distanza_tot,
            "sezione_convenzionale_case_sparse": sezioni[
                "sezione_convenzionale_case_sparse"
            ].to_numpy(),
        }
    )

    print("\n--- Controlli di coerenza ---")
    print(f"Sezioni totali in output: {len(risultato):,}")
    print(f"Sezioni con offerta NaN (escluse): {risultato['offerta_colonnine_500m'].isna().sum():,}")
    con_offerta_gt0 = (risultato["offerta_colonnine_500m"] > 0).sum()
    print(f"Sezioni con offerta_500m > 0: {con_offerta_gt0:,}")
    con_offerta_0 = (risultato["offerta_colonnine_500m"] == 0).sum()
    print(f"Sezioni con offerta_500m == 0: {con_offerta_0:,}")
    print(
        "Sezioni con offerta==0 ma distanza NaN (non dovrebbe accadere): "
        f"{((risultato['offerta_colonnine_500m'] == 0) & risultato['distanza_colonnina_piu_vicina_m'].isna()).sum():,}"
    )
    print(
        "Somma offerta_500m su tutte le sezioni (attesa > n. colonnine reali, per design): "
        f"{risultato['offerta_colonnine_500m'].sum(skipna=True):,.0f}"
    )

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    risultato.to_parquet(OUTPUT_PARQUET, index=False)
    risultato.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSalvato: {OUTPUT_PARQUET}")
    print(f"Salvato: {OUTPUT_CSV}")
    print(f"Tempo totale: {time.time() - inizio:.1f}s")


if __name__ == "__main__":
    main()
