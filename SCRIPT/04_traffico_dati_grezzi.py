"""
Step 4 della pipeline di siting per nuove colonnine (dati grezzi, nessun
punteggio).

Per il centroide di ciascuna delle 50 sezioni critiche di Milano,
interroga l'API TomTom "Flow Segment Data": restituisce la velocita'
attuale e la velocita' a flusso libero del segmento stradale piu' vicino
(TomTom effettua da solo lo snap al segmento piu' vicino al punto dato).
Nessuna elaborazione oltre al calcolo diretto della congestione
(1 - currentSpeed/freeFlowSpeed): e' un dato derivato immediato dai due
valori restituiti dall'API, non un punteggio pesato/combinato.

Indipendente dallo step 3 (POI): legge direttamente le sezioni dal
GeoJSON dello step 1.

La API key si legge da SCRIPT/tomtom_key.txt (non hardcoded nello script).

Output: traffico_dati_milano.csv (SEZ2011, COMUNE, gap_score,
        centroid_lat, centroid_lon, current_speed_kmh, free_flow_speed_kmh,
        congestione, road_closure)
"""

import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

CARTELLA_PROGETTO = r"C:\Users\fasanelli michele\OneDrive\Desktop\Contesto lavoro di gruppo ETL"
IN_GEOJSON = CARTELLA_PROGETTO + r"\SCRIPT\top50_sezioni_critiche_milano.geojson"
OUT_CSV = CARTELLA_PROGETTO + r"\SCRIPT\traffico_dati_milano.csv"
KEY_PATH = Path(CARTELLA_PROGETTO) / "SCRIPT" / "tomtom_key.txt"

TOMTOM_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
PAUSA_TRA_RICHIESTE_S = 0.3  # tier free: 5 richieste/secondo, stiamo larghi


def leggi_api_key():
    return KEY_PATH.read_text(encoding="utf-8").strip()


def query_flow_segment(lat, lon, api_key, tentativi=3):
    params = {"point": f"{lat},{lon}", "unit": "KMPH", "key": api_key}
    for tentativo in range(1, tentativi + 1):
        r = requests.get(TOMTOM_URL, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            print("    rate limit (429), attendo 10s e riprovo...")
            time.sleep(10)
            continue
        print(f"    tentativo {tentativo}: HTTP {r.status_code} - {r.text[:200]}")
        time.sleep(3)
    return None


def main(limite_righe=None):
    api_key = leggi_api_key()
    gdf = gpd.read_file(IN_GEOJSON)
    df = gdf[["SEZ2011", "COMUNE", "gap_score", "centroid_lat", "centroid_lon"]].copy()
    if limite_righe:
        df = df.head(limite_righe)

    risultati = []
    for i, row in df.iterrows():
        data = query_flow_segment(row["centroid_lat"], row["centroid_lon"], api_key)

        if data is None or "flowSegmentData" not in data:
            print(f"[{i+1}/{len(df)}] {row['SEZ2011']} ({row['COMUNE']}): nessun dato traffico")
            risultati.append({**row.to_dict(), "current_speed_kmh": None,
                               "free_flow_speed_kmh": None, "congestione": None,
                               "road_closure": None})
        else:
            seg = data["flowSegmentData"]
            current = seg.get("currentSpeed")
            free_flow = seg.get("freeFlowSpeed")
            congestione = (1 - current / free_flow) if (current is not None and free_flow) else None
            print(f"[{i+1}/{len(df)}] {row['SEZ2011']} ({row['COMUNE']}): "
                  f"current={current} free_flow={free_flow} congestione={congestione}")
            risultati.append({**row.to_dict(), "current_speed_kmh": current,
                               "free_flow_speed_kmh": free_flow, "congestione": congestione,
                               "road_closure": seg.get("roadClosure")})

        time.sleep(PAUSA_TRA_RICHIESTE_S)

    out = pd.DataFrame(risultati)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSalvato: {OUT_CSV} ({len(out)} righe)")


if __name__ == "__main__":
    import sys
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limite_righe=limite)
