"""
Step 6 della pipeline di siting per nuove colonnine.

Interroga TomTom Flow Segment Data sui 150 candidati stradali (3 per
sezione, output dello step 5) e APPENDE il risultato, con un timestamp,
a un CSV che cresce nel tempo. Pensato per essere eseguito ripetutamente
(es. ogni ora dalle 7 alle 22, vedi il workflow GitHub Actions) cosi' da
poter poi individuare, per ciascuna sezione, il candidato+ora con la
congestione piu' alta osservata durante la giornata: quello e' il punto
piu' trafficato dove valutare l'installazione della colonnina.

API key TomTom:
  1. variabile d'ambiente TOMTOM_API_KEY (usata su GitHub Actions, via
     secret) se presente;
  2. altrimenti SCRIPT/tomtom_key.txt (uso locale).

Output: traffico_serie_storica_milano.csv, in append (header scritto
        solo se il file non esiste ancora). Colonne: timestamp_utc,
        SEZ2011, COMUNE, gap_score, cand_id, lat, lon, road_class,
        is_intersection, current_speed_kmh, free_flow_speed_kmh,
        congestione, road_closure
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# percorso relativo alla posizione dello script: funziona sia in locale
# (Windows) sia sul runner GitHub Actions (Linux)
CARTELLA_SCRIPT = Path(__file__).resolve().parent
IN_CSV = CARTELLA_SCRIPT / "candidati_traffico_milano.csv"
OUT_CSV = CARTELLA_SCRIPT / "traffico_serie_storica_milano.csv"
KEY_PATH = CARTELLA_SCRIPT / "tomtom_key.txt"

TOMTOM_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
PAUSA_TRA_RICHIESTE_S = 0.3  # tier free: 5 richieste/secondo, stiamo larghi


def leggi_api_key():
    da_env = os.environ.get("TOMTOM_API_KEY")
    if da_env:
        return da_env.strip()
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
    df = pd.read_csv(IN_CSV)
    if limite_righe:
        df = df.head(limite_righe)

    timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    risultati = []
    for i, row in df.iterrows():
        data = query_flow_segment(row["lat"], row["lon"], api_key)

        base = {**row.to_dict(), "timestamp_utc": timestamp_utc}

        if data is None or "flowSegmentData" not in data:
            print(f"[{i+1}/{len(df)}] {row['SEZ2011']} cand{row['cand_id']}: nessun dato traffico")
            risultati.append({**base, "current_speed_kmh": None,
                               "free_flow_speed_kmh": None, "congestione": None,
                               "road_closure": None})
        else:
            seg = data["flowSegmentData"]
            current = seg.get("currentSpeed")
            free_flow = seg.get("freeFlowSpeed")
            congestione = (1 - current / free_flow) if (current is not None and free_flow) else None
            print(f"[{i+1}/{len(df)}] {row['SEZ2011']} cand{row['cand_id']} "
                  f"({row['road_class']}): current={current} free_flow={free_flow} "
                  f"congestione={congestione}")
            risultati.append({**base, "current_speed_kmh": current,
                               "free_flow_speed_kmh": free_flow, "congestione": congestione,
                               "road_closure": seg.get("roadClosure")})

        time.sleep(PAUSA_TRA_RICHIESTE_S)

    out = pd.DataFrame(risultati)
    colonne_ordinate = ["timestamp_utc", "SEZ2011", "COMUNE", "gap_score", "cand_id",
                         "lat", "lon", "road_class", "is_intersection",
                         "current_speed_kmh", "free_flow_speed_kmh", "congestione", "road_closure"]
    out = out[colonne_ordinate]

    file_esiste = Path(OUT_CSV).exists()
    out.to_csv(OUT_CSV, mode="a", header=not file_esiste, index=False)
    print(f"\n{'Aggiunte' if file_esiste else 'Salvate'} {len(out)} righe in: {OUT_CSV}")


if __name__ == "__main__":
    import sys
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limite_righe=limite)
