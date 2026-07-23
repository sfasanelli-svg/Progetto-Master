"""
Step 5 della pipeline di siting per nuove colonnine.

Interroga TomTom Flow Segment Data sui 150 candidati stradali (3 per
sezione, output dello step 4) e APPENDE il risultato, con un timestamp,
a un CSV che cresce nel tempo. Pensato per essere eseguito ripetutamente
(vedi il workflow GitHub Actions) cosi' da poter poi individuare, per
ciascuna sezione, il candidato+ora con la congestione piu' alta osservata
durante la giornata: quello e' il punto piu' trafficato dove valutare
l'installazione della colonnina.

Il trigger arriva da un cron esterno (cron-job.org, che chiama l'API
GitHub per lanciare il workflow: piu' affidabile dello scheduler interno
di GitHub Actions per cadenze fitte). Per evitare doppie letture se il
trigger esterno arriva piu' volte nello stesso INTERVALLO_MINUTI, lo
script salta l'esecuzione (nessuna chiamata API) se l'ultima riga
presente nel CSV di output cade gia' nello stesso intervallo temporale
di quello corrente: il costo resta cosi' al massimo 150 chiamate per
ogni intervallo effettivamente coperto.

Pensato per una raccolta intensiva di pochi giorni (non per durare un
mese): con INTERVALLO_MINUTI=15 e quota TomTom di 20.000 chiamate/mese,
il consumo (~8.550/giorno) esaurisce la quota residua in un paio di
giorni, coerente con l'obiettivo di raccogliere dati sufficienti in
tempi brevi piuttosto che sostenere la raccolta a lungo termine.

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
INTERVALLO_MINUTI = 15  # granularita' del controllo "gia' coperto"


def leggi_api_key():
    da_env = os.environ.get("TOMTOM_API_KEY")
    if da_env:
        return da_env.strip()
    return KEY_PATH.read_text(encoding="utf-8").strip()


def bucket(dt, intervallo_minuti):
    """Chiave dell'intervallo temporale (data, ora, minuto arrotondato
    per difetto a multipli di intervallo_minuti) a cui appartiene dt."""
    minuto_arrotondato = (dt.minute // intervallo_minuti) * intervallo_minuti
    return (dt.date(), dt.hour, minuto_arrotondato)


def intervallo_gia_coperto(out_path, adesso, intervallo_minuti):
    """True se l'ultima riga del CSV di output cade nello stesso
    intervallo temporale (bucket) di 'adesso'. Si guarda solo l'ultima
    riga perche' le righe sono scritte in ordine cronologico crescente."""
    if not out_path.exists():
        return False
    with open(out_path, "rb") as f:
        f.seek(0, 2)
        dimensione = f.tell()
        if dimensione == 0:
            return False
        blocco = min(dimensione, 4096)
        f.seek(-blocco, 2)
        ultime_righe = f.read().decode("utf-8", errors="ignore").strip().splitlines()
    if not ultime_righe:
        return False
    ultimo_timestamp = ultime_righe[-1].split(",", 1)[0]
    try:
        ultimo_dt = datetime.fromisoformat(ultimo_timestamp)
    except ValueError:
        return False
    return bucket(ultimo_dt, intervallo_minuti) == bucket(adesso, intervallo_minuti)


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


def main(limite_righe=None, forza=False):
    ora_corrente = datetime.now(timezone.utc)

    if not forza and intervallo_gia_coperto(OUT_CSV, ora_corrente, INTERVALLO_MINUTI):
        print(f"Intervallo di {INTERVALLO_MINUTI} minuti gia' coperto da "
              f"un'esecuzione precedente ({ora_corrente.isoformat(timespec='minutes')}): "
              f"nessuna chiamata TomTom, esco.")
        return

    api_key = leggi_api_key()
    df = pd.read_csv(IN_CSV)
    if limite_righe:
        df = df.head(limite_righe)

    timestamp_utc = ora_corrente.isoformat(timespec="seconds")

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
    argomenti = sys.argv[1:]
    forza = "--forza" in argomenti
    argomenti = [a for a in argomenti if a != "--forza"]
    limite = int(argomenti[0]) if argomenti else None
    main(limite_righe=limite, forza=forza)
