"""
Step 4 della pipeline di siting per nuove colonnine.

Per ciascuna delle 50 sezioni critiche di Milano, seleziona un numero
ridotto di punti stradali candidati (default 3) da monitorare nel tempo
con TomTom per trovare il punto piu' trafficato della sezione.

La cache Overpass (way["highway"=...]) contiene in media ~244 nodi
stradali unici per sezione: troppi per interrogarli tutti ad ogni run
restando nel tier free TomTom (2.500 chiamate/giorno). Si selezionano
quindi solo i nodi piu' promettenti:

  1. gerarchia stradale: primary > secondary > tertiary > unclassified >
     residential > living_street > service (le strade di rango piu' alto
     tendono a portare piu' traffico);
  2. tra nodi di pari rango, si preferiscono le intersezioni (nodo
     condiviso da 2+ way), tipicamente piu' trafficate e piu' facili da
     individuare come luogo reale;
  3. i candidati vengono deduplicati spazialmente (min 30m di distanza
     tra loro) per non scegliere piu' punti che sono di fatto lo stesso
     incrocio.

Output: candidati_traffico_milano.csv (SEZ2011, COMUNE, gap_score,
        cand_id, lat, lon, road_class, is_intersection)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer

CARTELLA_PROGETTO = r"C:\Users\fasanelli michele\OneDrive\Desktop\Contesto lavoro di gruppo ETL"
IN_GEOJSON = CARTELLA_PROGETTO + r"\SCRIPT\top50_sezioni_critiche_milano.geojson"
RAW_DIR = Path(CARTELLA_PROGETTO) / "SCRIPT" / "overpass_raw"
OUT_CSV = CARTELLA_PROGETTO + r"\SCRIPT\candidati_traffico_milano.csv"

N_CANDIDATI_PER_SEZIONE = 3
DIST_MIN_TRA_CANDIDATI_M = 30

# rango stradale: valore piu' basso = strada piu' importante
RANGO_STRADALE = {
    "primary": 0, "secondary": 1, "tertiary": 2, "unclassified": 3,
    "residential": 4, "living_street": 5, "service": 6,
}

CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32632"
_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True).transform


def estrai_nodi_stradali(data):
    """Ritorna un dict {(lat,lon) arrotondati: {'rango':int,'grado':int}}
    con il rango stradale migliore e il grado (n. di way a cui appartiene,
    proxy di intersezione) per ciascun nodo stradale della sezione."""
    nodi = {}
    for el in data["elements"]:
        if el["type"] != "way" or "geometry" not in el:
            continue
        rango = RANGO_STRADALE.get(el.get("tags", {}).get("highway"), 99)
        for nd in el["geometry"]:
            key = (round(nd["lat"], 6), round(nd["lon"], 6))
            info = nodi.setdefault(key, {"rango": rango, "grado": 0})
            info["rango"] = min(info["rango"], rango)
            info["grado"] += 1
    return nodi


def seleziona_candidati(nodi):
    """Ordina i nodi per (rango asc, grado desc) e ne tiene i migliori N,
    scartando quelli troppo vicini a un candidato gia' scelto."""
    lista = [
        {"lat": lat, "lon": lon, "rango": info["rango"], "grado": info["grado"]}
        for (lat, lon), info in nodi.items()
    ]
    lista.sort(key=lambda r: (r["rango"], -r["grado"]))

    scelti = []
    scelti_xy = []
    for cand in lista:
        x, y = _to_utm(cand["lon"], cand["lat"])
        troppo_vicino = any(
            np.hypot(x - sx, y - sy) < DIST_MIN_TRA_CANDIDATI_M for sx, sy in scelti_xy
        )
        if troppo_vicino:
            continue
        scelti.append(cand)
        scelti_xy.append((x, y))
        if len(scelti) >= N_CANDIDATI_PER_SEZIONE:
            break
    return scelti


def main():
    gdf = gpd.read_file(IN_GEOJSON)
    righe = []

    for _, row in gdf.iterrows():
        sez = row["SEZ2011"]
        raw_path = RAW_DIR / f"{sez}.json"
        if not raw_path.exists():
            print(f"{sez} ({row['COMUNE']}): nessuna cache Overpass, salto")
            continue

        with open(raw_path, encoding="utf-8") as f:
            data = json.load(f)

        nodi = estrai_nodi_stradali(data)
        candidati = seleziona_candidati(nodi)

        classi_inverse = {v: k for k, v in RANGO_STRADALE.items()}
        print(f"{sez} ({row['COMUNE']}): {len(nodi)} nodi totali -> "
              f"{len(candidati)} candidati "
              f"[{', '.join(classi_inverse.get(c['rango'], '?') for c in candidati)}]")

        for i, cand in enumerate(candidati, start=1):
            righe.append({
                "SEZ2011": sez, "COMUNE": row["COMUNE"], "gap_score": row["gap_score"],
                "cand_id": i, "lat": cand["lat"], "lon": cand["lon"],
                "road_class": classi_inverse.get(cand["rango"], "sconosciuta"),
                "is_intersection": cand["grado"] > 1,
            })

    df = pd.DataFrame(righe)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSalvato: {OUT_CSV} ({len(df)} candidati totali, "
          f"{df['SEZ2011'].nunique()} sezioni)")


if __name__ == "__main__":
    main()
