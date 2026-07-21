"""
Step 3 della pipeline di siting per nuove colonnine (dati grezzi, nessun
punteggio).

Per ciascuna sezione (cache Overpass dello step 2), estrae i POI trovati
entro un raggio fisso dal centroide e li conta per categoria (tipo di
amenity/shop/office). Nessun peso, nessun punteggio combinato: solo dati
grezzi da poter analizzare o pesare in un secondo momento.

Output: poi_dati_milano.csv (SEZ2011, COMUNE, gap_score, centroid_lat,
        centroid_lon, poi_count_totale, poi_count_<categoria> per ogni
        categoria osservata)
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer

CARTELLA_PROGETTO = r"C:\Users\fasanelli michele\OneDrive\Desktop\Contesto lavoro di gruppo ETL"
IN_GEOJSON = CARTELLA_PROGETTO + r"\SCRIPT\top50_sezioni_critiche_milano.geojson"
RAW_DIR = Path(CARTELLA_PROGETTO) / "SCRIPT" / "overpass_raw"
OUT_CSV = CARTELLA_PROGETTO + r"\SCRIPT\poi_dati_milano.csv"

RAGGIO_POI_METRI = 150

CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32632"


def categoria_poi(tags):
    """Ritorna la categoria del POI (valore del tag amenity/shop, oppure
    'office' generico), senza alcuna logica di peso."""
    for chiave in ("amenity", "shop", "office"):
        val = tags.get(chiave)
        if val is None:
            continue
        return val if chiave != "office" else "office"
    return "altro"


def estrai_poi_entro_raggio(data, centro_x_utm, centro_y_utm, to_utm):
    poi = []
    for el in data["elements"]:
        if el["type"] != "node":
            continue
        x, y = to_utm(el["lon"], el["lat"])
        dist = np.hypot(x - centro_x_utm, y - centro_y_utm)
        if dist <= RAGGIO_POI_METRI:
            poi.append(categoria_poi(el.get("tags", {})))
    return poi


def main():
    gdf = gpd.read_file(IN_GEOJSON)
    to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True).transform

    righe = []
    for _, row in gdf.iterrows():
        sez = row["SEZ2011"]
        raw_path = RAW_DIR / f"{sez}.json"

        cx, cy = to_utm(row["centroid_lon"], row["centroid_lat"])

        if not raw_path.exists():
            print(f"{sez} ({row['COMUNE']}): nessuna cache Overpass")
            conteggi = Counter()
        else:
            with open(raw_path, encoding="utf-8") as f:
                data = json.load(f)
            categorie = estrai_poi_entro_raggio(data, cx, cy, to_utm)
            conteggi = Counter(categorie)
            print(f"{sez} ({row['COMUNE']}): {sum(conteggi.values())} POI entro {RAGGIO_POI_METRI}m "
                  f"- {dict(conteggi)}")

        riga = {
            "SEZ2011": sez, "COMUNE": row["COMUNE"], "gap_score": row["gap_score"],
            "centroid_lat": row["centroid_lat"], "centroid_lon": row["centroid_lon"],
            "poi_count_totale": sum(conteggi.values()),
        }
        riga.update({f"poi_count_{cat}": n for cat, n in conteggi.items()})
        righe.append(riga)

    df = pd.DataFrame(righe).fillna(0)
    # colonne poi_count_* come interi
    for col in df.columns:
        if col.startswith("poi_count_"):
            df[col] = df[col].astype(int)

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSalvato: {OUT_CSV} ({len(df)} righe, {len(df.columns)} colonne)")


if __name__ == "__main__":
    main()
