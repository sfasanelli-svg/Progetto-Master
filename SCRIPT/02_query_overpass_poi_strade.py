"""
Step 2 della pipeline di siting per nuove colonnine.

Per ciascuna delle 50 sezioni critiche di Milano (output dello step 1),
interroga Overpass API (OpenStreetMap) per:
  - POI rilevanti come proxy di domanda/visibilita' (parcheggi, negozi,
    uffici, servizi)
  - rete stradale carrabile (nodi delle way taggate highway=...)

Usa il filtro poligonale "poly" di Overpass QL sul poligono di ciascuna
sezione (bufferizzato di qualche decina di metri, calcolato in UTM per
correttezza metrica) cosi' i risultati sono gia' assegnati alla sezione
corretta senza bisogno di join spaziale locale.

Una singola sezione = una singola richiesta HTTP (rispetto ai limiti del
server pubblico Overpass: pausa tra le richieste).

Output: overpass_raw/<SEZ2011>.json (cache grezza, una per sezione)
"""

import json
import time
from pathlib import Path

import geopandas as gpd
import requests
from pyproj import Transformer
from shapely.ops import transform

CARTELLA_PROGETTO = r"C:\Users\fasanelli michele\OneDrive\Desktop\Contesto lavoro di gruppo ETL"
IN_GEOJSON = CARTELLA_PROGETTO + r"\SCRIPT\top50_sezioni_critiche_milano.geojson"
OUT_DIR = Path(CARTELLA_PROGETTO) / "SCRIPT" / "overpass_raw"

OVERPASS_URL = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
HEADERS = {
    "User-Agent": "ProgettoEVChargeDesert/1.0 (progetto universitario Unimib; contatto simofasa01@gmail.com)"
}
BUFFER_METRI = 80  # margine attorno al poligono della sezione
PAUSA_TRA_RICHIESTE_S = 1.5  # cortesia verso il server pubblico Overpass

# tag POI usati come proxy di domanda/visibilita' per il siting
AMENITY_TAGS = ("parking", "supermarket", "fuel", "bank", "pharmacy",
                "marketplace", "restaurant", "cafe", "school", "hospital",
                "fast_food", "post_office")
HIGHWAY_TAGS = ("motorway", "trunk", "primary", "secondary", "tertiary",
                "unclassified", "residential", "living_street", "service")

CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32632"

_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True).transform
_to_wgs84 = Transformer.from_crs(CRS_UTM, CRS_WGS84, always_xy=True).transform


def poly_string(geom_wgs84):
    """Buffer in UTM (metri) del poligono, poi stringa 'lat lon lat lon ...'
    per il filtro Overpass QL `poly:"..."`."""
    geom_utm = transform(_to_utm, geom_wgs84)
    geom_utm_buff = geom_utm.buffer(BUFFER_METRI)
    geom_buff_wgs84 = transform(_to_wgs84, geom_utm_buff)

    coords = list(geom_buff_wgs84.exterior.coords)
    return " ".join(f"{lat:.6f} {lon:.6f}" for lon, lat in coords)


def build_query(poly):
    amenity_regex = "^(" + "|".join(AMENITY_TAGS) + ")$"
    highway_regex = "^(" + "|".join(HIGHWAY_TAGS) + ")$"
    return f"""
[out:json][timeout:60];
(
  node["amenity"~"{amenity_regex}"](poly:"{poly}");
  node["shop"](poly:"{poly}");
  node["office"](poly:"{poly}");
);
out body;
(
  way["highway"~"{highway_regex}"](poly:"{poly}");
);
out geom;
""".strip()


def query_overpass(query, tentativi=3):
    for tentativo in range(1, tentativi + 1):
        r = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=90)
        if r.status_code == 200:
            return r.json()
        print(f"  tentativo {tentativo}: HTTP {r.status_code}, riprovo tra 5s...")
        time.sleep(5)
    r.raise_for_status()


def main(limite_sezioni=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(IN_GEOJSON)
    if limite_sezioni:
        gdf = gdf.head(limite_sezioni)

    for i, row in gdf.iterrows():
        sez = row["SEZ2011"]
        out_path = OUT_DIR / f"{sez}.json"
        if out_path.exists():
            print(f"[{i+1}/{len(gdf)}] {sez} ({row['COMUNE']}): gia' in cache, salto")
            continue

        poly = poly_string(row["geometry"])
        query = build_query(poly)

        print(f"[{i+1}/{len(gdf)}] {sez} ({row['COMUNE']}): interrogo Overpass...")
        data = query_overpass(query)

        n_nodes = sum(1 for e in data["elements"] if e["type"] == "node")
        n_ways = sum(1 for e in data["elements"] if e["type"] == "way")
        print(f"    -> {n_nodes} nodi POI, {n_ways} way stradali")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        time.sleep(PAUSA_TRA_RICHIESTE_S)

    print("\nCompletato.")


if __name__ == "__main__":
    import sys
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limite_sezioni=limite)
