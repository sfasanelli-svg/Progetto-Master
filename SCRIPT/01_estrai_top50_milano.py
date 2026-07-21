"""
Step 1 della pipeline di siting per nuove colonnine.

Estrae le 50 sezioni di censimento di Milano con gap_score piu' alto
(dal file sezioni_gap_score_DEFINITIVO.parquet, gia' usato per generare
sezioni_critiche_milano.csv) e ne calcola il poligono e il centroide in
coordinate geografiche (WGS84), a partire dal sistema di riferimento
nativo del parquet (EPSG:32632 - UTM 32N).

Output: top50_sezioni_critiche_milano.geojson (poligoni WGS84 + attributi)
"""

import pandas as pd
from shapely import wkb
from shapely.ops import transform
from pyproj import Transformer
import geopandas as gpd

CARTELLA_PROGETTO = r"C:\Users\fasanelli michele\OneDrive\Desktop\Contesto lavoro di gruppo ETL"
PATH_PARQUET = CARTELLA_PROGETTO + r"\sezioni_gap_score_DEFINITIVO.parquet"
N_SEZIONI = 50
CRS_NATIVO = "EPSG:32632"
CRS_WGS84 = "EPSG:4326"

OUT = CARTELLA_PROGETTO + r"\SCRIPT\top50_sezioni_critiche_milano.geojson"


def main():
    cols = ["SEZ2011", "PROVINCIA", "COMUNE", "gap_score",
            "popolazione_eta_guida_stimata", "geometry"]
    df = pd.read_parquet(PATH_PARQUET, columns=cols)

    milano = df[df["PROVINCIA"] == "Milano"].dropna(subset=["gap_score"]).copy()
    top = milano.sort_values("gap_score", ascending=False).head(N_SEZIONI).copy()

    top["geometry"] = top["geometry"].apply(wkb.loads)

    # centroide calcolato nel CRS proiettato nativo (UTM), poi riproiettato:
    # il centroide di un poligono va calcolato in un CRS metrico, non in gradi
    centroide_utm = top["geometry"].apply(lambda g: g.centroid)

    transformer = Transformer.from_crs(CRS_NATIVO, CRS_WGS84, always_xy=True)

    def to_wgs84(geom):
        return transform(transformer.transform, geom)

    top["geometry"] = top["geometry"].apply(to_wgs84)
    centroide_wgs84 = centroide_utm.apply(to_wgs84)

    gdf = gpd.GeoDataFrame(top, geometry="geometry", crs=CRS_WGS84)
    gdf["centroid_lat"] = centroide_wgs84.apply(lambda p: p.y)
    gdf["centroid_lon"] = centroide_wgs84.apply(lambda p: p.x)

    gdf.to_file(OUT, driver="GeoJSON")

    print(f"Sezioni estratte: {len(gdf)}")
    print(gdf[["SEZ2011", "COMUNE", "gap_score", "centroid_lat", "centroid_lon"]].head(10).to_string())
    print(f"\nSalvato: {OUT}")


if __name__ == "__main__":
    main()
