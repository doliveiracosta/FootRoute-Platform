from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from math import asin, cos, radians, sin, sqrt
import csv

import pandas as pd
import pydeck as pdk
import streamlit as st


LONG_TRIP_DEFAULT = 1500.0


@dataclass(frozen=True)
class Place:
    id: str
    name: str
    city: str
    state: str
    region: str
    lat: float
    lon: float

    @property
    def city_label(self) -> str:
        return f"{self.city}-{self.state}"


CLUBS = [
    Place("atletico_mg", "Atlético-MG", "Belo Horizonte", "MG", "Sudeste", -19.9167, -43.9345),
    Place("bahia", "Bahia", "Salvador", "BA", "Nordeste", -12.9777, -38.5016),
    Place("botafogo", "Botafogo", "Rio de Janeiro", "RJ", "Sudeste", -22.9068, -43.1729),
    Place("corinthians", "Corinthians", "São Paulo", "SP", "Sudeste", -23.5505, -46.6333),
    Place("cruzeiro", "Cruzeiro", "Belo Horizonte", "MG", "Sudeste", -19.9167, -43.9345),
    Place("flamengo", "Flamengo", "Rio de Janeiro", "RJ", "Sudeste", -22.9068, -43.1729),
    Place("fluminense", "Fluminense", "Rio de Janeiro", "RJ", "Sudeste", -22.9068, -43.1729),
    Place("gremio", "Grêmio", "Porto Alegre", "RS", "Sul", -30.0346, -51.2177),
    Place("internacional", "Internacional", "Porto Alegre", "RS", "Sul", -30.0346, -51.2177),
    Place("palmeiras", "Palmeiras", "São Paulo", "SP", "Sudeste", -23.5505, -46.6333),
    Place("santos", "Santos", "Santos", "SP", "Sudeste", -23.9608, -46.3336),
    Place("sao_paulo", "São Paulo", "São Paulo", "SP", "Sudeste", -23.5505, -46.6333),
    Place("vasco", "Vasco", "Rio de Janeiro", "RJ", "Sudeste", -22.9068, -43.1729),
]


REGION_COLORS = {
    "Nordeste": "#f28e2b",
    "Sudeste": "#4e79a7",
    "Sul": "#e15759",
}


def haversine_km(a: Place, b: Place) -> float:
    radius = 6371.0
    lat1, lon1 = radians(a.lat), radians(a.lon)
    lat2, lon2 = radians(b.lat), radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * radius * asin(sqrt(h))


def route_distance(route: list[Place]) -> float:
    return sum(haversine_km(a, b) for a, b in zip(route, route[1:]))


def held_karp(start: Place, destinations: list[Place], return_to_start: bool) -> tuple[list[Place], float]:
    if not destinations:
        route = [start, start] if return_to_start else [start]
        return route, 0.0

    n = len(destinations)
    dp: dict[tuple[int, int], tuple[float, int | None]] = {}

    for j, destination in enumerate(destinations):
        dp[(1 << j, j)] = (haversine_km(start, destination), None)

    for mask in range(1, 1 << n):
        for j in range(n):
            if not mask & (1 << j) or (mask, j) not in dp:
                continue
            current_cost, _ = dp[(mask, j)]
            for k in range(n):
                if mask & (1 << k):
                    continue
                new_mask = mask | (1 << k)
                new_cost = current_cost + haversine_km(destinations[j], destinations[k])
                previous = dp.get((new_mask, k))
                if previous is None or new_cost < previous[0]:
                    dp[(new_mask, k)] = (new_cost, j)

    full_mask = (1 << n) - 1
    best_last = 0
    best_cost = float("inf")

    for j in range(n):
        cost, _ = dp[(full_mask, j)]
        if return_to_start:
            cost += haversine_km(destinations[j], start)
        if cost < best_cost:
            best_cost = cost
            best_last = j

    ordered: list[Place] = []
    mask = full_mask
    last: int | None = best_last
    while last is not None:
        ordered.append(destinations[last])
        _, previous = dp[(mask, last)]
        mask ^= 1 << last
        last = previous

    route = [start] + list(reversed(ordered))
    if return_to_start:
        route.append(start)
    return route, best_cost


def nearest_neighbor(start: Place, destinations: list[Place], return_to_start: bool) -> list[Place]:
    unvisited = destinations[:]
    route = [start]
    current = start
    while unvisited:
        next_place = min(unvisited, key=lambda place: haversine_km(current, place))
        route.append(next_place)
        unvisited.remove(next_place)
        current = next_place
    if return_to_start:
        route.append(start)
    return route


def two_opt(route: list[Place], fixed_cycle: bool) -> list[Place]:
    best = route[:]
    improved = True
    start_index = 1
    end_limit = len(best) - 1 if fixed_cycle else len(best)
    while improved:
        improved = False
        for i in range(start_index, end_limit - 1):
            for k in range(i + 1, end_limit):
                candidate = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                if route_distance(candidate) + 1e-9 < route_distance(best):
                    best = candidate
                    improved = True
    return best


def heuristic_route(start: Place, destinations: list[Place], return_to_start: bool) -> tuple[list[Place], float]:
    route = nearest_neighbor(start, destinations, return_to_start)
    optimized = two_opt(route, fixed_cycle=return_to_start)
    return optimized, route_distance(optimized)


def route_rows(route: list[Place], long_trip_km: float) -> list[dict[str, object]]:
    rows = []
    for idx, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
        distance = haversine_km(origin, destination)
        rows.append(
            {
                "Ordem": idx,
                "Origem": origin.name,
                "Destino": destination.name,
                "Cidade origem": origin.city_label,
                "Cidade destino": destination.city_label,
                "Região origem": origin.region,
                "Região destino": destination.region,
                "Distância (km)": round(distance, 1),
                "Inter-regional": "Sim" if origin.region != destination.region else "Não",
                "Viagem longa": "Sim" if distance >= long_trip_km else "Não",
            }
        )
    return rows


def route_map_points(route: list[Place]) -> pd.DataFrame:
    rows = []
    for idx, place in enumerate(route, start=1):
        is_origin = idx == 1
        is_final_return = idx == len(route) and place.id == route[0].id and len(route) > 1
        rows.append(
            {
                "ordem": idx,
                "label": str(idx),
                "clube": place.name,
                "cidade": place.city_label,
                "regiao": place.region,
                "lat": place.lat,
                "lon": place.lon,
                "size": 260 if is_origin or is_final_return else 150,
                "color": "#d62728" if is_origin or is_final_return else "#1f77b4",
                "fill_color": [214, 39, 40, 230] if is_origin or is_final_return else [31, 119, 180, 220],
            }
        )
    return pd.DataFrame(rows)


def route_map_segments(route: list[Place]) -> list[dict[str, object]]:
    segments = []
    for idx, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
        distance = haversine_km(origin, destination)
        segments.append(
            {
                "ordem": idx,
                "origem": origin.name,
                "destino": destination.name,
                "distancia_km": round(distance, 1),
                "path": [[origin.lon, origin.lat], [destination.lon, destination.lat]],
                "color": [17, 24, 39, 210],
                "width": 5 if distance >= 1000 else 3,
            }
        )
    return segments


def route_map_layers(route: list[Place]) -> pdk.Deck:
    points = route_map_points(route)
    segments = route_map_segments(route)
    center_lat = float(points["lat"].mean())
    center_lon = float(points["lon"].mean())

    path_layer = pdk.Layer(
        "PathLayer",
        data=segments,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=2,
        rounded=True,
        pickable=True,
    )
    node_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[lon, lat]",
        get_fill_color="fill_color",
        get_line_color="[17, 24, 39, 255]",
        get_radius="size",
        radius_min_pixels=7,
        radius_max_pixels=22,
        line_width_min_pixels=1,
        stroked=True,
        pickable=True,
    )
    label_layer = pdk.Layer(
        "TextLayer",
        data=points,
        get_position="[lon, lat]",
        get_text="label",
        get_size=15,
        get_color=[255, 255, 255, 255],
        get_text_anchor="'middle'",
        get_alignment_baseline="'center'",
    )

    return pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=4.2,
            pitch=0,
        ),
        layers=[path_layer, node_layer, label_layer],
        tooltip={
            "html": "<b>{origem}</b> → <b>{destino}</b><br/>{distancia_km} km",
            "style": {"backgroundColor": "#111827", "color": "white"},
        },
    )


def summary_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    distances = [float(row["Distância (km)"]) for row in rows]
    return {
        "total_km": sum(distances),
        "maior_trecho_km": max(distances, default=0.0),
        "trechos": float(len(rows)),
        "viagens_longas": float(sum(1 for row in rows if row["Viagem longa"] == "Sim")),
        "interregionais": float(sum(1 for row in rows if row["Inter-regional"] == "Sim")),
    }


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def format_km(value: float) -> str:
    return f"{value:,.1f} km".replace(",", "X").replace(".", ",").replace("X", ".")


def format_percent(value: float) -> str:
    return f"{value:.1f}%".replace(".", ",")


st.set_page_config(page_title="FootRoute", layout="wide")
clubs_by_name = {club.name: club for club in CLUBS}
club_names = list(clubs_by_name)

st.title("FootRoute")
st.caption("Painel de otimização de rotas logísticas entre clubes de futebol.")

with st.sidebar:
    st.header("Configuração")
    start_name = st.selectbox("Clube de origem", club_names, index=club_names.index("Flamengo"))
    available = [name for name in club_names if name != start_name]
    selected_names = st.multiselect("Clubes a visitar", available, default=available)
    return_to_start = st.checkbox("Retornar ao clube de origem", value=True)
    algorithm = st.radio(
        "Algoritmo",
        ["Exato (Held-Karp)", "Heurístico (vizinho mais próximo + 2-opt)"],
        index=0,
    )
    long_trip_km = st.slider("Limiar de viagem longa (km)", 500, 3000, int(LONG_TRIP_DEFAULT), 100)

start = clubs_by_name[start_name]
destinations = [clubs_by_name[name] for name in selected_names]
route, total_distance = (
    held_karp(start, destinations, return_to_start)
    if algorithm.startswith("Exato")
    else heuristic_route(start, destinations, return_to_start)
)

rows = route_rows(route, float(long_trip_km))
metrics = summary_metrics(rows)
baseline = [start] + destinations + ([start] if return_to_start else [])
baseline_distance = route_distance(baseline)
gain = 0.0 if baseline_distance == 0 else (baseline_distance - total_distance) / baseline_distance * 100

metric_cols = st.columns(5)
metric_cols[0].metric("Distância total", format_km(metrics["total_km"]))
metric_cols[1].metric("Trechos", int(metrics["trechos"]))
metric_cols[2].metric("Viagens longas", int(metrics["viagens_longas"]))
metric_cols[3].metric("Inter-regionais", int(metrics["interregionais"]))
metric_cols[4].metric("Ganho vs. ordem inicial", format_percent(gain))

tab_map, tab_legs, tab_model = st.tabs(["Mapa", "Trechos", "Modelo"])

with tab_map:
    st.subheader("Mapa com grafo da rota")
    map_points = route_map_points(route)
    st.pydeck_chart(route_map_layers(route), height=720)
    st.dataframe(
        map_points[["ordem", "clube", "cidade", "regiao"]],
        width="stretch",
        hide_index=True,
    )

with tab_legs:
    st.subheader("Sequência recomendada")
    sequence_rows = pd.DataFrame(
        {
            "Ordem": range(1, len(route) + 1),
            "Ponto": [place.name for place in route],
            "Cidade": [place.city for place in route],
            "Região": [place.region for place in route],
        }
    )
    st.dataframe(sequence_rows, width="stretch", hide_index=True)
    st.write(" → ".join(place.name for place in route))
    st.download_button(
        "Baixar rota em CSV",
        data=csv_bytes(rows),
        file_name="footroute_rota_otimizada.csv",
        mime="text/csv",
    )
    st.divider()
    st.subheader("Trechos calculados")
    st.dataframe(rows, width="stretch", hide_index=True)

with tab_model:
    st.subheader("Formulação matemática")
    st.write("Grafo ponderado de deslocamento:")
    st.latex(r"G=(V,E)")
    st.write("Peso da aresta entre dois pontos:")
    st.latex(r"d_{ij}=\operatorname{dist}(i,j)")
    st.write("Rota de visitação:")
    st.latex(r"\pi=(\pi_0,\pi_1,\ldots,\pi_n)")
    st.write("Função objetivo sem retorno obrigatório:")
    st.latex(r"\min Z=\sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}}")
    st.write("Função objetivo com retorno à origem:")
    st.latex(
        r"Z_{\mathrm{ciclo}}=\sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}}+d_{\pi_n,\pi_0}"
    )
    st.write(
        "O algoritmo exato utiliza programação dinâmica Held-Karp; a alternativa "
        "heurística combina vizinho mais próximo e melhoria local 2-opt."
    )
