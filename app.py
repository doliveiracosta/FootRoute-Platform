from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
import csv

import pandas as pd
import pydeck as pdk
import streamlit as st


RECIFE_CENTER = (-8.0632, -34.8711)
DEFAULT_URBAN_FACTOR = 1.32
DEFAULT_AVG_SPEED_KMH = 22
DEFAULT_COST_PER_KM = 1.15
DEFAULT_STOP_MINUTES = 4
EXACT_LIMIT = 12
DATA_DIR = Path(__file__).resolve().parent / "data"
NEIGHBORHOOD_SOURCE_URL = "https://pt.wikipedia.org/wiki/Lista_de_bairros_do_Recife"


@dataclass(frozen=True)
class DeliveryPoint:
    id: str
    name: str
    neighborhood: str
    point_type: str
    lat: float
    lon: float


POINTS = [
    DeliveryPoint("rest_bv", "Restaurante Boa Viagem", "Boa Viagem", "Origem", -8.1265, -34.9006),
    DeliveryPoint("hub_marco_zero", "Hub Marco Zero", "Recife", "Origem", -8.0632, -34.8711),
    DeliveryPoint("mercado_madalena", "Mercado Madalena", "Madalena", "Origem", -8.0529, -34.9069),
    DeliveryPoint("pedido_pina", "Pedido Pina", "Pina", "Entrega", -8.0924, -34.8837),
    DeliveryPoint("pedido_imibiribeira", "Pedido Imbiribeira", "Imbiribeira", "Entrega", -8.1103, -34.9126),
    DeliveryPoint("pedido_afogados", "Pedido Afogados", "Afogados", "Entrega", -8.0707, -34.9092),
    DeliveryPoint("pedido_derby", "Pedido Derby", "Derby", "Entrega", -8.0583, -34.8966),
    DeliveryPoint("pedido_gracas", "Pedido Graças", "Graças", "Entrega", -8.0435, -34.8943),
    DeliveryPoint("pedido_espinheiro", "Pedido Espinheiro", "Espinheiro", "Entrega", -8.0366, -34.8919),
    DeliveryPoint("pedido_jaqueira", "Pedido Jaqueira", "Jaqueira", "Entrega", -8.0371, -34.9042),
    DeliveryPoint("pedido_casa_forte", "Pedido Casa Forte", "Casa Forte", "Entrega", -8.0361, -34.9217),
    DeliveryPoint("pedido_torre", "Pedido Torre", "Torre", "Entrega", -8.0430, -34.9116),
    DeliveryPoint("pedido_cordeiro", "Pedido Cordeiro", "Cordeiro", "Entrega", -8.0511, -34.9302),
    DeliveryPoint("pedido_iputinga", "Pedido Iputinga", "Iputinga", "Entrega", -8.0426, -34.9362),
    DeliveryPoint("pedido_varzea", "Pedido Várzea", "Várzea", "Entrega", -8.0476, -34.9606),
    DeliveryPoint("pedido_casa_amarela", "Pedido Casa Amarela", "Casa Amarela", "Entrega", -8.0261, -34.9170),
    DeliveryPoint("pedido_encruzilhada", "Pedido Encruzilhada", "Encruzilhada", "Entrega", -8.0278, -34.8925),
    DeliveryPoint("pedido_aflitos", "Pedido Aflitos", "Aflitos", "Entrega", -8.0394, -34.8982),
    DeliveryPoint("pedido_boa_vista", "Pedido Boa Vista", "Boa Vista", "Entrega", -8.0589, -34.8833),
    DeliveryPoint("pedido_santo_amaro", "Pedido Santo Amaro", "Santo Amaro", "Entrega", -8.0487, -34.8793),
    DeliveryPoint("pedido_ibura", "Pedido Ibura", "Ibura", "Entrega", -8.1143, -34.9478),
]


@st.cache_data
def load_neighborhood_reference() -> pd.DataFrame:
    path = DATA_DIR / "bairros_recife.csv"
    if path.exists():
        neighborhoods = pd.read_csv(path)
    else:
        neighborhoods = pd.DataFrame({"bairro": sorted({point.neighborhood for point in POINTS})})

    mapped_neighborhoods = {point.neighborhood for point in POINTS}
    neighborhoods["ponto_simulado"] = neighborhoods["bairro"].apply(
        lambda value: "Sim" if value in mapped_neighborhoods else "Não"
    )
    return neighborhoods


def haversine_km(a: DeliveryPoint, b: DeliveryPoint) -> float:
    radius = 6371.0
    lat1, lon1 = radians(a.lat), radians(a.lon)
    lat2, lon2 = radians(b.lat), radians(b.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * radius * asin(sqrt(h))


def estimated_distance_km(a: DeliveryPoint, b: DeliveryPoint, urban_factor: float) -> float:
    return haversine_km(a, b) * urban_factor


def route_distance(route: list[DeliveryPoint], urban_factor: float) -> float:
    return sum(estimated_distance_km(a, b, urban_factor) for a, b in zip(route, route[1:]))


def held_karp(
    start: DeliveryPoint,
    destinations: list[DeliveryPoint],
    return_to_start: bool,
    urban_factor: float,
) -> tuple[list[DeliveryPoint], float]:
    if not destinations:
        route = [start, start] if return_to_start else [start]
        return route, 0.0

    n = len(destinations)
    dp: dict[tuple[int, int], tuple[float, int | None]] = {}

    for j, destination in enumerate(destinations):
        dp[(1 << j, j)] = (estimated_distance_km(start, destination, urban_factor), None)

    for mask in range(1, 1 << n):
        for j in range(n):
            if not mask & (1 << j) or (mask, j) not in dp:
                continue
            current_cost, _ = dp[(mask, j)]
            for k in range(n):
                if mask & (1 << k):
                    continue
                new_mask = mask | (1 << k)
                new_cost = current_cost + estimated_distance_km(destinations[j], destinations[k], urban_factor)
                previous = dp.get((new_mask, k))
                if previous is None or new_cost < previous[0]:
                    dp[(new_mask, k)] = (new_cost, j)

    full_mask = (1 << n) - 1
    best_last = 0
    best_cost = float("inf")
    for j in range(n):
        cost, _ = dp[(full_mask, j)]
        if return_to_start:
            cost += estimated_distance_km(destinations[j], start, urban_factor)
        if cost < best_cost:
            best_cost = cost
            best_last = j

    ordered: list[DeliveryPoint] = []
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


def nearest_neighbor(
    start: DeliveryPoint,
    destinations: list[DeliveryPoint],
    return_to_start: bool,
    urban_factor: float,
) -> list[DeliveryPoint]:
    unvisited = destinations[:]
    route = [start]
    current = start
    while unvisited:
        next_place = min(unvisited, key=lambda place: estimated_distance_km(current, place, urban_factor))
        route.append(next_place)
        unvisited.remove(next_place)
        current = next_place
    if return_to_start:
        route.append(start)
    return route


def two_opt(route: list[DeliveryPoint], fixed_cycle: bool, urban_factor: float) -> list[DeliveryPoint]:
    best = route[:]
    improved = True
    start_index = 1
    end_limit = len(best) - 1 if fixed_cycle else len(best)

    while improved:
        improved = False
        for i in range(start_index, end_limit - 1):
            for k in range(i + 1, end_limit):
                candidate = best[:i] + best[i : k + 1][::-1] + best[k + 1 :]
                if route_distance(candidate, urban_factor) + 1e-9 < route_distance(best, urban_factor):
                    best = candidate
                    improved = True
    return best


def heuristic_route(
    start: DeliveryPoint,
    destinations: list[DeliveryPoint],
    return_to_start: bool,
    urban_factor: float,
) -> tuple[list[DeliveryPoint], float]:
    route = nearest_neighbor(start, destinations, return_to_start, urban_factor)
    optimized = two_opt(route, fixed_cycle=return_to_start, urban_factor=urban_factor)
    return optimized, route_distance(optimized, urban_factor)


def route_rows(
    route: list[DeliveryPoint],
    urban_factor: float,
    avg_speed_kmh: float,
    cost_per_km: float,
    stop_minutes: int,
) -> list[dict[str, object]]:
    rows = []
    for idx, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
        distance = estimated_distance_km(origin, destination, urban_factor)
        travel_minutes = distance / avg_speed_kmh * 60 if avg_speed_kmh > 0 else 0
        has_delivery_stop = destination.id != route[0].id
        service_minutes = stop_minutes if has_delivery_stop else 0
        total_minutes = travel_minutes + service_minutes
        cost = distance * cost_per_km
        rows.append(
            {
                "Ordem": idx,
                "Origem": origin.name,
                "Destino": destination.name,
                "Bairro origem": origin.neighborhood,
                "Bairro destino": destination.neighborhood,
                "Distancia estimada (km)": round(distance, 2),
                "Tempo deslocamento (min)": round(travel_minutes, 1),
                "Tempo parada (min)": service_minutes,
                "Tempo total (min)": round(total_minutes, 1),
                "Custo estimado (R$)": round(cost, 2),
            }
        )
    return rows


def sequence_rows(route: list[DeliveryPoint]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ordem": range(1, len(route) + 1),
            "Ponto": [place.name for place in route],
            "Bairro": [place.neighborhood for place in route],
            "Tipo": [place.point_type for place in route],
        }
    )


def map_points(route: list[DeliveryPoint]) -> pd.DataFrame:
    rows = []
    for idx, point in enumerate(route, start=1):
        is_start = idx == 1
        is_return = idx == len(route) and point.id == route[0].id and len(route) > 1
        rows.append(
            {
                "order": idx,
                "label": str(idx),
                "name": point.name,
                "neighborhood": point.neighborhood,
                "lat": point.lat,
                "lon": point.lon,
                "radius": 95 if is_start or is_return else 65,
                "fill_color": [220, 38, 38, 235] if is_start or is_return else [37, 99, 235, 225],
                "tooltip_title": f"{idx}. {point.name}",
                "tooltip_body": point.neighborhood,
            }
        )
    return pd.DataFrame(rows)


def map_segments(
    route: list[DeliveryPoint],
    points: pd.DataFrame,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    segments = []
    for idx, row in enumerate(rows, start=1):
        origin_point = points.iloc[idx - 1]
        destination_point = points.iloc[idx]
        segments.append(
            {
                "path": [
                    [float(origin_point["lon"]), float(origin_point["lat"])],
                    [float(destination_point["lon"]), float(destination_point["lat"])],
                ],
                "color": [15, 23, 42, 210],
                "width": 4,
                "tooltip_title": f'{idx}. {row["Origem"]} -> {row["Destino"]}',
                "tooltip_body": (
                    f'{row["Distancia estimada (km)"]} km | '
                    f'{row["Tempo total (min)"]} min | '
                    f'R$ {row["Custo estimado (R$)"]}'
                ),
            }
        )
    return segments


def route_map(route: list[DeliveryPoint], rows: list[dict[str, object]]) -> pdk.Deck:
    points = map_points(route)
    segments = map_segments(route, points, rows)
    center_lat = float(points["lat"].mean()) if not points.empty else RECIFE_CENTER[0]
    center_lon = float(points["lon"].mean()) if not points.empty else RECIFE_CENTER[1]

    path_layer = pdk.Layer(
        "PathLayer",
        data=segments,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=3,
        width_max_pixels=7,
        rounded=True,
        pickable=True,
    )
    node_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points,
        get_position="[lon, lat]",
        get_fill_color="fill_color",
        get_line_color=[15, 23, 42, 255],
        get_radius="radius",
        radius_min_pixels=8,
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
        get_size=14,
        get_color=[255, 255, 255, 255],
        get_text_anchor="'middle'",
        get_alignment_baseline="'center'",
    )

    return pdk.Deck(
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=11.25,
            pitch=0,
        ),
        layers=[path_layer, node_layer, label_layer],
        tooltip={
            "html": "<b>{tooltip_title}</b><br/>{tooltip_body}",
            "style": {"backgroundColor": "#111827", "color": "white"},
        },
    )


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def format_km(value: float) -> str:
    return f"{value:,.2f} km".replace(",", "X").replace(".", ",").replace("X", ".")


def format_money(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_minutes(value: float) -> str:
    return f"{value:,.1f} min".replace(",", "X").replace(".", ",").replace("X", ".")


st.set_page_config(page_title="RotaRecife", layout="wide")

neighborhood_reference = load_neighborhood_reference()
points_by_name = {point.name: point for point in POINTS}
origin_names = [point.name for point in POINTS if point.point_type == "Origem"]
delivery_names = [point.name for point in POINTS if point.point_type == "Entrega"]
default_deliveries = [
    "Pedido Pina",
    "Pedido Derby",
    "Pedido Graças",
    "Pedido Jaqueira",
    "Pedido Casa Forte",
    "Pedido Afogados",
    "Pedido Imbiribeira",
]

st.title("RotaRecife")
st.caption("Roteirizacao urbana para apoiar entregadores de pedidos no Recife.")

with st.sidebar:
    st.header("Configuracao")
    start_name = st.selectbox("Ponto de partida", origin_names, index=0)
    selected_names = st.multiselect("Pedidos a entregar", delivery_names, default=default_deliveries)
    return_to_start = st.checkbox("Retornar ao ponto de partida", value=True)
    algorithm = st.radio(
        "Algoritmo",
        ["Exato (Held-Karp)", "Heuristico (vizinho mais proximo + 2-opt)"],
        index=0,
    )

    st.divider()
    st.header("Parametros operacionais")
    avg_speed_kmh = st.slider("Velocidade media urbana (km/h)", 8, 45, DEFAULT_AVG_SPEED_KMH, 1)
    urban_factor = st.slider("Fator de ajuste viario", 1.00, 1.80, DEFAULT_URBAN_FACTOR, 0.02)
    cost_per_km = st.slider("Custo operacional por km (R$)", 0.30, 4.00, DEFAULT_COST_PER_KM, 0.05)
    stop_minutes = st.slider("Tempo medio por entrega (min)", 0, 15, DEFAULT_STOP_MINUTES, 1)

    st.divider()
    st.header("Bairros")
    mapped_count = int((neighborhood_reference["ponto_simulado"] == "Sim").sum())
    st.caption(
        f"{len(neighborhood_reference)} bairros na referencia; "
        f"{mapped_count} com ponto simulado nesta versao."
    )
    with st.expander("Ver cobertura por bairro"):
        st.dataframe(neighborhood_reference, width="stretch", hide_index=True)
        st.markdown(f"[Fonte da lista de bairros]({NEIGHBORHOOD_SOURCE_URL})")

start = points_by_name[start_name]
destinations = [points_by_name[name] for name in selected_names]

use_exact = algorithm.startswith("Exato") and len(destinations) <= EXACT_LIMIT
if algorithm.startswith("Exato") and len(destinations) > EXACT_LIMIT:
    st.warning(
        f"O modo exato foi trocado pela heuristica porque ha {len(destinations)} pedidos. "
        f"Use ate {EXACT_LIMIT} pedidos para Held-Karp."
    )

if use_exact:
    route, optimized_distance = held_karp(start, destinations, return_to_start, urban_factor)
else:
    route, optimized_distance = heuristic_route(start, destinations, return_to_start, urban_factor)

rows = route_rows(route, urban_factor, avg_speed_kmh, cost_per_km, stop_minutes)
total_distance = sum(float(row["Distancia estimada (km)"]) for row in rows)
total_travel_minutes = sum(float(row["Tempo deslocamento (min)"]) for row in rows)
total_stop_minutes = sum(float(row["Tempo parada (min)"]) for row in rows)
total_minutes = sum(float(row["Tempo total (min)"]) for row in rows)
total_cost = sum(float(row["Custo estimado (R$)"]) for row in rows)

metric_cols = st.columns(5)
metric_cols[0].metric("Distancia total", format_km(total_distance))
metric_cols[1].metric("Tempo total", format_minutes(total_minutes))
metric_cols[2].metric("Custo estimado", format_money(total_cost))
metric_cols[3].metric("Pedidos", len(destinations))
metric_cols[4].metric("Trechos", len(rows))

tab_map, tab_legs, tab_model = st.tabs(["Mapa", "Trechos", "Modelo"])

with tab_map:
    st.subheader("Mapa da rota recomendada")
    st.pydeck_chart(route_map(route, rows), height=720)

with tab_legs:
    st.subheader("Sequencia recomendada")
    st.dataframe(sequence_rows(route), width="stretch", hide_index=True)
    st.write(" -> ".join(point.name for point in route))
    st.download_button(
        "Baixar trechos em CSV",
        data=csv_bytes(rows),
        file_name="rotarecife_trechos.csv",
        mime="text/csv",
    )
    st.divider()
    st.subheader("Trechos calculados")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tab_model:
    st.subheader("Modelo matematico")
    st.write("Considere um grafo ponderado em que cada vertice representa o ponto de partida ou um pedido.")
    st.latex(r"G=(V,E)")
    st.write("A distancia urbana estimada combina distancia geodesica e fator de ajuste viario.")
    st.latex(r"d_{ij}=f\cdot \operatorname{dist}_{geo}(i,j)")
    st.write("O tempo e o custo de cada deslocamento sao estimados por:")
    st.latex(r"t_{ij}=60\cdot\frac{d_{ij}}{v}+s_j")
    st.latex(r"c_{ij}=d_{ij}\cdot c_{km}")
    st.write("A funcao objetivo minimiza o deslocamento total da rota.")
    st.latex(r"\min Z=\sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}}")
    st.write("Quando ha retorno ao ponto de partida, a rota e tratada como ciclo.")
    st.latex(r"Z_{\mathrm{ciclo}}=\sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}}+d_{\pi_n,\pi_0}")
    st.write("Uma formulacao operacional com tempo e custo pode ser expressa como:")
    st.latex(
        r"\min Z=\sum_{i\in V}\sum_{j\in V,\,j\neq i}x_{ij}"
        r"\left(\alpha d_{ij}+\beta t_{ij}+\gamma c_{ij}\right)"
    )
    st.write(
        "A versao exata usa programacao dinamica Held-Karp. A alternativa heuristica usa "
        "vizinho mais proximo seguido de melhoria local 2-opt. As distancias sao estimativas "
        "baseadas em coordenadas e nao substituem dados reais de transito ou malha viaria."
    )
