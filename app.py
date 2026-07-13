from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from math import asin, cos, radians, sin, sqrt
import csv
import html
import json

import streamlit as st
import streamlit.components.v1 as components


RECIFE_CENTER = (-8.0632, -34.8711)
DEFAULT_URBAN_FACTOR = 1.32
DEFAULT_AVG_SPEED_KMH = 22
DEFAULT_COST_PER_KM = 1.15
DEFAULT_STOP_MINUTES = 4
EXACT_LIMIT = 12
DISTANCE_COLUMN = "Distancia aprox. ajustada (km)"
DEVELOPER_NAME = "David de Oliveira Costa"
DEVELOPER_ROLE = "Doutorando em Engenharia de Computacao"
ORCID_URL = "https://orcid.org/0000-0002-6138-7451"
LINKEDIN_URL = "https://www.linkedin.com/in/daviddeoliveiracosta/"


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


def build_start_points_by_neighborhood(points: list[DeliveryPoint]) -> dict[str, DeliveryPoint]:
    start_points: dict[str, DeliveryPoint] = {}
    for point in points:
        if point.neighborhood not in start_points:
            start_points[point.neighborhood] = DeliveryPoint(
                f"partida_{point.neighborhood.lower().replace(' ', '_')}",
                f"Partida - {point.neighborhood}",
                point.neighborhood,
                "Origem",
                point.lat,
                point.lon,
            )
    return dict(sorted(start_points.items()))


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
                DISTANCE_COLUMN: round(distance, 2),
                "Tempo deslocamento (min)": round(travel_minutes, 1),
                "Tempo parada (min)": service_minutes,
                "Tempo total (min)": round(total_minutes, 1),
                "Custo estimado (R$)": round(cost, 2),
            }
        )
    return rows


def sequence_rows(route: list[DeliveryPoint]) -> list[dict[str, object]]:
    rows = []
    for idx, point in enumerate(route, start=1):
        is_final_return = idx == len(route) and len(route) > 1 and point.id == route[0].id
        if is_final_return:
            continue
        rows.append(
            {
                "Ordem": idx,
                "Ponto": point.name,
                "Bairro": point.neighborhood,
                "Tipo": point.point_type,
            }
        )
    return rows


def route_map_html(route: list[DeliveryPoint], rows: list[dict[str, object]]) -> str:
    visible_route = []
    coordinate_counts: dict[tuple[float, float], int] = {}
    for idx, point in enumerate(route, start=1):
        is_final_return = idx == len(route) and len(route) > 1 and point.id == route[0].id
        if is_final_return:
            continue
        key = (round(point.lat, 5), round(point.lon, 5))
        coordinate_counts[key] = coordinate_counts.get(key, 0) + 1
        visible_route.append((idx, point, key))

    coordinate_seen: dict[tuple[float, float], int] = {}
    duplicate_offsets = [
        (-0.00055, -0.00055),
        (0.00055, 0.00055),
        (-0.00055, 0.00055),
        (0.00055, -0.00055),
        (0.00000, 0.00080),
        (0.00000, -0.00080),
    ]
    points = []
    for idx, point, key in visible_route:
        display_lat = point.lat
        display_lon = point.lon
        if coordinate_counts[key] > 1:
            duplicate_index = coordinate_seen.get(key, 0)
            offset_lat, offset_lon = duplicate_offsets[duplicate_index % len(duplicate_offsets)]
            display_lat += offset_lat
            display_lon += offset_lon
            coordinate_seen[key] = duplicate_index + 1

        points.append(
            {
                "label": str(idx),
                "name": point.name,
                "neighborhood": point.neighborhood,
                "lat": display_lat,
                "lon": display_lon,
                "kind": "origin" if idx == 1 else "delivery",
            }
        )

    route_coordinates = [[point.lat, point.lon] for point in route]
    points_json = json.dumps(points, ensure_ascii=False)
    coordinates_json = json.dumps(route_coordinates)

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    html, body, #map {{
      height: 100%;
      width: 100%;
      margin: 0;
      padding: 0;
      font-family: Arial, sans-serif;
    }}
    .route-marker {{
      align-items: center;
      border: 2px solid #111827;
      border-radius: 999px;
      color: #fff;
      display: flex;
      font-size: 12px;
      font-weight: 700;
      height: 26px;
      justify-content: center;
      width: 26px;
    }}
    .route-origin {{
      background: #dc2626;
    }}
    .route-delivery {{
      background: #2563eb;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const points = {points_json};
    const routeCoordinates = {coordinates_json};
    const map = L.map('map', {{ scrollWheelZoom: true }});

    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const routeLine = L.polyline(routeCoordinates, {{
      color: '#111827',
      weight: 4,
      opacity: 0.82
    }}).addTo(map);

    points.forEach((point) => {{
      const markerClass = point.kind === 'origin' ? 'route-origin' : 'route-delivery';
      const icon = L.divIcon({{
        className: '',
        html: `<div class="route-marker ${{markerClass}}">${{point.label}}</div>`,
        iconSize: [26, 26],
        iconAnchor: [13, 13]
      }});
      L.marker([point.lat, point.lon], {{ icon }})
        .bindPopup(`<strong>${{point.label}}. ${{point.name}}</strong><br>${{point.neighborhood}}`)
        .addTo(map);
    }});

    if (routeCoordinates.length > 1) {{
      map.fitBounds(routeLine.getBounds(), {{ padding: [28, 28] }});
    }} else {{
      map.setView(routeCoordinates[0] || [{RECIFE_CENTER[0]}, {RECIFE_CENTER[1]}], 12);
    }}
  </script>
</body>
</html>
"""


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        return b""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_Sem registros._"

    headers = list(rows[0].keys())

    def clean(value: object) -> str:
        return html.escape(str(value)).replace("|", "\\|").replace("\n", " ")

    header_line = "| " + " | ".join(clean(header) for header in headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [
        "| " + " | ".join(clean(row.get(header, "")) for header in headers) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def format_km(value: float) -> str:
    return f"{value:,.2f} km".replace(",", "X").replace(".", ",").replace("X", ".")


def format_money(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_minutes(value: float) -> str:
    return f"{value:,.1f} min".replace(",", "X").replace(".", ",").replace("X", ".")


st.set_page_config(page_title="RotaRecife", layout="wide")

st.markdown(
    """
    <style>
      .developer-line {
        color: #111827;
        font-size: 0.96rem;
        font-weight: 700;
        margin: 0.95rem 0 0.8rem 0;
      }
      .app-subtitle {
        color: #8b919e;
        font-size: 0.92rem;
        line-height: 1.45;
        margin: -0.15rem 0 0.2rem 0;
      }
      .profile-links {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin: 0.1rem 0 1.3rem 0;
      }
      .profile-links a {
        align-items: center;
        color: #475569;
        display: inline-flex;
        font-size: 0.92rem;
        gap: 0.42rem;
        text-decoration: none;
      }
      .profile-links a:hover {
        color: #111827;
        text-decoration: underline;
      }
      .profile-badge {
        align-items: center;
        border-radius: 4px;
        color: white;
        display: inline-flex;
        font-size: 0.72rem;
        font-weight: 700;
        height: 18px;
        justify-content: center;
        line-height: 1;
        width: 18px;
      }
      .orcid-badge {
        background: #a6ce39;
        border-radius: 999px;
      }
      .linkedin-badge {
        background: #0a66c2;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

points_by_name = {point.name: point for point in POINTS}
start_points_by_neighborhood = build_start_points_by_neighborhood(POINTS)
start_neighborhoods = list(start_points_by_neighborhood)
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
st.markdown(
    """
    <div class="app-subtitle">
      Roteirizacao urbana para apoiar entregadores de pedidos na Região Metropolitana do Recife
      com o Algoritmo Held-Karp.
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="developer-line">
      Desenvolvido por {DEVELOPER_NAME}, {DEVELOPER_ROLE}, 2026.
    </div>
    <div class="profile-links">
      <a href="{ORCID_URL}" target="_blank" rel="noopener noreferrer">
        <span class="profile-badge orcid-badge">iD</span> Perfil academico
      </a>
      <a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer">
        <span class="profile-badge linkedin-badge">in</span> Perfil profissional
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuracao")
    default_start_index = start_neighborhoods.index("Boa Viagem") if "Boa Viagem" in start_neighborhoods else 0
    start_neighborhood = st.selectbox("Bairro de partida", start_neighborhoods, index=default_start_index)
    start = start_points_by_neighborhood[start_neighborhood]
    selected_names = st.multiselect("Pedidos a entregar", delivery_names, default=default_deliveries)
    return_to_start = st.checkbox("Retornar ao ponto de partida", value=True)
    st.caption("Algoritmo: Held-Karp exato.")

    st.divider()
    st.header("Parametros operacionais")
    avg_speed_kmh = st.slider("Velocidade media urbana (km/h)", 8, 45, DEFAULT_AVG_SPEED_KMH, 1)
    urban_factor = st.slider("Fator de aproximacao viaria", 1.00, 1.80, DEFAULT_URBAN_FACTOR, 0.02)
    cost_per_km = st.slider("Custo operacional por km (R$)", 0.30, 4.00, DEFAULT_COST_PER_KM, 0.05)
    stop_minutes = st.slider("Tempo medio por entrega (min)", 0, 15, DEFAULT_STOP_MINUTES, 1)
    st.divider()
    st.caption(f"Desenvolvido por {DEVELOPER_NAME}.")
    st.markdown(f"[ORCID]({ORCID_URL}) | [LinkedIn]({LINKEDIN_URL})")

destinations = [points_by_name[name] for name in selected_names]

if len(destinations) > EXACT_LIMIT:
    st.error(
        f"O Held-Karp esta habilitado para ate {EXACT_LIMIT} pedidos nesta versao. "
        f"Reduza a selecao para calcular a rota exata."
    )
    st.stop()

route, optimized_distance = held_karp(start, destinations, return_to_start, urban_factor)

rows = route_rows(route, urban_factor, avg_speed_kmh, cost_per_km, stop_minutes)
total_distance = sum(float(row[DISTANCE_COLUMN]) for row in rows)
total_travel_minutes = sum(float(row["Tempo deslocamento (min)"]) for row in rows)
total_stop_minutes = sum(float(row["Tempo parada (min)"]) for row in rows)
total_minutes = sum(float(row["Tempo total (min)"]) for row in rows)
total_cost = sum(float(row["Custo estimado (R$)"]) for row in rows)

metric_cols = st.columns(5)
metric_cols[0].metric("Distancia aprox.", format_km(total_distance))
metric_cols[1].metric("Tempo estimado", format_minutes(total_minutes))
metric_cols[2].metric("Custo estimado", format_money(total_cost))
metric_cols[3].metric("Pedidos", len(destinations))
metric_cols[4].metric("Trechos", len(rows))
st.caption(
    "Distancia, tempo e custo sao estimativas. O mapa exibe ligacoes retas entre pontos, "
    "nao o percurso real pela malha viaria."
)

tab_map, tab_legs, tab_model = st.tabs(["Mapa", "Trechos", "Modelo"])

with tab_map:
    st.subheader("Mapa da rota recomendada")
    st.caption("Tracado esquematico da ordem de visita; nao representa o caminho real pelas ruas.")
    components.html(route_map_html(route, rows), height=720)

with tab_legs:
    st.subheader("Sequencia recomendada")
    st.markdown(markdown_table(sequence_rows(route)))
    visible_route = route[:-1] if len(route) > 1 and route[-1].id == route[0].id else route
    st.write(" -> ".join(point.name for point in visible_route))
    st.download_button(
        "Baixar trechos em CSV",
        data=csv_bytes(rows),
        file_name="rotarecife_trechos.csv",
        mime="text/csv",
    )
    st.divider()
    st.subheader("Trechos calculados")
    st.markdown(markdown_table(rows))

with tab_model:
    st.subheader("Modelo matematico")
    st.write("Considere um grafo ponderado em que cada vertice representa o ponto de partida ou um pedido.")
    st.latex(r"G=(V,E)")
    st.write("A distancia apresentada e uma aproximacao: combina distancia geodesica e fator de ajuste viario.")
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
        "A versao atual usa programacao dinamica Held-Karp para calcular a rota exata. "
        "As distancias sao estimativas baseadas em coordenadas e nao substituem dados reais "
        "de transito ou malha viaria."
    )
