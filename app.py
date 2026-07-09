from __future__ import annotations

from dataclasses import dataclass
from base64 import b64encode
from io import StringIO
from math import asin, cos, radians, sin, sqrt
import csv

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


CAPITALS = [
    Place("salvador", "Salvador", "Salvador", "BA", "Nordeste", -12.9777, -38.5016),
    Place("belo_horizonte", "Belo Horizonte", "Belo Horizonte", "MG", "Sudeste", -19.9167, -43.9345),
    Place("rio_de_janeiro", "Rio de Janeiro", "Rio de Janeiro", "RJ", "Sudeste", -22.9068, -43.1729),
    Place("porto_alegre", "Porto Alegre", "Porto Alegre", "RS", "Sul", -30.0346, -51.2177),
    Place("sao_paulo_capital", "São Paulo", "São Paulo", "SP", "Sudeste", -23.5505, -46.6333),
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


def nearest_capital_rows() -> list[dict[str, object]]:
    rows = []
    for club in CLUBS:
        nearest = min(CAPITALS, key=lambda capital: haversine_km(club, capital))
        rows.append(
            {
                "Clube": club.name,
                "Cidade": club.city_label,
                "Região": club.region,
                "Capital": nearest.name,
                "UF": nearest.state,
                "Distância até capital (km)": round(haversine_km(club, nearest), 1),
            }
        )
    return rows


def project_points(places: list[Place], width: int, height: int, margin: int) -> dict[str, tuple[float, float]]:
    min_lat = min(place.lat for place in places)
    max_lat = max(place.lat for place in places)
    min_lon = min(place.lon for place in places)
    max_lon = max(place.lon for place in places)
    positions = {}
    for place in places:
        x = margin + (place.lon - min_lon) / (max_lon - min_lon) * (width - 2 * margin)
        y = margin + (max_lat - place.lat) / (max_lat - min_lat) * (height - 2 * margin)
        positions[place.id] = (x, y)
    return positions


def svg_escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def route_svg(route: list[Place]) -> str:
    width, height, margin = 1040, 660, 92
    positions = project_points(CLUBS, width, height, margin)
    route_ids = {place.id for place in route}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#111827" opacity="0.86" />',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#fbfbf8" />',
        '<text x="36" y="42" font-family="Arial" font-size="22" font-weight="700" fill="#111827">Rota otimizada</text>',
    ]
    for idx, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
        x1, y1 = positions[origin.id]
        x2, y2 = positions[destination.id]
        distance = haversine_km(origin, destination)
        stroke_width = 2.0 + min(distance / 1200.0, 3.5)
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#111827" stroke-width="{stroke_width:.2f}" opacity="0.58" marker-end="url(#arrow)" />'
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        lines.append(
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="11" fill="#ffffff" stroke="#111827" stroke-width="1" />'
            f'<text x="{mx:.1f}" y="{my + 4:.1f}" text-anchor="middle" font-family="Arial" font-size="10" font-weight="700" fill="#111827">{idx}</text>'
        )
    for place in CLUBS:
        x, y = positions[place.id]
        color = REGION_COLORS.get(place.region, "#6b7280")
        radius = 11 if place.id in route_ids else 7
        opacity = "1" if place.id in route_ids else "0.35"
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" stroke="#111827" stroke-width="1.4" opacity="{opacity}" />'
        )
        lines.append(
            f'<text x="{x + 14:.1f}" y="{y - 2:.1f}" font-family="Arial" font-size="12" font-weight="700" fill="#111827" opacity="{opacity}">{svg_escape(place.name)}</text>'
        )
        lines.append(
            f'<text x="{x + 14:.1f}" y="{y + 13:.1f}" font-family="Arial" font-size="10" fill="#4b5563" opacity="{opacity}">{svg_escape(place.city_label)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def complete_graph_svg() -> str:
    width, height, margin = 1040, 660, 92
    positions = project_points(CLUBS, width, height, margin)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbf8" />',
        '<text x="36" y="42" font-family="Arial" font-size="22" font-weight="700" fill="#111827">Grafo completo dos clubes</text>',
        '<text x="36" y="68" font-family="Arial" font-size="13" fill="#4b5563">Nós representam clubes; arestas indicam deslocamentos possíveis entre cidades-sede.</text>',
    ]

    for i, origin in enumerate(CLUBS):
        for destination in CLUBS[i + 1 :]:
            x1, y1 = positions[origin.id]
            x2, y2 = positions[destination.id]
            distance = haversine_km(origin, destination)
            opacity = 0.10 if distance < 1000 else 0.18
            stroke_width = 0.8 + min(distance / 1800.0, 2.4)
            lines.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#374151" stroke-width="{stroke_width:.2f}" opacity="{opacity:.2f}" />'
            )

    for place in CLUBS:
        x, y = positions[place.id]
        color = REGION_COLORS.get(place.region, "#6b7280")
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{color}" stroke="#111827" stroke-width="1.4" />'
        )
        lines.append(
            f'<text x="{x + 15:.1f}" y="{y - 2:.1f}" font-family="Arial" font-size="12" font-weight="700" fill="#111827">{svg_escape(place.name)}</text>'
        )
        lines.append(
            f'<text x="{x + 15:.1f}" y="{y + 13:.1f}" font-family="Arial" font-size="10" fill="#4b5563">{svg_escape(place.city_label)}</text>'
        )

    legend_x, legend_y = 36, 106
    for idx, (region, color) in enumerate(REGION_COLORS.items()):
        y = legend_y + idx * 22
        lines.append(f'<circle cx="{legend_x}" cy="{y}" r="7" fill="{color}" />')
        lines.append(
            f'<text x="{legend_x + 15}" y="{y + 4}" font-family="Arial" font-size="12" fill="#374151">{svg_escape(region)}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def render_svg(svg: str) -> None:
    encoded = b64encode(svg.encode("utf-8")).decode("ascii")
    st.markdown(
        f'<img src="data:image/svg+xml;base64,{encoded}" style="width: 100%; height: auto;" />',
        unsafe_allow_html=True,
    )


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

tab_route, tab_graph, tab_legs, tab_reference, tab_model = st.tabs(
    ["Rota", "Grafo", "Trechos", "Referenciais", "Modelo"]
)

with tab_route:
    graph_col, route_col = st.columns([1.55, 1])
    with graph_col:
        render_svg(route_svg(route))
    with route_col:
        st.subheader("Sequência recomendada")
        st.write(" → ".join(place.name for place in route))
        st.subheader("Leitura operacional")
        st.write(
            f"O percurso parte de {start.name}, visita {len(destinations)} clube(s) e "
            f"{'retorna à origem' if return_to_start else 'encerra no último destino'}. "
            f"O maior trecho individual tem {format_km(metrics['maior_trecho_km'])}."
        )
        st.download_button(
            "Baixar rota em CSV",
            data=csv_bytes(rows),
            file_name="footroute_rota_otimizada.csv",
            mime="text/csv",
        )

with tab_graph:
    st.subheader("Grafo completo")
    render_svg(complete_graph_svg())
    st.caption(
        "O grafo completo representa todos os deslocamentos possíveis entre os clubes. "
        "No painel de rota, apenas o percurso otimizado é destacado."
    )

with tab_legs:
    st.dataframe(rows, width="stretch", hide_index=True)

with tab_reference:
    st.subheader("Clubes")
    st.dataframe(
        [
            {
                "Clube": club.name,
                "Cidade": club.city_label,
                "Região": club.region,
                "Latitude": club.lat,
                "Longitude": club.lon,
            }
            for club in CLUBS
        ],
        width="stretch",
        hide_index=True,
    )
    st.subheader("Capital de referência")
    st.dataframe(nearest_capital_rows(), width="stretch", hide_index=True)

with tab_model:
    st.markdown(
        r"""
### Formulação resumida

Considere um grafo ponderado \(G=(V,E)\), em que cada vértice representa um
clube e cada aresta representa o deslocamento entre duas cidades-sede. O peso
\(d_{ij}\) é a distância geodésica aproximada entre os clubes \(i\) e \(j\).

Para uma rota \(\pi=(\pi_0,\pi_1,\ldots,\pi_n)\), a função objetivo é:

\[
\min Z = \sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}}
\]

Se houver retorno à origem:

\[
Z_{\text{ciclo}} =
\sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}} + d_{\pi_n,\pi_0}
\]

O algoritmo exato usa programação dinâmica Held-Karp. A heurística usa vizinho
mais próximo seguido de melhoria local 2-opt.
"""
    )
