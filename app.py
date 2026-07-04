from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Place:
    name: str
    city: str
    state: str
    region: str
    lat: float
    lon: float

    @property
    def city_label(self) -> str:
        return f"{self.city}/{self.state}"


def load_places(path: Path) -> list[Place]:
    frame = pd.read_csv(path)
    required = {"name", "city", "state", "region", "lat", "lon"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Arquivo {path} sem coluna(s): {', '.join(sorted(missing))}")
    return [
        Place(
            name=str(row["name"]),
            city=str(row["city"]),
            state=str(row["state"]),
            region=str(row["region"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
        )
        for _, row in frame.iterrows()
    ]


EARTH_RADIUS_KM = 6371.0088
def haversine_km(a: Place, b: Place) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [a.lat, a.lon, b.lat, b.lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(h))


def route_distance(route: list[Place]) -> float:
    return sum(haversine_km(a, b) for a, b in zip(route, route[1:]))


def distance_matrix(places: list[Place]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for origin in places:
        row: dict[str, object] = {"Clube": origin.name}
        for destination in places:
            row[destination.name] = round(haversine_km(origin, destination), 1)
        rows.append(row)
    return rows


def route_rows(route: list[Place], long_trip_km: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
        distance = haversine_km(origin, destination)
        rows.append(
            {
                "ordem": order,
                "origem": origin.name,
                "destino": destination.name,
                "cidade_origem": origin.city_label,
                "cidade_destino": destination.city_label,
                "regiao_origem": origin.region,
                "regiao_destino": destination.region,
                "distancia_km": round(distance, 1),
                "interregional": origin.region != destination.region,
                "viagem_longa": distance >= long_trip_km,
            }
        )
    return rows


def summary_metrics(rows: list[dict[str, object]]) -> dict[str, float | int]:
    distances = [float(row["distancia_km"]) for row in rows]
    return {
        "total_km": sum(distances),
        "trechos": len(rows),
        "viagens_longas": sum(1 for row in rows if bool(row["viagem_longa"])),
        "interregionais": sum(1 for row in rows if bool(row["interregional"])),
        "maior_trecho_km": max(distances, default=0.0),
    }


def _two_opt(route: list[Place], fixed_start: bool = True, fixed_end: bool = False) -> list[Place]:
    if len(route) < 4:
        return route
    best = route[:]
    improved = True
    while improved:
        improved = False
        start_i = 1 if fixed_start else 0
        end_k = len(best) - (1 if fixed_end else 0)
        for i, k in combinations(range(start_i, end_k), 2):
            if k - i < 2:
                continue
            candidate = best[:i] + list(reversed(best[i:k])) + best[k:]
            if route_distance(candidate) + 1e-9 < route_distance(best):
                best = candidate
                improved = True
                break
        if improved:
            continue
    return best


def heuristic_route(start: Place, destinations: list[Place], return_to_start: bool) -> tuple[list[Place], float]:
    remaining = destinations[:]
    route = [start]
    current = start
    while remaining:
        nxt = min(remaining, key=lambda place: haversine_km(current, place))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    if return_to_start:
        route.append(start)
        route = _two_opt(route, fixed_start=True, fixed_end=True)
        if route[-1] != start:
            route.append(start)
    else:
        route = _two_opt(route, fixed_start=True, fixed_end=False)
    return route, route_distance(route)


def held_karp(start: Place, destinations: list[Place], return_to_start: bool) -> tuple[list[Place], float]:
    places = [start] + destinations
    n = len(places)
    if n == 1:
        return [start], 0.0
    if len(destinations) > 12:
        return heuristic_route(start, destinations, return_to_start)

    dist = [[haversine_km(places[i], places[j]) for j in range(n)] for i in range(n)]

    @lru_cache(maxsize=None)
    def dp(mask: int, last: int) -> tuple[float, tuple[int, ...]]:
        if mask == (1 << last):
            return dist[0][last], (0, last)
        prev_mask = mask & ~(1 << last)
        best_cost = float("inf")
        best_path: tuple[int, ...] = ()
        for prev in range(1, n):
            if prev_mask & (1 << prev):
                cost, path = dp(prev_mask, prev)
                candidate = cost + dist[prev][last]
                if candidate < best_cost:
                    best_cost = candidate
                    best_path = path + (last,)
        return best_cost, best_path

    full_mask = sum(1 << i for i in range(1, n))
    best_cost = float("inf")
    best_path: tuple[int, ...] = ()
    for last in range(1, n):
        cost, path = dp(full_mask, last)
        if return_to_start:
            cost += dist[last][0]
        if cost < best_cost:
            best_cost = cost
            best_path = path

    route = [places[index] for index in best_path]
    if return_to_start:
        route.append(start)
    return route, best_cost


def nearest_capital_rows(clubs: list[Place], capitals: list[Place]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for club in clubs:
        capital = min(capitals, key=lambda item: haversine_km(club, item))
        rows.append(
            {
                "clube": club.name,
                "cidade_clube": club.city_label,
                "regiao_clube": club.region,
                "capital_referencia": capital.city,
                "uf_capital": capital.state,
                "regiao_capital": capital.region,
                "distancia_capital_km": round(haversine_km(club, capital), 1),
            }
        )
    return rows


def route_svg(clubs: list[Place], route: list[Place]) -> str:
    if not clubs:
        return ""
    width = 980
    height = 620
    margin = 50
    min_lat = min(place.lat for place in clubs)
    max_lat = max(place.lat for place in clubs)
    min_lon = min(place.lon for place in clubs)
    max_lon = max(place.lon for place in clubs)

    def scale_x(lon: float) -> float:
        if max_lon == min_lon:
            return width / 2
        return margin + (lon - min_lon) / (max_lon - min_lon) * (width - 2 * margin)

    def scale_y(lat: float) -> float:
        if max_lat == min_lat:
            return height / 2
        return height - margin - (lat - min_lat) / (max_lat - min_lat) * (height - 2 * margin)

    points = {place.name: (scale_x(place.lon), scale_y(place.lat)) for place in clubs}
    svg_parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="620" xmlns="http://www.w3.org/2000/svg">',
        '<rect x="0" y="0" width="100%" height="100%" rx="18" fill="#f8fafc"/>',
        '<text x="28" y="36" font-size="22" font-family="Arial" font-weight="700" fill="#111827">Grafo da rota otimizada</text>',
        '<text x="28" y="62" font-size="13" font-family="Arial" fill="#475569">Vértices = clubes/cidades-sede; arestas = deslocamentos selecionados</text>',
    ]

    for idx, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
        x1, y1 = points[origin.name]
        x2, y2 = points[destination.name]
        svg_parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            'stroke="#2563eb" stroke-width="3" stroke-linecap="round" opacity="0.8"/>'
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        svg_parts.append(
            f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="11" fill="#1d4ed8" opacity="0.92"/>'
            f'<text x="{mx:.1f}" y="{my + 4:.1f}" font-size="10" text-anchor="middle" font-family="Arial" fill="white">{idx}</text>'
        )

    route_names = {place.name for place in route}
    for place in clubs:
        x, y = points[place.name]
        selected = place.name in route_names
        radius = 8 if selected else 5
        fill = "#ef4444" if selected else "#64748b"
        svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="white" stroke-width="2"/>')
        svg_parts.append(
            f'<text x="{x + 10:.1f}" y="{y - 8:.1f}" font-size="12" font-family="Arial" fill="#0f172a">{place.name}</text>'
        )
        svg_parts.append(
            f'<text x="{x + 10:.1f}" y="{y + 7:.1f}" font-size="10" font-family="Arial" fill="#64748b">{place.city_label}</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


LONG_TRIP_DEFAULT = 1500.0


@st.cache_data
def load_data() -> tuple[list[Place], list[Place]]:
    clubs = load_places(DATA_DIR / "clubes.csv")
    capitals = load_places(DATA_DIR / "capitais_brasil.csv")
    return clubs, capitals


def format_km(value: float) -> str:
    return f"{value:,.1f} km".replace(",", "X").replace(".", ",").replace("X", ".")


def format_percent(value: float) -> str:
    return f"{value:.1f}%".replace(".", ",")


def bool_label(value: bool) -> str:
    return "Sim" if value else "Não"


def display_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "ordem": "Ordem",
            "origem": "Origem",
            "destino": "Destino",
            "cidade_origem": "Cidade origem",
            "cidade_destino": "Cidade destino",
            "regiao_origem": "Região origem",
            "regiao_destino": "Região destino",
            "distancia_km": "Distância (km)",
            "interregional": "Inter-regional",
            "viagem_longa": "Viagem longa",
        }
    )
    frame["Inter-regional"] = frame["Inter-regional"].map(bool_label)
    frame["Viagem longa"] = frame["Viagem longa"].map(bool_label)
    return frame


def baseline_route(start: Place, destinations: list[Place], return_to_start: bool) -> list[Place]:
    route = [start] + destinations
    if return_to_start:
        route.append(start)
    return route


def solve_route(
    algorithm: str,
    start: Place,
    destinations: list[Place],
    return_to_start: bool,
) -> tuple[list[Place], float]:
    if algorithm.startswith("Exato"):
        return held_karp(start, destinations, return_to_start)
    return heuristic_route(start, destinations, return_to_start)



# Relação clube -> capital/cidade de referência informada pelo usuário.
# Esta camada separa a entidade esportiva (clube), a cidade-sede e a capital
# usada como referência territorial no modelo.
CLUB_CAPITAL_PROFILE: dict[str, dict[str, str]] = {
    "Corinthians": {"capital_referencia": "São Paulo", "uf": "SP", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "São Paulo", "observacao": "Equipe sediada na capital."},
    "Palmeiras": {"capital_referencia": "São Paulo", "uf": "SP", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "São Paulo", "observacao": "Equipe sediada na capital."},
    "São Paulo": {"capital_referencia": "São Paulo", "uf": "SP", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "São Paulo", "observacao": "Equipe sediada na capital."},
    "Botafogo": {"capital_referencia": "Rio de Janeiro", "uf": "RJ", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Rio de Janeiro", "observacao": "Equipe sediada na capital."},
    "Flamengo": {"capital_referencia": "Rio de Janeiro", "uf": "RJ", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Rio de Janeiro", "observacao": "Equipe sediada na capital."},
    "Fluminense": {"capital_referencia": "Rio de Janeiro", "uf": "RJ", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Rio de Janeiro", "observacao": "Equipe sediada na capital."},
    "Vasco": {"capital_referencia": "Rio de Janeiro", "uf": "RJ", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Rio de Janeiro", "observacao": "Equipe sediada na capital. Nome equivalente: Vasco da Gama."},
    "Vasco da Gama": {"capital_referencia": "Rio de Janeiro", "uf": "RJ", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Rio de Janeiro", "observacao": "Equipe sediada na capital."},
    "Atlético-MG": {"capital_referencia": "Belo Horizonte", "uf": "MG", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Belo Horizonte", "observacao": "Equipe sediada na capital."},
    "Cruzeiro": {"capital_referencia": "Belo Horizonte", "uf": "MG", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Belo Horizonte", "observacao": "Equipe sediada na capital."},
    "Athletico-PR": {"capital_referencia": "Curitiba", "uf": "PR", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Curitiba", "observacao": "Equipe sediada na capital."},
    "Coritiba": {"capital_referencia": "Curitiba", "uf": "PR", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Curitiba", "observacao": "Equipe sediada na capital."},
    "Grêmio": {"capital_referencia": "Porto Alegre", "uf": "RS", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Porto Alegre", "observacao": "Equipe sediada na capital."},
    "Internacional": {"capital_referencia": "Porto Alegre", "uf": "RS", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Porto Alegre", "observacao": "Equipe sediada na capital."},
    "Bahia": {"capital_referencia": "Salvador", "uf": "BA", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Salvador", "observacao": "Equipe sediada na capital."},
    "Vitória": {"capital_referencia": "Salvador", "uf": "BA", "tipo_territorial": "Capital com múltiplas equipes", "cidade_sede": "Salvador", "observacao": "Equipe sediada na capital."},
    "Remo": {"capital_referencia": "Belém", "uf": "PA", "tipo_territorial": "Capital com apenas uma equipe", "cidade_sede": "Belém", "observacao": "Equipe sediada na capital."},
    "Red Bull Bragantino": {"capital_referencia": "São Paulo", "uf": "SP", "tipo_territorial": "Interior", "cidade_sede": "Bragança Paulista", "observacao": "Equipe localizada em cidade do interior; capital estadual usada como referência."},
    "Chapecoense": {"capital_referencia": "Florianópolis", "uf": "SC", "tipo_territorial": "Interior", "cidade_sede": "Chapecó", "observacao": "Equipe localizada em cidade do interior; capital estadual usada como referência."},
    "Mirassol": {"capital_referencia": "São Paulo", "uf": "SP", "tipo_territorial": "Interior", "cidade_sede": "Mirassol", "observacao": "Equipe localizada em cidade do interior; capital estadual usada como referência."},
    "Santos": {"capital_referencia": "São Paulo", "uf": "SP", "tipo_territorial": "Interior", "cidade_sede": "Santos", "observacao": "Equipe localizada em cidade do interior; capital estadual usada como referência."},
}


def capital_profile(place: Place) -> dict[str, str]:
    """Return the territorial profile for a club, falling back to the loaded data."""
    profile = CLUB_CAPITAL_PROFILE.get(place.name, {})
    return {
        "clube": place.name,
        "cidade_sede": profile.get("cidade_sede", place.city_label),
        "capital_referencia": profile.get("capital_referencia", place.city_label),
        "uf": profile.get("uf", "-"),
        "tipo_territorial": profile.get("tipo_territorial", "Não classificado"),
        "regiao": place.region,
        "observacao": profile.get("observacao", "Classificação não informada na relação de capitais."),
    }


def capital_profile_rows(places: list[Place]) -> list[dict[str, str]]:
    return [capital_profile(place) for place in places]


def selected_capital_summary(places: list[Place]) -> dict[str, int]:
    rows = capital_profile_rows(places)
    return {
        "clubes_em_capitais": sum(1 for row in rows if row["tipo_territorial"] != "Interior"),
        "clubes_do_interior": sum(1 for row in rows if row["tipo_territorial"] == "Interior"),
        "capitais_distintas": len({row["capital_referencia"] + row["uf"] for row in rows}),
    }


def route_arc_rows(route: list[Place]) -> list[dict[str, object]]:
    """Return the binary decision variables x_ij selected by the optimized route."""
    rows: list[dict[str, object]] = []
    for order, (origin, destination) in enumerate(zip(route, route[1:]), start=1):
        rows.append(
            {
                "ordem": order,
                "variavel": f"x_{{{origin.name},{destination.name}}}",
                "origem": origin.name,
                "destino": destination.name,
                "valor": 1,
            }
        )
    return rows


def complete_arc_count(vertices: list[Place]) -> int:
    """Number of directed arcs in a complete graph without self-loops."""
    return len(vertices) * (len(vertices) - 1)


def model_summary_rows(
    vertices: list[Place],
    route: list[Place],
    return_to_start: bool,
    algorithm: str,
) -> list[dict[str, object]]:
    """Human-readable mapping between user choices and mathematical objects."""
    return [
        {"Elemento da interface": "Clube de origem", "Objeto matemático": "vértice inicial s", "Valor atual": route[0].name if route else "-"},
        {"Elemento da interface": "Clubes a visitar", "Objeto matemático": "conjunto de vértices V", "Valor atual": len(vertices)},
        {"Elemento da interface": "Deslocamentos possíveis", "Objeto matemático": "conjunto de arcos A", "Valor atual": complete_arc_count(vertices)},
        {"Elemento da interface": "Retornar à origem", "Objeto matemático": "rota fechada/ciclo" if return_to_start else "rota aberta/caminho", "Valor atual": bool_label(return_to_start)},
        {"Elemento da interface": "Algoritmo selecionado", "Objeto matemático": "método de solução", "Valor atual": algorithm},
        {"Elemento da interface": "Sequência recomendada", "Objeto matemático": "arcos com x_ij = 1", "Valor atual": max(len(route) - 1, 0)},
    ]


def objective_terms(rows: list[dict[str, object]]) -> str:
    """Build a readable numerical objective expression from route legs."""
    if not rows:
        return "Z = 0"
    terms = []
    for row in rows:
        distance = float(row["distancia_km"])
        origin = str(row["origem"])
        destination = str(row["destino"])
        terms.append(f"{distance:.1f}·x_{{{origin},{destination}}}")
    return "Z = " + " + ".join(terms)


st.set_page_config(page_title="FootRoute", layout="wide")

clubs, capitals = load_data()
clubs_by_name = {club.name: club for club in clubs}
club_names = list(clubs_by_name)

st.title("FootRoute")
st.caption("Painel de otimização de rotas logísticas entre clubes de futebol.")

with st.sidebar:
    st.header("Configuração")
    start_name = st.selectbox("Clube de origem", club_names, index=club_names.index("Flamengo") if "Flamengo" in club_names else 0)
    available = [name for name in club_names if name != start_name]
    selected_names = st.multiselect("Clubes a visitar", available, default=available)
    return_to_start = st.checkbox("Retornar ao clube de origem", value=True)
    algorithm = st.radio(
        "Algoritmo",
        ["Exato (Held-Karp)", "Heurístico (vizinho mais próximo + 2-opt)"],
        index=0,
    )
    long_trip_km = st.slider("Limiar de viagem longa (km)", 500, 3000, int(LONG_TRIP_DEFAULT), 100)
    territorial_layer = st.radio(
        "Camada territorial do modelo",
        ["Cidade-sede do clube", "Capital de referência"],
        index=0,
        help="Define como a aba Modelo interpreta os vértices: pela cidade real do clube ou pela capital estadual de referência.",
    )

start = clubs_by_name[start_name]
destinations = [clubs_by_name[name] for name in selected_names]
if algorithm.startswith("Exato") and len(destinations) > 12:
    st.sidebar.warning(
        "O método exato Held-Karp foi substituído pela heurística porque a seleção atual tem mais de 12 destinos."
    )
    effective_algorithm = "Heurístico (vizinho mais próximo + 2-opt)"
else:
    effective_algorithm = algorithm
route, total_distance = solve_route(effective_algorithm, start, destinations, return_to_start)
rows = route_rows(route, long_trip_km=float(long_trip_km))
metrics = summary_metrics(rows)
territorial_summary = selected_capital_summary([start] + destinations)

baseline = baseline_route(start, destinations, return_to_start)
baseline_distance = route_distance(baseline)
gain = 0.0 if baseline_distance == 0 else (baseline_distance - total_distance) / baseline_distance * 100

metric_cols = st.columns(6)
metric_cols[0].metric("Distância total", format_km(metrics["total_km"]))
metric_cols[1].metric("Trechos", int(metrics["trechos"]))
metric_cols[2].metric("Viagens longas", int(metrics["viagens_longas"]))
metric_cols[3].metric("Inter-regionais", int(metrics["interregionais"]))
metric_cols[4].metric("Interior", int(territorial_summary["clubes_do_interior"]))
metric_cols[5].metric("Ganho vs. ordem inicial", format_percent(gain))

tab_route, tab_legs, tab_reference, tab_model = st.tabs(
    ["Rota", "Trechos", "Referenciais", "Modelo"]
)

with tab_route:
    graph_col, route_col = st.columns([1.55, 1])
    with graph_col:
        st.html(route_svg(clubs, route))
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
            data=display_rows(rows).to_csv(index=False).encode("utf-8-sig"),
            file_name="footroute_rota_otimizada.csv",
            mime="text/csv",
        )

with tab_legs:
    st.dataframe(display_rows(rows), width="stretch", hide_index=True)

with tab_reference:
    ref_cols = st.columns([1, 1])
    with ref_cols[0]:
        st.subheader("Clubes")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Clube": club.name,
                        "Cidade": club.city_label,
                        "Região": club.region,
                        "Latitude": club.lat,
                        "Longitude": club.lon,
                    }
                    for club in clubs
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.subheader("Classificação territorial")
        st.dataframe(
            pd.DataFrame(capital_profile_rows(clubs)).rename(
                columns={
                    "clube": "Clube",
                    "cidade_sede": "Cidade-sede",
                    "capital_referencia": "Capital de referência",
                    "uf": "UF",
                    "tipo_territorial": "Tipo territorial",
                    "regiao": "Região",
                    "observacao": "Observação",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    with ref_cols[1]:
        st.subheader("Capital de referência")
        st.dataframe(
            pd.DataFrame(nearest_capital_rows(clubs, capitals)).rename(
                columns={
                    "clube": "Clube",
                    "cidade_clube": "Cidade",
                    "regiao_clube": "Região",
                    "capital_referencia": "Capital",
                    "uf_capital": "UF",
                    "regiao_capital": "Região capital",
                    "distancia_capital_km": "Distância até capital (km)",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Matriz de distâncias entre clubes"):
        st.dataframe(pd.DataFrame(distance_matrix(clubs)), width="stretch", hide_index=True)

with tab_model:
    model_vertices = [start] + destinations
    selected_profiles = capital_profile_rows(model_vertices)
    model_rows = model_summary_rows(model_vertices, route, return_to_start, effective_algorithm)
    model_rows.append({"Elemento da interface": "Camada territorial do modelo", "Objeto matemático": "interpretação do vértice v_i", "Valor atual": territorial_layer})
    model_rows.append({"Elemento da interface": "Clubes do interior", "Objeto matemático": "vértices com cidade-sede diferente da capital de referência", "Valor atual": territorial_summary["clubes_do_interior"]})
    selected_arcs = route_arc_rows(route)
    solution_edges = display_rows(rows)

    st.subheader("Modelagem matemática interativa")
    st.write(
        "Esta seção traduz as escolhas feitas na barra lateral para os elementos formais "
        "do problema de otimização. Ao alterar origem, destinos, retorno ou algoritmo, "
        "o grafo, as variáveis e a função objetivo são atualizados automaticamente."
    )

    model_col, graph_col = st.columns([1.05, 1])
    with model_col:
        st.markdown("#### 1. Da interface para o modelo")
        st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)

        st.markdown("#### 2. Conjuntos")
        if territorial_layer == "Capital de referência":
            vertex_labels = [f"{row['clube']} → {row['capital_referencia']}/{row['uf']}" for row in selected_profiles]
            vertex_interpretation = "cada clube é interpretado pela capital estadual de referência"
        else:
            vertex_labels = [f"{row['clube']} → {row['cidade_sede']}" for row in selected_profiles]
            vertex_interpretation = "cada clube é interpretado pela sua cidade-sede real"

        st.markdown(
            f"""
- Vértices selecionados:  
  $V = \\{{{', '.join(place.name for place in model_vertices)}\\}}$
- Interpretação territorial dos vértices:  
  {vertex_interpretation}.
- Vértice inicial:  
  $s = {start.name}$
- Quantidade de arcos direcionados possíveis, sem laços:  
  $|A| = |V|(|V|-1) = {len(model_vertices)}({len(model_vertices)-1}) = {complete_arc_count(model_vertices)}$
"""
        )
        st.dataframe(
            pd.DataFrame({"Vértice do modelo": [place.name for place in model_vertices], "Interpretação territorial": vertex_labels}),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### 3. Variáveis de decisão")
        st.write("Variável binária que indica se o arco entre dois vértices foi selecionado na rota:")
        st.latex(
            r"""
x_{ij}=\begin{cases}
1, & \text{se o deslocamento de } i \text{ para } j \text{ for escolhido}\\
0, & \text{caso contrário}
\end{cases}
"""
        )
        st.write("Variável auxiliar de ordenação, usada para representar a posição de visitação:")
        st.latex(r"u_i = \text{posição de visitação do vértice } i \text{ na rota}")

    with graph_col:
        st.markdown("#### Grafo associado à solução")
        st.html(route_svg(clubs, route))
        st.caption("As arestas exibidas representam os arcos selecionados na solução atual.")
        if territorial_summary["clubes_do_interior"] > 0:
            st.warning(
                "A seleção atual contém equipe(s) do interior. Se a análise for feita apenas por capitais, "
                "a capital de referência deve ser interpretada como agregação territorial, não como a cidade real de jogo."
            )

    st.divider()

    formulation_col, solution_col = st.columns([1, 1])
    with formulation_col:
        st.markdown("#### 4. Função objetivo")
        st.write("O objetivo é minimizar a distância total percorrida:")
        st.latex(r"\min Z = \sum_{i \in V}\sum_{j \in V,\ j \neq i} d_{ij}x_{ij}")
        st.code(objective_terms(rows), language="text")
        st.metric("Valor de Z na solução atual", format_km(metrics["total_km"]))

        st.markdown("#### 5. Restrições principais")
        if return_to_start:
            st.write("Como a opção de retorno está ativa, o problema é tratado como uma rota fechada.")
            st.write("Cada vértice deve ter exatamente um arco de saída:")
            st.latex(r"\sum_{j \in V,\ j \neq i}x_{ij}=1, \quad \forall i \in V")
            st.write("Cada vértice deve ter exatamente um arco de entrada:")
            st.latex(r"\sum_{i \in V,\ i \neq j}x_{ij}=1, \quad \forall j \in V")
            st.write("Para evitar subciclos, pode-se usar uma restrição de ordenação do tipo MTZ:")
            st.latex(r"u_i-u_j+|V|x_{ij}\leq |V|-1, \quad i \neq j,\ i,j \in V\setminus\{s\}")
        else:
            st.write("Como a opção de retorno está desativada, o problema é tratado como uma rota aberta.")
            st.write("O vértice inicial deve ter um arco de saída:")
            st.latex(r"\sum_{j \in V,\ j \neq s}x_{sj}=1")
            st.write("Nenhum arco precisa retornar ao vértice inicial:")
            st.latex(r"\sum_{i \in V,\ i \neq s}x_{is}=0")
            st.write("Cada destino selecionado deve ser visitado uma única vez:")
            st.latex(r"\sum_{i \in V,\ i \neq j}x_{ij}=1, \quad \forall j \in V\setminus\{s\}")
            st.write("Assim, cada destino é visitado uma única vez e a rota não precisa retornar ao vértice inicial.")

    with solution_col:
        st.markdown("#### 6. Cidade-sede e capital de referência")
        st.dataframe(
            pd.DataFrame(selected_profiles).rename(
                columns={
                    "clube": "Clube",
                    "cidade_sede": "Cidade-sede",
                    "capital_referencia": "Capital de referência",
                    "uf": "UF",
                    "tipo_territorial": "Tipo territorial",
                    "regiao": "Região",
                    "observacao": "Observação",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### 7. Variáveis ativadas na solução")
        st.dataframe(pd.DataFrame(selected_arcs), width="stretch", hide_index=True)

        st.markdown("#### 8. Trechos que compõem o valor da função objetivo")
        st.dataframe(solution_edges, width="stretch", hide_index=True)

    with st.expander("Interpretação didática da solução"):
        st.write(
            f"A solução atual seleciona {len(selected_arcs)} arco(s) com valor x_ij = 1. "
            f"Esses arcos formam a sequência {' → '.join(place.name for place in route)}. "
            f"A soma dos pesos d_ij desses arcos resulta em {format_km(metrics['total_km'])}."
        )
        st.write(
            "Na prática, isso torna o modelo auditável: o usuário não vê apenas a rota final, "
            "mas também como cada escolha da interface altera os elementos matemáticos usados "
            "na otimização. A classificação territorial acrescenta uma segunda leitura: a rota pode "
            "ser analisada pela cidade-sede real do clube ou pela capital de referência do estado."
        )
