from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from footroute.models import Place, load_places  # noqa: E402
from footroute.optimization import (  # noqa: E402
    distance_matrix,
    held_karp,
    heuristic_route,
    route_distance,
    route_rows,
    summary_metrics,
)
from footroute.visualization import nearest_capital_rows, route_svg  # noqa: E402


DATA_DIR = ROOT / "data"
LONG_TRIP_DEFAULT = 1500.0


@st.cache_data
def load_data() -> tuple[list[Place], list[Place]]:
    clubs = load_places(DATA_DIR / "clubes_13.csv")
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


st.set_page_config(page_title="FootRoute", layout="wide")

clubs, capitals = load_data()
clubs_by_name = {club.name: club for club in clubs}
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
route, total_distance = solve_route(algorithm, start, destinations, return_to_start)
rows = route_rows(route, long_trip_km=float(long_trip_km))
metrics = summary_metrics(rows)

baseline = baseline_route(start, destinations, return_to_start)
baseline_distance = route_distance(baseline)
gain = 0.0 if baseline_distance == 0 else (baseline_distance - total_distance) / baseline_distance * 100

metric_cols = st.columns(5)
metric_cols[0].metric("Distância total", format_km(metrics["total_km"]))
metric_cols[1].metric("Trechos", int(metrics["trechos"]))
metric_cols[2].metric("Viagens longas", int(metrics["viagens_longas"]))
metric_cols[3].metric("Inter-regionais", int(metrics["interregionais"]))
metric_cols[4].metric("Ganho vs. ordem inicial", format_percent(gain))

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
    st.markdown("### Formulação resumida")

    st.write(
        "Considere um grafo ponderado, em que cada vértice representa um clube "
        "e cada aresta representa o deslocamento entre duas cidades-sede."
    )

    st.latex(r"G=(V,E)")

    st.write(
        "O peso associado a cada aresta é a distância geodésica aproximada entre "
        "os clubes i e j."
    )

    st.latex(r"d_{ij}=\text{distância geodésica entre os clubes } i \text{ e } j")

    st.write("Para uma rota representada por uma sequência de visitação:")

    st.latex(r"\pi=(\pi_0,\pi_1,\ldots,\pi_n)")

    st.write("A função objetivo é minimizar a distância total percorrida:")

    st.latex(r"\min Z = \sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}}")

    if return_to_start:
        st.write("Como há retorno à origem, a rota é tratada como um ciclo:")

        st.latex(
            r"Z_{\text{ciclo}} = "
            r"\sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}} + d_{\pi_n,\pi_0}"
        )
    else:
        st.write("Como não há retorno à origem, a rota é tratada como caminho aberto:")

        st.latex(r"Z_{\text{aberto}} = \sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}}")

    st.write(
        "O algoritmo exato usa programação dinâmica Held-Karp. "
        "A heurística usa vizinho mais próximo seguido de melhoria local 2-opt."
    )
