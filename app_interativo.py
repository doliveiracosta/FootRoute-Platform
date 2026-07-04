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
    model_vertices = [start] + destinations
    model_rows = model_summary_rows(model_vertices, route, return_to_start, algorithm)
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
        st.markdown(
            f"""
- Vértices selecionados:  
  $V = \{{{', '.join(place.name for place in model_vertices)}\}}$
- Vértice inicial:  
  $s = {start.name}$
- Quantidade de arcos direcionados possíveis, sem laços:  
  $|A| = |V|(|V|-1) = {len(model_vertices)}({len(model_vertices)-1}) = {complete_arc_count(model_vertices)}$
"""
        )

        st.markdown("#### 3. Variáveis de decisão")
        st.markdown(
            r"""
\[
x_{ij}=\begin{cases}
1, & \text{se o deslocamento de } i \text{ para } j \text{ for escolhido}\\
0, & \text{caso contrário}
\end{cases}
\]

\[
u_i = \text{posição de visitação do vértice } i \text{ na rota}
\]
"""
        )

    with graph_col:
        st.markdown("#### Grafo associado à solução")
        st.html(route_svg(clubs, route))
        st.caption("As arestas exibidas representam os arcos selecionados na solução atual.")

    st.divider()

    formulation_col, solution_col = st.columns([1, 1])
    with formulation_col:
        st.markdown("#### 4. Função objetivo")
        st.markdown(
            r"""
O objetivo é minimizar a distância total percorrida:

\[
\min Z = \sum_{i \in V}\sum_{j \in V, j \neq i} d_{ij}x_{ij}
\]
"""
        )
        st.code(objective_terms(rows), language="text")
        st.metric("Valor de Z na solução atual", format_km(metrics["total_km"]))

        st.markdown("#### 5. Restrições principais")
        if return_to_start:
            st.markdown(
                r"""
Como a opção de retorno está ativa, o problema é tratado como uma rota fechada:

\[
\sum_{j \in V, j \neq i}x_{ij}=1, \quad \forall i \in V
\]

\[
\sum_{i \in V, i \neq j}x_{ij}=1, \quad \forall j \in V
\]

As restrições acima impõem uma saída e uma entrada para cada vértice. Para evitar
subciclos, pode-se usar uma restrição de ordenação do tipo MTZ:

\[
u_i-u_j+|V|x_{ij}\leq |V|-1, \quad i \neq j,\ i,j \in V\setminus\{s\}
\]
"""
            )
        else:
            st.markdown(
                r"""
Como a opção de retorno está desativada, o problema é tratado como uma rota aberta:

\[
\sum_{j \in V, j \neq s}x_{sj}=1
\]

\[
\sum_{i \in V, i \neq s}x_{is}=0
\]

\[
\sum_{i \in V, i \neq j}x_{ij}=1, \quad \forall j \in V\setminus\{s\}
\]

Cada destino é visitado uma única vez e a rota não precisa retornar ao vértice inicial.
"""
            )

    with solution_col:
        st.markdown("#### 6. Variáveis ativadas na solução")
        st.dataframe(pd.DataFrame(selected_arcs), width="stretch", hide_index=True)

        st.markdown("#### 7. Trechos que compõem o valor da função objetivo")
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
            "na otimização."
        )
