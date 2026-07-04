# Modelo Matemático

O sistema representa os clubes como vértices de um grafo ponderado:

\[
G=(V,E)
\]

em que:

- \(V\) é o conjunto de clubes;
- \(E\) é o conjunto de deslocamentos possíveis entre clubes;
- \(d_{ij}\) é a distância entre o clube \(i\) e o clube \(j\).

Para uma sequência de visitação:

\[
\pi=(\pi_0,\pi_1,\ldots,\pi_n)
\]

a distância total é:

\[
Z = \sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}}
\]

Quando a rota deve retornar ao ponto de origem:

\[
Z_{\text{ciclo}} =
\sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}} + d_{\pi_n,\pi_0}
\]

O objetivo do modelo é:

\[
\min Z
\]

ou, no caso de retorno:

\[
\min Z_{\text{ciclo}}
\]

O painel disponibiliza duas estratégias:

- algoritmo exato Held-Karp, baseado em programação dinâmica;
- heurística vizinho mais próximo com melhoria local 2-opt.
