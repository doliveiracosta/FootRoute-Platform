# Modelo matematico

Considere um grafo ponderado:

```math
G=(V,E)
```

em que cada vertice representa o ponto de partida ou um pedido a ser entregue, e cada aresta representa um deslocamento urbano entre dois pontos.

A distancia urbana estimada e definida por:

```math
d_{ij}=f\cdot \operatorname{dist}_{geo}(i,j)
```

em que `f` e o fator de ajuste viario e `dist_geo(i,j)` e a distancia geodesica entre os pontos `i` e `j`.

O tempo estimado de deslocamento e atendimento e:

```math
t_{ij}=60\cdot\frac{d_{ij}}{v}+s_j
```

em que `v` e a velocidade media urbana e `s_j` e o tempo medio de parada no destino `j`.

O custo operacional estimado e:

```math
c_{ij}=d_{ij}\cdot c_{km}
```

Para uma rota:

```math
\pi=(\pi_0,\pi_1,\ldots,\pi_n)
```

a funcao objetivo basica e:

```math
\min Z=\sum_{k=0}^{n-1} d_{\pi_k,\pi_{k+1}}
```

Quando ha retorno ao ponto de partida, usa-se:

```math
Z_{\mathrm{ciclo}}=\sum_{k=0}^{n-1}d_{\pi_k,\pi_{k+1}}+d_{\pi_n,\pi_0}
```

Uma extensao multicriterio pode combinar distancia, tempo e custo:

```math
\min Z=\sum_{i\in V}\sum_{j\in V,\,j\neq i}x_{ij}
\left(\alpha d_{ij}+\beta t_{ij}+\gamma c_{ij}\right)
```

com:

```math
\sum_{j\in V,\,j\neq i}x_{ij}=1,\quad \forall i\in V
```

```math
\sum_{i\in V,\,i\neq j}x_{ij}=1,\quad \forall j\in V
```

A versao atual da plataforma usa Held-Karp para instancias pequenas e uma heuristica baseada em vizinho mais proximo e 2-opt para instancias maiores.
