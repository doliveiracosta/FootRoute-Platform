# Modelo matemático resumido

Considere um grafo ponderado:

$$
G=(V,A)
$$

em que cada vértice representa uma equipe associada à sua cidade-sede ou à sua capital de referência, conforme a camada territorial selecionada pelo usuário.

A variável de decisão é:

$$
x_{ij}=\begin{cases}
1, & \text{se o deslocamento de } i \text{ para } j \text{ for escolhido}\\
0, & \text{caso contrário}
\end{cases}
$$

A função objetivo minimiza a distância total percorrida:

$$
\min Z = \sum_{i \in V}\sum_{j \in V, j \neq i} d_{ij}x_{ij}
$$

Para rota fechada, cada vértice deve ter exatamente uma entrada e uma saída:

$$
\sum_{j \in V, j \neq i}x_{ij}=1, \quad \forall i \in V
$$

$$
\sum_{i \in V, i \neq j}x_{ij}=1, \quad \forall j \in V
$$

Para evitar subciclos, pode-se usar uma restrição de ordenação do tipo MTZ:

$$
u_i-u_j+|V|x_{ij}\leq |V|-1, \quad i \neq j,\ i,j \in V\setminus\{s\}
$$
