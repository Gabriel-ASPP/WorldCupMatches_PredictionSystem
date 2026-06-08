# 📄 Relatório Técnico

**Projeto:** Super Projeto — Estatística e Probabilidade
**Tema:** Previsão do resultado de partidas da Copa do Mundo FIFA
**Aluno:** Gabriel Augusto Silva Pinho Pereira

---

## 1. Descrição do dataset

| Item | Descrição |
|---|---|
| **Nome** | FIFA World Cup — *World Cup Matches* |
| **Origem** | [Kaggle — FIFA World Cup](https://www.kaggle.com/datasets/abecklas/fifa-world-cup) |
| **Domínio** | Esporte / futebol (dados reais, não sintéticos) |
| **Instâncias (brutas)** | 4.572 linhas no arquivo original |
| **Instâncias (válidas)** | **852 partidas** após a limpeza |
| **Período** | Copas de **1930 a 2014** |
| **Atributos** | 20 colunas: ano, data, fase, estádio, cidade, seleções mandante/visitante, gols, condições de vitória, público, gols no 1º tempo, arbitragem, IDs, etc. |

### Variável alvo

A variável categórica de interesse é o **resultado da partida** (`Match Result`), **derivada** a partir do placar, com três classes:

| Classe | Condição | Frequência |
|---|---|---|
| `Home win` (vitória do mandante) | gols mandante > gols visitante | 488 partidas (**57,3%**) |
| `Draw` (empate) | gols iguais | 190 partidas (**22,3%**) |
| `Away Win` (vitória do visitante) | gols mandante < gols visitante | 174 partidas (**20,4%**) |

### Variáveis preditoras

| Preditora | Tipo | Descrição |
|---|---|---|
| `Stage_cat` | Categórica | Fase: *Fase de Grupos* (639) ou *Mata-mata* (213) |
| `Attendance_cat` | Categórica | Faixa de público em tercis: *Baixo*, *Médio*, *Alto* |
| `Home Team Name` | Categórica | Seleção mandante |
| `Away Team Name` | Categórica | Seleção visitante *(atributo adicionado para reforçar os modelos)* |
| `Decada` | Categórica | Década da partida, ex.: *1990s* *(captura a "época" do futebol)* |

---

## 2. Justificativa da escolha

O dataset é adequado aos objetivos do projeto pelos seguintes motivos:

- **Dados reais e confiáveis:** partidas oficiais da Copa do Mundo, obtidas de fonte aberta (Kaggle), sem qualquer geração aleatória.
- **Mistura de variáveis qualitativas e quantitativas:** possui categóricas (fase, cidade, seleção) e numéricas (gols, público, ano), permitindo uma EDA rica.
- **Necessidade real de limpeza:** o arquivo original contém milhares de linhas vazias e valores ausentes — atendendo à exigência de tratamento obrigatório.
- **Variável alvo categórica clara:** o resultado da partida é uma variável de três classes, ideal para análise probabilística (Bayes) e classificação.
- **Permite engenharia de atributos:** a variável alvo e as faixas de público foram derivadas dos dados brutos, exercitando *feature engineering*.

---

## 3. Tratamentos aplicados

Cada tratamento abaixo está implementado na função `carregar_dados()` do arquivo `app.py` (e no notebook `SuperProjeto.ipynb`).

### 3.1 Remoção de linhas vazias

```python
df.dropna(subset=["Year", "Datetime", "Home Team Name", "Away Team Name"], inplace=True)
```

**Justificativa:** o CSV original possui 4.572 linhas, mas a maioria são linhas em branco no fim do arquivo (sem ano, data ou times). Elas não representam partidas e distorceriam todas as estatísticas. Após a remoção, restam **852 partidas válidas**.

### 3.2 Imputação de valores ausentes (público)

```python
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())
```

**Justificativa:** algumas partidas não têm o público registrado. Em vez de descartá-las (perdendo informação válida das demais colunas), o valor ausente foi substituído pela **média da coluna** (≈ 45.165 espectadores), uma imputação simples e não enviesada para a tendência central.

### 3.3 Conversão de tipos

```python
df["Year"] = df["Year"].astype(int)
df["Home Team Goals"] = df["Home Team Goals"].astype(int)
df["Away Team Goals"] = df["Away Team Goals"].astype(int)
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
```

**Justificativa:** garante que gols e ano sejam inteiros (operações aritméticas e agrupamentos corretos) e que a data seja um tipo temporal real (ordenação e análises por período).

### 3.4 Engenharia de atributos

```python
df["Goal Difference"] = df["Home Team Goals"] - df["Away Team Goals"]
df["Match Result"] = df.apply(lambda r: "Home win" if r["Home Team Goals"] > r["Away Team Goals"]
                              else ("Away Win" if r["Home Team Goals"] < r["Away Team Goals"] else "Draw"), axis=1)
df["Stage_cat"] = df["Stage"].apply(lambda s: "Fase de Grupos"
                  if str(s).startswith(("Group", "First", "Preliminary")) else "Mata-mata")
df["Decada"] = (df["Year"] // 10 * 10).astype(str) + "s"   # ex.: 1990s
```

**Justificativa:** cria a **variável alvo** (`Match Result`) a partir do placar, o saldo de gols (`Goal Difference`), simplifica a coluna `Stage` em duas categorias interpretáveis e deriva a **década** da partida. A década, junto com a **seleção visitante** (já presente no dataset), foram incorporadas como preditoras dos modelos de classificação — essa engenharia de atributos foi decisiva para o desempenho final (ver Seção 6).

### 3.5 Discretização do público (tercis)

```python
df["Attendance_cat"] = pd.qcut(df["Attendance"], q=3, labels=["Baixo", "Médio", "Alto"])
```

**Justificativa:** o público é uma variável contínua e muito assimétrica. Transformá-la em três faixas de igual frequência (tercis) viabiliza o cálculo das **verossimilhanças** no Teorema de Bayes, que trabalha com categorias.

---

## 4. Insights da Análise Exploratória

As visualizações estão na **Seção 1 do dashboard** (sete gráficos com filtros interativos).

- **Forte "fator casa":** o mandante vence **57,3%** das partidas, contra apenas 20,4% do visitante — o achado mais marcante da análise e a base do desbalanceamento da variável alvo (*Gráfico 2*).
- **Mandante marca mais gols:** média de **1,81 gol** para o mandante contra **1,02** do visitante por partida; a média geral é de **2,83 gols por jogo** (*Gráficos 1 e 7*).
- **O fator casa é ainda maior no mata-mata:** a vitória do mandante sobe de 55,1% (fase de grupos) para **63,8%** no mata-mata, enquanto os empates caem de 24,7% para 15,0% (*relação fase × resultado*).
- **Potências ofensivas históricas:** as seleções que mais marcaram são **Brasil (225 gols)**, Argentina (133), Alemanha FR (131), Itália (128) e França (108) (*Gráfico 5*).
- **Público assimétrico:** a distribuição do público é assimétrica à direita, com média próxima de 45 mil — justificando a discretização em tercis (*Gráfico 3*).
- **Evolução temporal:** o saldo de gols médio favorável ao mandante varia entre as Copas, permitindo observar tendências de equilíbrio ao longo da história (*Gráfico 4*).

---

## 5. Análise probabilística — Teorema de Bayes

Foi implementado um **classificador Naive Bayes manualmente** (sem usar a classe pronta do scikit-learn), aplicando:

$$P(C \mid X) = \frac{P(C) \cdot P(X \mid C)}{P(X)} \quad\text{com}\quad P(X \mid C) = \prod_i P(x_i \mid C)$$

**Etapas calculadas a partir do dataset:**

- **Probabilidades a priori** `P(C)` — frequência relativa de cada classe:
  - `P(Home win) = 57,3%`
  - `P(Draw) = 22,3%`
  - `P(Away Win) = 20,4%`
- **Verossimilhanças** `P(xᵢ | C)` — para cada preditora, a proporção de cada categoria dentro de cada classe, com **suavização de Laplace** (`alpha = 1`) para evitar probabilidade zero quando uma combinação não aparece no treino.
- **Probabilidades a posteriori** `P(C | X)` — produto das anteriores, normalizado para somar 100%.

### Exemplo de cálculo

Para uma partida com **mandante = Brasil, fase = Fase de Grupos, público = Alto**, o modelo retorna:

| Resultado | Probabilidade a posteriori |
|---|---|
| **Home win** | **74,5%** |
| Draw | 15,6% |
| Away Win | 9,9% |

O resultado é coerente: o Brasil, como maior artilheiro histórico e jogando como mandante, tem probabilidade de vitória bem acima da priori geral (57,3% → 74,5%), demonstrando que o modelo aprende o efeito das evidências.

> Este exemplo usa a versão **conceitual** do Bayes (3 atributos), para clareza didática. No dashboard e na comparação final (Seção 6), o Bayes utiliza o mesmo conjunto enriquecido de preditoras dos demais modelos. Implementação na função `bayes_probabilidades()` em `app.py`.

---

## 6. Resultados da classificação

Além do Bayes, foram treinados dois algoritmos do scikit-learn — **Árvore de Decisão** e **KNN** — avaliados em conjunto de teste (25% dos dados, estratificado). Para extrair o máximo deles, aplicou-se:

1. **Tuning de hiperparâmetros** via `GridSearchCV` com validação cruzada (5-fold), buscando a melhor combinação automaticamente:
   - Árvore: `criterion="entropy"`, `max_depth=None`, `min_samples_leaf=10`;
   - KNN: `n_neighbors=21`, `weights="distance"`, `metric="manhattan"`.
2. **Engenharia de atributos:** as preditoras foram enriquecidas com a **seleção visitante** e a **década** da partida (de ~83 para 174 colunas após o one-hot encoding).

### Métricas no conjunto de teste

| Modelo | Acurácia | Precisão (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| **KNN** (k=21, manhattan) | **0,606** | **0,50** | **0,44** | **0,43** |
| **Árvore de Decisão** (entropy) | 0,606 | 0,47 | 0,43 | 0,41 |
| Teorema de Bayes | 0,582 | 0,42 | 0,43 | 0,41 |
| *Baseline (chutar sempre "Home win")* | *0,573* | — | — | — |

**Comparação e interpretação:**

- Após o tuning **e** a engenharia de atributos, **a Árvore e o KNN passaram a superar tanto o baseline (57,3%) quanto o Teorema de Bayes (58,2%)**, atingindo **60,6%** de acurácia. O **KNN foi o melhor modelo geral** (F1 macro = 0,43).
- O ganho veio principalmente das **novas preditoras**: a Árvore saltou de 0,24 para **0,41** de F1 macro e o KNN de 0,32 para **0,43**. Os atributos mais informativos da Árvore passaram a ser a **década** da partida e a **fase** (mata-mata), confirmando que contexto agrega sinal real.
- Os modelos agora **acertam parte das classes minoritárias** (o KNN recupera 15% dos empates e 25% das vitórias do visitante), algo que antes era praticamente nulo.
- A **abordagem bayesiana** continua competitiva (empata em F1 macro) e mantém a vantagem de fornecer **probabilidades interpretáveis por classe**, útil para o dashboard.

### Comunicação dos resultados — o dashboard

Todos os resultados são apresentados num **dashboard interativo** (`app.py`, em Streamlit), pensado para leitura clara durante a apresentação:

- **Seção 1 — Análise dos Dados:** os sete gráficos da EDA com filtros interativos (período, fase, seleção, Top N) e KPIs que reagem aos filtros.
- **Seção 2 — Classificação Probabilística:** o usuário informa os atributos de uma partida (fase, público, **mandante**, **visitante** e **ano**) e recebe a previsão dos **três métodos** lado a lado, com a probabilidade por classe e um **destaque de consenso** (o resultado mais provável e quantos métodos concordam).

A apresentação visual segue uma **identidade consistente**: cada resultado tem uma **cor semântica fixa** (mandante = teal, empate = âmbar, visitante = coral) repetida em todos os gráficos, facilitando a leitura; o tema claro e a tipografia foram escolhidos para boa legibilidade em projeção. O desempenho dos modelos fica acessível num painel de métricas dentro da própria Seção 2.

---

## 7. Conclusões

**Aprendizados:**

- O **fator casa** é o sinal mais forte do dataset: o mandante vence quase 6 em cada 10 partidas, e esse efeito é ainda maior nas fases de mata-mata.
- A implementação **manual do Teorema de Bayes** consolidou a compreensão de prioris, verossimilhanças, posterioris e da necessidade de suavização de Laplace — indo além do simples uso de uma biblioteca.
- **A engenharia de atributos teve mais impacto que o tuning isolado:** adicionar a seleção visitante e a década fez a Árvore e o KNN finalmente superarem o baseline e o Bayes. Princípio central de ML: *o modelo só aprende o que os atributos permitem*.
- A análise por **F1 macro, precisão e recall por classe** revelou comportamentos que a acurácia esconde — essencial num problema desbalanceado.

**Limitações identificadas:**

- **Teto de desempenho:** mesmo com tuning + feature engineering, a acurácia estaciona em ~61%. Resultados de futebol têm forte componente aleatório que os atributos disponíveis não capturam. Ranking/força das seleções, confronto direto histórico ou forma recente poderiam ir além.
- **Desbalanceamento das classes:** a predominância de vitórias do mandante ainda penaliza a previsão de empates.
- **Recorte temporal:** os dados vão até 2014, sem as edições mais recentes.
- **Naive Bayes assume independência** entre as preditoras, o que nem sempre é verdade (público e fase, por exemplo, podem estar correlacionados).

> **Reprodutibilidade:** todos os números deste relatório foram calculados diretamente sobre o `WorldCupMatches.csv` com o código do projeto.
