#  ⚽ Super Projeto — Estatística e Probabilidade

**Previsão do resultado de partidas da Copa do Mundo FIFA**
Análise exploratória, classificação probabilística (Teorema de Bayes) e algoritmos de Machine Learning, apresentados em um dashboard interativo.

---

## 👤 Integrante

| Nome completo |
|---|
| Gabriel Augusto Silva Pinho Pereira |

---

## 📊 Sobre o projeto

Este projeto integra todas as etapas da disciplina de **Estatística e Probabilidade**:

1. **Tratamento e limpeza** de um dataset real de partidas da Copa do Mundo;
2. **Análise Exploratória de Dados (EDA)** com visualizações interativas;
3. **Classificação probabilística** usando o **Teorema de Bayes** implementado manualmente;
4. **Dois algoritmos de classificação** (Árvore de Decisão e KNN) para comparação;
5. **Dashboard interativo** em Streamlit reunindo tudo.

O problema investigado: **dado o contexto de uma partida (fase do torneio, público e seleção mandante), qual o resultado mais provável — vitória do mandante, empate ou vitória do visitante?**

---

## 🗂️ Dataset

- **Nome:** FIFA World Cup — World Cup Matches
- **Fonte:** [Kaggle — FIFA World Cup](https://www.kaggle.com/datasets/abecklas/fifa-world-cup)
- **Arquivo:** `WorldCupMatches.csv`
- **Domínio:** Esporte / futebol (dados reais, não sintéticos)
- **Instâncias:** ~850 partidas válidas (Copas de 1930 a 2014)
- **Atributos:** 20 colunas (ano, fase, estádio, cidade, seleções, gols, público, arbitragem, etc.)

### Variável alvo (categórica)

`Match Result` — variável **derivada** a partir dos gols, com três classes:

| Classe | Condição |
|---|---|
| `Home win` | Gols do mandante > gols do visitante |
| `Draw` | Gols iguais |
| `Away Win` | Gols do mandante < gols do visitante |

### Variáveis preditoras

| Preditora | Tipo | Descrição |
|---|---|---|
| `Stage_cat` | Categórica | Fase da partida: *Fase de Grupos* ou *Mata-mata* |
| `Attendance_cat` | Categórica | Faixa de público: *Baixo*, *Médio* ou *Alto* (tercis) |
| `Home Team Name` | Categórica | Seleção mandante |

---

## 🧹 Tratamentos aplicados

Cada tratamento foi escolhido por uma razão técnica:

| Tratamento | Por quê |
|---|---|
| **Remoção de linhas vazias** (`dropna` em `Year`, `Datetime`, times) | O CSV original traz milhares de linhas em branco no fim do arquivo que não representam partidas. |
| **Imputação de público faltante** (média da coluna) | Evita descartar partidas válidas só por falta do dado de público. |
| **Conversão de tipos** (`Year`, gols → `int`; `Datetime` → datetime) | Permite operações numéricas, ordenação temporal e agrupamentos corretos. |
| **Engenharia de atributos** (`Goal Difference`, `Match Result`, `Stage_cat`) | Cria a variável alvo e simplifica a fase em duas categorias interpretáveis. |
| **Discretização do público** (`pd.qcut` em tercis) | Transforma a variável contínua em faixas, viabilizando o cálculo das verossimilhanças no Teorema de Bayes. |

---

## 🔍 Análise Exploratória (Seção 1 do dashboard)

Sete visualizações, cada uma com objetivo analítico explícito e filtros interativos (período, fase, seleção, Top N):

1. Distribuição de gols por partida (mandante × visitante);
2. Distribuição dos resultados — evidencia o **fator casa** e o desbalanceamento da variável alvo;
3. Distribuição do público (assimetria);
4. Saldo de gols médio por Copa (tendência temporal);
5. Top N seleções que mais marcaram;
6. Top N cidades que mais sediaram jogos;
7. Gols de mandante × visitante por Copa (barras agrupadas).

Há ainda **KPIs** que reagem aos filtros e uma tabela exportável das partidas filtradas.

---

## 🎲 Análise probabilística — Teorema de Bayes

O núcleo do projeto é um **Naive Bayes implementado manualmente** (sem usar a classe pronta do scikit-learn), para demonstrar compreensão genuína do método:

$$P(C \mid X) = \frac{P(C)\cdot P(X \mid C)}{P(X)}$$

- **Prioris** `P(C)`: frequência relativa de cada classe no dataset;
- **Verossimilhanças** `P(X|C)`: contagem condicional de cada atributo por classe;
- **Suavização de Laplace** (`alpha=1`): evita probabilidade zero quando uma combinação não aparece nos dados;
- **Posterioris** `P(C|X)`: normalizadas para somar 1.

Implementação em `bayes_probabilidades()` no arquivo [app.py](app.py).

---

## 🤖 Algoritmos de classificação

Para comparação com a abordagem bayesiana, foram treinados dois modelos do scikit-learn:

| Modelo | Configuração |
|---|---|
| **Árvore de Decisão** | `max_depth=6`, `min_samples_leaf=10` |
| **KNN** | `n_neighbors = √N` (regra prática) |

Na **Seção 2 do dashboard**, o usuário informa os atributos de uma partida e vê:
- A predição de cada um dos **três métodos** (Bayes, Árvore, KNN);
- As probabilidades por classe segundo o Teorema de Bayes;
- Uma **comparação visual** lado a lado entre os três métodos.

---

## 🚀 Como executar

### Pré-requisitos

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

### Rodar o dashboard

```bash
streamlit run app.py
```

O navegador abrirá automaticamente (geralmente em `http://localhost:8501`).

### Explorar o desenvolvimento

O notebook [SuperProjeto.ipynb](SuperProjeto.ipynb) contém o passo a passo da limpeza, EDA e modelagem.

---

## 📁 Estrutura do repositório

```
SuperProjeto/
├── app.py               # Dashboard interativo (Streamlit)
├── SuperProjeto.ipynb   # Notebook: limpeza, EDA e modelagem
├── WorldCupMatches.csv  # Dataset (Copa do Mundo FIFA)
└── README.md            # Este arquivo
```

---

## 🛠️ Tecnologias

- **Python**
- **pandas / numpy** — manipulação de dados
- **matplotlib** — visualizações
- **scikit-learn** — Árvore de Decisão e KNN
- **Streamlit** — dashboard interativo
- **Jupyter Notebook** — desenvolvimento e análise

---

