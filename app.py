# -*- coding: utf-8 -*-
# Dashboard - Copa do Mundo (Trabalho de Faculdade)
# Rodar com:  streamlit run app.py
#,

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Copa do Mundo - Dashboard", layout="wide")

CSV_PATH = "WorldCupMatches.csv"
PREDITORAS = ["Stage_cat", "Attendance_cat", "Home Team Name"]
ALVO = "Match Result"
ROTULOS = ["Home win", "Draw", "Away Win"]


# ============================================================
# 1) CARREGAR E LIMPAR OS DADOS (mesma lógica do notebook)
# ============================================================
@st.cache_data
def carregar_dados():
    df = pd.read_csv(CSV_PATH)

    # Remove as linhas completamente vazias do arquivo
    df.dropna(subset=["Year", "Datetime", "Home Team Name", "Away Team Name"],
              inplace=True)

    # Público faltante -> média da coluna
    df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())

    # Tipos corretos
    df["Year"] = df["Year"].astype(int)
    df["Home Team Goals"] = df["Home Team Goals"].astype(int)
    df["Away Team Goals"] = df["Away Team Goals"].astype(int)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    # Colunas derivadas
    df["Goal Difference"] = df["Home Team Goals"] - df["Away Team Goals"]
    df["Match Result"] = df.apply(
        lambda r: "Home win" if r["Home Team Goals"] > r["Away Team Goals"]
        else ("Away Win" if r["Home Team Goals"] < r["Away Team Goals"] else "Draw"),
        axis=1,
    )

    # Preditoras categóricas
    df["Stage_cat"] = df["Stage"].apply(
        lambda s: "Fase de Grupos"
        if str(s).startswith(("Group", "First", "Preliminary")) else "Mata-mata"
    )
    df["Attendance_cat"], faixas = pd.qcut(
        df["Attendance"], q=3, labels=["Baixo", "Médio", "Alto"], retbins=True
    )
    return df, faixas


# ============================================================
# 2) TREINAR OS MODELOS
# ============================================================
@st.cache_resource
def treinar_modelos(df):
    X = pd.get_dummies(df[PREDITORAS])
    y = df[ALVO]

    arvore = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
    arvore.fit(X, y)

    knn = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(X))))
    knn.fit(X, y)

    return X.columns, arvore, knn


# --- Teorema de Bayes (Naive Bayes manual, com suavização de Laplace) ---
def bayes_probabilidades(df, entrada, alpha=1):
    classes = df[ALVO].unique()
    priori = df[ALVO].value_counts(normalize=True).to_dict()
    k = {f: df[f].nunique() for f in PREDITORAS}

    scores = {}
    for c in classes:
        sub = df[df[ALVO] == c]
        p = priori[c]
        for f in PREDITORAS:
            cont = (sub[f] == entrada[f]).sum()
            p *= (cont + alpha) / (len(sub) + alpha * k[f])
        scores[c] = p

    total = sum(scores.values())
    return {c: scores[c] / total for c in classes}


def proba_sklearn(modelo, colunas, entrada):
    """Monta a linha one-hot da entrada do usuário e devolve as probabilidades."""
    linha = pd.get_dummies(pd.DataFrame([entrada]))
    linha = linha.reindex(columns=colunas, fill_value=0)
    probs = modelo.predict_proba(linha)[0]
    return dict(zip(modelo.classes_, probs))


# ============================================================
# APP
# ============================================================
df, faixas = carregar_dados()
colunas, arvore, knn = treinar_modelos(df)

st.title("⚽ Dashboard - Copa do Mundo")
st.caption("Análise de dados e classificação probabilística do resultado das partidas")

secao = st.sidebar.radio(
    "Navegação",
    ["Seção 1 — Análise dos Dados", "Seção 2 — Classificação Probabilística"],
)


# ------------------------------------------------------------
# SEÇÃO 1 — ANÁLISE DOS DADOS
# ------------------------------------------------------------
if secao.startswith("Seção 1"):
    st.header("Seção 1 — Análise dos Dados")

    # ===== Filtros interativos (barra lateral) =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Filtros da Seção 1")

    ano_min, ano_max = int(df["Year"].min()), int(df["Year"].max())
    anos = st.sidebar.slider("Período (Copas)", ano_min, ano_max,
                             (ano_min, ano_max), step=4)

    fases = st.sidebar.multiselect(
        "Fase da partida", ["Fase de Grupos", "Mata-mata"],
        default=["Fase de Grupos", "Mata-mata"],
    )

    todos_times = sorted(set(df["Home Team Name"]) | set(df["Away Team Name"]))
    times_sel = st.sidebar.multiselect(
        "Seleção envolvida (mandante ou visitante)", todos_times,
        help="Deixe vazio para considerar todas as seleções.",
    )

    top_n = st.sidebar.slider("Top N nos rankings", 5, 20, 10)

    # ===== Aplica os filtros =====
    mask = df["Year"].between(anos[0], anos[1])
    if fases:
        mask &= df["Stage_cat"].isin(fases)
    if times_sel:
        mask &= (df["Home Team Name"].isin(times_sel) |
                 df["Away Team Name"].isin(times_sel))
    dff = df[mask]

    if dff.empty:
        st.warning("Nenhuma partida corresponde aos filtros selecionados. "
                   "Ajuste o período, a fase ou as seleções na barra lateral.")
        st.stop()

    # ===== KPIs que reagem aos filtros =====
    st.caption(f"Mostrando **{len(dff)}** de {len(df)} partidas "
               f"(período {anos[0]}–{anos[1]}).")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Partidas", len(dff))
    k2.metric("Gols por jogo (média)",
              f"{(dff['Home Team Goals'] + dff['Away Team Goals']).mean():.2f}")
    k3.metric("Público médio", f"{dff['Attendance'].mean():,.0f}")
    k4.metric("Vitórias do mandante",
              f"{(dff[ALVO] == 'Home win').mean()*100:.0f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # --- Gráfico 1: distribuição de gols (mandante x visitante) ---
    with col1:
        st.subheader("1. Distribuição de gols por partida")
        st.caption("Objetivo: mostrar quantos gols costumam ser marcados e comparar "
                   "mandante x visitante (variável de contagem → histograma).")
        quem = st.radio("Exibir gols de:", ["Ambos", "Mandante", "Visitante"],
                        horizontal=True, key="g1")
        fig, ax = plt.subplots()
        if quem in ("Ambos", "Mandante"):
            ax.hist(dff["Home Team Goals"], bins=range(0, 12), alpha=0.6,
                    label="Mandante", color="#2a9d8f")
        if quem in ("Ambos", "Visitante"):
            ax.hist(dff["Away Team Goals"], bins=range(0, 12), alpha=0.6,
                    label="Visitante", color="#e76f51")
        ax.set_xlabel("Gols na partida")
        ax.set_ylabel("Nº de partidas")
        ax.legend()
        st.pyplot(fig)

    # --- Gráfico 2: distribuição dos resultados (variável alvo) ---
    with col2:
        st.subheader("2. Distribuição dos resultados")
        st.caption("Objetivo: revelar o desbalanceamento da variável alvo e o "
                   "fator casa (variável categórica → barras ou pizza).")
        tipo = st.radio("Tipo de gráfico:", ["Barras", "Pizza"],
                        horizontal=True, key="g2")
        contagem = dff[ALVO].value_counts().reindex(ROTULOS).fillna(0)
        cores = ["#2a9d8f", "#e9c46a", "#e76f51"]
        fig, ax = plt.subplots()
        if tipo == "Barras":
            ax.bar(contagem.index, contagem.values, color=cores)
            ax.set_ylabel("Nº de partidas")
            for i, v in enumerate(contagem.values):
                ax.text(i, v, str(int(v)), ha="center", va="bottom")
        else:
            ax.pie(contagem.values, labels=contagem.index, colors=cores,
                   autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    # --- Gráfico 3: público (distribuição) ---
    with col3:
        st.subheader("3. Distribuição do público")
        st.caption("Objetivo: entender a assimetria do público pagante "
                   "(variável contínua → histograma).")
        bins = st.slider("Nº de faixas (bins)", 10, 60, 30, key="g3")
        fig, ax = plt.subplots()
        ax.hist(dff["Attendance"], bins=bins, color="#264653")
        ax.axvline(dff["Attendance"].mean(), color="#e76f51", linestyle="--",
                   label=f"Média: {dff['Attendance'].mean():,.0f}")
        ax.set_xlabel("Público")
        ax.set_ylabel("Nº de partidas")
        ax.legend()
        st.pyplot(fig)

    # --- Gráfico 4: saldo de gols médio por ano (tendência temporal) ---
    with col4:
        st.subheader("4. Saldo de gols médio por Copa")
        st.caption("Objetivo: analisar a evolução temporal do equilíbrio das "
                   "partidas (série temporal → gráfico de linha).")
        serie = dff.groupby("Year")["Goal Difference"].mean()
        fig, ax = plt.subplots()
        ax.plot(serie.index, serie.values, marker="o", color="#2a9d8f")
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xlabel("Ano")
        ax.set_ylabel("Saldo médio (mandante)")
        st.pyplot(fig)

    col5, col6 = st.columns(2)

    # --- Gráfico 5: maiores artilheiros ---
    with col5:
        st.subheader(f"5. Top {top_n} seleções que mais marcaram")
        st.caption("Objetivo: ranquear as maiores potências ofensivas "
                   "(comparação entre categorias → barras horizontais).")
        gols_casa = dff.groupby("Home Team Name")["Home Team Goals"].sum()
        gols_fora = dff.groupby("Away Team Name")["Away Team Goals"].sum()
        total = (gols_casa.add(gols_fora, fill_value=0)
                 .sort_values(ascending=False).head(top_n))
        fig, ax = plt.subplots()
        ax.barh(total.index[::-1], total.values[::-1], color="#2a9d8f")
        ax.set_xlabel("Total de gols")
        st.pyplot(fig)

    # --- Gráfico 6: cidades que mais sediaram jogos ---
    with col6:
        st.subheader(f"6. Top {top_n} cidades que mais sediaram")
        st.caption("Objetivo: identificar os principais polos-sede "
                   "(comparação de frequência → barras).")
        cidades = dff["City"].value_counts().head(top_n)
        fig, ax = plt.subplots()
        ax.barh(cidades.index[::-1], cidades.values[::-1], color="#e76f51")
        ax.set_xlabel("Nº de partidas")
        st.pyplot(fig)

    # --- Gráfico 7: gols mandante x visitante por ano (comparação) ---
    st.subheader("7. Gols de mandante x visitante por Copa")
    st.caption("Objetivo: evidenciar o 'fator casa' ao longo da história "
               "(comparação de duas séries → barras agrupadas).")
    gols_ano = dff.groupby("Year")[["Home Team Goals", "Away Team Goals"]].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    gols_ano.plot(kind="bar", ax=ax, color=["#2a9d8f", "#e76f51"])
    ax.set_xlabel("Ano")
    ax.set_ylabel("Total de gols")
    ax.legend(["Mandante", "Visitante"])
    st.pyplot(fig)

    # --- Tabela opcional dos dados filtrados ---
    with st.expander("📋 Ver tabela das partidas filtradas"):
        st.dataframe(
            dff[["Year", "Stage", "City", "Home Team Name", "Home Team Goals",
                 "Away Team Goals", "Away Team Name", "Attendance", "Match Result"]]
            .sort_values("Year").reset_index(drop=True)
        )


# ------------------------------------------------------------
# SEÇÃO 2 — CLASSIFICAÇÃO PROBABILÍSTICA
# ------------------------------------------------------------
else:
    st.header("Seção 2 — Classificação Probabilística")
    st.write("Informe os dados de uma partida e veja a previsão dos **três métodos**: "
             "Teorema de Bayes, Árvore de Decisão e KNN.")

    # ---- Interface de entrada do usuário ----
    st.subheader("Atributos da partida")
    c1, c2, c3 = st.columns(3)
    with c1:
        fase = st.selectbox("Fase", ["Fase de Grupos", "Mata-mata"])
    with c2:
        publico = st.slider("Público", int(df["Attendance"].min()),
                            int(df["Attendance"].max()), 45000, step=1000)
    with c3:
        times = sorted(df["Home Team Name"].unique())
        time_casa = st.selectbox("Seleção mandante", times,
                                 index=times.index("Brazil") if "Brazil" in times else 0)

    # Converte o público para a faixa categórica (Baixo / Médio / Alto)
    publico_cat = str(pd.cut([publico], bins=faixas,
                             labels=["Baixo", "Médio", "Alto"],
                             include_lowest=True)[0])
    entrada = {"Stage_cat": fase, "Attendance_cat": publico_cat,
               "Home Team Name": time_casa}
    st.info(f"Entrada interpretada → Fase: **{fase}** | "
            f"Público: **{publico_cat}** | Mandante: **{time_casa}**")

    if st.button("Classificar partida", type="primary"):
        # ---- Calcula as probabilidades dos três métodos ----
        p_bayes = bayes_probabilidades(df, entrada)
        p_arvore = proba_sklearn(arvore, colunas, entrada)
        p_knn = proba_sklearn(knn, colunas, entrada)

        metodos = {
            "Teorema de Bayes": p_bayes,
            "Árvore de Decisão": p_arvore,
            "KNN": p_knn,
        }

        # ---- Predição de cada método ----
        st.subheader("Predição de cada método")
        cols = st.columns(3)
        for col, (nome, probs) in zip(cols, metodos.items()):
            pred = max(probs, key=probs.get)
            col.metric(nome, pred, f"{probs[pred]*100:.1f}% de confiança")

        # ---- Probabilidades por classe (Bayes em destaque) ----
        st.subheader("Probabilidades pelo Teorema de Bayes")
        tabela_bayes = (pd.Series(p_bayes).reindex(ROTULOS) * 100).round(1)
        st.bar_chart(tabela_bayes)

        # ---- Comparação visual entre os três métodos ----
        st.subheader("Comparação visual entre os três métodos")
        st.caption("Probabilidade atribuída a cada resultado por cada método.")
        comp = pd.DataFrame(
            {nome: [probs.get(r, 0) * 100 for r in ROTULOS]
             for nome, probs in metodos.items()},
            index=ROTULOS,
        )

        fig, ax = plt.subplots(figsize=(9, 4.5))
        comp.plot(kind="bar", ax=ax, rot=0, edgecolor="black")
        ax.set_ylabel("Probabilidade (%)")
        ax.set_xlabel("Resultado")
        ax.set_title("Probabilidade por classe — Bayes x Árvore x KNN")
        ax.legend(title="Método")
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.0f", fontsize=8, padding=2)
        st.pyplot(fig)

        st.dataframe(comp.round(1).astype(str) + " %")
