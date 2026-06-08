# -*- coding: utf-8 -*-
# Dashboard - Copa do Mundo (Trabalho de Faculdade)
# Rodar com:  streamlit run app.py
#,

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(
    page_title="Copa do Mundo · Dashboard",
    page_icon="⚽",
    layout="wide",
)

# ============================================================
# IDENTIDADE VISUAL — paleta semântica, estilo dos gráficos e CSS
# ============================================================
PALETA = {
    "home": "#2a9d8f",   # vitória do mandante
    "draw": "#e3b23c",   # empate
    "away": "#e76f51",   # vitória do visitante
    "ink":  "#2b3a42",   # texto e séries neutras
    "soft": "#5b6b73",   # rótulos secundários
}
CORES_RESULTADO = [PALETA["home"], PALETA["draw"], PALETA["away"]]
ROTULO_PT = {"Home win": "Vitória do mandante", "Draw": "Empate",
             "Away Win": "Vitória do visitante"}
COR_RESULTADO = {"Home win": PALETA["home"], "Draw": PALETA["draw"],
                 "Away Win": PALETA["away"]}

plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "axes.edgecolor": "#cdbfa8",
    "axes.labelcolor": PALETA["ink"],
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "text.color": PALETA["ink"],
    "xtick.color": PALETA["soft"],
    "ytick.color": PALETA["soft"],
    "font.size": 10,
})


def estilo_eixo(ax, grid="y"):
    """Remove molduras e aplica grid suave — visual consistente em todo o dashboard."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#e7dccb", linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
    return ax


def aplicar_css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; }
          h1, h2, h3 { letter-spacing: -0.015em; }
          div[data-testid="stMetric"], div[data-testid="metric-container"] {
              background: #ffffff;
              border: 1px solid #efe6d6;
              border-bottom: 3px solid #2a9d8f;
              border-radius: 10px;
              padding: 14px 18px;
          }
          div[data-testid="stMetricLabel"] p { color: #5b6b73; font-weight: 600; }
          section[data-testid="stSidebar"] { border-right: 1px solid #efe6d6; }
          hr { border-color: #efe6d6; }
        </style>
        """,
        unsafe_allow_html=True,
    )


CSV_PATH = "WorldCupMatches.csv"
# Preditoras enriquecidas: além da fase, público e mandante, o modelo agora usa
# a seleção visitante e a década da partida — mais sinal para os algoritmos.
PREDITORAS = ["Stage_cat", "Attendance_cat", "Home Team Name", "Away Team Name", "Decada"]
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

    # Década da partida (ex.: 1990s) -> captura a "época" do futebol como atributo
    df["Decada"] = (df["Year"] // 10 * 10).astype(str) + "s"
    return df, faixas


# ============================================================
# 2) TREINAR OS MODELOS
# ============================================================
@st.cache_resource
def treinar_modelos(df):
    X = pd.get_dummies(df[PREDITORAS])

    # Rótulos codificados como inteiros: necessário para o KNN com métrica
    # manhattan (evita um bug do scikit-learn com rótulos de texto) e uniformiza
    # o tratamento das classes entre os modelos.
    le = LabelEncoder()
    y = le.fit_transform(df[ALVO])

    # Split treino/teste estratificado -> avaliação honesta (sem vazamento de dados)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Hiperparâmetros otimizados via GridSearchCV (5-fold) no notebook SuperProjeto.ipynb
    arvore = DecisionTreeClassifier(
        criterion="entropy", max_depth=None, min_samples_leaf=10, random_state=42
    )
    arvore.fit(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=21, weights="distance", metric="manhattan")
    knn.fit(X_train, y_train)

    # Métricas no conjunto de teste (acurácia + F1 macro, que enxerga as 3 classes)
    metricas = {}
    for nome, modelo in [("Árvore de Decisão", arvore), ("KNN", knn)]:
        pred = modelo.predict(X_test)
        metricas[nome] = {
            "Acurácia": accuracy_score(y_test, pred),
            "F1 (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
        }
    rotulo_home = le.transform(["Home win"])[0]
    baseline = (y_test == rotulo_home).mean()  # acurácia de "chutar sempre o mandante"

    return X.columns, arvore, knn, le, metricas, baseline


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


def proba_sklearn(modelo, colunas, entrada, le):
    """Monta a linha one-hot da entrada do usuário e devolve as probabilidades."""
    linha = pd.get_dummies(pd.DataFrame([entrada]))
    linha = linha.reindex(columns=colunas, fill_value=0)
    probs = modelo.predict_proba(linha)[0]
    classes = le.inverse_transform(modelo.classes_)  # inteiros -> rótulos de texto
    return dict(zip(classes, probs))


# ============================================================
# APP
# ============================================================
df, faixas = carregar_dados()
colunas, arvore, knn, le, metricas, baseline = treinar_modelos(df)

aplicar_css()

st.title("⚽ Copa do Mundo · Dashboard")
st.caption(
    f"Análise exploratória e classificação do resultado das partidas  ·  "
    f"{len(df):,} jogos  ·  {df['Year'].min()}–{df['Year'].max()}".replace(",", ".")
)
st.divider()

st.sidebar.markdown("### ⚽ Navegação")
secao = st.sidebar.radio(
    "Seção",
    ["Seção 1 — Análise dos Dados", "Seção 2 — Classificação Probabilística"],
    label_visibility="collapsed",
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
    pct_mandante = (dff[ALVO] == "Home win").mean() * 100
    pct_geral = (df[ALVO] == "Home win").mean() * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Partidas", f"{len(dff)}")
    k2.metric("Gols por jogo",
              f"{(dff['Home Team Goals'] + dff['Away Team Goals']).mean():.2f}")
    k3.metric("Público médio", f"{dff['Attendance'].mean():,.0f}".replace(",", "."))
    k4.metric("Vitórias do mandante", f"{pct_mandante:.0f}%",
              delta=f"{pct_mandante - pct_geral:+.0f} pp vs. geral",
              delta_color="off")

    st.divider()
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
            ax.hist(dff["Home Team Goals"], bins=range(0, 12), alpha=0.75,
                    label="Mandante", color=PALETA["home"])
        if quem in ("Ambos", "Visitante"):
            ax.hist(dff["Away Team Goals"], bins=range(0, 12), alpha=0.75,
                    label="Visitante", color=PALETA["away"])
        ax.set_xlabel("Gols na partida")
        ax.set_ylabel("Nº de partidas")
        ax.legend(frameon=False)
        estilo_eixo(ax)
        st.pyplot(fig)

    # --- Gráfico 2: distribuição dos resultados (variável alvo) ---
    with col2:
        st.subheader("2. Distribuição dos resultados")
        st.caption("Objetivo: revelar o desbalanceamento da variável alvo e o "
                   "fator casa (variável categórica → barras ou pizza).")
        tipo = st.radio("Tipo de gráfico:", ["Barras", "Pizza"],
                        horizontal=True, key="g2")
        contagem = dff[ALVO].value_counts().reindex(ROTULOS).fillna(0)
        fig, ax = plt.subplots()
        if tipo == "Barras":
            ax.bar(contagem.index, contagem.values, color=CORES_RESULTADO)
            ax.set_ylabel("Nº de partidas")
            for i, v in enumerate(contagem.values):
                ax.text(i, v, str(int(v)), ha="center", va="bottom",
                        color=PALETA["ink"], fontweight="bold")
            estilo_eixo(ax)
        else:
            ax.pie(contagem.values, labels=contagem.index, colors=CORES_RESULTADO,
                   autopct="%1.1f%%", startangle=90,
                   wedgeprops={"edgecolor": "#faf7f2", "linewidth": 2},
                   textprops={"color": PALETA["ink"]})
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
        ax.hist(dff["Attendance"], bins=bins, color=PALETA["ink"])
        ax.axvline(dff["Attendance"].mean(), color=PALETA["away"], linestyle="--",
                   linewidth=2, label=f"Média: {dff['Attendance'].mean():,.0f}".replace(",", "."))
        ax.set_xlabel("Público")
        ax.set_ylabel("Nº de partidas")
        ax.legend(frameon=False)
        estilo_eixo(ax)
        st.pyplot(fig)

    # --- Gráfico 4: saldo de gols médio por ano (tendência temporal) ---
    with col4:
        st.subheader("4. Saldo de gols médio por Copa")
        st.caption("Objetivo: analisar a evolução temporal do equilíbrio das "
                   "partidas (série temporal → gráfico de linha).")
        serie = dff.groupby("Year")["Goal Difference"].mean()
        fig, ax = plt.subplots()
        ax.fill_between(serie.index, serie.values, 0, color=PALETA["home"], alpha=0.12)
        ax.plot(serie.index, serie.values, marker="o", color=PALETA["home"], linewidth=2)
        ax.axhline(0, color=PALETA["soft"], linestyle="--", linewidth=1)
        ax.set_xlabel("Ano")
        ax.set_ylabel("Saldo médio (mandante)")
        estilo_eixo(ax)
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
        ax.barh(total.index[::-1], total.values[::-1], color=PALETA["home"])
        ax.set_xlabel("Total de gols")
        estilo_eixo(ax, grid="x")
        st.pyplot(fig)

    # --- Gráfico 6: cidades que mais sediaram jogos ---
    with col6:
        st.subheader(f"6. Top {top_n} cidades que mais sediaram")
        st.caption("Objetivo: identificar os principais polos-sede "
                   "(comparação de frequência → barras).")
        cidades = dff["City"].value_counts().head(top_n)
        fig, ax = plt.subplots()
        ax.barh(cidades.index[::-1], cidades.values[::-1], color=PALETA["away"])
        ax.set_xlabel("Nº de partidas")
        estilo_eixo(ax, grid="x")
        st.pyplot(fig)

    # --- Gráfico 7: gols mandante x visitante por ano (comparação) ---
    st.subheader("7. Gols de mandante x visitante por Copa")
    st.caption("Objetivo: evidenciar o 'fator casa' ao longo da história "
               "(comparação de duas séries → barras agrupadas).")
    gols_ano = dff.groupby("Year")[["Home Team Goals", "Away Team Goals"]].sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    gols_ano.plot(kind="bar", ax=ax, color=[PALETA["home"], PALETA["away"]], width=0.8)
    ax.set_xlabel("Ano")
    ax.set_ylabel("Total de gols")
    ax.legend(["Mandante", "Visitante"], frameon=False)
    estilo_eixo(ax)
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

    # ---- Desempenho dos modelos (avaliação honesta em conjunto de teste) ----
    with st.expander("📈 Desempenho dos modelos no conjunto de teste"):
        tabela_m = (pd.DataFrame(metricas).T * 100).round(1).astype(str) + " %"
        st.table(tabela_m)
        st.caption(
            f"Avaliado em 25% dos dados (split estratificado). "
            f"Baseline — chutar sempre 'Home win': **{baseline*100:.1f}%**. "
            "Hiperparâmetros escolhidos por GridSearchCV (validação cruzada 5-fold)."
        )

    # ---- Interface de entrada do usuário ----
    st.subheader("Atributos da partida")
    times = sorted(set(df["Home Team Name"]) | set(df["Away Team Name"]))
    c1, c2, c3 = st.columns(3)
    with c1:
        fase = st.selectbox("Fase", ["Fase de Grupos", "Mata-mata"])
        publico = st.slider("Público", int(df["Attendance"].min()),
                            int(df["Attendance"].max()), 45000, step=1000)
    with c2:
        time_casa = st.selectbox("Seleção mandante", times,
                                 index=times.index("Brazil") if "Brazil" in times else 0)
        time_fora = st.selectbox("Seleção visitante", times,
                                 index=times.index("Italy") if "Italy" in times else 1)
    with c3:
        anos_disp = sorted(df["Year"].unique())
        ano = st.selectbox("Ano da Copa", anos_disp, index=len(anos_disp) - 1)

    # Converte o público para a faixa categórica (Baixo / Médio / Alto)
    publico_cat = str(pd.cut([publico], bins=faixas,
                             labels=["Baixo", "Médio", "Alto"],
                             include_lowest=True)[0])
    decada = f"{ano // 10 * 10}s"
    entrada = {"Stage_cat": fase, "Attendance_cat": publico_cat,
               "Home Team Name": time_casa, "Away Team Name": time_fora,
               "Decada": decada}
    st.info(f"Entrada interpretada → Fase: **{fase}** | Público: **{publico_cat}** | "
            f"Mandante: **{time_casa}** | Visitante: **{time_fora}** | Década: **{decada}**")

    if st.button("Classificar partida", type="primary"):
        # ---- Calcula as probabilidades dos três métodos ----
        p_bayes = bayes_probabilidades(df, entrada)
        p_arvore = proba_sklearn(arvore, colunas, entrada, le)
        p_knn = proba_sklearn(knn, colunas, entrada, le)

        metodos = {
            "Teorema de Bayes": p_bayes,
            "Árvore de Decisão": p_arvore,
            "KNN": p_knn,
        }
        predicoes = {nome: max(probs, key=probs.get) for nome, probs in metodos.items()}

        # ---- Consenso em destaque ----
        votos = pd.Series(list(predicoes.values())).value_counts()
        consenso, n_votos = votos.index[0], int(votos.iloc[0])
        cor = COR_RESULTADO[consenso]
        concordam = "os 3 métodos concordam" if n_votos == 3 else f"{n_votos} de 3 métodos concordam"
        st.markdown(
            f"""
            <div style="background:{cor}14; border:1px solid {cor}55;
                        border-radius:12px; padding:18px 22px; margin:6px 0 4px;">
              <span style="color:{PALETA['soft']}; font-size:0.85rem; font-weight:600;
                           text-transform:uppercase; letter-spacing:.04em;">Resultado mais provável</span>
              <div style="color:{cor}; font-size:1.6rem; font-weight:800; margin-top:2px;">
                {ROTULO_PT[consenso]}
              </div>
              <span style="color:{PALETA['soft']}; font-size:0.9rem;">{concordam}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Predição de cada método ----
        st.markdown("##### Previsão de cada método")
        cols = st.columns(3)
        for col, (nome, probs) in zip(cols, metodos.items()):
            pred = max(probs, key=probs.get)
            col.metric(nome, ROTULO_PT[pred], f"{probs[pred]*100:.1f}% de confiança",
                       delta_color="off")

        st.divider()

        # ---- Comparação visual entre os três métodos ----
        st.markdown("##### Probabilidade por classe — Bayes × Árvore × KNN")
        st.caption("Cada grupo de barras mostra a probabilidade que os três métodos "
                   "atribuem a um mesmo resultado.")
        comp = pd.DataFrame(
            {nome: [probs.get(r, 0) * 100 for r in ROTULOS]
             for nome, probs in metodos.items()},
            index=[ROTULO_PT[r] for r in ROTULOS],
        )

        fig, ax = plt.subplots(figsize=(9, 4.2))
        comp.plot(kind="bar", ax=ax, rot=0, width=0.8,
                  color=[PALETA["home"], PALETA["ink"], PALETA["away"]])
        ax.set_ylabel("Probabilidade (%)")
        ax.set_xlabel("")
        ax.legend(title="Método", frameon=False)
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.0f", fontsize=8, padding=2, color=PALETA["soft"])
        estilo_eixo(ax)
        st.pyplot(fig)

        with st.expander("🔢 Ver tabela de probabilidades (%)"):
            st.dataframe(comp.round(1).astype(str) + " %", width="stretch")
