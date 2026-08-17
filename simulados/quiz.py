"""
Simulado Interativo — Databricks Certified Machine Learning Associate

Material experimental e não oficial. Não há garantia de aprovação.
Execução: streamlit run simulados/quiz.py
"""

import json
import random
import time
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulado — Databricks ML Associate",
    page_icon="🔷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Constantes da prova real (exam guide de 1º de março de 2025)
# ---------------------------------------------------------------------------
PROVA_N_QUESTOES = 48
PROVA_MINUTOS = 90
META_APROVACAO = 70          # não é a nota de corte oficial — o Databricks não a publica

# Peso oficial de cada seção
PESOS_SECAO = {1: 38, 2: 19, 3: 31, 4: 12}

SECOES = {
    1: "Databricks Machine Learning",
    2: "Data Processing",
    3: "Model Development",
    4: "Model Deployment",
}

DIF_LABEL = {"facil": "🟢 Fácil", "media": "🟡 Média", "dificil": "🔴 Difícil"}

# ---------------------------------------------------------------------------
# Carregar questões
# ---------------------------------------------------------------------------
QUESTOES_PATH = Path(__file__).parent / "questoes.json"


@st.cache_data
def carregar_questoes():
    with open(QUESTOES_PATH, encoding="utf-8") as f:
        questoes = json.load(f)
    # Normaliza a resposta correta para sempre ser uma lista de índices
    for q in questoes:
        correta = q["resposta_correta"]
        q["_corretas"] = correta if isinstance(correta, list) else [correta]
    return questoes


QUESTOES_TODAS = carregar_questoes()
MODULOS_DISPONIVEIS = sorted({q["modulo_nome"] for q in QUESTOES_TODAS})
SECOES_DISPONIVEIS = sorted({q["secao"] for q in QUESTOES_TODAS})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def embaralhar(questoes):
    """Embaralha as alternativas de cada questão, reindexando as corretas."""
    resultado = []
    for q in questoes:
        ordem = list(range(len(q["alternativas"])))
        random.shuffle(ordem)
        resultado.append({
            **q,
            "alternativas": [q["alternativas"][i] for i in ordem],
            "_corretas": sorted(ordem.index(c) for c in q["_corretas"]),
        })
    return resultado


def iniciar(questoes, modo, limite_segundos=None):
    if not questoes:
        st.error("Nenhuma questão encontrada com esses filtros. Ajuste e tente novamente.")
        return False
    st.session_state.questoes = embaralhar(questoes)
    st.session_state.idx = 0
    st.session_state.respostas = []      # (escolhas:list, corretas:list, questao)
    st.session_state.modo = modo
    st.session_state.inicio = time.time()
    st.session_state.limite = limite_segundos
    st.session_state.fase = "pergunta"
    return True


def iniciar_modo_estudo(n_questoes, modulos, dificuldades, secoes):
    pool = [
        q for q in QUESTOES_TODAS
        if q["modulo_nome"] in modulos
        and q["dificuldade"] in dificuldades
        and q["secao"] in secoes
    ]
    if not pool:
        st.error("Nenhuma questão encontrada com esses filtros. Ajuste e tente novamente.")
        return False
    return iniciar(random.sample(pool, min(n_questoes, len(pool))), modo="estudo")


def iniciar_modo_prova():
    """Sorteia 48 questões respeitando a distribuição oficial por seção."""
    # 38% / 19% / 31% / 12% de 48 questões
    cotas = {1: 18, 2: 9, 3: 15, 4: 6}
    selecionadas = []
    for secao, cota in cotas.items():
        pool = [q for q in QUESTOES_TODAS if q["secao"] == secao]
        selecionadas += random.sample(pool, min(cota, len(pool)))

    # Se alguma seção tiver questões insuficientes, completa com o restante do banco
    if len(selecionadas) < PROVA_N_QUESTOES:
        ids = {q["id"] for q in selecionadas}
        resto = [q for q in QUESTOES_TODAS if q["id"] not in ids]
        selecionadas += random.sample(resto, min(PROVA_N_QUESTOES - len(selecionadas), len(resto)))

    random.shuffle(selecionadas)
    return iniciar(selecionadas, modo="prova", limite_segundos=PROVA_MINUTOS * 60)


def tempo_restante():
    if not st.session_state.get("limite"):
        return None
    decorrido = time.time() - st.session_state.inicio
    return max(0, st.session_state.limite - decorrido)


def formatar_tempo(segundos):
    minutos, seg = divmod(int(segundos), 60)
    return f"{minutos:02d}:{seg:02d}"


def registrar_resposta(escolhas):
    q = st.session_state.questoes[st.session_state.idx]
    st.session_state.respostas.append((sorted(escolhas), q["_corretas"], q))
    if st.session_state.modo == "prova":
        avancar()
    else:
        st.session_state.fase = "feedback"


def avancar():
    st.session_state.idx += 1
    if st.session_state.idx >= len(st.session_state.questoes):
        st.session_state.fase = "resultado"
    else:
        st.session_state.fase = "pergunta"


def reiniciar():
    for k in ["fase", "questoes", "idx", "respostas", "modo", "inicio", "limite"]:
        st.session_state.pop(k, None)


def render_alternativas(q, escolhas, corretas):
    """Mostra as alternativas coloridas conforme acerto/erro."""
    for i, alt in enumerate(q["alternativas"]):
        letra = chr(65 + i)
        marcou = i in escolhas
        eh_correta = i in corretas
        if eh_correta and marcou:
            st.success(f"✅ **{letra}) {alt}** ← sua resposta (correta)")
        elif eh_correta:
            st.success(f"✅ **{letra}) {alt}** ← resposta correta")
        elif marcou:
            st.error(f"❌ {letra}) {alt} ← sua resposta")
        else:
            st.markdown(f"&nbsp;&nbsp;&nbsp;{letra}) {alt}")


# ---------------------------------------------------------------------------
# Estado inicial
# ---------------------------------------------------------------------------
if "fase" not in st.session_state:
    st.session_state.fase = "home"

# Encerra automaticamente quando o tempo do Modo Prova acaba
if st.session_state.fase in ("pergunta", "feedback"):
    restante = tempo_restante()
    if restante is not None and restante <= 0:
        st.session_state.fase = "resultado"

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if st.session_state.fase == "home":
    st.title("🔷 Simulado — Databricks ML Associate")
    st.caption(
        f"Banco com {len(QUESTOES_TODAS)} questões, distribuídas pelas 4 seções do "
        "exam guide oficial (versão de 1º de março de 2025)."
    )

    st.warning(
        "**Material experimental e não oficial.** Não foi produzido nem endossado pela "
        "Databricks e **não garante aprovação**. As questões são autorais, escritas no estilo "
        "da prova — não são questões reais do exame. Use como complemento ao exam guide oficial."
    )

    tab_prova, tab_estudo = st.tabs(["🎯 Modo Prova", "📚 Modo Estudo"])

    # --- MODO PROVA ---
    with tab_prova:
        st.markdown(
            f"""
**Simula as condições da prova real:**

- **{PROVA_N_QUESTOES} questões** sorteadas com a distribuição oficial por seção
- **{PROVA_MINUTOS} minutos** de cronômetro
- **Sem feedback durante a prova** — a revisão vem só no final
- Algumas questões são de **múltipla seleção**, como no exame real
"""
        )
        cols = st.columns(4)
        for col, (secao, peso) in zip(cols, PESOS_SECAO.items()):
            col.metric(f"Seção {secao}", f"{peso}%", help=SECOES[secao])

        st.caption(
            "⏱️ O cronômetro é verificado a cada interação. Ao zerar, o simulado é "
            "encerrado e o resultado é exibido com o que foi respondido até ali."
        )

        if st.button("▶ Iniciar Modo Prova", type="primary", use_container_width=True):
            if iniciar_modo_prova():
                st.rerun()

    # --- MODO ESTUDO ---
    with tab_estudo:
        st.markdown("Escolha o recorte que você quer treinar. O feedback aparece a cada questão.")

        col1, col2 = st.columns(2)
        with col1:
            n_questoes = st.slider(
                "Número de questões", min_value=5,
                max_value=min(60, len(QUESTOES_TODAS)), value=20, step=5,
            )
        with col2:
            dificuldades = st.multiselect(
                "Dificuldade",
                options=["facil", "media", "dificil"],
                default=["facil", "media", "dificil"],
                format_func=lambda x: DIF_LABEL[x],
            )

        secoes_sel = st.multiselect(
            "Seções oficiais",
            options=SECOES_DISPONIVEIS,
            default=SECOES_DISPONIVEIS,
            format_func=lambda s: f"Seção {s} — {SECOES[s]} ({PESOS_SECAO[s]}%)",
        )

        modulos_sel = st.multiselect(
            "Módulos", options=MODULOS_DISPONIVEIS, default=MODULOS_DISPONIVEIS,
            help="Deixe todos selecionados para cobrir o guia inteiro.",
        )

        disponiveis = len([
            q for q in QUESTOES_TODAS
            if q["modulo_nome"] in modulos_sel
            and q["dificuldade"] in dificuldades
            and q["secao"] in secoes_sel
        ])
        st.caption(f"📚 {disponiveis} questões disponíveis com esses filtros.")

        if st.button("▶ Iniciar Modo Estudo", type="primary", use_container_width=True):
            if not modulos_sel or not dificuldades or not secoes_sel:
                st.error("Selecione ao menos um módulo, uma dificuldade e uma seção.")
            elif iniciar_modo_estudo(n_questoes, modulos_sel, dificuldades, secoes_sel):
                st.rerun()

# ---------------------------------------------------------------------------
# PERGUNTA
# ---------------------------------------------------------------------------
elif st.session_state.fase == "pergunta":
    total = len(st.session_state.questoes)
    idx = st.session_state.idx
    q = st.session_state.questoes[idx]
    multipla = len(q["_corretas"]) > 1

    col_prog, col_info = st.columns([3, 1])
    with col_prog:
        st.progress(idx / total, text=f"Questão {idx + 1} de {total}")
    with col_info:
        restante = tempo_restante()
        if restante is not None:
            st.metric("Tempo", formatar_tempo(restante))
        else:
            acertos = sum(1 for e, c, _ in st.session_state.respostas if e == c)
            st.metric("Acertos", f"{acertos}/{idx}")

    st.divider()
    st.caption(
        f"Seção {q['secao']} — {q['secao_nome']}  ·  {q['modulo_nome']}  ·  "
        f"{DIF_LABEL.get(q['dificuldade'], q['dificuldade'])}"
    )
    st.markdown(f"### {q['pergunta']}")

    if multipla:
        st.info(f"☑️ Questão de múltipla seleção — marque **{len(q['_corretas'])}** alternativas.")
        escolhas = []
        for i, alt in enumerate(q["alternativas"]):
            if st.checkbox(f"{chr(65 + i)}) {alt}", key=f"chk_{idx}_{i}"):
                escolhas.append(i)
    else:
        opcao = st.radio(
            "Selecione sua resposta:",
            options=list(range(len(q["alternativas"]))),
            format_func=lambda i: f"{chr(65 + i)}) {q['alternativas'][i]}",
            index=None, key=f"radio_{idx}", label_visibility="collapsed",
        )
        escolhas = [opcao] if opcao is not None else []

    st.markdown("")
    rotulo = "✔ Confirmar e avançar" if st.session_state.modo == "prova" else "✔ Confirmar resposta"
    if st.button(rotulo, type="primary", use_container_width=True):
        if not escolhas:
            st.warning("Selecione ao menos uma alternativa antes de confirmar.")
        elif multipla and len(escolhas) != len(q["_corretas"]):
            st.warning(f"Esta questão pede exatamente {len(q['_corretas'])} alternativas.")
        else:
            registrar_resposta(escolhas)
            st.rerun()

    with st.sidebar:
        st.markdown("### Progresso")
        if st.session_state.modo == "prova":
            st.caption("As respostas são reveladas apenas ao final.")
            for i, _ in enumerate(st.session_state.respostas):
                st.markdown(f"▪️ Q{i + 1} respondida")
        else:
            for i, (e, c, qr) in enumerate(st.session_state.respostas):
                st.markdown(f"{'✅' if e == c else '❌'} Q{i + 1}: {qr['modulo_nome'][:22]}")
        if st.button("🏠 Encerrar e voltar", use_container_width=True):
            reiniciar()
            st.rerun()

# ---------------------------------------------------------------------------
# FEEDBACK (somente Modo Estudo)
# ---------------------------------------------------------------------------
elif st.session_state.fase == "feedback":
    total = len(st.session_state.questoes)
    idx = st.session_state.idx
    q = st.session_state.questoes[idx]
    escolhas, corretas, _ = st.session_state.respostas[-1]

    st.progress((idx + 1) / total, text=f"Questão {idx + 1} de {total}")
    st.divider()
    st.caption(
        f"Seção {q['secao']} — {q['secao_nome']}  ·  {q['modulo_nome']}  ·  "
        f"{DIF_LABEL.get(q['dificuldade'], q['dificuldade'])}"
    )
    st.markdown(f"### {q['pergunta']}")
    st.markdown("")

    render_alternativas(q, escolhas, corretas)

    st.markdown("")
    if escolhas == corretas:
        st.info(f"💡 **Explicação:** {q['explicacao']}")
    else:
        st.warning(f"💡 **Explicação:** {q['explicacao']}")

    st.divider()
    rotulo = "➡ Próxima questão" if idx + 1 < total else "📊 Ver resultado"
    if st.button(rotulo, type="primary", use_container_width=True):
        avancar()
        st.rerun()

# ---------------------------------------------------------------------------
# RESULTADO
# ---------------------------------------------------------------------------
elif st.session_state.fase == "resultado":
    respostas = st.session_state.respostas
    total_apresentadas = len(st.session_state.questoes)
    respondidas = len(respostas)
    acertos = sum(1 for e, c, _ in respostas if e == c)
    pct = acertos / respondidas * 100 if respondidas else 0.0
    modo_prova = st.session_state.get("modo") == "prova"

    st.title("📊 Resultado")

    if modo_prova and respondidas < total_apresentadas:
        st.error(
            f"⏱️ O tempo acabou com {total_apresentadas - respondidas} questões não respondidas. "
            "Na prova real, questões em branco contam como erro — controle o ritmo."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Acertos", f"{acertos}/{respondidas}")
    col2.metric("Score", f"{pct:.1f}%")
    col3.metric(
        "Meta de segurança",
        "✅ Atingida" if pct >= META_APROVACAO else "❌ Não atingida",
        delta=f"≥ {META_APROVACAO}%",
    )
    st.progress(acertos / respondidas if respondidas else 0)

    st.caption(
        "⚠️ O Databricks **não publica** a nota de corte deste exame. Os 70% usados aqui são "
        "apenas uma meta de segurança adotada por este material, não um critério oficial."
    )

    if pct >= 80:
        st.success("🏆 Ótimo resultado. Repita em outros dias para confirmar a consistência.")
    elif pct >= META_APROVACAO:
        st.info("👍 Resultado razoável. Revise as seções com menor desempenho antes de marcar a prova.")
    elif pct >= 50:
        st.warning("⚠️ Abaixo do esperado. Volte aos módulos das seções mais fracas.")
    else:
        st.error("📚 Resultado crítico. Refaça os módulos antes de simular de novo.")

    # ---- Desempenho por seção oficial ----
    st.divider()
    st.subheader("📈 Desempenho por seção oficial")

    stats_secao = {}
    for escolhas, corretas, q in respostas:
        s = q["secao"]
        stats_secao.setdefault(s, {"acertos": 0, "total": 0})
        stats_secao[s]["total"] += 1
        if escolhas == corretas:
            stats_secao[s]["acertos"] += 1

    for secao in sorted(stats_secao):
        st_ = stats_secao[secao]
        pct_s = st_["acertos"] / st_["total"] * 100
        emoji = "✅" if pct_s >= META_APROVACAO else ("⚠️" if pct_s >= 50 else "❌")
        st.markdown(
            f"{emoji} **Seção {secao} — {SECOES[secao]}** "
            f"({PESOS_SECAO[secao]}% da prova) — {st_['acertos']}/{st_['total']} ({pct_s:.0f}%)"
        )
        st.progress(st_["acertos"] / st_["total"])

    # ---- Desempenho por módulo ----
    st.divider()
    st.subheader("📚 Desempenho por módulo")

    stats_modulo = {}
    for escolhas, corretas, q in respostas:
        m = q["modulo_nome"]
        stats_modulo.setdefault(m, {"acertos": 0, "total": 0})
        stats_modulo[m]["total"] += 1
        if escolhas == corretas:
            stats_modulo[m]["acertos"] += 1

    for modulo, s in sorted(stats_modulo.items(), key=lambda x: x[1]["acertos"] / x[1]["total"]):
        pct_m = s["acertos"] / s["total"] * 100
        emoji = "✅" if pct_m >= META_APROVACAO else ("⚠️" if pct_m >= 50 else "❌")
        st.markdown(f"{emoji} **{modulo}** — {s['acertos']}/{s['total']} ({pct_m:.0f}%)")

    # ---- Revisão dos erros ----
    st.divider()
    erros = [(e, c, q) for e, c, q in respostas if e != c]
    if erros:
        st.subheader(f"❌ Questões erradas ({len(erros)})")
        st.caption("Revise a explicação de cada uma antes de simular novamente.")
        for i, (escolhas, corretas, q) in enumerate(erros, 1):
            with st.expander(f"Erro {i}: {q['pergunta'][:80]}..."):
                st.markdown(f"**{q['pergunta']}**")
                st.markdown("")
                render_alternativas(q, escolhas, corretas)
                st.info(f"💡 **Explicação:** {q['explicacao']}")
                st.caption(
                    f"Seção {q['secao']} — {q['secao_nome']} · {q['modulo_nome']} · "
                    f"{q['dificuldade']}"
                )
    else:
        st.success("🏆 Você acertou todas as questões.")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Novo simulado", type="primary", use_container_width=True):
            reiniciar()
            st.rerun()
    with col_b:
        if st.button("🏠 Voltar ao início", use_container_width=True):
            reiniciar()
            st.rerun()
