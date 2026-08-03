"""
Simulado Interativo — Databricks Certified Machine Learning Associate
Execução: streamlit run simulados/quiz.py
"""

import json
import random
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
# Carregar questões
# ---------------------------------------------------------------------------
QUESTOES_PATH = Path(__file__).parent / "questoes.json"

@st.cache_data
def carregar_questoes():
    with open(QUESTOES_PATH, encoding="utf-8") as f:
        return json.load(f)

QUESTOES_TODAS = carregar_questoes()
MODULOS_DISPONIVEIS = sorted({q["modulo_nome"] for q in QUESTOES_TODAS})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def iniciar_quiz(n_questoes: int, modulos_selecionados: list, dificuldades: list):
    """Filtra, embaralha questões e alternativas, inicializa o estado."""
    pool = [
        q for q in QUESTOES_TODAS
        if q["modulo_nome"] in modulos_selecionados
        and q["dificuldade"] in dificuldades
    ]

    if not pool:
        st.error("Nenhuma questão encontrada com esses filtros. Ajuste os filtros e tente novamente.")
        return

    n = min(n_questoes, len(pool))
    questoes = random.sample(pool, n)

    # Embaralha alternativas de cada questão mantendo rastreio da correta
    questoes_embaralhadas = []
    for q in questoes:
        ordem = list(range(len(q["alternativas"])))
        random.shuffle(ordem)
        nova_correta = ordem.index(q["resposta_correta"])
        questoes_embaralhadas.append({
            **q,
            "alternativas": [q["alternativas"][i] for i in ordem],
            "resposta_correta": nova_correta,
        })

    st.session_state.questoes  = questoes_embaralhadas
    st.session_state.idx       = 0
    st.session_state.respostas = []   # lista de (escolha_idx, correta_idx, q)
    st.session_state.escolha   = None
    st.session_state.fase      = "pergunta"


def confirmar_resposta():
    escolha = st.session_state.escolha
    if escolha is None:
        st.warning("Selecione uma alternativa antes de confirmar.")
        return
    q = st.session_state.questoes[st.session_state.idx]
    st.session_state.respostas.append((escolha, q["resposta_correta"], q))
    st.session_state.fase = "feedback"


def proxima_pergunta():
    st.session_state.idx += 1
    st.session_state.escolha = None
    total = len(st.session_state.questoes)
    if st.session_state.idx >= total:
        st.session_state.fase = "resultado"
    else:
        st.session_state.fase = "pergunta"


def reiniciar():
    for k in ["fase", "questoes", "idx", "respostas", "escolha"]:
        st.session_state.pop(k, None)


# ---------------------------------------------------------------------------
# Inicialização do session state
# ---------------------------------------------------------------------------
if "fase" not in st.session_state:
    st.session_state.fase = "home"

# ---------------------------------------------------------------------------
# TELA HOME
# ---------------------------------------------------------------------------
if st.session_state.fase == "home":
    st.title("🔷 Simulado — Databricks ML Associate")
    st.markdown(
        "Prepare-se para a certificação com questões no mesmo estilo da prova real. "
        "Configure o simulado abaixo e clique em **Iniciar**."
    )
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        n_questoes = st.slider(
            "Número de questões",
            min_value=5,
            max_value=len(QUESTOES_TODAS),
            value=20,
            step=5,
            help=f"Banco total: {len(QUESTOES_TODAS)} questões",
        )

    with col2:
        dificuldades = st.multiselect(
            "Dificuldade",
            options=["facil", "media", "dificil"],
            default=["facil", "media", "dificil"],
            format_func=lambda x: {"facil": "Fácil", "media": "Média", "dificil": "Difícil"}[x],
        )

    modulos_sel = st.multiselect(
        "Módulos",
        options=MODULOS_DISPONIVEIS,
        default=MODULOS_DISPONIVEIS,
        help="Deixe todos selecionados para um simulado completo.",
    )

    questoes_disponiveis = len([
        q for q in QUESTOES_TODAS
        if q["modulo_nome"] in modulos_sel and q["dificuldade"] in dificuldades
    ])
    st.caption(f"📚 {questoes_disponiveis} questões disponíveis com esses filtros.")

    st.divider()

    if st.button("▶ Iniciar Simulado", type="primary", use_container_width=True):
        if not modulos_sel:
            st.error("Selecione ao menos um módulo.")
        elif not dificuldades:
            st.error("Selecione ao menos uma dificuldade.")
        else:
            iniciar_quiz(n_questoes, modulos_sel, dificuldades)
            st.rerun()

# ---------------------------------------------------------------------------
# TELA DE PERGUNTA
# ---------------------------------------------------------------------------
elif st.session_state.fase == "pergunta":
    total   = len(st.session_state.questoes)
    idx     = st.session_state.idx
    q       = st.session_state.questoes[idx]
    acertos = sum(1 for e, c, _ in st.session_state.respostas if e == c)

    # Cabeçalho
    col_prog, col_score = st.columns([3, 1])
    with col_prog:
        st.progress((idx) / total, text=f"Questão {idx + 1} de {total}")
    with col_score:
        st.metric("Acertos", f"{acertos}/{idx}")

    st.divider()

    # Badge módulo + dificuldade
    dif_label = {"facil": "🟢 Fácil", "media": "🟡 Média", "dificil": "🔴 Difícil"}
    st.caption(f"{q['modulo_nome']}  ·  {dif_label.get(q['dificuldade'], q['dificuldade'])}")

    st.markdown(f"### {q['pergunta']}")
    st.markdown("")

    # Alternativas como radio
    opcao = st.radio(
        "Selecione sua resposta:",
        options=list(range(len(q["alternativas"]))),
        format_func=lambda i: f"{chr(65 + i)}) {q['alternativas'][i]}",
        index=None,
        key=f"radio_{idx}",
        label_visibility="collapsed",
    )
    st.session_state.escolha = opcao

    st.markdown("")
    if st.button("✔ Confirmar Resposta", type="primary", use_container_width=True):
        confirmar_resposta()
        st.rerun()

    with st.sidebar:
        st.markdown("### Progresso")
        for i, (e, c, qr) in enumerate(st.session_state.respostas):
            emoji = "✅" if e == c else "❌"
            st.markdown(f"{emoji} Q{i+1}: {qr['modulo_nome'][:20]}")
        if st.button("🏠 Voltar ao início", use_container_width=True):
            reiniciar()
            st.rerun()

# ---------------------------------------------------------------------------
# TELA DE FEEDBACK
# ---------------------------------------------------------------------------
elif st.session_state.fase == "feedback":
    total   = len(st.session_state.questoes)
    idx     = st.session_state.idx
    q       = st.session_state.questoes[idx]
    escolha, correta, _ = st.session_state.respostas[-1]
    acertou = escolha == correta

    acertos = sum(1 for e, c, _ in st.session_state.respostas if e == c)
    st.progress((idx + 1) / total, text=f"Questão {idx + 1} de {total}")
    st.divider()

    dif_label = {"facil": "🟢 Fácil", "media": "🟡 Média", "dificil": "🔴 Difícil"}
    st.caption(f"{q['modulo_nome']}  ·  {dif_label.get(q['dificuldade'], q['dificuldade'])}")
    st.markdown(f"### {q['pergunta']}")
    st.markdown("")

    # Mostrar alternativas com cores
    for i, alt in enumerate(q["alternativas"]):
        letra = chr(65 + i)
        if i == correta and i == escolha:
            st.success(f"✅ **{letra}) {alt}** ← Sua resposta (Correta!)")
        elif i == correta:
            st.success(f"✅ **{letra}) {alt}** ← Resposta correta")
        elif i == escolha:
            st.error(f"❌ {letra}) {alt} ← Sua resposta")
        else:
            st.markdown(f"&nbsp;&nbsp;&nbsp;{letra}) {alt}")

    st.markdown("")
    if acertou:
        st.info(f"💡 **Explicação:** {q['explicacao']}")
    else:
        st.warning(f"💡 **Explicação:** {q['explicacao']}")

    st.divider()
    proxima_label = "➡ Próxima Questão" if idx + 1 < total else "📊 Ver Resultado"
    if st.button(proxima_label, type="primary", use_container_width=True):
        proxima_pergunta()
        st.rerun()

# ---------------------------------------------------------------------------
# TELA DE RESULTADO
# ---------------------------------------------------------------------------
elif st.session_state.fase == "resultado":
    respostas = st.session_state.respostas
    total     = len(respostas)
    acertos   = sum(1 for e, c, _ in respostas if e == c)
    pct       = acertos / total * 100

    st.title("📊 Resultado Final")

    # Score geral
    col1, col2, col3 = st.columns(3)
    col1.metric("Acertos", f"{acertos}/{total}")
    col2.metric("Score", f"{pct:.1f}%")
    aprovado = pct >= 70
    col3.metric("Status", "✅ Aprovado" if aprovado else "❌ Reprovação", delta="≥ 70%" if aprovado else "< 70%")

    # Barra de progresso geral
    st.progress(acertos / total)

    if pct >= 80:
        st.success("🏆 Excelente! Você está muito bem preparado para a prova.")
    elif pct >= 70:
        st.info("👍 Bom resultado! Revise os módulos com mais erros antes da prova.")
    elif pct >= 50:
        st.warning("⚠️ Resultado abaixo do esperado. Dedique mais tempo aos módulos abaixo.")
    else:
        st.error("📚 Resultado crítico. Volte ao guia e revise os fundamentos antes de simular novamente.")

    st.divider()

    # Performance por módulo
    st.subheader("📈 Performance por Módulo")

    modulo_stats: dict = {}
    for escolha, correta, q in respostas:
        m = q["modulo_nome"]
        if m not in modulo_stats:
            modulo_stats[m] = {"acertos": 0, "total": 0}
        modulo_stats[m]["total"] += 1
        if escolha == correta:
            modulo_stats[m]["acertos"] += 1

    # Ordenar pelos piores resultados primeiro
    modulos_ordenados = sorted(
        modulo_stats.items(),
        key=lambda x: x[1]["acertos"] / x[1]["total"]
    )

    for modulo, stats in modulos_ordenados:
        ac = stats["acertos"]
        tot = stats["total"]
        pct_m = ac / tot * 100
        emoji = "✅" if pct_m >= 70 else ("⚠️" if pct_m >= 50 else "❌")
        st.markdown(f"{emoji} **{modulo}** — {ac}/{tot} ({pct_m:.0f}%)")
        st.progress(ac / tot)

    st.divider()

    # Revisão das questões erradas
    erros = [(e, c, q) for e, c, q in respostas if e != c]
    if erros:
        st.subheader(f"❌ Questões Erradas ({len(erros)})")
        st.caption("Revise cada questão e entenda a explicação antes de simular novamente.")

        for i, (escolha, correta, q) in enumerate(erros, 1):
            with st.expander(f"Erro {i}: {q['pergunta'][:70]}..."):
                st.markdown(f"**Pergunta:** {q['pergunta']}")
                st.markdown("")

                for j, alt in enumerate(q["alternativas"]):
                    letra = chr(65 + j)
                    if j == correta:
                        st.success(f"✅ **{letra}) {alt}** ← Correta")
                    elif j == escolha:
                        st.error(f"❌ {letra}) {alt} ← Sua resposta")
                    else:
                        st.markdown(f"&nbsp;&nbsp;&nbsp;{letra}) {alt}")

                st.info(f"💡 **Explicação:** {q['explicacao']}")
                st.caption(f"Módulo: {q['modulo_nome']} · Dificuldade: {q['dificuldade']}")
    else:
        st.success("🏆 Parabéns — você acertou todas as questões!")

    st.divider()

    # Ações finais
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Novo Simulado", type="primary", use_container_width=True):
            reiniciar()
            st.rerun()
    with col_b:
        if st.button("🏠 Voltar ao Início", use_container_width=True):
            reiniciar()
            st.rerun()
