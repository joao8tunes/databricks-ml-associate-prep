"""
Valida o banco de questões antes de abrir um pull request.

Uso:
    python simulados/validar_questoes.py

Sai com código 1 se encontrar qualquer problema.
"""

import collections
import json
import random
import sys
from pathlib import Path

QUESTOES_PATH = Path(__file__).parent / "questoes.json"

MODULOS_VALIDOS = {
    "plataforma", "pyspark", "eda", "mlflow", "feature_engineering",
    "treinamento", "hyperopt", "automl", "feature_store", "deployment",
    "mlops", "revisao",
}
DIFICULDADES_VALIDAS = {"facil", "media", "dificil"}
SECOES_VALIDAS = {1, 2, 3, 4}
CAMPOS_OBRIGATORIOS = {
    "id", "modulo", "modulo_nome", "secao", "secao_nome",
    "dificuldade", "pergunta", "alternativas", "resposta_correta", "explicacao",
}

# Cotas usadas pelo Modo Prova do quiz.py
COTAS_MODO_PROVA = {1: 18, 2: 9, 3: 15, 4: 6}


def corretas_de(questao):
    valor = questao["resposta_correta"]
    return valor if isinstance(valor, list) else [valor]


def validar(questoes):
    problemas = []

    contagem_ids = collections.Counter(q.get("id") for q in questoes)
    for qid, n in contagem_ids.items():
        if n > 1:
            problemas.append(f"id {qid} aparece {n} vezes")

    for q in questoes:
        qid = q.get("id", "?")

        faltando = CAMPOS_OBRIGATORIOS - set(q)
        if faltando:
            problemas.append(f"#{qid}: campos faltando: {sorted(faltando)}")
            continue

        if q["modulo"] not in MODULOS_VALIDOS:
            problemas.append(f"#{qid}: módulo inválido: {q['modulo']}")
        if q["dificuldade"] not in DIFICULDADES_VALIDAS:
            problemas.append(f"#{qid}: dificuldade inválida: {q['dificuldade']}")
        if q["secao"] not in SECOES_VALIDAS:
            problemas.append(f"#{qid}: seção inválida: {q['secao']}")

        alternativas = q["alternativas"]
        if len(alternativas) < 2:
            problemas.append(f"#{qid}: precisa de ao menos 2 alternativas")
        if len(set(alternativas)) != len(alternativas):
            problemas.append(f"#{qid}: alternativas duplicadas")

        corretas = corretas_de(q)
        if not corretas:
            problemas.append(f"#{qid}: sem resposta correta")
        if len(set(corretas)) != len(corretas):
            problemas.append(f"#{qid}: índices de resposta repetidos")
        if len(corretas) >= len(alternativas):
            problemas.append(f"#{qid}: todas as alternativas marcadas como corretas")
        for c in corretas:
            if not isinstance(c, int) or not (0 <= c < len(alternativas)):
                problemas.append(f"#{qid}: índice de resposta fora do intervalo: {c}")

        if not q["pergunta"].strip():
            problemas.append(f"#{qid}: enunciado vazio")
        if not q["explicacao"].strip():
            problemas.append(f"#{qid}: explicação vazia")

    # O embaralhamento do quiz precisa preservar a resposta correta
    random.seed(0)
    for q in questoes:
        corretas = corretas_de(q)
        if not all(isinstance(c, int) and 0 <= c < len(q["alternativas"]) for c in corretas):
            continue
        ordem = list(range(len(q["alternativas"])))
        random.shuffle(ordem)
        novas_alternativas = [q["alternativas"][i] for i in ordem]
        novos_indices = [ordem.index(c) for c in corretas]
        texto_antes = {q["alternativas"][c] for c in corretas}
        texto_depois = {novas_alternativas[c] for c in novos_indices}
        if texto_antes != texto_depois:
            problemas.append(f"#{q['id']}: embaralhamento nao preserva a resposta correta")

    # O Modo Prova precisa de questões suficientes em cada seção
    por_secao = collections.Counter(q.get("secao") for q in questoes)
    for secao, cota in COTAS_MODO_PROVA.items():
        if por_secao[secao] < cota:
            problemas.append(
                f"seção {secao}: {por_secao[secao]} questões, "
                f"abaixo da cota {cota} exigida pelo Modo Prova"
            )

    return problemas, por_secao


def main():
    questoes = json.loads(QUESTOES_PATH.read_text(encoding="utf-8"))
    problemas, por_secao = validar(questoes)

    total = len(questoes)
    pesos = {1: 38, 2: 19, 3: 31, 4: 12}

    print(f"Questões: {total}\n")
    print("Distribuição por seção (alvo = peso oficial):")
    for secao in sorted(SECOES_VALIDAS):
        n = por_secao[secao]
        print(f"  Seção {secao}: {n:3d}  ({n / total:5.1%}  | alvo {pesos[secao]}%)")

    dificuldades = collections.Counter(q.get("dificuldade") for q in questoes)
    print("\nPor dificuldade:", dict(sorted(dificuldades.items())))
    multiplas = sum(1 for q in questoes if isinstance(q.get("resposta_correta"), list))
    print(f"Múltipla seleção: {multiplas}")

    if problemas:
        print(f"\n{len(problemas)} PROBLEMA(S):")
        for p in problemas:
            print("  -", p)
        sys.exit(1)

    print("\nBanco válido.")


if __name__ == "__main__":
    main()
