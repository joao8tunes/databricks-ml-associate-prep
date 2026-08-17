# Simulados

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).

Sistema de simulados para a certificação Databricks Certified Machine Learning Associate.

**As questões são autorais.** Foram escritas no estilo do exame a partir dos objetivos públicos do [exam guide oficial](https://www.databricks.com/sites/default/files/2025-02/databricks-certified-machine-learning-associate-exam-guide-1-mar-2025.pdf). **Não são questões reais da prova** — reproduzi-las violaria o acordo de confidencialidade da certificação.

---

## Como rodar

```bash
pip install -r simulados/requirements.txt
```

```bash
streamlit run simulados/quiz.py
```

---

## Os dois modos

| | **Modo Prova** | **Modo Estudo** |
|---|---|---|
| Questões | 48, sorteadas na distribuição oficial | 5 a 60, você escolhe |
| Tempo | 90 minutos, cronometrado | Livre |
| Feedback | Só no final | Imediato, a cada questão |
| Filtros | Nenhum — cobre tudo | Seção, módulo, dificuldade |
| Para quê | Medir prontidão real | Fixar conteúdo de um tópico |

A distribuição do Modo Prova segue os pesos oficiais das quatro seções:

| Seção | Nome | Peso | Questões |
|---|---|---|---|
| 1 | Databricks Machine Learning | 38% | 18 |
| 2 | Data Processing | 19% | 9 |
| 3 | Model Development | 31% | 15 |
| 4 | Model Deployment | 12% | 6 |

> A meta de 70% usada nos relatórios é **uma referência deste material**, não a nota de corte oficial — o Databricks não publica a nota mínima deste exame.

---

## Formato do banco de questões

`questoes.json` é uma lista de objetos:

```json
{
  "id": 42,
  "modulo": "mlflow",
  "modulo_nome": "04 - MLflow",
  "secao": 1,
  "secao_nome": "Databricks Machine Learning",
  "dificuldade": "media",
  "pergunta": "Enunciado da questão?",
  "alternativas": ["Alternativa A", "Alternativa B", "Alternativa C", "Alternativa D"],
  "resposta_correta": 1,
  "explicacao": "Por que a resposta correta está certa e por que as outras estão erradas."
}
```

| Campo | Valores |
|---|---|
| `id` | Inteiro único |
| `modulo` | `plataforma`, `pyspark`, `eda`, `mlflow`, `feature_engineering`, `treinamento`, `hyperopt`, `automl`, `feature_store`, `deployment`, `mlops`, `revisao` |
| `secao` | `1`, `2`, `3` ou `4` — a seção oficial do exam guide |
| `dificuldade` | `facil`, `media` ou `dificil` |
| `resposta_correta` | **Inteiro** (índice, questão de resposta única) ou **lista de inteiros** (múltipla seleção) |

**Questão de múltipla seleção** — basta usar uma lista:

```json
"resposta_correta": [0, 2]
```

O app detecta automaticamente e renderiza checkboxes, informando ao usuário quantas alternativas marcar. Não há crédito parcial: o conjunto marcado precisa ser exatamente igual ao esperado — mesma regra da prova real.

---

## Contribuindo com questões

Ao adicionar uma questão:

1. **Amarre a um objetivo oficial.** Se ela não corresponde a nenhuma linha do [mapa do exam guide](../GUIA_DA_PROVA.md), provavelmente não cai na prova.
2. **Escreva os distratores plausíveis.** Alternativas obviamente absurdas não treinam ninguém.
3. **Nunca cite a posição ou a letra de uma alternativa.** As alternativas são embaralhadas a cada execução, então "as alternativas B e C estão corretas", "todas as anteriores" e "nenhuma das acima" perdem o sentido na tela. Quando a resposta envolve mais de uma opção, use múltipla seleção.
4. **Mantenha o comprimento parecido entre as alternativas.** Se só a correta traz a justificativa completa, dá para acertar escolhendo a mais longa sem ler o enunciado. A justificativa vai na `explicacao`.
5. **Explique também por que as outras estão erradas.** É na explicação que o estudo acontece.
6. **Use `id` maior que o último existente** e mantenha `secao` coerente com `modulo`.
7. Rode a validação antes de abrir o PR:

```bash
python simulados/validar_questoes.py
```

O script confere campos obrigatórios, IDs duplicados, índices de resposta fora do intervalo, alternativas repetidas, explicações vazias, referências à posição ou à letra de alternativas, se o reindexamento do embaralhamento preserva a resposta correta e se cada seção tem questões suficientes para o Modo Prova. Ele sai com código 1 em caso de problema, então serve direto em CI.
