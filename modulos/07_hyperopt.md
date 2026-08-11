# Módulo 7 — Tuning de Hiperparâmetros com Hyperopt

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Seção oficial:** 3 — Model Development (**31% da prova**)
>
> **Notebook sugerido:** `07_hyperopt`
>
> **Pré-requisito:** Módulos 2, 4 e 6.

**Objetivos oficiais cobertos neste módulo:**
- Usar a operação `fmin` do Hyperopt para tunar hiperparâmetros
- Executar busca aleatória, em grade ou bayesiana como método de tuning
- Paralelizar modelos single-node para tuning de hiperparâmetros

---

> ⚠️ **Antes de rodar o código:** o Hyperopt **não vem mais pré-instalado** nos runtimes recentes do Databricks (foi removido a partir do Databricks Runtime ML 17.0) nem no compute serverless. Instale na primeira célula do notebook:
>
> ```python
> %pip install hyperopt
> dbutils.library.restartPython()
> ```
>
> **Isso não muda nada para a prova.** O exam guide vigente cobra explicitamente "usar a operação `fmin` do Hyperopt". Para trabalho novo, o Databricks hoje recomenda **Optuna** (single-node) e **Ray Tune** (distribuído) — ver seção 7.7.

---

## 7.1 As três estratégias de busca

Objetivo oficial. A prova pergunta qual estratégia usar e por quê.

```
GRID SEARCH (busca em grade)
├── Testa TODAS as combinações de uma lista pré-definida
├── Exaustivo e determinístico
├── Custo explode: 4 params × 5 valores = 625 combinações
└── Desperdiça tempo em regiões ruins do espaço

RANDOM SEARCH (busca aleatória)
├── Sorteia N combinações aleatórias do espaço
├── Você controla o orçamento: "faça 50 tentativas"
├── Em geral SUPERA grid search com o mesmo orçamento
│     (porque poucos hiperparâmetros realmente importam,
│      e o random explora mais valores distintos dos que importam)
└── Não aprende com as tentativas anteriores

BAYESIAN SEARCH (otimização bayesiana — TPE)
├── Constrói um modelo probabilístico de "quais regiões são promissoras"
├── Cada tentativa APRENDE com as anteriores
├── Encontra bons resultados com muito menos tentativas
└── É sequencial por natureza (paralelizar reduz a qualidade da informação)
```

| | Grid | Random | Bayesiana (TPE) |
|---|---|---|---|
| Cobre todas as combinações | ✅ | ❌ | ❌ |
| Orçamento controlável | ❌ | ✅ | ✅ |
| Aprende com tentativas anteriores | ❌ | ❌ | ✅ |
| Eficiência por tentativa | Baixa | Média | **Alta** |
| Paralelização | Perfeita | Perfeita | Parcial |
| Espaço contínuo | Precisa discretizar | ✅ Natural | ✅ Natural |
| No Spark ML | `ParamGridBuilder` + `CrossValidator` | — | — |
| No Hyperopt | `hp.choice` + `atpe`/exaustivo | `rand.suggest` | **`tpe.suggest`** |

> **Pegadinha:** "quero o melhor resultado com o menor número de tentativas" → **bayesiana (TPE)**. "Quero garantir que todas as combinações foram testadas" → **grid**. "Tenho muitos workers ociosos e quero paralelismo máximo sem complicação" → **random**.

---

## 7.2 Conceitos essenciais do Hyperopt

```
ESPAÇO DE BUSCA
hp.choice("nome", [a, b, c])       → escolha discreta de uma lista
hp.randint("nome", upper)          → inteiro aleatório em [0, upper)
hp.uniform("nome", low, high)      → float uniforme contínuo
hp.quniform("nome", low, high, q)  → float em passos de q (converter com int())
hp.loguniform("nome", log(low), log(high))  → escala logarítmica
hp.qloguniform(...)                → log + quantizado

ALGORITMO DE BUSCA
tpe.suggest   → Tree of Parzen Estimators (bayesiano) — PADRÃO
rand.suggest  → busca aleatória (baseline)
anneal.suggest→ simulated annealing

OBJETO DE TRIALS
Trials()       → execução sequencial, em uma máquina
SparkTrials()  → distribui os trials nos workers Spark
```

> **`hp.loguniform` recebe os limites já em log.** Escrever `hp.loguniform("lr", 0.001, 0.1)` é um erro clássico — o correto é `hp.loguniform("lr", np.log(0.001), np.log(0.1))`.

---

## 7.3 Hyperopt na prática

```python
%pip install hyperopt
dbutils.library.restartPython()
```

```python
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# --- Dados (tabela criada no Módulo 2) ---
df = spark.table("workspace.default.churn_clientes")
df_pd = df.select("tenure_months", "monthly_charges", "total_charges", "churn").toPandas()
X = df_pd.drop("churn", axis=1)
y = df_pd["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 1. Espaço de busca ---
search_space = {
    "n_estimators":      hp.choice("n_estimators", [50, 100, 200, 300]),
    "max_depth":         hp.quniform("max_depth", 2, 10, 1),         # float → int()
    "min_samples_leaf":  hp.quniform("min_samples_leaf", 1, 20, 1),  # float → int()
    "max_features":      hp.choice("max_features", ["sqrt", "log2", 0.5]),
    "min_samples_split": hp.uniform("min_samples_split", 0.01, 0.3),
}

# --- 2. Função objetivo ---
# fmin MINIMIZA o valor em "loss".
# Para MAXIMIZAR o AUC, retornamos -AUC.
def objetivo(params):
    params = dict(params)
    params["max_depth"]       = int(params["max_depth"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])

    with mlflow.start_run(nested=True):     # run filho dentro do run pai
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("auc_roc", auc)

        return {"loss": -auc, "status": STATUS_OK}

# --- 3. Executar a busca ---
usuario = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{usuario}/churn-hyperopt")

with mlflow.start_run(run_name="hyperopt_rf_search"):   # run PAI
    trials = Trials()

    best_params = fmin(
        fn=objetivo,
        space=search_space,
        algo=tpe.suggest,        # bayesiana
        max_evals=25,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

# hp.choice retorna ÍNDICES em best_params — space_eval converte para os valores reais
best = space_eval(search_space, best_params)
print("Melhores parâmetros:")
for k, v in best.items():
    print(f"  {k}: {v}")

melhor_auc = -min(t["result"]["loss"] for t in trials.trials)
print(f"\nMelhor AUC-ROC: {melhor_auc:.4f}")
```

### As três coisas que a prova cobra sobre `fmin`

```
1. fmin MINIMIZA.
   Para maximizar AUC/accuracy/F1 → retorne {"loss": -metrica}
   Para minimizar RMSE/log loss   → retorne {"loss": rmse} direto

2. A função objetivo retorna um DICIONÁRIO
   {"loss": valor, "status": STATUS_OK}
   (pode retornar só o float, mas o dicionário é o padrão)

3. hp.choice devolve o ÍNDICE no resultado final
   → sempre passe best_params por space_eval() antes de usar
```

---

## 7.4 Comparar as três estratégias no mesmo espaço

```python
from hyperopt import rand

# --- Bayesiana (TPE) ---
with mlflow.start_run(run_name="busca_tpe"):
    best_tpe = fmin(fn=objetivo, space=search_space, algo=tpe.suggest,
                    max_evals=25, trials=Trials(), rstate=np.random.default_rng(42))

# --- Aleatória ---
with mlflow.start_run(run_name="busca_random"):
    best_rand = fmin(fn=objetivo, space=search_space, algo=rand.suggest,
                     max_evals=25, trials=Trials(), rstate=np.random.default_rng(42))

print("TPE   :", space_eval(search_space, best_tpe))
print("Random:", space_eval(search_space, best_rand))
```

```python
# --- Grid search: no Spark ML é ParamGridBuilder + CrossValidator (Módulo 6) ---
# from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# Em sklearn, é GridSearchCV.
# Hyperopt não é a ferramenta natural para grid exaustivo.
```

---

## 7.5 Treinar o modelo final

```python
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_registry_uri("databricks-uc")

best_final = space_eval(search_space, best_params)
best_final["max_depth"]        = int(best_final["max_depth"])
best_final["min_samples_leaf"] = int(best_final["min_samples_leaf"])

with mlflow.start_run(run_name="RF_final_hyperopt"):
    mlflow.log_params(best_final)
    mlflow.log_params({"otimizador": "hyperopt_tpe", "max_evals": 25})

    model_final = RandomForestClassifier(**best_final, random_state=42)
    model_final.fit(X_train, y_train)

    proba = model_final.predict_proba(X_test)[:, 1]
    preds = model_final.predict(X_test)

    mlflow.log_metrics({
        "auc_roc":  roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, preds),
        "f1":       f1_score(y_test, preds),
    })

    mlflow.sklearn.log_model(
        model_final,
        name="model",
        input_example=X_train.head(3),
        registered_model_name="workspace.default.churn_predictor_hyperopt",  # sem hífen
    )

    print(f"AUC-ROC final: {roc_auc_score(y_test, proba):.4f}")
```

---

## 7.6 Paralelizar o tuning: SparkTrials

Objetivo oficial: "paralelizar modelos single-node para tuning de hiperparâmetros". Esse é exatamente o propósito do `SparkTrials`.

```
O cenário:
├── O modelo é SINGLE-NODE (sklearn, XGBoost single-machine)
├── Ele cabe na memória de uma máquina
├── Mas você quer testar 100 combinações
└── → SparkTrials distribui cada TRIAL para um worker diferente

O anti-cenário:
├── O modelo JÁ É distribuído (Spark MLlib, Horovod)
├── Ele já usa o cluster inteiro para um único treino
└── → use Trials() normal. SparkTrials aqui competiria por recursos
```

```python
from hyperopt import SparkTrials

# SparkTrials NÃO suporta mlflow.start_run(nested=True) na função objetivo —
# ele cria um run MLflow por trial automaticamente.
def objetivo_spark(params):
    params = dict(params)
    params["max_depth"]        = int(params["max_depth"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return {"loss": -auc, "status": STATUS_OK}

with mlflow.start_run(run_name="hyperopt_spark_trials"):
    best_spark = fmin(
        fn=objetivo_spark,
        space=search_space,
        algo=tpe.suggest,
        max_evals=25,
        trials=SparkTrials(parallelism=4),   # 4 trials simultâneos
    )
```

> ⚠️ **`SparkTrials` provavelmente não vai rodar na Free Edition.** Ele depende das APIs de RDD do Spark, que não são suportadas no compute serverless (só Spark Connect). Se der erro, troque por `Trials()` — o conceito da prova continua o mesmo. Este é um dos poucos trechos deste guia que você pode não conseguir executar.

### O trade-off do paralelismo

```
parallelism = 1   → totalmente sequencial; o TPE aproveita TODA a informação
                    das tentativas anteriores. Melhor qualidade, mais lento.

parallelism = N   → N trials em voo ao mesmo tempo; os N em execução não
                    conhecem os resultados uns dos outros. Mais rápido,
                    a busca bayesiana fica um pouco "mais cega".

Recomendação prática: parallelism ≈ raiz quadrada de max_evals,
limitado pelo número de workers.
```

| | `Trials()` | `SparkTrials()` |
|---|---|---|
| Execução | Sequencial, uma máquina | Paralela, nos workers |
| Modelo alvo | Qualquer um | **Single-node** (sklearn, XGBoost) |
| Modelo Spark MLlib | ✅ Use este | ❌ Não use |
| `nested=True` no MLflow | ✅ Sim | ❌ **Não** — cria os runs sozinho |
| Qualidade da busca TPE | Máxima | Levemente menor com paralelismo alto |

---

## 7.7 Nota sobre o futuro: Optuna e Ray Tune

O Hyperopt de código aberto não é mais mantido ativamente, e o Databricks o removeu dos runtimes ML a partir da versão 17.0. As recomendações atuais são:

| Ferramenta | Papel |
|---|---|
| **Optuna** | Substituto para tuning single-node. API mais moderna, mesma ideia de otimização bayesiana |
| **Ray Tune** | Substituto para tuning distribuído (o papel que o `SparkTrials` cumpria) |

```python
# Equivalência conceitual (Optuna) — só para você reconhecer, NÃO cai na prova
# import optuna
# def objetivo(trial):
#     params = {
#         "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200]),
#         "max_depth":    trial.suggest_int("max_depth", 2, 10),
#     }
#     model = RandomForestClassifier(**params, random_state=42).fit(X_train, y_train)
#     return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
#
# estudo = optuna.create_study(direction="maximize")   # ← maximiza direto,
# estudo.optimize(objetivo, n_trials=25)               #   sem o truque do -AUC
```

> **Para a prova: estude Hyperopt.** O exam guide vigente nomeia `fmin` explicitamente. Esta seção existe só para você não se confundir se encontrar Optuna na documentação atual do Databricks.

---

## Exercício prático

```python
# 1. Rode a busca com tpe.suggest e max_evals=25. Anote o melhor AUC.
# 2. Rode a MESMA busca com rand.suggest e max_evals=25. Compare.
# 3. Rode tpe.suggest com max_evals=10 e depois com max_evals=50.
#    Quanto o AUC melhorou? O ganho compensou o tempo?
# 4. Abra o experimento na MLflow UI: os 25 trials aparecem como runs
#    FILHOS do run pai. Expanda o run pai para vê-los.
# 5. Mude o retorno da função objetivo de -auc para +auc e rode de novo.
#    O que acontece? Por quê?  (dica: fmin minimiza)
```

---

## Pontos-chave para a prova

| Conceito | Detalhe |
|---|---|
| **`fmin()`** | **Minimiza** a função objetivo |
| Maximizar uma métrica | Retornar `{"loss": -metrica, "status": STATUS_OK}` |
| Retorno da objetivo | Dicionário com `loss` e `status` |
| `hp.choice(lista)` | Escolha discreta — devolve **índice**, use `space_eval` |
| `hp.quniform(l, h, q)` | Float quantizado — converter com `int()` |
| `hp.uniform(l, h)` | Float contínuo |
| `hp.loguniform(log(l), log(h))` | Escala log — para learning rate e regularização. **Limites já em log** |
| `tpe.suggest` | Otimização bayesiana — padrão e mais eficiente por tentativa |
| `rand.suggest` | Busca aleatória — baseline, paraleliza perfeitamente |
| Grid search | `ParamGridBuilder` + `CrossValidator` (Spark ML) ou `GridSearchCV` (sklearn) |
| `Trials()` | Sequencial — use com modelos **distribuídos** (Spark MLlib) |
| `SparkTrials(parallelism=N)` | Paralelo — use com modelos **single-node** (sklearn) |
| `nested=True` | Com `Trials()` sim; com `SparkTrials` **não** |
| `space_eval(space, best)` | Converte índices do `hp.choice` nos valores reais |
| Paralelismo alto | Mais rápido, mas o TPE perde informação entre trials |
| Hyperopt hoje | Não vem pré-instalado — `%pip install hyperopt`. Sucessores: Optuna e Ray Tune |

---

→ Próximo: [08_automl.md](08_automl.md)
