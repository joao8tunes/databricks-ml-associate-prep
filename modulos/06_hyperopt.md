# Módulo 6 — Hyperopt — Tuning Distribuído

> **Notebook sugerido:** `06_hyperopt`
>
> Hyperopt é a forma recomendada no Databricks para hyperparameter tuning.
> Em vez de busca exaustiva (GridSearch), usa otimização Bayesiana — encontra bons parâmetros com muito menos tentativas.

---

## 6.1 Por que Hyperopt em vez de GridSearch?

```
GridSearchCV (sklearn):
├── Testa TODAS as combinações da grade
├── 3 params × 4 valores cada = 64 combinações × k-folds = muito lento
└── Single-machine (não distribui no Spark)

Hyperopt:
├── Busca inteligente (Bayesiana) — aprende quais regiões são promissoras
├── Encontra bons resultados com 20-50 trials vs 64+ do grid
└── Com SparkTrials: distribui os trials nos workers do Spark
```

---

## 6.2 Conceitos essenciais

```
hp.choice("nome", [a, b, c])           → escolha discreta de uma lista
hp.randint("nome", upper)              → inteiro aleatório [0, upper)
hp.quniform("nome", low, high, q)     → float em passos de q (use int() depois)
hp.uniform("nome", low, high)         → float uniforme contínuo
hp.loguniform("nome", log(low), log(high)) → escala logarítmica (para learning rate)

tpe.suggest  → Tree of Parzen Estimators (Bayesiano) — PADRÃO, mais eficiente
rand.suggest → busca aleatória (baseline)

Trials()        → execução sequencial (funciona em qualquer ambiente)
SparkTrials()   → execução paralela nos workers Spark (conta paga)
```

---

## 6.3 Hyperopt com sklearn + MLflow

```python
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd

# --- Preparar dados ---
df = spark.table("workspace.default.churn_clientes")
df_pd = df.select("tenure_months", "monthly_charges", "total_charges", "churn").toPandas()
X = df_pd.drop("churn", axis=1)
y = df_pd["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 1. Definir espaço de busca ---
search_space = {
    "n_estimators":     hp.choice("n_estimators", [50, 100, 200, 300]),
    "max_depth":        hp.quniform("max_depth", 2, 10, 1),        # int via quniform
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 20, 1), # int via quniform
    "max_features":     hp.choice("max_features", ["sqrt", "log2", 0.5]),
    "min_samples_split":hp.uniform("min_samples_split", 0.01, 0.3),
}

# --- 2. Função objetivo ---
# IMPORTANTE: a função MINIMIZA o valor retornado em "loss"
# → Para maximizar AUC, retornamos -AUC

def objetivo(params):
    # hp.quniform retorna float → converter para int onde necessário
    params["max_depth"]        = int(params["max_depth"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])

    with mlflow.start_run(nested=True):   # nested=True: run filho dentro do run pai
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, proba)

        mlflow.log_metric("auc_roc", auc)

        return {
            "loss":   -auc,        # minimizar → negativo da métrica que queremos maximizar
            "status": STATUS_OK,
        }

# --- 3. Executar a busca ---
mlflow.set_experiment("/churn-hyperopt")

with mlflow.start_run(run_name="hyperopt_rf_search"):   # run pai
    trials = Trials()  # sequencial (Free Edition)

    best_params = fmin(
        fn=objetivo,
        space=search_space,
        algo=tpe.suggest,
        max_evals=25,
        trials=trials,
        rstate=np.random.default_rng(42)
    )

# Converter indices do hp.choice para os valores reais
best = space_eval(search_space, best_params)
print("Melhores parâmetros:")
for k, v in best.items():
    print(f"  {k}: {v}")

melhor_auc = -min(t["result"]["loss"] for t in trials.trials)
print(f"\nMelhor AUC-ROC: {melhor_auc:.4f}")
```

---

## 6.4 SparkTrials — paralelizar nos workers

```python
from hyperopt import SparkTrials

# SparkTrials distribui os trials nos workers do Spark
# parallelism = quantos trials rodam em simultâneo (≤ número de workers)
# Na Free Edition funciona mas sem ganho de velocidade (1 worker)

mlflow.set_experiment("/churn-hyperopt")

with mlflow.start_run(run_name="hyperopt_paralelo"):
    spark_trials = SparkTrials(parallelism=4)

    best_params_spark = fmin(
        fn=objetivo,             # MESMA função objetivo de antes
        space=search_space,
        algo=tpe.suggest,
        max_evals=25,
        trials=spark_trials      # ← única diferença: SparkTrials em vez de Trials
    )

# ATENÇÃO: SparkTrials NÃO suporta nested=True no MLflow
# Cada trial cria automaticamente um MLflow run separado
# → Remova o mlflow.start_run(nested=True) da função objetivo ao usar SparkTrials
```

```python
# Versão da função objetivo compatível com SparkTrials (sem nested=True):
def objetivo_spark(params):
    params["max_depth"]        = int(params["max_depth"])
    params["min_samples_leaf"] = int(params["min_samples_leaf"])

    # Sem mlflow.start_run(nested=True) — SparkTrials cria o run automaticamente
    mlflow.log_params(params)

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    mlflow.log_metric("auc_roc", auc)

    return {"loss": -auc, "status": STATUS_OK}

with mlflow.start_run(run_name="hyperopt_spark_trials"):
    best = fmin(
        fn=objetivo_spark,
        space=search_space,
        algo=tpe.suggest,
        max_evals=25,
        trials=SparkTrials(parallelism=4)
    )
```

---

## 6.5 Outros algoritmos de busca

```python
from hyperopt import rand

# rand.suggest: busca aleatória (baseline — tpe.suggest é sempre melhor)
best_random = fmin(
    fn=objetivo,
    space=search_space,
    algo=rand.suggest,   # ← aleatório em vez de Bayesiano
    max_evals=25,
    trials=Trials()
)
```

---

## 6.6 Treinar modelo final com os melhores parâmetros

```python
from hyperopt import space_eval
from sklearn.metrics import accuracy_score, f1_score

# Converter params de volta para os valores reais
best_final = space_eval(search_space, best_params)
best_final["max_depth"]        = int(best_final["max_depth"])
best_final["min_samples_leaf"] = int(best_final["min_samples_leaf"])

# Treinar modelo final com os melhores params
with mlflow.start_run(run_name="RF_final_hyperopt"):
    mlflow.log_params(best_final)
    mlflow.log_param("otimizador", "hyperopt_tpe")
    mlflow.log_param("max_evals", 25)

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
        artifact_path="model",
        registered_model_name="churn-predictor-hyperopt"
    )

    print(f"AUC-ROC final: {roc_auc_score(y_test, proba):.4f}")
```

---

## 6.7 Hyperopt com GradientBoosting — exemplo adicional

```python
from sklearn.ensemble import GradientBoostingClassifier

space_gbt = {
    "n_estimators":  hp.choice("n_estimators", [50, 100, 200]),
    "max_depth":     hp.quniform("max_depth", 2, 6, 1),
    "learning_rate": hp.loguniform("learning_rate", np.log(0.001), np.log(0.3)),
    "subsample":     hp.uniform("subsample", 0.5, 1.0),
}

def objetivo_gbt(params):
    params["max_depth"] = int(params["max_depth"])

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("auc_roc", auc)
        return {"loss": -auc, "status": STATUS_OK}

with mlflow.start_run(run_name="hyperopt_gbt"):
    best_gbt = fmin(
        fn=objetivo_gbt,
        space=space_gbt,
        algo=tpe.suggest,
        max_evals=20,
        trials=Trials()
    )
```

---

## Pontos-chave para a prova

| Conceito | Detalhe |
|---|---|
| `fmin()` | **Minimiza** a função objetivo |
| Maximizar AUC | Retornar `{"loss": -auc, ...}` |
| `hp.choice(lista)` | Escolha discreta (retorna índice — usar `space_eval` depois) |
| `hp.quniform(l,h,q)` | Float em passos — converter com `int()` |
| `hp.loguniform` | Melhor para learning rate e regularização |
| `tpe.suggest` | Otimização Bayesiana (padrão, mais eficiente) |
| `rand.suggest` | Busca aleatória (baseline) |
| `Trials()` | Execução sequencial |
| `SparkTrials(parallelism=N)` | Execução paralela nos workers |
| `nested=True` | Para Trials() — **não usar** com SparkTrials |
| `space_eval(space, best)` | Converte índices do `hp.choice` para valores reais |

---

→ Próximo: [07_automl.md](07_automl.md)
