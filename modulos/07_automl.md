# Módulo 7 — AutoML

> **Notebook sugerido:** `07_automl`
>
> AutoML treina dezenas de modelos automaticamente e gera notebooks com o código do melhor modelo.
>
> ⚠️ **Não disponível na Free Edition.** O código abaixo funciona em workspaces pagos ou trial. Estude mesmo assim — cai na prova.

---

## 7.1 O que o AutoML faz automaticamente

```
Databricks AutoML:
├── Analisa o dataset (nulos, distribuição, cardinalidade, desbalanceamento)
├── Feature engineering automático (encoding, scaling, imputação)
├── Treina múltiplos algoritmos (sklearn, XGBoost, LightGBM)
├── Hyperparameter tuning com early stopping
├── Cross-validation interna
├── Gera notebook de exploração de dados (EDA)
├── Gera notebook do melhor modelo (código completo e editável!)
└── Registra todos os runs no MLflow automaticamente
```

---

## 7.2 Tipos de problema suportados

| Tipo | Função | Métricas disponíveis |
|---|---|---|
| **Classificação** | `automl.classify()` | `f1`, `accuracy`, `log_loss`, `roc_auc` |
| **Regressão** | `automl.regress()` | `r2`, `mae`, `rmse`, `mse` |
| **Forecasting** | `automl.forecast()` | `smape`, `mse`, `rmse`, `mae`, `mdape` |

---

## 7.3 AutoML via código

```python
from databricks import automl

df = spark.table("workspace.default.churn_clientes")

# --- Classificação ---
summary = automl.classify(
    dataset=df,                              # DataFrame Spark ou pandas
    target_col="churn",                      # coluna alvo
    primary_metric="roc_auc",               # métrica para otimizar o melhor modelo
    exclude_cols=["customer_id"],            # colunas para ignorar (IDs, etc.)
    timeout_minutes=30,                      # tempo máximo de busca
    experiment_name="/churn-automl",         # onde salvar os runs no MLflow
)

# --- Regressão ---
summary_reg = automl.regress(
    dataset=df,
    target_col="monthly_charges",
    primary_metric="r2",
    timeout_minutes=30,
)

# --- Forecasting (séries temporais) ---
# (precisa de um dataset com coluna de tempo)
summary_ts = automl.forecast(
    dataset=df_vendas,
    target_col="vendas",
    time_col="data",
    frequency="D",     # D=diário, W=semanal, M=mensal, Y=anual, h=hora
    horizon=30,        # quantos períodos futuros prever
    primary_metric="smape",
    timeout_minutes=30,
)
```

---

## 7.4 AutoML via UI

```
1. Menu esquerdo → "Machine Learning" → "AutoML"
2. Clique em "New AutoML experiment"
3. Configure:
   ├── Cluster: selecionar
   ├── ML problem type: Classification / Regression / Forecasting
   ├── Input training dataset: selecionar tabela Delta
   ├── Prediction target: coluna alvo
   └── Advanced config:
       ├── Evaluation metric (primary_metric)
       ├── Training frameworks (sklearn, xgboost, lightgbm)
       └── Timeout (minutos)
4. Clique "Start AutoML"
5. Acompanhar na interface:
   ├── "View data exploration notebook" — análise automática do dataset
   └── Clicar no melhor run → "Open notebook" — código completo editável
```

---

## 7.5 Usar o output do AutoML

```python
# Acessar o melhor trial
print(type(summary))           # databricks.automl.AutoMLSummary
print(summary.best_trial)      # objeto AutoMLTrial
print(summary.best_trial.model_path)   # path MLflow do melhor modelo

# Carregar o melhor modelo
import mlflow
best_model = mlflow.sklearn.load_model(summary.best_trial.model_path)

# Fazer predições
import pandas as pd
novos = pd.DataFrame({
    "tenure_months": [3, 48],
    "monthly_charges": [85.0, 50.0],
    "total_charges": [255.0, 2400.0],
    "contract_type": ["Month-to-month", "Two year"],
    "internet_service": ["Fiber optic", "DSL"],
    "tech_support": ["No", "Yes"],
    "payment_method": ["Electronic check", "Credit card"],
})
predicoes = best_model.predict(novos)
print(f"Predições: {predicoes}")   # [1, 0] — 1=churn, 0=fica
```

```python
# Ver todos os trials ordenados por performance
import mlflow

runs = mlflow.search_runs(
    experiment_names=["/churn-automl"],
    order_by=["metrics.val_roc_auc_score DESC"]
)
display(runs[["tags.mlflow.runName", "metrics.val_roc_auc_score"]].head(10))

# Registrar o melhor modelo no Registry
from mlflow.tracking import MlflowClient
client = MlflowClient()

model_uri = summary.best_trial.model_path
result = mlflow.register_model(model_uri, "churn-automl-best")
client.transition_model_version_stage(
    name="churn-automl-best",
    version=result.version,
    stage="Production"
)
```

---

## 7.6 O notebook gerado pelo AutoML

O AutoML gera um notebook Python completo com todo o código do melhor modelo. Ele inclui:

```python
# Exemplo simplificado do tipo de código que o AutoML gera:

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

# 1. Carregar e preparar os dados
df_loaded = spark.table("churn_project.clientes").toPandas()

# 2. Preprocessamento (gerado automaticamente com as melhores transformações)
# StringIndexer para categorias, StandardScaler para numéricas, etc.
# (código completo gerado pelo AutoML)

# 3. Treinar o modelo com os melhores hiperparâmetros encontrados
with mlflow.start_run():
    model = LGBMClassifier(
        n_estimators=237,
        max_depth=7,
        learning_rate=0.056,
        num_leaves=63,
        # ... outros params encontrados pelo AutoML
    )
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")

# → Você pode editar este notebook e re-treinar com seus próprios ajustes!
```

---

## 7.7 Parâmetros importantes do AutoML

```python
summary = automl.classify(
    dataset=df,
    target_col="churn",

    # Métricas disponíveis para classificação:
    primary_metric="roc_auc",   # "f1", "accuracy", "log_loss", "roc_auc"

    # Excluir colunas
    exclude_cols=["customer_id", "data_cadastro"],

    # Excluir frameworks específicos
    exclude_frameworks=["sklearn"],  # usa só xgboost e lightgbm

    # Configurações de tempo
    timeout_minutes=30,
    max_trials=20,              # limita número de trials

    # Onde salvar no MLflow
    experiment_name="/meu-experimento-automl",

    # Dados de validação customizados
    # (por padrão AutoML faz split interno)
    # split_col="split",  # coluna que indica treino/validação
)
```

---

## Pontos-chave para a prova

- AutoML suporta **3 tipos**: Classification, Regression, Forecasting
- `primary_metric` controla qual métrica seleciona o melhor modelo
- `automl.forecast()` exige `time_col`, `frequency` e `horizon`
- `summary.best_trial.model_path` → URI do MLflow para carregar o modelo
- AutoML gera **2 notebooks**: exploração de dados + melhor modelo (editável)
- Todos os trials são **automaticamente logados no MLflow**
- AutoML **não está disponível na Free Edition**
- `exclude_cols` → remover IDs e colunas que não devem ser features
- O notebook gerado pode ser editado e re-treinado

---

→ Próximo: [08_feature_store.md](08_feature_store.md)
