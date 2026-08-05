# Módulo 3 — MLflow — Tracking & Registry

> **Notebook sugerido:** `03_mlflow`
>
> **Este é o módulo mais importante da prova — vale ~25% das questões.**
>
> MLflow é o sistema de gerenciamento de experimentos do Databricks. Aprenda bem cada função.

---

## 3.1 O problema que MLflow resolve

```
Sem MLflow:
  "Qual modelo deu 94% de AUC? Era 100 ou 200 árvores?
   Qual threshold eu usei? Qual versão do dataset?
   Onde está o arquivo do modelo treinado na semana passada?"

Com MLflow:
  → Tudo registrado e versionado automaticamente.
  → Interface visual para comparar experimentos.
  → Repositório central para versões de modelos.
```

### Os 4 componentes do MLflow

| Componente | O que faz | Peso na prova |
|---|---|---|
| **Tracking** | Registra params, métricas, artefatos de cada run | ★★★★★ |
| **Models** | Formato padrão para empacotar qualquer modelo | ★★★★☆ |
| **Registry** | Catálogo de versões — teoria: `None → Staging → Production`; prática (Unity Catalog): versão + aliases | ★★★★☆ |
| **Projects** | Empacota código para reprodutibilidade | ★★☆☆☆ |

---

## 3.2 Tracking — registrar experimentos

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Preparar dados (converter Spark para pandas para usar sklearn)
df = spark.table("workspace.default.churn_clientes")

df_pd = df.select("tenure_months", "monthly_charges", "total_charges", "churn").toPandas()
X = df_pd.drop("churn", axis=1)
y = df_pd["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Estrutura básica de um MLflow run ---
mlflow.set_experiment("/churn-prediction")  # cria ou usa experimento existente

with mlflow.start_run(run_name="RF_baseline"):

    # 1. Definir e logar parâmetros (hiperparâmetros — não mudam durante o run)
    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    mlflow.log_params(params)           # vários de uma vez
    # mlflow.log_param("n_estimators", 100)  # ou um por um

    # 2. Treinar
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 3. Avaliar
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # 4. Logar métricas (valores que medem a qualidade — podem ter step)
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, preds),
        "f1":       f1_score(y_test, preds),
        "auc_roc":  roc_auc_score(y_test, proba),
    })

    # 5. Logar modelo
    mlflow.sklearn.log_model(model, artifact_path="model")

    # 6. Logar artefatos extras (qualquer arquivo)
    import tempfile, os
    feat_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        feat_df.to_csv(f.name, index=False)
        mlflow.log_artifact(f.name, artifact_path="feature_importance")

    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    print(f"AUC-ROC: {roc_auc_score(y_test, proba):.4f}")
```

---

## 3.3 autolog — logging automático (muito cobrado!)

```python
# autolog() registra TUDO automaticamente:
# parâmetros, métricas de treino, modelo, feature importances, assinatura do modelo...

# Opção A: autolog específico por framework
mlflow.sklearn.autolog()   # para sklearn
mlflow.xgboost.autolog()   # para XGBoost
mlflow.keras.autolog()     # para Keras/TensorFlow
mlflow.pytorch.autolog()   # para PyTorch

# Opção B: autolog genérico (detecta o framework automaticamente)
mlflow.autolog()

mlflow.set_experiment("/churn-prediction")

with mlflow.start_run(run_name="LR_autolog"):
    from sklearn.linear_model import LogisticRegression
    model_lr = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    model_lr.fit(X_train, y_train)
    # MLflow logou parâmetros, métricas, modelo — automaticamente!

# Quando usar autolog vs log manual?
# autolog()    → exploração rápida, prototipagem
# log manual  → controle total sobre o que registrar (produção)
```

---

## 3.4 Comparar múltiplos modelos

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

configuracoes = [
    ("RF_100_depth5",   RandomForestClassifier(n_estimators=100, max_depth=5,  random_state=42)),
    ("RF_200_depth10",  RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ("GBT_100",         GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("LR_C01",          LogisticRegression(C=0.1, max_iter=500, random_state=42)),
    ("LR_C10",          LogisticRegression(C=10.0, max_iter=500, random_state=42)),
]

mlflow.set_experiment("/churn-prediction")

for run_name, modelo in configuracoes:
    with mlflow.start_run(run_name=run_name):
        mlflow.sklearn.autolog(log_models=True)
        modelo.fit(X_train, y_train)

        preds = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)[:, 1]

        # autolog não loga métricas de teste — logar manualmente
        mlflow.log_metrics({
            "test_accuracy": accuracy_score(y_test, preds),
            "test_f1":       f1_score(y_test, preds),
            "test_auc_roc":  roc_auc_score(y_test, proba),
        })

print("Veja os resultados em: Experiments > /churn-prediction")
```

---

## 3.5 Buscar e comparar runs programaticamente

```python
# mlflow.search_runs() retorna um DataFrame pandas com todos os runs
runs = mlflow.search_runs(
    experiment_names=["/churn-prediction"],
    order_by=["metrics.test_auc_roc DESC"]
)

# Colunas importantes:
# run_id, status, tags.mlflow.runName
# params.n_estimators, params.max_depth, ...
# metrics.test_accuracy, metrics.test_f1, metrics.test_auc_roc, ...

display(
    runs[["tags.mlflow.runName", "metrics.test_auc_roc",
          "metrics.test_accuracy", "metrics.test_f1"]]
    .head(10)
)

# Pegar o ID do melhor run
melhor_run_id = runs.iloc[0]["run_id"]
melhor_auc    = runs.iloc[0]["metrics.test_auc_roc"]
print(f"\nMelhor run: {melhor_run_id}")
print(f"Melhor AUC-ROC: {melhor_auc:.4f}")

# Filtrar por métrica
bons = mlflow.search_runs(
    experiment_names=["/churn-prediction"],
    filter_string="metrics.test_auc_roc > 0.80"
)
print(f"Runs com AUC > 0.80: {len(bons)}")
```

---

## 3.6 Model Registry — ciclo de vida do modelo (teoria para a prova)

```
Stages do Registry legado (Workspace Model Registry):
None ──→ Staging ──→ Production ──→ Archived

None:       versão recém-registrada, sem stage
Staging:    em validação / QA
Production: versão ativa em uso
Archived:   versão inativa (mantida para histórico)
```

> ⚠️ **Desde abril de 2024, o Databricks desabilita por padrão o Workspace Model Registry (stages None/Staging/Production) em qualquer workspace com Unity Catalog habilitado** — o que inclui o Free Edition. `client.transition_model_version_stage(...)` e URIs como `models:/nome/Production` **não funcionam** nesses workspaces. Guarde os stages para a prova (ainda caem), mas na prática — aqui e em qualquer workspace moderno — o registro de modelos usa o **Unity Catalog Models**: nome de 3 níveis (`catalog.schema.modelo`) + **aliases** no lugar de stages.

### Na prática: registrando no Unity Catalog

```python
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")   # garante o registro no Unity Catalog
client = MlflowClient()

# Nomes de modelo no Unity Catalog seguem as mesmas regras de tabela:
# só letras, números e underscore — SEM hífen — e sempre catalog.schema.nome
MODEL_NAME = "workspace.default.churn_predictor"

# --- Registrar modelo ---

# Opção A: registrar de um run existente
model_uri = f"runs:/{melhor_run_id}/model"
versao = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Registrado: versão {versao.version}")

# Opção B: registrar direto no log_model (mais comum)
with mlflow.start_run(run_name="RF_para_producao"):
    mlflow.sklearn.autolog()
    model_final = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model_final.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        model_final,
        artifact_path="model",
        registered_model_name=MODEL_NAME  # registra automaticamente ao logar
    )
```

```python
# --- Aliases: o equivalente moderno de "promover para Production" ---

# "champion" é só um nome de alias — pode ser qualquer string (ex: "producao", "campeao")
client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=1)

# Trocar o alias para outra versão = "promover" a nova versão
client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=2)

# Remover um alias
client.delete_registered_model_alias(name=MODEL_NAME, alias="champion")
```

```python
# --- Metadados e tags (funciona igual, com nome de 3 níveis) ---
client.update_model_version(
    name=MODEL_NAME,
    version=1,
    description="RF 200 árvores, otimizado com Hyperopt. AUC-ROC: 0.87"
)

client.set_model_version_tag(
    name=MODEL_NAME, version=1,
    key="validado_por", value="time-ml"
)

client.set_registered_model_tag(
    name=MODEL_NAME,
    key="projeto", value="churn-telecom"
)
```

---

## 3.7 Carregar modelo do Registry

```python
# Por alias (equivalente moderno de "por stage")
model_prod = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

# Por versão específica
model_v2 = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/2")

# Por run ID (sem precisar registrar)
model_run = mlflow.sklearn.load_model(f"runs:/{melhor_run_id}/model")

# Formato genérico pyfunc (funciona com qualquer framework)
model_pyfunc = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")

# Usar qualquer dos modelos carregados
novos_clientes = pd.DataFrame({
    "tenure_months":   [2, 48, 15],
    "monthly_charges": [85.0, 50.0, 70.0],
    "total_charges":   [170.0, 2400.0, 1050.0]
})
print("sklearn:", model_prod.predict(novos_clientes))
print("pyfunc:", model_pyfunc.predict(novos_clientes))
```

> **Resumo teoria vs. prática:** a prova pergunta sobre `Staging`/`Production`/`transition_model_version_stage`. Na sua conta Free Edition (e em qualquer workspace com Unity Catalog), você usa nome de 3 níveis + `set_registered_model_alias` + `models:/nome@alias`.

---

## 3.8 Logar métricas por step (épocas)

```python
# log_metric com step → útil para acompanhar curvas de aprendizado

import numpy as np

with mlflow.start_run(run_name="simulacao_treinamento_por_epoca"):
    mlflow.log_param("epochs", 50)
    mlflow.log_param("learning_rate", 0.01)

    for epoch in range(50):
        train_loss = 1.0 / (epoch + 1) + np.random.uniform(0, 0.05)
        val_loss   = 1.2 / (epoch + 1) + np.random.uniform(0, 0.08)

        mlflow.log_metrics({
            "train_loss":     train_loss,
            "val_loss":       val_loss,
            "train_accuracy": 1 - train_loss / 2,
            "val_accuracy":   1 - val_loss   / 2,
        }, step=epoch)

print("Veja os gráficos de loss/accuracy no MLflow UI!")
```

---

## 3.9 Logar artefatos variados

```python
with mlflow.start_run(run_name="exemplo_artefatos"):

    # Arquivo qualquer
    mlflow.log_artifact("relatorio_analise.html")

    # Dicionário como JSON
    mlflow.log_dict({"threshold": 0.45, "versao_dados": "v2"}, "config.json")

    # Texto
    mlflow.log_text("Modelo treinado em 2024-01. AUC: 0.87", "notas.txt")

    # Imagem matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.bar(["Ficou", "Churn"], [y_test.value_counts()[0], y_test.value_counts()[1]])
    ax.set_title("Distribuição do Target")
    mlflow.log_figure(fig, "distribuicao_target.png")
    plt.close()
```

---

## Pontos-chave para a prova

| Função | O que faz |
|---|---|
| `mlflow.set_experiment("/nome")` | Define o experimento ativo (cria se não existir) |
| `mlflow.start_run(run_name="x")` | Inicia um run (contexto `with`) |
| `mlflow.log_param("k", v)` | Loga um hiperparâmetro |
| `mlflow.log_params({...})` | Loga múltiplos parâmetros |
| `mlflow.log_metric("k", v, step=n)` | Loga uma métrica (opcionalmente por step) |
| `mlflow.log_metrics({...})` | Loga múltiplas métricas |
| `mlflow.sklearn.log_model(model, "path")` | Loga o modelo sklearn |
| `mlflow.log_artifact("arquivo.csv")` | Loga um arquivo qualquer |
| `mlflow.autolog()` | Ativa logging automático completo |
| `mlflow.search_runs(...)` | Retorna DataFrame pandas com os runs |
| `mlflow.register_model(uri, nome)` | Registra modelo no Registry (nome de 3 níveis no Unity Catalog) |
| `mlflow.sklearn.load_model("models:/x/Production")` | Carrega por stage — **teoria**; Registry legado desabilitado por padrão em workspaces UC |
| `mlflow.pyfunc.load_model(uri)` | Carrega qualquer modelo (formato genérico) |
| `client.transition_model_version_stage(...)` | Muda o stage de uma versão — **teoria**; use `set_registered_model_alias` na prática |
| `client.set_registered_model_alias(nome, alias, versao)` | Aponta um alias (ex: `champion`) para uma versão — equivalente moderno de "promover" |
| `models:/catalog.schema.modelo@alias` | URI para carregar por alias no Unity Catalog |
| `nested=True` em `start_run()` | Cria run filho dentro de um run pai (Hyperopt!) |

---

→ Próximo: [04_feature_engineering.md](04_feature_engineering.md)
