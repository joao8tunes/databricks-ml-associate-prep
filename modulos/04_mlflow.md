# Módulo 4 — MLflow: Tracking & Registry

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Seção oficial:** 1 — Databricks Machine Learning (**38% da prova**)
>
> **Notebook sugerido:** `04_mlflow`
>
> **Pré-requisito:** Módulo 2 (tabela `workspace.default.churn_clientes` criada).

**Este é o módulo mais importante da prova.** Oito objetivos oficiais estão aqui. Se você só puder estudar um módulo, estude este.

**Objetivos oficiais cobertos neste módulo:**
- Logar manualmente métricas, artefatos e modelos em um MLflow Run
- Identificar as informações disponíveis na MLflow UI
- Identificar o melhor run usando a MLflow Client API
- Registrar um modelo com a MLflow Client API no registry do Unity Catalog
- Identificar os benefícios de registrar modelos no Unity Catalog em vez do workspace registry
- Definir ou remover uma tag de um modelo
- Promover um modelo challenger a champion usando aliases

---

## 4.1 O problema que o MLflow resolve

```
Sem MLflow:
  "Qual modelo deu 94% de AUC? Era 100 ou 200 árvores?
   Qual threshold eu usei? Qual versão do dataset?
   Onde está o arquivo do modelo treinado semana passada?"

Com MLflow:
  → Tudo registrado e versionado automaticamente
  → Interface visual para comparar experimentos
  → Repositório central de versões de modelos
```

### Os componentes do MLflow

| Componente | O que faz | Peso na prova |
|---|---|---|
| **Tracking** | Registra params, métricas e artefatos de cada run | ★★★★★ |
| **Models** | Formato padrão para empacotar qualquer modelo (`pyfunc`) | ★★★★☆ |
| **Model Registry** | Catálogo de versões de modelos — no Unity Catalog | ★★★★★ |

### Vocabulário — errar isso custa questão

| Termo | O que é |
|---|---|
| **Experiment** | A pasta que agrupa runs de um mesmo projeto |
| **Run** | Uma execução de treino: um conjunto de params + métricas + artefatos |
| **Parameter** | Configuração definida **antes** do treino, imutável durante o run (`n_estimators`) |
| **Metric** | Valor medido, pode variar ao longo do run via `step` (`loss` por época) |
| **Artifact** | Qualquer arquivo: modelo, gráfico, CSV, JSON |
| **Model** | Artefato especial, com assinatura e formato padronizado |
| **Registered Model** | Um modelo publicado no Registry, com versões |

---

## 4.2 Preparar os dados

Rode isto uma vez — as seções seguintes reutilizam estas variáveis.

```python
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Tabela criada no Módulo 2 — rode aquele notebook primeiro se ainda não rodou
df = spark.table("workspace.default.churn_clientes")

df_pd = df.select("tenure_months", "monthly_charges", "total_charges", "churn").toPandas()
X = df_pd.drop("churn", axis=1)
y = df_pd["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Definir o experimento ---
# Criar experimento na RAIZ do workspace ("/churn-prediction") pode falhar por permissão.
# O caminho seguro é dentro da sua pasta de usuário — resolvido dinamicamente:
usuario = spark.sql("SELECT current_user()").first()[0]
EXPERIMENTO = f"/Users/{usuario}/churn-prediction"

mlflow.set_experiment(EXPERIMENTO)
print("Experimento:", EXPERIMENTO)
```

---

## 4.3 Tracking — registrar um experimento manualmente

```python
with mlflow.start_run(run_name="RF_baseline"):

    # 1. Logar parâmetros (hiperparâmetros — não mudam durante o run)
    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    mlflow.log_params(params)              # vários de uma vez
    # mlflow.log_param("n_estimators", 100)   # ou um a um

    # 2. Treinar
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 3. Avaliar
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # 4. Logar métricas
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, preds),
        "f1":       f1_score(y_test, preds),
        "auc_roc":  roc_auc_score(y_test, proba),
    })

    # 5. Logar o modelo
    #    MLflow 3 usa name= (artifact_path= está depreciado mas ainda funciona)
    #    input_example gera a assinatura do modelo automaticamente
    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_train.head(3),
    )

    # 6. Logar artefatos extras
    feat_df = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    mlflow.log_table(feat_df, artifact_file="feature_importance.json")

    run_id_baseline = mlflow.active_run().info.run_id
    print(f"Run ID : {run_id_baseline}")
    print(f"AUC-ROC: {roc_auc_score(y_test, proba):.4f}")
```

> **`name=` vs `artifact_path=`:** a partir do MLflow 3, `log_model()` usa `name=`. O parâmetro `artifact_path=` continua funcionando, mas emite aviso de depreciação. Na prova, os dois formatos podem aparecer — o conceito é o mesmo: é o subdiretório do run onde o modelo é salvo.

### Cada função de log

| Função | Loga |
|---|---|
| `mlflow.log_param("k", v)` | Um hiperparâmetro |
| `mlflow.log_params({...})` | Vários hiperparâmetros |
| `mlflow.log_metric("k", v, step=n)` | Uma métrica, opcionalmente por step |
| `mlflow.log_metrics({...})` | Várias métricas |
| `mlflow.log_artifact("arquivo.csv")` | Um arquivo que já existe em disco |
| `mlflow.log_artifacts("pasta/")` | Uma pasta inteira |
| `mlflow.log_dict({...}, "config.json")` | Um dicionário como JSON |
| `mlflow.log_text("...", "notas.txt")` | Texto puro |
| `mlflow.log_figure(fig, "grafico.png")` | Uma figura matplotlib |
| `mlflow.log_table(df, "tabela.json")` | Um DataFrame |
| `mlflow.sklearn.log_model(m, name="model")` | Um modelo sklearn |
| `mlflow.spark.log_model(m, name="model")` | Um modelo Spark ML |

---

## 4.4 autolog — logging automático

```python
# autolog() registra automaticamente: parâmetros, métricas de TREINO,
# o modelo e a assinatura — sem chamadas explícitas.

# Opção A: autolog por framework
mlflow.sklearn.autolog()
# mlflow.xgboost.autolog()
# mlflow.pytorch.autolog()

# Opção B: autolog genérico (detecta o framework em uso)
# mlflow.autolog()

with mlflow.start_run(run_name="LR_autolog"):
    from sklearn.linear_model import LogisticRegression
    model_lr = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    model_lr.fit(X_train, y_train)
    # params, métricas de treino e modelo logados automaticamente

mlflow.sklearn.autolog(disable=True)   # desligar para as próximas seções
```

> **A pegadinha do autolog:** ele loga métricas calculadas **durante o `fit()`** — ou seja, sobre os dados de treino. Métricas do conjunto de **teste** ele não tem como saber, porque você avalia depois. Se a questão perguntar "o autolog registrou a acurácia de teste?", a resposta é **não** — você loga isso manualmente.

| | `autolog()` | Log manual |
|---|---|---|
| Esforço | Nenhum | Você escreve cada linha |
| Métricas de teste | ❌ Não | ✅ Sim |
| Controle sobre o que é logado | Baixo | Total |
| Quando usar | Exploração, prototipagem | Produção |

---

## 4.5 O que você vê na MLflow UI

Objetivo oficial: "identificar as informações disponíveis na MLflow UI". Abra sua UI (menu esquerdo → **Experiments** → seu experimento) e localize cada item:

```
LISTA DE RUNS (tela do experimento)
├── Run Name, Run ID, usuário, data/hora de início, duração
├── Status: RUNNING / FINISHED / FAILED
├── Colunas de parâmetros e métricas (configuráveis)
├── Ordenação e filtro por qualquer métrica ou parâmetro
├── Comparar runs selecionados lado a lado
└── Gráfico de coordenadas paralelas (params × métrica)

DENTRO DE UM RUN
├── Parameters      → todos os params logados
├── Metrics         → valores finais + GRÁFICO por step, se houver
├── Artifacts       → árvore de arquivos (modelo, gráficos, CSVs)
├── Model           → assinatura (schema de entrada/saída) e exemplo
├── Tags            → metadados chave-valor
└── Source          → notebook/commit que gerou o run

COMPARAR RUNS
├── Tabela lado a lado de params e métricas
├── Scatter plot: um param × uma métrica
└── Coordenadas paralelas: várias dimensões ao mesmo tempo
```

> **O que a UI NÃO faz:** ela não treina modelos, não cria endpoints de serving e não edita o código do run. Alternativas de prova que sugerem isso estão erradas.

---

## 4.6 Encontrar o melhor run com a MLflow Client API

Primeiro, gere vários runs para comparar:

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

configuracoes = [
    ("RF_100_depth5",  RandomForestClassifier(n_estimators=100, max_depth=5,  random_state=42)),
    ("RF_200_depth10", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ("GBT_100",        GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("LR_C01",         LogisticRegression(C=0.1,  max_iter=500, random_state=42)),
    ("LR_C10",         LogisticRegression(C=10.0, max_iter=500, random_state=42)),
]

for run_name, modelo in configuracoes:
    with mlflow.start_run(run_name=run_name):
        modelo.fit(X_train, y_train)
        preds = modelo.predict(X_test)
        proba = modelo.predict_proba(X_test)[:, 1]

        mlflow.log_params(modelo.get_params())
        mlflow.log_metrics({
            "test_accuracy": accuracy_score(y_test, preds),
            "test_f1":       f1_score(y_test, preds),
            "test_auc_roc":  roc_auc_score(y_test, proba),
        })
        mlflow.sklearn.log_model(modelo, name="model", input_example=X_train.head(3))
```

### Caminho 1 — `mlflow.search_runs()` (retorna DataFrame **pandas**)

```python
runs = mlflow.search_runs(
    experiment_names=[EXPERIMENTO],
    order_by=["metrics.test_auc_roc DESC"],
)

display(runs[["tags.mlflow.runName", "metrics.test_auc_roc",
              "metrics.test_accuracy", "metrics.test_f1"]].head(10))

melhor_run_id = runs.iloc[0]["run_id"]
melhor_auc    = runs.iloc[0]["metrics.test_auc_roc"]
print(f"Melhor run: {melhor_run_id} | AUC: {melhor_auc:.4f}")

# Filtrar por métrica
bons = mlflow.search_runs(
    experiment_names=[EXPERIMENTO],
    filter_string="metrics.test_auc_roc > 0.70",
)
print(f"Runs com AUC > 0.70: {len(bons)}")
```

> **Pegadinha:** `mlflow.search_runs()` retorna um DataFrame **pandas**, não Spark. As colunas vêm prefixadas: `params.*`, `metrics.*`, `tags.*`.

### Caminho 2 — `MlflowClient().search_runs()` (retorna objetos `Run`)

Este é o caminho que o objetivo oficial chama de "MLflow Client API":

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

experimento = client.get_experiment_by_name(EXPERIMENTO)

melhores = client.search_runs(
    experiment_ids=[experimento.experiment_id],
    filter_string="metrics.test_auc_roc > 0",
    order_by=["metrics.test_auc_roc DESC"],
    max_results=1,                     # só o melhor
)

melhor = melhores[0]
print("Run ID :", melhor.info.run_id)
print("Nome   :", melhor.data.tags.get("mlflow.runName"))
print("AUC    :", melhor.data.metrics["test_auc_roc"])
print("Params :", {k: melhor.data.params[k] for k in list(melhor.data.params)[:3]})

melhor_run_id = melhor.info.run_id
```

| | `mlflow.search_runs()` | `MlflowClient().search_runs()` |
|---|---|---|
| Retorna | DataFrame **pandas** | Lista de objetos `Run` |
| Identifica o experimento por | `experiment_names` | `experiment_ids` |
| Acesso aos valores | `runs.iloc[0]["metrics.x"]` | `run.data.metrics["x"]` |
| Bom para | Análise tabular, comparação visual | Automação, scripts, CI/CD |

> **A ordenação faz o trabalho.** Para "identificar o melhor run", o padrão é `order_by=["metrics.<metrica> DESC"]` + `max_results=1`. Repare que `DESC` maximiza — para métricas onde menor é melhor (RMSE, log loss), use `ASC`.

---

## 4.7 Model Registry no Unity Catalog

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")    # ← registra no Unity Catalog
client = MlflowClient()

# Nomes de modelo no UC seguem as regras de tabela:
# catalog.schema.nome — só letras, números e underscore. SEM HÍFEN.
MODEL_NAME = "workspace.default.churn_predictor"
```

### Registrar — três formas

```python
# Forma A: a partir de um run existente
model_uri = f"runs:/{melhor_run_id}/model"
versao = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Registrado como versão {versao.version}")
```

```python
# Forma B: direto no log_model (a mais comum)
with mlflow.start_run(run_name="RF_para_producao"):
    model_final = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model_final.fit(X_train, y_train)

    proba = model_final.predict_proba(X_test)[:, 1]
    mlflow.log_metric("test_auc_roc", roc_auc_score(y_test, proba))

    mlflow.sklearn.log_model(
        model_final,
        name="model",
        input_example=X_train.head(3),
        registered_model_name=MODEL_NAME,   # registra automaticamente ao logar
    )
```

```python
# Forma C: via MLflow Client API (o objetivo oficial menciona esta)
# create_model_version exige que o registered model já exista
try:
    client.create_registered_model(MODEL_NAME)
except Exception:
    pass   # já existe

nova = client.create_model_version(
    name=MODEL_NAME,
    source=f"runs:/{melhor_run_id}/model",
    run_id=melhor_run_id,
)
print(f"Versão criada via Client API: {nova.version}")
```

### Consultar o Registry

```python
# Listar todas as versões do modelo
for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
    print(f"v{mv.version} | run_id={mv.run_id[:8]}... | aliases={list(mv.aliases)}")

# Metadados de uma versão específica
info = client.get_model_version(name=MODEL_NAME, version=1)
print(info.description, info.tags)
```

---

## 4.8 Por que Unity Catalog e não o Workspace Registry

Objetivo oficial: "identificar os benefícios de registrar modelos no Unity Catalog em vez do workspace registry".

| | **Workspace Model Registry** (legado) | **Unity Catalog Model Registry** (atual) |
|---|---|---|
| **Nome do modelo** | `nome_do_modelo` (1 nível) | **`catalog.schema.modelo`** (3 níveis) |
| **Escopo** | Um workspace | **A conta inteira** (metastore) |
| **Ciclo de vida** | Stages: `None → Staging → Production → Archived` | **Aliases** (`champion`, `challenger`, ...) |
| **Permissões** | Por workspace, pouco granular | **Granular** — `GRANT` por modelo/schema/catalog |
| **Linhagem** | Não | **Sim** — dados → features → modelo |
| **Governança unificada com dados** | Não | **Sim** — mesmo sistema de tabelas e volumes |
| **Status** | Desabilitado por padrão em workspaces com UC | Padrão |

**Os benefícios que caem na prova:**
1. **Escopo de conta** — o mesmo modelo é acessível de qualquer workspace, viabilizando a separação dev/staging/prod
2. **Governança granular** — permissões por modelo, no mesmo sistema dos dados
3. **Linhagem** — rastreia qual dado gerou qual feature que treinou qual modelo
4. **Aliases em vez de stages** — nomes livres e semânticos, não quatro estados fixos
5. **Namespace unificado** — dados, features e modelos no mesmo padrão de 3 níveis

> ⚠️ **Desde abril de 2024 o Databricks desabilita por padrão o Workspace Model Registry** em qualquer workspace com Unity Catalog — o que inclui a Free Edition. `client.transition_model_version_stage(...)` e URIs como `models:/nome/Production` **não funcionam** ali.

> 📌 **Os stages saíram do exam guide vigente.** O guia de março de 2025 não menciona `Staging`/`Production` em nenhum objetivo — só aliases. Material antigo que insiste em stages está baseado no guide anterior. Reconheça os termos se aparecerem como distratores, mas não invista tempo neles.

---

## 4.9 Tags e descrições

Objetivo oficial: "definir **ou remover** uma tag de um modelo". Repare no *remover* — a prova cobra as duas operações.

```python
# --- Descrição da versão ---
client.update_model_version(
    name=MODEL_NAME,
    version=1,
    description="RF com 200 árvores, otimizado com Hyperopt. AUC-ROC de teste: 0.87",
)

# --- Tag em uma VERSÃO específica ---
client.set_model_version_tag(name=MODEL_NAME, version=1, key="validado_por", value="time-ml")
client.delete_model_version_tag(name=MODEL_NAME, version=1, key="validado_por")

# --- Tag no MODELO REGISTRADO (vale para todas as versões) ---
client.set_registered_model_tag(name=MODEL_NAME, key="projeto", value="churn-telecom")
client.delete_registered_model_tag(name=MODEL_NAME, key="projeto")

# --- Tag em um RUN (não confundir com tag de modelo) ---
client.set_tag(melhor_run_id, "revisado", "sim")
client.delete_tag(melhor_run_id, "revisado")
```

| Escopo | Definir | Remover |
|---|---|---|
| Run | `client.set_tag(run_id, k, v)` | `client.delete_tag(run_id, k)` |
| Versão do modelo | `client.set_model_version_tag(name, version, k, v)` | `client.delete_model_version_tag(name, version, k)` |
| Modelo registrado | `client.set_registered_model_tag(name, k, v)` | `client.delete_registered_model_tag(name, k)` |

---

## 4.10 Aliases: champion e challenger

**Este é o objetivo oficial mais específico do módulo:** "promover um modelo challenger a champion usando aliases".

```python
# Um alias é um ponteiro nomeado para uma versão.
# "champion" e "challenger" são só convenções — o nome pode ser qualquer string.

client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=1)

# Uma nova versão entra como candidata
client.set_registered_model_alias(name=MODEL_NAME, alias="challenger", version=2)

# ... aqui rodam as validações comparando as duas ...

# PROMOVER: aponta "champion" para a versão do challenger
client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=2)

# Limpar o alias de challenger
client.delete_registered_model_alias(name=MODEL_NAME, alias="challenger")

# Consultar qual versão está por trás de um alias
mv = client.get_model_version_by_alias(name=MODEL_NAME, alias="champion")
print(f"champion → versão {mv.version}")
```

> **Por que isso importa:** o código de inferência sempre carrega `models:/{MODEL_NAME}@champion`. Promover um modelo vira uma operação de metadados — sem redeploy, sem mudar código, e reversível na hora. Aprofundado no [Módulo 11](11_mlops.md#114-champion-e-challenger).

---

## 4.11 Carregar modelos

```python
import pandas as pd

# Por ALIAS — o padrão em produção (repare no @)
model_champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")

# Por VERSÃO específica (repare na /)
model_v1 = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/1")

# Por RUN, sem precisar registrar
model_run = mlflow.sklearn.load_model(f"runs:/{melhor_run_id}/model")

# Formato genérico pyfunc — funciona com qualquer framework
model_pyfunc = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")

novos_clientes = pd.DataFrame({
    "tenure_months":   [2, 48, 15],
    "monthly_charges": [85.0, 50.0, 70.0],
    "total_charges":   [170.0, 2400.0, 1050.0],
})
print("sklearn:", model_champion.predict(novos_clientes))
print("pyfunc :", model_pyfunc.predict(novos_clientes))
```

| URI | Significa |
|---|---|
| `models:/catalog.schema.nome@champion` | Versão apontada pelo **alias** — `@` |
| `models:/catalog.schema.nome/3` | **Versão** 3 — `/` |
| `runs:/<run_id>/model` | Modelo logado naquele run, sem registro |

> **Pegadinha:** `@` é alias, `/` é versão. Trocar os dois é erro de sintaxe de URI e aparece como distrator.

| | `mlflow.sklearn.load_model()` | `mlflow.pyfunc.load_model()` |
|---|---|---|
| Retorna | O objeto sklearn original | Wrapper genérico |
| Métodos disponíveis | `.predict()`, `.predict_proba()`, `.feature_importances_`... | Só `.predict()` |
| Funciona com | Só sklearn | **Qualquer framework** |
| Use quando | Precisa de recursos específicos do framework | Serving, código agnóstico de framework |

---

## 4.12 Métricas por step

```python
import numpy as np

with mlflow.start_run(run_name="curva_de_aprendizado"):
    mlflow.log_params({"epochs": 50, "learning_rate": 0.01})

    for epoch in range(50):
        train_loss = 1.0 / (epoch + 1) + np.random.uniform(0, 0.05)
        val_loss   = 1.2 / (epoch + 1) + np.random.uniform(0, 0.08)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss":   val_loss,
        }, step=epoch)     # ← o step vira o eixo X do gráfico na UI

print("Abra o run na UI e veja o gráfico de loss por época.")
```

> `step` é o que transforma uma métrica em **curva** na MLflow UI. Sem ele, cada `log_metric` sobrescreve visualmente o anterior no resumo.

---

## 4.13 Artefatos variados

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

with mlflow.start_run(run_name="exemplo_artefatos"):

    # Dicionário como JSON
    mlflow.log_dict({"threshold": 0.45, "versao_dados": "v2"}, "config.json")

    # Texto puro
    mlflow.log_text("Modelo treinado com dados até 2024-06. AUC: 0.87", "notas.txt")

    # Figura matplotlib (não precisa salvar em disco antes)
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Ficou", "Churn"]).plot(ax=ax)
    mlflow.log_figure(fig, "matriz_confusao.png")
    plt.close(fig)

    # Arquivo em disco — só funciona se o arquivo EXISTIR
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        caminho = os.path.join(tmp, "resumo.csv")
        pd.DataFrame({"metrica": ["auc"], "valor": [0.87]}).to_csv(caminho, index=False)
        mlflow.log_artifact(caminho)
```

> `log_artifact()` exige um arquivo **já existente em disco**. `log_dict`, `log_text`, `log_figure` e `log_table` criam o arquivo por você.

---

## Exercício prático

```python
# 1. Treine 3 modelos diferentes, logando params e métricas de TESTE manualmente
# 2. Use MlflowClient().search_runs() para achar o melhor por AUC
# 3. Registre esse modelo no Unity Catalog via Client API
# 4. Coloque o alias "champion" nele
# 5. Adicione uma tag "validado_por" na versão e depois remova
# 6. Carregue o modelo por alias e faça uma predição
# 7. Abra a MLflow UI e localize: params, métricas, artefatos, assinatura e tags
```

---

## Pontos-chave para a prova

| Função | O que faz |
|---|---|
| `mlflow.set_experiment("/Users/.../nome")` | Define o experimento ativo (cria se não existir) |
| `mlflow.start_run(run_name="x")` | Inicia um run (usar com `with`) |
| `mlflow.log_param(s)` | Hiperparâmetros — imutáveis no run |
| `mlflow.log_metric(s)(..., step=n)` | Métricas — podem variar por step |
| `mlflow.log_artifact("f.csv")` | Arquivo que **já existe** em disco |
| `mlflow.log_dict / log_text / log_figure / log_table` | Cria o arquivo e loga |
| `mlflow.sklearn.log_model(m, name="model")` | Loga o modelo (MLflow 3 usa `name=`) |
| `mlflow.autolog()` | Logging automático — **não pega métricas de teste** |
| `mlflow.search_runs()` | DataFrame **pandas** dos runs |
| `MlflowClient().search_runs(order_by=[...], max_results=1)` | Melhor run como objeto `Run` |
| `mlflow.set_registry_uri("databricks-uc")` | Aponta o Registry para o Unity Catalog |
| `mlflow.register_model(uri, "cat.schema.nome")` | Registra um modelo |
| `client.create_model_version(name, source, run_id)` | Registra via Client API |
| `client.set_registered_model_alias(name, alias, version)` | **Promove** — aponta alias para versão |
| `client.get_model_version_by_alias(name, alias)` | Qual versão está atrás do alias |
| `client.set/delete_model_version_tag(...)` | Tag de uma versão |
| `client.set/delete_registered_model_tag(...)` | Tag do modelo registrado |
| `models:/cat.schema.nome@alias` | Carregar por **alias** (`@`) |
| `models:/cat.schema.nome/3` | Carregar por **versão** (`/`) |
| `runs:/<run_id>/model` | Carregar direto de um run |
| `mlflow.pyfunc.load_model(uri)` | Formato genérico — qualquer framework |
| `nested=True` em `start_run()` | Run filho dentro de um pai (Hyperopt!) |

**E os quatro pontos conceituais:**
- Nome no UC é `catalog.schema.modelo`, **sem hífen**
- Benefícios do UC Registry: escopo de conta, governança granular, linhagem, aliases
- `autolog()` não loga métricas de teste
- Stages (`Staging`/`Production`) saíram do exam guide — o atual usa **aliases**

---

→ Próximo: [05_feature_engineering.md](05_feature_engineering.md)
