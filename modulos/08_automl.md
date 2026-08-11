# Módulo 8 — AutoML

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Seção oficial:** 1 — Databricks Machine Learning (**38% da prova**)
>
> **Notebook sugerido:** `08_automl`

**Objetivos oficiais cobertos neste módulo:**
- Identificar como o AutoML facilita a seleção de modelos e de features
- Identificar as vantagens que o AutoML traz ao processo de desenvolvimento de modelos

> 📌 **Repare que os dois objetivos começam com "identificar".** A prova cobra AutoML de forma **conceitual** — o que ele faz, o que ele gera, quando usar. Não espere questões pedindo a assinatura exata de `automl.classify()`. Isso é uma boa notícia, porque você provavelmente não vai conseguir rodá-lo.

---

> ⚠️ **Disponibilidade na Free Edition**
>
> O AutoML de **classificação e regressão** roda em **compute clássico com Databricks Runtime ML**, em modo de acesso *Dedicated* ou *No isolation shared*. A Free Edition oferece **apenas compute serverless**, sem opção de cluster clássico — então essas duas modalidades não rodam ali.
>
> No serverless, o Databricks suporta AutoML apenas para **forecasting**. Vale tentar; se a interface não oferecer a opção na sua conta, siga com a teoria — é o que a prova cobra.

---

## 8.1 O que o AutoML faz automaticamente

```
Você entrega:  uma tabela + o nome da coluna alvo
O AutoML faz:

├── 1. ANÁLISE DO DATASET
│      nulos, cardinalidade, tipos, distribuições, desbalanceamento
│
├── 2. FEATURE ENGINEERING AUTOMÁTICO
│      imputação, encoding de categóricas, scaling,
│      tratamento de datas e de texto
│
├── 3. SELEÇÃO DE FEATURES
│      descarta colunas inúteis: constantes, IDs de alta cardinalidade,
│      colunas com nulos demais
│
├── 4. SELEÇÃO DE MODELOS
│      treina várias famílias em paralelo:
│      scikit-learn, XGBoost, LightGBM (e Prophet/ARIMA em forecasting)
│
├── 5. TUNING DE HIPERPARÂMETROS
│      busca automática com early stopping
│
├── 6. VALIDAÇÃO
│      split e cross-validation internos
│
└── 7. SAÍDAS
       ├── todos os trials logados no MLflow
       ├── notebook de EXPLORAÇÃO DE DADOS (EDA)
       └── notebook do MELHOR MODELO — código Python completo e EDITÁVEL
```

### Como ele facilita a seleção de **modelos**

O AutoML treina dezenas de combinações algoritmo × hiperparâmetros em paralelo, avalia todas com a mesma métrica (`primary_metric`) e no mesmo esquema de validação, e ordena o resultado. O que você faria em dias de trabalho manual — e provavelmente com menos rigor de comparação — sai em minutos, com todos os resultados rastreados no MLflow.

### Como ele facilita a seleção de **features**

- Descarta automaticamente colunas sem poder preditivo (constantes, quase-constantes)
- Detecta e sinaliza colunas de alta cardinalidade que se comportam como identificadores
- Aplica a transformação adequada a cada tipo de coluna
- Reporta a **importância das features** do melhor modelo
- Permite excluir colunas manualmente via `exclude_cols`

---

## 8.2 As vantagens que caem na prova

| Vantagem | Detalhe |
|---|---|
| **Baseline rápido e forte** | Em minutos você tem uma referência de performance. Se seu modelo manual não bate o AutoML, o problema é o seu modelo |
| **É "glass-box", não black-box** | ⭐ **A vantagem mais característica do AutoML do Databricks:** ele gera o **código-fonte completo** do melhor modelo em um notebook editável. Você não recebe um binário misterioso — recebe código que pode ler, entender, modificar e re-treinar |
| **Reduz o tempo de prototipagem** | Elimina o trabalho repetitivo de encoding, imputação e busca de hiperparâmetros |
| **Democratiza o ML** | Analistas sem experiência profunda em ML conseguem produzir um modelo válido |
| **Tudo rastreado no MLflow** | Todos os trials viram runs — comparáveis, reproduzíveis e auditáveis |
| **Consistência na comparação** | Todos os modelos avaliados com a mesma métrica e o mesmo esquema de validação |
| **Notebook de EDA gratuito** | Análise exploratória automática do dataset, útil mesmo que você descarte o modelo |

> **A pegadinha clássica:** alternativas que dizem que o AutoML "substitui o cientista de dados", "dispensa entendimento do problema de negócio" ou "produz um modelo pronto para produção sem revisão" estão **erradas**. O AutoML gera um **ponto de partida**, não um produto final.

### Quando **não** usar AutoML

- O problema não é classificação, regressão ou forecasting (ex.: clustering, ranking, NLP complexo)
- Você precisa de feature engineering específica de domínio que o AutoML não infere
- O dataset exige tratamento especial (temporal com regras próprias, agregações complexas)
- Você já tem um modelo em produção bem tunado — o AutoML dificilmente vai superá-lo

---

## 8.3 Os três tipos de problema

| Tipo | Função | Métricas (`primary_metric`) |
|---|---|---|
| **Classificação** | `automl.classify()` | `f1`, `accuracy`, `log_loss`, `roc_auc`, `precision`, `recall` |
| **Regressão** | `automl.regress()` | `r2`, `mae`, `rmse`, `mse` |
| **Forecasting** | `automl.forecast()` | `smape`, `mse`, `rmse`, `mae`, `mdape` |

> **O que só existe em `forecast()`:** `time_col`, `frequency` e `horizon`. Se a questão citar qualquer um desses três, é forecasting.

---

## 8.4 AutoML via código

```python
from databricks import automl

df = spark.table("workspace.default.churn_clientes")

# --- Classificação ---
summary = automl.classify(
    dataset=df,                          # DataFrame Spark ou pandas
    target_col="churn",                  # coluna alvo
    primary_metric="roc_auc",            # métrica que define o "melhor" modelo
    exclude_cols=["customer_id"],        # colunas que não devem virar feature
    timeout_minutes=30,                  # orçamento de tempo
    experiment_name="/Users/.../churn-automl",
)

# --- Regressão ---
summary_reg = automl.regress(
    dataset=df,
    target_col="monthly_charges",
    primary_metric="r2",
    timeout_minutes=30,
)
```

```python
# --- Forecasting: precisa de uma série temporal ---
# O dataset de churn não serve. Criamos vendas diárias só para ilustrar.
import datetime

df_vendas = spark.createDataFrame(
    [(datetime.date(2024, 1, 1) + datetime.timedelta(days=i), 100.0 + i * 2)
     for i in range(180)],
    ["data", "vendas"],
)

summary_ts = automl.forecast(
    dataset=df_vendas,
    target_col="vendas",
    time_col="data",       # ← exclusivo do forecast
    frequency="D",         # ← exclusivo do forecast: D, W, M, Q, Y, h, min
    horizon=30,            # ← exclusivo do forecast: períodos futuros a prever
    primary_metric="smape",
    timeout_minutes=30,
)
```

### Outros parâmetros que aparecem em questões

```python
summary = automl.classify(
    dataset=df,
    target_col="churn",
    primary_metric="roc_auc",
    exclude_cols=["customer_id", "data_cadastro"],
    exclude_frameworks=["sklearn"],      # usar só xgboost e lightgbm
    timeout_minutes=30,
    max_trials=20,                       # limite de tentativas
    experiment_name="/Users/.../meu-automl",
    # split_col="split",                 # coluna que define treino/validação/teste
    # time_col="data",                   # em classify/regress: faz split TEMPORAL
    #                                    #   em vez de aleatório
)
```

> **Detalhe que cai:** `time_col` em `classify()`/`regress()` **não** transforma o problema em forecasting. Ele apenas faz o AutoML dividir treino/validação/teste **cronologicamente** em vez de aleatoriamente — o que evita data leakage temporal.

---

## 8.5 AutoML pela interface

```
Menu esquerdo → "Experiments" → "Create" → "AutoML Experiment"

Configuração:
├── Compute: cluster com Databricks Runtime ML
├── ML problem type: Classification / Regression / Forecasting
├── Dataset: selecionar tabela do Unity Catalog
├── Prediction target: coluna alvo
└── Advanced:
    ├── Evaluation metric (primary_metric)
    ├── Training frameworks (sklearn, xgboost, lightgbm)
    ├── Timeout e nº máximo de trials
    └── Colunas a excluir

Durante e depois da execução:
├── "View data exploration notebook"  → EDA automática
├── Lista de trials ordenada pela primary_metric
└── Melhor trial → "View notebook for best model" → código completo editável
```

---

## 8.6 Usar o resultado

```python
# O objeto de retorno
print(type(summary))                     # databricks.automl.AutoMLSummary
print(summary.best_trial)                # o melhor trial
print(summary.best_trial.model_path)     # URI MLflow: runs:/<run_id>/model
print(summary.experiment.experiment_id)  # experimento onde tudo foi logado

# Carregar o melhor modelo
import mlflow
best_model = mlflow.pyfunc.load_model(summary.best_trial.model_path)
```

```python
import pandas as pd

novos = pd.DataFrame({
    "tenure_months":    [3, 48],
    "monthly_charges":  [85.0, 50.0],
    "total_charges":    [255.0, 2400.0],
    "contract_type":    ["Month-to-month", "Two year"],
    "internet_service": ["Fiber optic", "DSL"],
    "tech_support":     ["No", "Yes"],
    "payment_method":   ["Electronic check", "Credit card"],
})
print("Predições:", best_model.predict(novos))   # 1 = churn, 0 = fica
```

```python
# Registrar o melhor modelo no Unity Catalog
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

MODEL_NAME = "workspace.default.churn_automl_best"   # sem hífen
resultado = mlflow.register_model(summary.best_trial.model_path, MODEL_NAME)

# "Promover" = apontar o alias (ver Módulo 4)
client.set_registered_model_alias(name=MODEL_NAME, alias="champion",
                                  version=resultado.version)
```

---

## 8.7 O notebook gerado

O maior diferencial do AutoML do Databricks. O notebook do melhor modelo contém:

```
1. Carregamento dos dados a partir da tabela original
2. Todo o pré-processamento aplicado, explicitado em código
   (imputers, encoders, scalers — nada escondido)
3. A definição do modelo com os hiperparâmetros vencedores
4. O bloco de treino, com o logging MLflow
5. Avaliação com a primary_metric e outras métricas
6. Gráficos de interpretabilidade (SHAP, importância de features)
7. Matriz de confusão / gráficos de resíduos, conforme o tipo de problema
```

Você pode clonar esse notebook, ajustar o que quiser e re-treinar. **Este é o argumento de "glass-box" — e é o ponto que a prova mais gosta de cobrar sobre AutoML.**

---

## Pontos-chave para a prova

- AutoML suporta **3 tipos**: `classify()`, `regress()`, `forecast()`
- `primary_metric` define qual modelo é considerado o **melhor**
- **Só `forecast()`** usa `time_col`, `frequency` e `horizon`
- Em `classify()`/`regress()`, `time_col` só faz o split ser **temporal**, não vira forecasting
- `exclude_cols` remove colunas que não devem ser features (IDs, chaves)
- `summary.best_trial.model_path` → URI MLflow do melhor modelo
- Gera **2 notebooks**: exploração de dados (EDA) + melhor modelo (editável)
- **Glass-box:** entrega o código-fonte, não um binário opaco — é a principal vantagem
- Todos os trials são logados automaticamente no **MLflow**
- Automatiza: análise, feature engineering, seleção de features, seleção de modelos, tuning e validação
- Serve como **baseline forte**, não como produto final — não substitui o cientista de dados
- **Classificação/regressão exigem compute clássico com Runtime ML** — na Free Edition (serverless) só forecasting é suportado

---

→ Próximo: [09_feature_store.md](09_feature_store.md)
