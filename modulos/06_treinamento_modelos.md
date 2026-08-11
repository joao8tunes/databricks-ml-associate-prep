# Módulo 6 — Treinamento, Avaliação e Seleção de Modelos

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Seção oficial:** 3 — Model Development (**31% da prova**)
>
> **Notebook sugerido:** `06_treinamento_modelos`
>
> **Pré-requisito:** Módulos 2, 3 e 5.

Este módulo concentra o maior número de questões **conceituais** da prova: escolher algoritmo, escolher métrica, contar modelos treinados, entender viés-variância. Muita coisa aqui você não roda — você raciocina.

**Objetivos oficiais cobertos neste módulo:**
- Usar fundamentos de ML para escolher o algoritmo adequado a um cenário
- Identificar métodos para mitigar desbalanceamento de classes
- Usar métricas comuns de classificação (F1, Log Loss, ROC/AUC)
- Usar métricas comuns de regressão (RMSE, MAE, R²)
- Escolher a métrica mais apropriada para o objetivo de um cenário
- Descrever benefícios e desvantagens de cross-validation vs train-validation split
- Executar cross-validation como parte do ajuste do modelo
- Identificar quantos modelos são treinados em grid-search com cross-validation
- Identificar a necessidade de exponenciar variáveis log-transformadas antes de calcular métricas
- Avaliar o impacto da complexidade do modelo e do trade-off viés-variância

---

## 6.1 Escolher o algoritmo certo

A prova apresenta um cenário e pergunta qual algoritmo usar. O primeiro filtro é sempre o **tipo de problema**:

```
O que você quer prever?
├── Uma categoria (sim/não, A/B/C)      → CLASSIFICAÇÃO
├── Um número contínuo (preço, tempo)   → REGRESSÃO
├── Grupos sem rótulo prévio            → CLUSTERING (não supervisionado)
└── Itens para um usuário               → RECOMENDAÇÃO
```

```python
# --- Classificação ---
from pyspark.ml.classification import (
    LogisticRegression,              # rápido, interpretável, bom baseline
    RandomForestClassifier,          # robusto, pouco tuning, não precisa de scaling
    GBTClassifier,                   # geralmente a melhor acurácia em dados tabulares
    DecisionTreeClassifier,          # muito interpretável, mas sozinha faz overfitting
    LinearSVC,                       # SVM linear
    NaiveBayes,                      # rápido, clássico para texto
    MultilayerPerceptronClassifier,  # rede neural simples
)

# --- Regressão ---
from pyspark.ml.regression import (
    LinearRegression,
    RandomForestRegressor,
    GBTRegressor,
    DecisionTreeRegressor,
    GeneralizedLinearRegression,     # GLM: famílias Poisson, Gamma, Binomial...
)

# --- Clustering (não supervisionado) ---
from pyspark.ml.clustering import (
    KMeans,
    BisectingKMeans,                 # hierárquico divisivo
    GaussianMixture,
    LDA,                             # Latent Dirichlet Allocation — tópicos em texto
)

# --- Recomendação ---
from pyspark.ml.recommendation import ALS   # Alternating Least Squares
```

### O que cada cenário pede

| O enunciado diz… | Algoritmo indicado |
|---|---|
| "precisa **explicar** a decisão ao regulador/cliente" | Regressão Logística ou Decision Tree — coeficientes/regras legíveis |
| "quer o **melhor desempenho possível** em dados tabulares" | GBT / XGBoost / LightGBM |
| "quer um **baseline rápido**" | Regressão Logística (classificação) ou Linear (regressão) |
| "poucos dados, muitas features" | Modelos lineares com regularização |
| "features em escalas muito diferentes e não quer normalizar" | Modelos de árvore |
| "relação claramente **não-linear** entre features e target" | Árvores, ensembles ou redes neurais |
| "não tem rótulos, quer **segmentar** clientes" | KMeans (clustering) |
| "quer **recomendar** produtos com base no histórico" | ALS |
| "prever contagem de eventos (nº de acessos, sinistros)" | GLM com família Poisson |
| "dados de texto, precisa ser muito rápido" | Naive Bayes |

> **Pegadinha:** "interpretabilidade" é a palavra-chave para modelos lineares e árvores individuais. "Máxima performance" é a palavra-chave para ensembles (Random Forest, GBT). Se as duas aparecem juntas, o enunciado quase sempre pede que você escolha o trade-off explicitamente.

---

## 6.2 Desbalanceamento de classes

Objetivo oficial. Um dataset com 1% de fraudes ou 5% de churn quebra tanto o treino quanto a avaliação.

```
O problema: com 95% de classe negativa, o modelo aprende que
"prever sempre NEGATIVO" já acerta 95% das vezes.
Ele nunca aprende a reconhecer a classe minoritária.
```

### Os quatro grupos de solução

| Estratégia | Como funciona | Cuidado |
|---|---|---|
| **1. Class weights / cost-sensitive learning** | Dá mais peso ao erro na classe minoritária durante o treino | ✅ **Geralmente a melhor primeira escolha** — não altera os dados |
| **2. Oversampling da minoritária** | Duplica ou sintetiza exemplos da classe rara (ex.: SMOTE) | Risco de overfitting nos exemplos duplicados |
| **3. Undersampling da majoritária** | Remove exemplos da classe comum | Descarta dados e informação |
| **4. Trocar a métrica de avaliação** | Usar F1, AUC-PR ou recall em vez de accuracy | Não resolve o treino, mas evita a ilusão de "99% de acurácia" |

**Outras que podem aparecer:**
- **Ajustar o threshold de decisão** (usar 0.3 em vez de 0.5) — barato e frequentemente eficaz
- **Coletar mais dados da classe minoritária** — ideal quando possível
- **Tratar como detecção de anomalia** — quando a classe rara é *muito* rara (<0.1%)

```python
# --- Class weights no Spark ML: crie uma coluna de peso e passe em weightCol ---
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression

df = spark.table("workspace.default.churn_clientes")

n_total = df.count()
n_pos   = df.filter(F.col("churn") == 1).count()
n_neg   = n_total - n_pos

peso_pos = n_total / (2.0 * n_pos)
peso_neg = n_total / (2.0 * n_neg)
print(f"Positivos: {n_pos} (peso {peso_pos:.2f}) | Negativos: {n_neg} (peso {peso_neg:.2f})")

df_pesado = df.withColumn(
    "class_weight",
    F.when(F.col("churn") == 1, peso_pos).otherwise(peso_neg),
)

# lr = LogisticRegression(featuresCol="features", labelCol="churn", weightCol="class_weight")
```

```python
# --- Em sklearn, é um parâmetro direto ---
# RandomForestClassifier(class_weight="balanced")
# LogisticRegression(class_weight="balanced")
# XGBClassifier(scale_pos_weight = n_negativos / n_positivos)
```

> **Pegadinha da prova (é a Questão 3 do exam guide oficial):** o enunciado descreve um dataset com 10% de churn e pergunta o que **mitiga diretamente o viés do modelo** para a classe majoritária. A resposta é **cost-sensitive learning / class weights**. Normalizar features, coletar mais dados da classe *majoritária* ou simplificar o modelo **não** atacam o desbalanceamento.

> **Não confunda:** `weightCol` no Spark ML atribui peso **por linha**. Você calcula o peso a partir da classe e escreve na coluna. O Spark não tem um `class_weight="balanced"` pronto como o sklearn.

---

## 6.3 Métricas de classificação

Tudo parte da matriz de confusão:

```
                    PREDITO
                 Negativo  Positivo
        Negativo    TN        FP     ← FP = falso alarme
REAL                                   (erro tipo I)
        Positivo    FN        TP     ← FN = deixou passar
                                       (erro tipo II)
```

| Métrica | Fórmula | Responde a pergunta | Use quando |
|---|---|---|---|
| **Accuracy** | (TP+TN)/total | "Quantos acertei no total?" | Classes **balanceadas** |
| **Precision** | TP/(TP+FP) | "Dos que apontei como positivos, quantos eram?" | **Falso positivo é caro** |
| **Recall** (sensibilidade) | TP/(TP+FN) | "Dos positivos reais, quantos capturei?" | **Falso negativo é caro** |
| **F1** | 2·(P·R)/(P+R) | Média harmônica de precision e recall | Quer equilíbrio entre os dois |
| **ROC-AUC** | área sob TPR×FPR | "Quão bem o modelo separa as classes?" | Comparação geral; classes razoavelmente balanceadas |
| **PR-AUC** (average precision) | área sob precision×recall | Idem, focado na classe positiva | **Classes muito desbalanceadas** |
| **Log Loss** | −média(log da prob. da classe correta) | "Quão bem calibradas são as probabilidades?" | Você usa as **probabilidades**, não só o rótulo |

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, confusion_matrix,
)
```

### Log Loss — a métrica que a maioria ignora

Log Loss está listada explicitamente no exam guide, e é a única da lista que **penaliza confiança errada**:

```
Rótulo real = 1 (churn)

Modelo A prevê probabilidade 0.6  → acerta a classe, log loss ≈ 0.51
Modelo B prevê probabilidade 0.9  → acerta a classe, log loss ≈ 0.11  ← melhor
Modelo C prevê probabilidade 0.1  → ERRA a classe,  log loss ≈ 2.30   ← punição pesada

Accuracy trata A e B como idênticos. Log Loss não.
```

- **Menor é melhor** (ao contrário de accuracy, F1 e AUC)
- Vai de 0 (perfeito) a ∞
- Exige **probabilidades**, não só a classe predita
- É a função de perda que a regressão logística minimiza

### Avaliadores do Spark ML

```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# BINÁRIA — usa rawPrediction (scores contínuos), não a classe predita
bin_eval = BinaryClassificationEvaluator(
    labelCol="churn",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC",     # "areaUnderROC" | "areaUnderPR"
)

# MULTICLASSE — funciona para binária também; usa prediction (a classe)
mc_eval = MulticlassClassificationEvaluator(labelCol="churn", predictionCol="prediction")
# métricas: "accuracy", "f1", "weightedPrecision", "weightedRecall",
#           "weightedFMeasure", "logLoss", "truePositiveRateByLabel"
```

> **Pegadinha:** `BinaryClassificationEvaluator` só oferece **duas** métricas: `areaUnderROC` e `areaUnderPR`. Accuracy e F1 vêm do `MulticlassClassificationEvaluator` — que funciona para problemas binários também. Se a questão pede accuracy num problema binário, o avaliador é o **Multiclass**.

---

## 6.4 Métricas de regressão

| Métrica | O que é | Unidade | Sensível a outliers |
|---|---|---|---|
| **MSE** | média dos erros ao quadrado | unidade² | **Muito** |
| **RMSE** | raiz do MSE | **mesma do target** | Muito |
| **MAE** | média dos erros absolutos | mesma do target | **Pouco** (robusta) |
| **R²** | proporção da variância explicada | adimensional | Média |

```python
from pyspark.ml.evaluation import RegressionEvaluator

reg_eval = RegressionEvaluator(labelCol="monthly_charges", predictionCol="prediction")
# métricas: "rmse" (default), "mse", "mae", "r2", "var"
```

**Como ler o R²:**
```
R² = 1.0  → predições perfeitas
R² = 0.7  → o modelo explica 70% da variância do target
R² = 0.0  → equivale a sempre prever a média
R² < 0    → PIOR que sempre prever a média
```

> **RMSE vs MAE:** o RMSE eleva os erros ao quadrado, então um erro grande pesa desproporcionalmente. Se **erros grandes são especialmente ruins** (previsão de demanda, capacidade), use RMSE. Se **os outliers são ruído** e você não quer que dominem, use MAE.

---

## 6.5 Qual métrica usar em cada cenário

```
CLASSIFICAÇÃO
├── Classes balanceadas, todos os erros custam igual  → Accuracy
├── Classes desbalanceadas                            → F1, PR-AUC (nunca accuracy)
├── Falso positivo é caro                             → Precision
│     ex.: bloquear cartão de cliente legítimo, spam que era importante
├── Falso negativo é caro                             → Recall
│     ex.: não detectar fraude, não detectar doença
├── Quer equilíbrio entre os dois                     → F1
├── Comparar modelos independente de threshold        → ROC-AUC
├── Comparar modelos com classe positiva muito rara   → PR-AUC
└── As PROBABILIDADES importam (ranking, precificação) → Log Loss

REGRESSÃO
├── Erros grandes são desproporcionalmente ruins  → RMSE
├── Outliers são ruído e não devem dominar        → MAE
├── Quer uma medida comparável entre datasets     → R²
└── Precisa reportar na unidade do negócio        → RMSE ou MAE (nunca MSE)
```

> **A pegadinha número um da prova:** um enunciado com "detecção de fraude com 1% de fraudes" e uma alternativa dizendo "usar accuracy, pois 99% é um ótimo resultado". Está sempre errado — prever "não é fraude" para tudo já dá 99%.

---

## 6.6 Cross-validation vs train-validation split

```
TRAIN-VALIDATION SPLIT
┌───────────────────────────┬──────────┐
│         TREINO 75%        │  VAL 25% │   → treina 1 vez por combinação
└───────────────────────────┴──────────┘

CROSS-VALIDATION (k=3)
┌──────┬──────┬──────┐
│ VAL  │TREINO│TREINO│  fold 1
├──────┼──────┼──────┤
│TREINO│ VAL  │TREINO│  fold 2   → treina k vezes por combinação
├──────┼──────┼──────┤             e faz a média das k métricas
│TREINO│TREINO│ VAL  │  fold 3
└──────┴──────┴──────┘
```

| | **CrossValidator** | **TrainValidationSplit** |
|---|---|---|
| Modelos por combinação | **k** | **1** |
| Custo computacional | k vezes maior | Baixo |
| Confiabilidade da estimativa | **Alta** — média de k avaliações | Menor — depende de um único split sortudo/azarado |
| Usa todos os dados para validar | Sim (cada linha é validação uma vez) | Não |
| Bom para | Datasets **pequenos ou médios** | Datasets **grandes**, ou tuning caro |
| Parâmetro-chave | `numFolds` | `trainRatio` |

**Benefícios da cross-validation:** estimativa mais estável e menos dependente de sorte no split; aproveita todos os dados tanto para treino quanto para validação; reduz o risco de escolher hiperparâmetros que só funcionam num split específico.

**Desvantagens:** custa k vezes mais tempo e computação; em datasets muito grandes o ganho de estabilidade é marginal (um split de 20% de 10 milhões de linhas já é bem estimado); não é adequada em séries temporais no formato padrão, porque embaralha a ordem cronológica.

---

## 6.7 Cross-validation na prática

```python
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

df = spark.table("workspace.default.churn_clientes")

colunas_num = ["tenure_months", "monthly_charges", "total_charges"]
colunas_cat = ["contract_type", "internet_service", "tech_support", "payment_method"]

stages = [
    Imputer(inputCols=colunas_num, outputCols=[c + "_imp" for c in colunas_num]),
    *[StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in colunas_cat],
    *[OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_ohe") for c in colunas_cat],
    VectorAssembler(
        inputCols=[c + "_imp" for c in colunas_num] + [c + "_ohe" for c in colunas_cat],
        outputCol="features_raw",
    ),
    StandardScaler(inputCol="features_raw", outputCol="features"),
]

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
feat_model = Pipeline(stages=stages).fit(train_df)
train_feat = feat_model.transform(train_df)
test_feat  = feat_model.transform(test_df)
```

```python
rf = RandomForestClassifier(featuresCol="features", labelCol="churn", seed=42)

param_grid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [50, 100, 200])
    .addGrid(rf.maxDepth, [3, 5, 8])
    .build())
print(f"Combinações na grade: {len(param_grid)}")   # 3 × 3 = 9

evaluator = BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderROC")

cv = CrossValidator(
    estimator=rf,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    seed=42,
    parallelism=2,      # quantas combinações avaliar em paralelo
)

cv_model = cv.fit(train_feat)
melhor   = cv_model.bestModel

print(f"Melhor AUC médio na CV: {max(cv_model.avgMetrics):.4f}")
print(f"numTrees: {melhor.getNumTrees} | maxDepth: {melhor.getOrDefault('maxDepth')}")
print(f"AUC no conjunto de teste: {evaluator.evaluate(cv_model.transform(test_feat)):.4f}")
```

```python
# TrainValidationSplit — mesma API, um único split
tvs = TrainValidationSplit(
    estimator=rf,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    trainRatio=0.8,      # 80% treino, 20% validação interna
    seed=42,
)
tvs_model = tvs.fit(train_feat)
print(f"Melhor AUC (TVS): {max(tvs_model.validationMetrics):.4f}")
```

> **O que o `CrossValidator` faz no final:** depois de avaliar todas as combinações, ele **re-treina o melhor modelo usando 100% dos dados de treino**. O `bestModel` é esse modelo final, não um dos modelos dos folds.

---

## 6.8 Quantos modelos são treinados — a conta que cai na prova

Objetivo oficial explícito. É a questão mais mecânica da prova inteira e vale ponto garantido.

```
Nº de modelos = (nº de combinações de hiperparâmetros) × (nº de folds)

Nº de combinações = produto do tamanho de cada lista de parâmetros
```

### Exemplo do próprio exam guide (Questão 4)

```
SVM com GridSearchCV, 5-fold:
  C      = [0.1, 1, 10]      → 3 valores
  kernel = ['linear', 'rbf'] → 2 valores
  gamma  = [0.01, 0.1, 1]    → 3 valores

Combinações = 3 × 2 × 3 = 18
Modelos     = 18 × 5 folds = 90        ← resposta oficial: 90
```

### O detalhe do refit

```
scikit-learn GridSearchCV com refit=True (o padrão):
  90 modelos da busca + 1 refit final com todos os dados = 91

Mas a resposta oficial da questão do exam guide é 90.
→ A prova conta os modelos DA BUSCA. Não some o refit,
  a menos que o enunciado pergunte explicitamente por ele.
```

### Spark ML

```
CrossValidator:
  modelos = combinações × numFolds        (+1 refit do bestModel)

TrainValidationSplit:
  modelos = combinações × 1               (+1 refit do bestModel)
```

```python
# Confira você mesmo
grid = (ParamGridBuilder()
    .addGrid(rf.numTrees, [50, 100, 200])    # 3
    .addGrid(rf.maxDepth, [3, 5, 8])         # 3
    .build())

n_combinacoes = len(grid)
n_folds = 3
print(f"Combinações : {n_combinacoes}")                    # 9
print(f"CrossValidator     : {n_combinacoes * n_folds}")   # 27
print(f"TrainValidationSplit: {n_combinacoes * 1}")        # 9
```

> **Erro clássico:** somar em vez de multiplicar as listas de parâmetros (3+3 = 6 em vez de 3×3 = 9), ou esquecer de multiplicar pelos folds. Leia devagar e multiplique tudo.

---

## 6.9 Target log-transformado: o erro que quase todo mundo comete

Objetivo oficial: "identificar a necessidade de exponenciar variáveis log-transformadas antes de calcular métricas de avaliação ou interpretar predições".

```
Você treinou:   modelo.fit(X, log(y))
O modelo prevê: valores em ESCALA LOG

Se você calcular RMSE entre log(y_real) e pred_log:
  → o número resultante está em escala log
  → NÃO é o erro em reais, dias ou unidades do negócio
  → e não é comparável com o RMSE de um modelo treinado sem log
```

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_real   = np.array([1200.0, 3400.0, 800.0, 5600.0])
y_log    = np.log(y_real)

# Suponha que o modelo previu isto, em escala log
pred_log = np.array([7.05, 8.20, 6.75, 8.55])

# --- ERRADO: métrica calculada em escala log ---
rmse_log = np.sqrt(mean_squared_error(y_log, pred_log))
print(f"RMSE em escala log : {rmse_log:.4f}   ← não significa R$ 0,xx")

# --- CORRETO: exponenciar de volta ANTES de avaliar ---
pred_original = np.exp(pred_log)          # inverso de np.log
rmse_real = np.sqrt(mean_squared_error(y_real, pred_original))
mae_real  = mean_absolute_error(y_real, pred_original)

print(f"Predições em R$    : {np.round(pred_original, 2)}")
print(f"RMSE em R$         : {rmse_real:.2f}")
print(f"MAE  em R$         : {mae_real:.2f}")
```

| Transformação aplicada | Inverso a usar |
|---|---|
| `np.log(y)` / `F.log(col)` | `np.exp(pred)` |
| `np.log1p(y)` / `F.log1p(col)` | **`np.expm1(pred)`** |
| `np.log10(y)` | `10 ** pred` |

> **Não misture os pares.** Se você treinou com `log1p` e desfez com `exp`, o resultado fica errado por um fator constante — e o erro passa despercebido porque os números "parecem plausíveis".

**Isso vale também para interpretar a predição para o negócio.** Reportar "o modelo prevê 8.2" quando 8.2 é `log(receita)` não significa nada para ninguém. `exp(8.2) ≈ R$ 3.641` significa.

---

## 6.10 Viés, variância e complexidade do modelo

Objetivo oficial: "avaliar o impacto da complexidade do modelo e do trade-off viés-variância na performance".

```
Erro total = Viés² + Variância + Erro irredutível
                                  ↑
                        ruído inerente aos dados — não dá para eliminar
```

| | **Viés alto** | **Variância alta** |
|---|---|---|
| Nome comum | **Underfitting** | **Overfitting** |
| O que acontece | O modelo é simples demais para captar o padrão | O modelo decorou o treino, inclusive o ruído |
| Erro no **treino** | **Alto** | **Baixo** |
| Erro no **teste** | **Alto** | **Alto** |
| Diferença treino↔teste | Pequena | **Grande** |
| Exemplo | Reta em dados curvos | Árvore de profundidade 50 |

```
        Erro
          │╲                              ╱
          │ ╲                           ╱
          │  ╲___                    ╱      ← erro de TESTE
          │      ╲___          ___╱           (curva em U)
          │           ╲______╱
          │              ↑
          │        ponto ótimo
          │                    ╲___
          │                        ╲______   ← erro de TREINO
          │                                    (só cai)
          └──────────────────────────────────→
             simples    Complexidade    complexo
```

**O sinal de diagnóstico:**

```
Treino ruim  + Teste ruim              → VIÉS alto (underfitting)
Treino ótimo + Teste ruim              → VARIÂNCIA alta (overfitting)
Treino bom   + Teste bom e parecidos   → bom equilíbrio
```

### Como corrigir cada um

| Problema | O que fazer |
|---|---|
| **Underfitting** (viés alto) | Modelo mais complexo; mais features; menos regularização; treinar mais tempo/iterações |
| **Overfitting** (variância alta) | **Mais dados de treino**; mais regularização (L1/L2); reduzir profundidade/nº de features; early stopping; cross-validation; ensembles (bagging) |

> **O que aumentar dados faz:** reduz **variância**, não viés. Se o problema é underfitting, coletar mais dados não resolve — o modelo continua simples demais.

> **Ensembles e o trade-off:** *bagging* (Random Forest) reduz principalmente **variância** ao fazer a média de muitas árvores independentes. *Boosting* (GBT) reduz principalmente **viés** ao corrigir sequencialmente os erros do modelo anterior — por isso boosting é mais propenso a overfitting que bagging.

---

## 6.11 Treinar e avaliar na prática

```python
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
import mlflow

usuario = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{usuario}/churn-sparkml")

with mlflow.start_run(run_name="RF_Spark"):
    rf = RandomForestClassifier(
        featuresCol="features",   # coluna Vector, criada pelo VectorAssembler
        labelCol="churn",         # label numérica
        numTrees=100,
        maxDepth=5,
        seed=42,
    )
    mlflow.log_params({"numTrees": 100, "maxDepth": 5, "framework": "Spark MLlib"})

    rf_model    = rf.fit(train_feat)
    predictions = rf_model.transform(test_feat)
    # predictions agora tem: features, churn, rawPrediction, probability, prediction

    bin_eval = BinaryClassificationEvaluator(labelCol="churn", metricName="areaUnderROC")
    mc_eval  = MulticlassClassificationEvaluator(labelCol="churn", predictionCol="prediction")

    auc = bin_eval.evaluate(predictions)
    acc = mc_eval.setMetricName("accuracy").evaluate(predictions)
    f1  = mc_eval.setMetricName("f1").evaluate(predictions)

    mlflow.log_metrics({"auc_roc": auc, "accuracy": acc, "f1": f1})
    mlflow.spark.log_model(rf_model, name="spark_rf_model")

    print(f"AUC-ROC: {auc:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
```

### As colunas que os modelos de classificação adicionam

```
├── rawPrediction → scores brutos (log-odds na LR, contagens de votos na RF)
├── probability   → probabilidade por classe, como Vector: [P(0), P(1)]
└── prediction    → a classe escolhida (0.0 ou 1.0)
```

```python
# Extrair a probabilidade da classe positiva
display(
    predictions
    .withColumn("prob_churn", F.round(F.col("probability")[1], 4))
    .select("customer_id", "churn", "prediction", "prob_churn")
    .orderBy(F.col("prob_churn").desc())
    .limit(15)
)
```

> **Correção de um mito muito difundido:** você vai encontrar em vários materiais a afirmação de que "o `GBTClassifier` não produz a coluna `probability`". Isso **era** verdade no Spark 2.x. Desde o Spark 3.0 o `GBTClassifier` estende `ProbabilisticClassifier` e **produz sim** a coluna `probability`. Confira você mesmo:

```python
gbt = GBTClassifier(featuresCol="features", labelCol="churn",
                    maxIter=30, maxDepth=4, stepSize=0.1, seed=42)
gbt_model = gbt.fit(train_feat)
preds_gbt = gbt_model.transform(test_feat)

print("Colunas produzidas pelo GBT:", [c for c in preds_gbt.columns
                                        if c in ("rawPrediction", "probability", "prediction")])
# → ['rawPrediction', 'probability', 'prediction']
```

### Feature importance

```python
# featureImportances vem na ordem das colunas montadas pelo VectorAssembler.
# Para nomear corretamente, leia os metadados do vetor — não invente os nomes.
attrs = train_feat.schema["features"].metadata["ml_attr"]["attrs"]
nomes = [None] * train_feat.schema["features"].metadata["ml_attr"]["num_attrs"]
for tipo in attrs:
    for a in attrs[tipo]:
        nomes[a["idx"]] = a["name"]

importancias = sorted(zip(nomes, rf_model.featureImportances.toArray()),
                      key=lambda x: -x[1])
print("Top 8 features:")
for nome, imp in importancias[:8]:
    print(f"  {nome:40s}: {imp:.4f}")
```

### Regressão

```python
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Exemplo didático: prever monthly_charges (não tem valor de negócio real aqui)
gbt_reg = GBTRegressor(featuresCol="features", labelCol="monthly_charges",
                       maxIter=30, maxDepth=4, seed=42)
preds_reg = gbt_reg.fit(train_feat).transform(test_feat)

reg_eval = RegressionEvaluator(labelCol="monthly_charges", predictionCol="prediction")
print(f"RMSE: {reg_eval.setMetricName('rmse').evaluate(preds_reg):.4f}")
print(f"MAE : {reg_eval.setMetricName('mae').evaluate(preds_reg):.4f}")
print(f"R²  : {reg_eval.setMetricName('r2').evaluate(preds_reg):.4f}")
```

---

## Referência rápida dos avaliadores

> Bloco de consulta — `labelCol="label"` é ilustrativo (no projeto use `labelCol="churn"`).

```python
# CLASSIFICAÇÃO BINÁRIA — só duas métricas
from pyspark.ml.evaluation import BinaryClassificationEvaluator
BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")

# MULTICLASSE — serve para binária também
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# "accuracy" | "f1" | "weightedPrecision" | "weightedRecall"
# "weightedFMeasure" | "logLoss" | "truePositiveRateByLabel"

# REGRESSÃO
from pyspark.ml.evaluation import RegressionEvaluator
# "rmse" (default) | "mse" | "mae" | "r2" | "var"

# CLUSTERING
from pyspark.ml.evaluation import ClusteringEvaluator
# "silhouette" (default) — varia de -1 a 1, maior é melhor
```

---

## Pontos-chave para a prova

**Algoritmos**
- Interpretabilidade → Regressão Logística/Linear, Decision Tree
- Máxima performance em tabular → GBT / ensembles
- Sem rótulos, segmentar → KMeans | Recomendação → ALS | Contagens → GLM Poisson

**Desbalanceamento**
- **Class weights / cost-sensitive learning** é a resposta padrão
- Oversampling (SMOTE), undersampling, ajuste de threshold e troca de métrica também valem
- No Spark: `weightCol` com peso por linha (não existe `class_weight="balanced"`)

**Métricas**
- Accuracy só serve com classes balanceadas
- Precision → falso positivo caro | Recall → falso negativo caro | F1 → equilíbrio
- **Log Loss: menor é melhor**, exige probabilidades, penaliza confiança errada
- PR-AUC > ROC-AUC quando a classe positiva é muito rara
- RMSE na unidade do target e sensível a outliers; MAE robusta; R² comparável
- `BinaryClassificationEvaluator` só tem `areaUnderROC` e `areaUnderPR`

**Validação**
- CV: k modelos por combinação, estimativa estável, k× mais caro
- TVS: 1 modelo por combinação, mais rápido, menos confiável
- `CrossValidator` re-treina o `bestModel` com todos os dados de treino ao final

**Contagem**
- **modelos = combinações × folds**; combinações = produto das listas
- 3 × 2 × 3 params com 5 folds = **90** modelos

**Log transform**
- Exponencie (`exp` / `expm1`) **antes** de calcular a métrica ou interpretar a predição
- `log` ↔ `exp` | `log1p` ↔ `expm1`

**Viés-variância**
- Treino ruim + teste ruim = **underfitting** (viés alto)
- Treino ótimo + teste ruim = **overfitting** (variância alta)
- Mais dados reduz **variância**, não viés
- Bagging reduz variância; boosting reduz viés

**Spark ML**
- Exige coluna `features` do tipo **Vector**; `labelCol` numérico
- Classificadores adicionam `rawPrediction`, `probability` e `prediction`
- **`GBTClassifier` produz `probability`** desde o Spark 3 — a afirmação contrária está desatualizada

---

→ Próximo: [07_hyperopt.md](07_hyperopt.md)
