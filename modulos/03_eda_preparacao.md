# Módulo 3 — EDA e Preparação de Dados

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Seção oficial:** 2 — Data Processing (**19% da prova**)
>
> **Notebook sugerido:** `03_eda_preparacao`
>
> **Pré-requisito:** Módulo 2 (tabela `workspace.default.churn_clientes` criada).

Esta é a seção que mais gente ignora — e é quase 1 em cada 5 questões da prova. A boa notícia: quase tudo aqui é **conceito de estatística básica**, não API decorada. Se você entende *por que* usar mediana em vez de média, acerta a questão mesmo sem lembrar a sintaxe.

**Objetivos oficiais cobertos neste módulo:**
- Calcular estatísticas descritivas com `.summary()` ou dbutils data summaries
- Remover outliers com base em desvio padrão ou IQR
- Criar visualizações para features categóricas ou contínuas
- Comparar duas features categóricas ou duas contínuas
- Comparar e contrastar imputação com média, mediana ou moda
- Imputar valores faltantes com moda, média ou mediana
- Identificar quando one-hot encoding **não** é apropriado
- Identificar quando a transformação logarítmica é apropriada

---

## 3.1 Estatísticas descritivas

Existem três formas de obter estatísticas descritivas no Databricks. **A prova cobra os nomes e as diferenças entre elas.**

```python
from pyspark.sql import functions as F

df = spark.table("workspace.default.churn_clientes")

# --- 1. describe() — o básico: count, mean, stddev, min, max ---
display(df.describe())

# --- 2. summary() — describe() + os quartis (25%, 50%, 75%) ---
display(df.summary())
# → count, mean, stddev, min, 25%, 50% (mediana), 75%, max

# --- 3. summary() com percentis customizados ---
display(df.summary("count", "min", "10%", "50%", "90%", "max"))
```

| Método | O que retorna | Quando usar |
|---|---|---|
| `df.describe()` | count, mean, stddev, min, max | Visão rápida |
| `df.summary()` | tudo do describe **+ quartis 25%/50%/75%** | Padrão para EDA — você precisa dos quartis para IQR |
| `df.summary("min", "90%")` | apenas as estatísticas pedidas | Percentis específicos |
| `dbutils.data.summarize(df)` | relatório visual interativo | Exploração na UI |

> **Pegadinha da prova:** a diferença entre `describe()` e `summary()` é que **`summary()` inclui os quartis**. Se a questão pede mediana ou os dados para calcular IQR, a resposta é `summary()`.

```python
# --- dbutils.data.summarize() — relatório visual completo ---
# Gera um painel com distribuição, nulos, cardinalidade e valores mais frequentes
# de TODAS as colunas de uma vez. Não retorna DataFrame — renderiza na tela.
dbutils.data.summarize(df)
```

`dbutils.data.summarize()` mostra, para cada coluna: tipo, contagem de nulos, número de valores distintos, um histograma e as estatísticas descritivas. É o caminho mais rápido para uma primeira olhada em um dataset desconhecido.

```python
# --- Contagem de nulos por coluna ---
# ATENÇÃO: isnan() só funciona em colunas numéricas (float/double).
# Aplicá-la em coluna de texto causa erro. A forma segura é checar o tipo:
from pyspark.sql.types import DoubleType, FloatType

def contar_nulos(df):
    exprs = []
    for campo in df.schema.fields:
        c = F.col(campo.name)
        cond = c.isNull()
        if isinstance(campo.dataType, (DoubleType, FloatType)):
            cond = cond | F.isnan(c)          # NaN só existe em float/double
        exprs.append(F.count(F.when(cond, campo.name)).alias(campo.name))
    return df.select(exprs)

display(contar_nulos(df))
```

```python
# --- Cardinalidade: quantos valores distintos cada coluna categórica tem ---
# Isso decide se one-hot encoding é viável (ver seção 3.7)
for c in ["contract_type", "internet_service", "payment_method", "customer_id"]:
    print(f"{c:20s}: {df.select(c).distinct().count()} valores distintos")
```

---

## 3.2 Detectar e remover outliers

A prova cobra **dois métodos**, e cobra saber qual usar em cada situação.

### Método 1 — Desvio padrão (regra dos 3 sigmas)

Considera outlier tudo que está a mais de N desvios padrão da média (N = 3 é o padrão).

```
limite_inferior = média − 3 × desvio_padrão
limite_superior = média + 3 × desvio_padrão
```

```python
from pyspark.sql import functions as F

df = spark.table("workspace.default.churn_clientes")
COLUNA = "total_charges"

# Calcular média e desvio padrão em uma única passada
stats = df.select(
    F.mean(COLUNA).alias("media"),
    F.stddev(COLUNA).alias("desvio")
).first()

limite_inf = stats["media"] - 3 * stats["desvio"]
limite_sup = stats["media"] + 3 * stats["desvio"]

print(f"Média: {stats['media']:.2f} | Desvio: {stats['desvio']:.2f}")
print(f"Faixa aceita: [{limite_inf:.2f}, {limite_sup:.2f}]")

df_sem_outliers_std = df.filter(
    (F.col(COLUNA) >= limite_inf) & (F.col(COLUNA) <= limite_sup)
)

print(f"Antes: {df.count()} | Depois: {df_sem_outliers_std.count()}")
```

### Método 2 — IQR (intervalo interquartil)

Considera outlier tudo fora de 1.5 × IQR além dos quartis. **Este é o método por trás do boxplot.**

```
IQR = Q3 − Q1
limite_inferior = Q1 − 1.5 × IQR
limite_superior = Q3 + 1.5 × IQR
```

```python
# approxQuantile calcula quantis de forma distribuída e eficiente
# Assinatura: approxQuantile(coluna, [lista_de_quantis], erro_relativo)
q1, q3 = df.approxQuantile(COLUNA, [0.25, 0.75], 0.0)   # 0.0 = exato (mais lento)
iqr = q3 - q1

limite_inf = q1 - 1.5 * iqr
limite_sup = q3 + 1.5 * iqr

print(f"Q1: {q1:.2f} | Q3: {q3:.2f} | IQR: {iqr:.2f}")
print(f"Faixa aceita: [{limite_inf:.2f}, {limite_sup:.2f}]")

df_sem_outliers_iqr = df.filter(
    (F.col(COLUNA) >= limite_inf) & (F.col(COLUNA) <= limite_sup)
)

print(f"Antes: {df.count()} | Depois: {df_sem_outliers_iqr.count()}")
```

```python
# --- Em vez de remover: marcar os outliers como uma feature ---
# Muitas vezes o outlier é informativo (fraude, cliente atípico) e remover perde sinal
df_marcado = df.withColumn(
    "eh_outlier",
    F.when(
        (F.col(COLUNA) < limite_inf) | (F.col(COLUNA) > limite_sup), 1
    ).otherwise(0)
)
display(df_marcado.groupBy("eh_outlier").agg(
    F.count("*").alias("clientes"),
    F.round(F.mean("churn") * 100, 1).alias("taxa_churn_%")
))
```

### Qual método usar?

| | Desvio padrão (3σ) | IQR (1.5×) |
|---|---|---|
| **Pressupõe distribuição normal?** | Sim | Não |
| **Sensível aos próprios outliers?** | **Sim** — média e desvio são puxados pelos extremos | Não — quartis são robustos |
| **Distribuição assimétrica** | Ruim | Bom |
| **Distribuição simétrica/normal** | Bom | Bom |
| **Base do boxplot** | Não | Sim |

> **Pegadinha da prova:** a média e o desvio padrão são *eles mesmos* distorcidos por outliers extremos. Por isso o **IQR é a escolha mais segura** quando você não sabe a forma da distribuição, ou quando ela é claramente assimétrica.

---

## 3.3 Visualizações

A prova não pede para você escrever código de gráfico — pede para **escolher o gráfico certo para o tipo de dado**.

| Tipo de dado | Gráfico apropriado | Para quê |
|---|---|---|
| **1 feature contínua** | Histograma, boxplot, density plot | Ver distribuição, assimetria, outliers |
| **1 feature categórica** | Gráfico de barras, contagem | Ver frequência de cada categoria |
| **2 contínuas** | Scatter plot | Ver relação/correlação |
| **2 categóricas** | Barras agrupadas ou empilhadas, heatmap | Ver associação |
| **1 categórica + 1 contínua** | Boxplot por categoria, barras com média | Comparar distribuições entre grupos |
| **Muitas contínuas** | Matriz de correlação (heatmap), pair plot | Ver relações em bloco |

```python
# --- No Databricks: display() gera gráficos sem código ---
# Rode e clique no ícone de gráfico abaixo da tabela para escolher o tipo
display(df.select("tenure_months", "monthly_charges", "total_charges", "churn"))
```

```python
# --- Histograma de uma feature contínua (matplotlib) ---
import matplotlib.pyplot as plt

# Amostra pequena para o driver — nunca traga um dataset grande inteiro
pdf = df.select("monthly_charges", "total_charges", "contract_type", "churn").toPandas()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(pdf["monthly_charges"], bins=30, edgecolor="white")
ax[0].set_title("monthly_charges — distribuição")
ax[1].boxplot(pdf["total_charges"], vert=False)
ax[1].set_title("total_charges — boxplot (mostra os outliers pelo IQR)")
plt.tight_layout()
plt.show()
```

```python
# --- Barras: feature categórica ---
contagem = pdf["contract_type"].value_counts()
plt.figure(figsize=(6, 3.5))
plt.bar(contagem.index, contagem.values)
plt.title("contract_type — frequência por categoria")
plt.xticks(rotation=15)
plt.show()
```

```python
# --- Scatter: duas features contínuas ---
plt.figure(figsize=(6, 4))
plt.scatter(pdf["tenure_months"] if "tenure_months" in pdf else pdf["monthly_charges"],
            pdf["total_charges"], alpha=0.3, s=10)
plt.xlabel("monthly_charges")
plt.ylabel("total_charges")
plt.title("Relação entre duas features contínuas")
plt.show()
```

> **Pegadinha da prova:** boxplot serve para **uma variável contínua** (opcionalmente quebrada por categoria). Scatter plot serve para **duas contínuas**. Barra serve para **categórica**. Um enunciado do tipo "quer visualizar a distribuição de uma variável contínua e identificar outliers" tem como resposta **boxplot** — porque o boxplot já desenha os limites de 1.5×IQR.

---

## 3.4 Comparar duas features

Este objetivo aparece na prova como: "um cientista de dados quer comparar X e Y. Qual método é apropriado?". **A resposta depende do tipo das duas variáveis.**

| Combinação | Método apropriado | Estatística |
|---|---|---|
| **contínua × contínua** | Correlação + scatter plot | Pearson (linear) ou Spearman (monotônica) |
| **categórica × categórica** | Tabela de contingência (crosstab) | Teste **qui-quadrado** (chi-square) |
| **categórica × contínua** | Média/mediana da contínua por categoria, boxplot | Teste t (2 grupos) ou ANOVA (3+ grupos) |

### Contínua × contínua

```python
# Correlação de Pearson: mede relação LINEAR, varia de -1 a 1
print("Pearson :", df.stat.corr("tenure_months", "total_charges"))

# Spearman: mede relação MONOTÔNICA (não precisa ser linear)
print("Spearman:", df.stat.corr("tenure_months", "total_charges", method="spearman"))

# Matriz de correlação de várias colunas de uma vez
colunas_num = ["tenure_months", "monthly_charges", "total_charges", "churn"]
for i, a in enumerate(colunas_num):
    for b in colunas_num[i + 1:]:
        print(f"{a:16s} ↔ {b:16s}: {df.stat.corr(a, b):+.3f}")
```

> **Pegadinha:** correlação de Pearson **próxima de zero não significa "sem relação"** — significa "sem relação *linear*". Uma relação em U tem Pearson ≈ 0. Por isso o scatter plot acompanha a correlação.

### Categórica × categórica

```python
# Tabela de contingência: quantos clientes em cada combinação
display(df.stat.crosstab("contract_type", "internet_service"))

# Mesma coisa com groupBy + pivot, em formato mais legível
display(
    df.groupBy("contract_type")
      .pivot("internet_service")
      .count()
)
```

```python
# --- Teste qui-quadrado: as duas categóricas são independentes? ---
# H0: as variáveis são independentes. p-valor < 0.05 → há associação.
from scipy.stats import chi2_contingency
import pandas as pd

tabela = (df.stat.crosstab("contract_type", "internet_service")
            .toPandas()
            .set_index("contract_type_internet_service"))

chi2, p_valor, graus_lib, _ = chi2_contingency(tabela.values)
print(f"Qui-quadrado: {chi2:.2f} | p-valor: {p_valor:.4f}")
print("→ Há associação entre as variáveis" if p_valor < 0.05
      else "→ Não há evidência de associação")
```

> O Spark ML também tem `ChiSquareTest` (em `pyspark.ml.stat`), usado para testar features categóricas contra o label. Ele espera uma coluna `features` do tipo Vector — ou seja, é usado *depois* do `VectorAssembler`, não em colunas de texto cruas.

### Categórica × contínua

```python
# Comparar a distribuição de uma contínua entre categorias
display(
    df.groupBy("contract_type").agg(
        F.count("*").alias("n"),
        F.round(F.mean("monthly_charges"), 2).alias("media"),
        F.round(F.expr("percentile_approx(monthly_charges, 0.5)"), 2).alias("mediana"),
        F.round(F.stddev("monthly_charges"), 2).alias("desvio"),
    ).orderBy("media", ascending=False)
)
```

---

## 3.5 Imputação: média, mediana ou moda

Este é um objetivo **conceitual** — a prova pergunta *qual escolher*, não como codificar.

| Estratégia | Use quando | Cuidado |
|---|---|---|
| **Média** (`mean`) | Feature **numérica** com distribuição aproximadamente **simétrica**, sem outliers fortes | A média é puxada por outliers — imputar com ela distorce os dados |
| **Mediana** (`median`) | Feature **numérica** com distribuição **assimétrica** ou com outliers | Escolha mais segura por padrão em dados reais |
| **Moda** (`mode`) | Feature **categórica** (ou numérica discreta, como "número de filhos") | Concentra ainda mais massa na categoria já dominante |

**Regra prática para a prova:**
- Numérica + simétrica → **média**
- Numérica + assimétrica ou com outliers → **mediana**
- Categórica → **moda**

> **Pegadinha da prova (esta é a Questão 2 do exam guide oficial):** o enunciado diz "quer imputar com o **mínimo de esforço, mas com o resultado correto**". A resposta certa é **examinar a distribuição e escolher a imputação adequada** — não existe atalho automático que acerte sozinho. `SimpleImputer` do sklearn **não** escolhe a estratégia por você (o default é `mean`), e "média é sempre o correto para contínuas" é falso.

**Outras estratégias que podem aparecer como distratores:**

| Estratégia | Comentário |
|---|---|
| **Remover as linhas** (`dropna`) | Válido se os nulos forem poucos e aleatórios. Perde dados e pode enviesar se os nulos não forem aleatórios. |
| **Preencher com constante** (0, "Desconhecido") | Útil quando o nulo *significa* algo (ex.: "não contratou o serviço"). |
| **Criar flag de nulo** | Adicionar uma coluna `foi_imputado` preserva a informação de que o valor estava faltando. Frequentemente é a melhor prática. |
| **Imputação por modelo** (KNN, iterativa) | Mais preciso, muito mais caro. Raramente é a resposta em prova de nível Associate. |

---

## 3.6 Imputação na prática

```python
from pyspark.ml.feature import Imputer
from pyspark.sql import functions as F

df = spark.table("workspace.default.churn_clientes")

# Criar nulos artificiais para o exercício (o dataset original não tem)
import random
random.seed(0)
ids_nulos_num = [f"C{i:04d}" for i in random.sample(range(1, 1001), 50)]
ids_nulos_cat = [f"C{i:04d}" for i in random.sample(range(1, 1001), 30)]

df_com_nulos = (
    df
    .withColumn("monthly_charges",
        F.when(F.col("customer_id").isin(ids_nulos_num), None)
         .otherwise(F.col("monthly_charges")))
    .withColumn("contract_type",
        F.when(F.col("customer_id").isin(ids_nulos_cat), None)
         .otherwise(F.col("contract_type")))
)

print("Nulos em monthly_charges:", df_com_nulos.filter(F.col("monthly_charges").isNull()).count())
print("Nulos em contract_type :", df_com_nulos.filter(F.col("contract_type").isNull()).count())
```

### Colunas numéricas — `Imputer`

```python
# Imputer é um ESTIMATOR: fit() aprende a estatística, transform() aplica
imputer = Imputer(
    inputCols=["monthly_charges", "total_charges"],
    outputCols=["monthly_charges_imp", "total_charges_imp"],
    strategy="median",     # "mean" (default) | "median" | "mode"
)

imputer_model = imputer.fit(df_com_nulos)     # aprende a mediana
df_imputado = imputer_model.transform(df_com_nulos)

# O valor aprendido fica disponível no modelo
display(imputer_model.surrogateDF)   # mostra o valor usado para cada coluna

print("Nulos após imputação:",
      df_imputado.filter(F.col("monthly_charges_imp").isNull()).count())
```

> **`Imputer` só aceita colunas numéricas.** Em coluna de texto ele lança erro. Isso vale inclusive para `strategy="mode"`.

### Colunas categóricas — moda calculada e `fillna`

```python
# Calcular a moda: categoria mais frequente, ignorando nulos
moda = (df_com_nulos
        .filter(F.col("contract_type").isNotNull())
        .groupBy("contract_type")
        .count()
        .orderBy(F.desc("count"))
        .first()["contract_type"])

print(f"Moda de contract_type: {moda}")

df_imputado = df_imputado.fillna({"contract_type": moda})
print("Nulos restantes:", df_imputado.filter(F.col("contract_type").isNull()).count())
```

### A regra que vale ponto na prova

```python
# ERRADO: imputar antes do split — a mediana é calculada com dados de teste
df_imputado_errado = imputer.fit(df_com_nulos).transform(df_com_nulos)
train, test = df_imputado_errado.randomSplit([0.8, 0.2], seed=42)   # ← data leakage

# CORRETO: split primeiro, fit só no treino
train, test = df_com_nulos.randomSplit([0.8, 0.2], seed=42)
imputer_model = imputer.fit(train)          # aprende a mediana APENAS do treino
train_imp = imputer_model.transform(train)
test_imp  = imputer_model.transform(test)   # aplica a mediana do treino no teste
```

> Colocar o `Imputer` dentro de um `Pipeline` (Módulo 5) garante isso automaticamente.

---

## 3.7 Quando one-hot encoding é má ideia

O objetivo oficial é literalmente "identificar e explicar os tipos de modelo ou datasets para os quais o one-hot encoding **é ou não** apropriado". Ou seja: a prova quer que você saiba quando **não** usar.

### Quando OHE é apropriado

- **Modelos lineares** (Regressão Linear, Regressão Logística, SVM linear) — eles interpretam números como grandeza, então uma coluna com valores 0/1/2 seria lida como "2 é o dobro de 1". OHE resolve isso.
- **Redes neurais** — mesma lógica.
- **Variáveis nominais** (sem ordem natural): cor, cidade, método de pagamento.
- **Baixa cardinalidade** (poucas categorias distintas).

### Quando OHE é má ideia

| Situação | Por quê | O que fazer |
|---|---|---|
| **Modelos baseados em árvore** (Random Forest, GBT, Decision Tree) | Árvores fazem splits por valor, não precisam de OHE. Além disso, OHE **espalha o sinal** de uma variável por várias colunas binárias esparsas, o que enfraquece cada split e piora a performance | Usar apenas `StringIndexer` (índice numérico) |
| **Alta cardinalidade** (ex.: CEP, ID de produto, milhares de categorias) | Gera milhares de colunas: explosão de dimensionalidade, memória e tempo de treino | Target/mean encoding, hashing, embeddings, ou agrupar categorias raras |
| **Variáveis ordinais** (baixo < médio < alto; ruim < bom < ótimo) | OHE **destrói a ordem**, que é informação útil | Ordinal encoding com a ordem correta |
| **Datasets muito pequenos** | Muitas colunas com poucas linhas → overfitting | Reduzir cardinalidade antes |

> **Pegadinha da prova:** "Modelo de árvore + variável categórica com muitas categorias" é a combinação clássica em que **OHE é a resposta errada**. Árvores lidam bem com o índice numérico direto do `StringIndexer`.

> **Um detalhe que também cai:** `OneHotEncoder(dropLast=True)` (o padrão no Spark) descarta a última categoria. Isso evita **multicolinearidade** (a "armadilha da variável dummy") em modelos lineares — com N categorias, N−1 colunas já carregam toda a informação.

O código de OHE está no [Módulo 5 — Feature Engineering](05_feature_engineering.md#53-one-hot-encoding).

---

## 3.8 Transformação logarítmica

### Quando aplicar log

| Cenário | Por quê |
|---|---|
| **Distribuição com assimetria à direita** (cauda longa nos valores altos) | O log comprime a cauda e aproxima a distribuição de uma normal — o que modelos lineares assumem |
| **Variável que cresce multiplicativamente** (renda, preço, população, receita) | No espaço log, crescimento percentual vira crescimento aditivo |
| **Features em ordens de grandeza muito diferentes** (1 a 10.000.000) | O log reduz a amplitude e evita que uma feature domine as outras |
| **Relação multiplicativa entre variáveis** | O log transforma a relação em linear, que modelos lineares capturam |
| **Reduzir o impacto de outliers** | Sem removê-los: valores extremos ficam bem mais próximos do resto |

### Quando **não** aplicar

- Dados com **zeros ou negativos** — `log(0)` é indefinido e `log(negativo)` não existe. Use `log1p(x)` = `log(1+x)` para dados com zeros, ou desloque a variável.
- **Distribuição já simétrica** — não ganha nada e piora a interpretabilidade.
- **Modelos baseados em árvore** — são invariantes a transformações monotônicas. Aplicar log em uma feature **não muda os splits** de uma árvore. Não faz mal, mas não ajuda.

```python
from pyspark.sql import functions as F

df = spark.table("workspace.default.churn_clientes")

# log natural, e log1p para o caso de haver zeros
df_log = (df
    .withColumn("total_charges_log",  F.log(F.col("total_charges")))
    .withColumn("total_charges_log1p", F.log1p(F.col("total_charges")))
)

# Comparar a assimetria antes e depois
display(df_log.select(
    F.round(F.skewness("total_charges"), 3).alias("assimetria_original"),
    F.round(F.skewness("total_charges_log"), 3).alias("assimetria_log"),
))
# Assimetria próxima de 0 = distribuição simétrica
```

### A parte que a prova cobra de verdade

Se você aplica log **no target** (na variável que o modelo prevê), o modelo passa a prever valores em escala logarítmica. Antes de calcular RMSE, MAE ou interpretar a predição, **você precisa desfazer o log**:

```python
import numpy as np

# Treinou com y_log = log(y)
# O modelo prevê pred_log — que está em escala de log, NÃO em reais/unidades
pred_log = np.array([7.6, 8.1, 6.9])

# ERRADO: calcular RMSE direto sobre pred_log
# → o número resultante não tem significado na unidade original

# CORRETO: exponenciar de volta antes de avaliar
pred_original = np.exp(pred_log)          # inverso de log()
# Se você usou log1p, o inverso é expm1():
# pred_original = np.expm1(pred_log)

print("Predição em escala log     :", pred_log)
print("Predição na unidade original:", np.round(pred_original, 2))
```

> **Pegadinha da prova (objetivo oficial explícito):** um RMSE de 0.35 calculado em escala log **não é** um erro de R$ 0,35. Sempre exponencie (`exp` ou `expm1`) antes de calcular a métrica ou de reportar a predição para o negócio. Isso é revisitado no [Módulo 6](06_treinamento_modelos.md#69-target-log-transformado-o-erro-que-quase-todo-mundo-comete).

---

## Exercício prático

```python
from pyspark.sql import functions as F
from pyspark.ml.feature import Imputer

df = spark.table("workspace.default.churn_clientes")

# 1. Estatísticas com quartis
display(df.summary())

# 2. Detectar outliers em total_charges pelos DOIS métodos e comparar
q1, q3 = df.approxQuantile("total_charges", [0.25, 0.75], 0.0)
iqr = q3 - q1
lim_iqr = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

s = df.select(F.mean("total_charges").alias("m"), F.stddev("total_charges").alias("d")).first()
lim_std = (s["m"] - 3 * s["d"], s["m"] + 3 * s["d"])

n_iqr = df.filter((F.col("total_charges") < lim_iqr[0]) | (F.col("total_charges") > lim_iqr[1])).count()
n_std = df.filter((F.col("total_charges") < lim_std[0]) | (F.col("total_charges") > lim_std[1])).count()

print(f"IQR  → faixa [{lim_iqr[0]:.0f}, {lim_iqr[1]:.0f}] | outliers: {n_iqr}")
print(f"3σ   → faixa [{lim_std[0]:.0f}, {lim_std[1]:.0f}] | outliers: {n_std}")
print("\nPor que os números diferem? Veja a assimetria:")
display(df.select(F.round(F.skewness("total_charges"), 3).alias("assimetria")))

# 3. Comparar duas categóricas
display(df.stat.crosstab("contract_type", "tech_support"))

# 4. Comparar duas contínuas
print("Correlação tenure ↔ total_charges:", round(df.stat.corr("tenure_months", "total_charges"), 3))
```

**Pergunte-se ao final:**
- O IQR encontrou mais ou menos outliers que a regra dos 3σ? Por quê?
- `total_charges` é assimétrica? Isso muda sua escolha entre média e mediana para imputar?

---

## Pontos-chave para a prova

- `describe()` = count, mean, stddev, min, max — **sem quartis**
- `summary()` = describe **+ 25%, 50%, 75%** — é o que você usa para IQR
- `dbutils.data.summarize(df)` = relatório visual completo, todas as colunas
- **IQR:** `Q1 − 1.5×IQR` e `Q3 + 1.5×IQR` — robusto, base do boxplot
- **Desvio padrão:** `média ± 3×desvio` — pressupõe normalidade e é distorcido pelos próprios outliers
- `approxQuantile(col, [0.25, 0.75], erro)` → calcula quartis distribuídos
- **2 contínuas** → correlação + scatter | **2 categóricas** → crosstab + qui-quadrado | **1 de cada** → boxplot por grupo
- Correlação de Pearson mede relação **linear**; zero não significa "sem relação"
- Imputação: **simétrica → média**, **assimétrica/outliers → mediana**, **categórica → moda**
- `Imputer` do Spark ML aceita **só colunas numéricas** (mean/median/mode)
- Sempre **fit do imputer só no treino** — imputar antes do split é data leakage
- OHE é **ruim** para: modelos de árvore, alta cardinalidade, variáveis ordinais
- `dropLast=True` no OHE evita multicolinearidade em modelos lineares
- Log serve para: **assimetria à direita**, crescimento multiplicativo, ordens de grandeza distintas
- `log1p`/`expm1` para dados com zero; **árvores são indiferentes ao log**
- **Target log-transformado: exponencie antes de calcular a métrica**

---

→ Próximo: [04_mlflow.md](04_mlflow.md)
