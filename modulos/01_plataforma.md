# Módulo 1 — Plataforma Databricks

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Seção oficial:** 1 — Databricks Machine Learning (**38% da prova**)
>
> **Notebook sugerido:** `01_plataforma_databricks`
>
> **O que você aprende:** compute, Databricks Runtime ML, Unity Catalog, Delta Lake e dbutils — a base de tudo o que vem depois.

**Objetivo oficial coberto neste módulo:**
- Identificar as vantagens de usar os ML runtimes

---

## 1.1 Compute: driver e workers

```
Compute = o conjunto de máquinas que executa seu código Spark.

┌─────────────────────────────────────┐
│           DRIVER NODE               │  ← seu código roda aqui
│  (orquestra, monta o plano, coleta) │
└──────────────┬──────────────────────┘
               │ distribui tarefas
    ┌──────────┼──────────┐
    ▼          ▼          ▼
 Worker 1   Worker 2   Worker 3      ← processam partições em paralelo
```

| Nó | Papel |
|---|---|
| **Driver** | Executa seu código Python, constrói o plano de execução (DAG), distribui as tarefas e recebe os resultados. É para a memória **dele** que `collect()` e `toPandas()` trazem os dados |
| **Workers** | Processam as partições dos dados em paralelo. Nunca executam seu código Python de nível superior — só as tarefas que o driver distribui |

> **Por que isso cai na prova:** entender que o driver é uma máquina só explica por que `collect()` e `toPandas()` estouram a memória em datasets grandes, e por que `pandas_udf` (que roda nos workers) é a alternativa correta.

---

## 1.2 Tipos de compute

| Tipo | Uso | Custo | Quando usar |
|---|---|---|---|
| **All-Purpose** | Notebooks interativos | Maior — fica ligado esperando você | Desenvolvimento e exploração |
| **Job Cluster** | Jobs agendados | Menor — criado e destruído automaticamente | **Produção** |
| **Serverless** | Notebooks e jobs, sem configuração | Pago por uso, sem tempo ocioso | Padrão atual; **único disponível na Free Edition** |

> **Na prova:** para produção agendada, **Job Cluster** é a resposta clássica — ele nasce no início do job e é destruído ao final, então você paga só o tempo de execução. All-Purpose fica ligado acumulando custo mesmo sem ninguém usando.

> ⚠️ **Na Free Edition você não cria nem configura clusters.** Só existe compute **Serverless**: o Databricks decide o hardware por trás dos panos, sem tela de "Create Cluster", sem escolher tamanho de máquina, sem esperar o cluster subir. A tabela acima continua sendo teoria de prova.

---

## 1.3 Databricks Runtime e Runtime ML

> **Analogia:** se o **compute** é o hardware (quantas máquinas, de que tamanho), o **Databricks Runtime** é o sistema operacional + programas já instalados nelas. Você escolhe as duas coisas separadamente.

```
Databricks Runtime          → Python + Spark + Delta Lake
Databricks Runtime ML       → tudo acima + MLflow, scikit-learn, XGBoost,
                              LightGBM, PyTorch, TensorFlow, databricks-feature-engineering...
Databricks Runtime ML (GPU) → tudo acima + CUDA, cuDNN e drivers de GPU
```

### As vantagens do Runtime ML — objetivo oficial

| Vantagem | Detalhe |
|---|---|
| **Bibliotecas de ML já instaladas** | MLflow, scikit-learn, XGBoost, LightGBM, PyTorch, TensorFlow — sem `pip install` a cada cluster |
| **Versões testadas e compatíveis entre si** | O Databricks valida o conjunto. Você não perde horas resolvendo conflito de dependência entre numpy, scipy e sklearn |
| **Inicialização mais rápida** | As bibliotecas já vêm na imagem; instalar tudo manualmente adiciona minutos a cada start |
| **Integração pronta com MLflow** | Tracking configurado automaticamente para o workspace, sem apontar URI |
| **Suporte a GPU pré-configurado** | A variante GPU já traz CUDA e drivers compatíveis com os frameworks |
| **Habilita recursos da plataforma** | AutoML (classificação/regressão) exige compute clássico com Runtime ML |
| **Reprodutibilidade** | Fixar a versão do runtime fixa o ambiente inteiro — o mesmo código roda igual meses depois |

> **A pegadinha:** "por que usar Runtime ML em vez do Runtime padrão + `pip install`?" A resposta não é só "conveniência" — é **compatibilidade validada entre versões** e **tempo de inicialização**. Instalar manualmente funciona, mas você assume o risco de conflitos e paga o custo de instalação a cada cluster novo.

> No **Serverless** não existe seletor de versão de Runtime — o Databricks mantém o ambiente atualizado automaticamente, já com as bibliotecas de ML. Em contrapartida, algumas coisas não estão lá: `hyperopt` (Módulo 7) e `SparkTrials` são os casos que você vai encontrar neste guia.

---

## 1.4 Unity Catalog

O Unity Catalog organiza e governa dados, features e modelos em **três níveis**:

```
catalog  →  schema  →  objeto
   │           │         ├── tabela   (dados estruturados)
   │           │         ├── volume   (arquivos crus)
   │           │         ├── modelo   (MLflow Registry)
   │           │         └── função   (UDF registrada)
   │           └── equivale a "database" no mundo relacional
   └── o nível mais alto, compartilhado na conta
```

Toda conta Free Edition já vem com o catalog `workspace` e o schema `default` prontos — não é preciso criar nada para começar.

### Tabela gerenciada

```python
# Tabela gerenciada: você não lida com paths, só com o nome de 3 níveis
df_exemplo = spark.createDataFrame([("C001", 0), ("C002", 1)], ["customer_id", "churn"])

df_exemplo.write.format("delta").mode("overwrite").saveAsTable("workspace.default.exemplo_uc")

display(spark.table("workspace.default.exemplo_uc"))
# → deve mostrar as 2 linhas que você acabou de escrever
```

### Volume — para arquivos crus

```sql
-- Rodar uma vez (célula %sql ou spark.sql)
CREATE VOLUME IF NOT EXISTS workspace.default.churn_project;
```

O volume começa **vazio**. Os comandos abaixo criam um arquivo de teste e depois o manipulam, para você ver o efeito de cada operação:

```python
CAMINHO = "/Volumes/workspace/default/churn_project"

# 0. Criar um arquivo de teste (senão não há nada para listar/copiar/mover)
dbutils.fs.put(f"{CAMINHO}/origem.csv", "customer_id,churn\nC001,0\nC002,1\n", overwrite=True)

# 1. Listar → deve aparecer "origem.csv"
display(dbutils.fs.ls(CAMINHO))

# 2. Criar pasta → cria "data/", ainda vazia
dbutils.fs.mkdirs(f"{CAMINHO}/data/")

# 3. Copiar → agora existem "origem.csv" E "destino.csv"
dbutils.fs.cp(f"{CAMINHO}/origem.csv", f"{CAMINHO}/destino.csv")

# 4. Mover → "destino.csv" sai da raiz e vira "data/arquivo.csv"
dbutils.fs.mv(f"{CAMINHO}/destino.csv", f"{CAMINHO}/data/arquivo.csv")

# 5. Deletar um arquivo
dbutils.fs.rm(f"{CAMINHO}/origem.csv")

# 6. Deletar uma pasta inteira (recursivo)
dbutils.fs.rm(f"{CAMINHO}/data/", recurse=True)

# Conferir que o volume voltou a ficar vazio
display(dbutils.fs.ls(CAMINHO))
```

| Recurso | Para quê | Substitui |
|---|---|---|
| **Tabela gerenciada** | DataFrames como Delta, sem lidar com paths | `df.write.save(path)` em `dbfs:/` |
| **Volume** | Arquivos crus: CSV, imagens, checkpoints de streaming | `dbfs:/FileStore/` |

---

## 1.5 DBFS — teoria de prova

DBFS é o sistema de arquivos distribuído do Databricks, montado sobre o object storage da cloud. Persiste entre sessões (diferente do disco local do driver) e é acessível de todos os clusters do workspace. Caminhos começam com `dbfs:/`, e `dbfs:/FileStore/` era a área tradicional de upload.

```
dbutils.fs.ls("dbfs:/")             → listar arquivos
dbutils.fs.mkdirs("dbfs:/caminho")  → criar pasta
dbutils.fs.cp / mv / rm             → copiar, mover, remover
dbfs:/FileStore/                    → área tradicional de upload
```

| | **DBFS** (legado) | **Unity Catalog** (atual) |
|---|---|---|
| O que é | Um sistema de arquivos por workspace | Catálogo governado: `catalog.schema.objeto` e `/Volumes/...` |
| Como você referencia | Path (`dbfs:/pasta/arquivo.csv`) | Nome lógico ou path de Volume |
| Permissões | Por workspace, pouco granular | Granular por catalog/schema/objeto |
| Entre workspaces | Não | Sim |
| Neste guia | Só teoria | O que você usa na prática |

> ⚠️ **Na Free Edition o "public DBFS root" vem desabilitado.** `dbutils.fs.ls("dbfs:/")` ainda lista três mounts especiais (`Volumes/`, `Workspace/`, `databricks-datasets/`), mas **escrever** fora deles retorna `DBFS_DISABLED: Public DBFS root is disabled`. Isso não é bug — é a direção da plataforma, migrando armazenamento de arquivos para o Unity Catalog. Guarde o conceito de DBFS para a prova; use Unity Catalog na prática.

---

## 1.6 Delta Lake

Delta Lake é o formato padrão de armazenamento no Databricks. Pense nele como Parquet com superpoderes.

```
Delta Lake resolve os problemas do CSV/Parquet puro:
├── Transações ACID    → escrita atômica, sem corrupção em falhas
├── Time Travel        → ler qualquer versão anterior da tabela
├── Schema enforcement → rejeita dados que não batem com o schema
├── Schema evolution   → permite evoluir o schema de forma controlada
└── Otimizações        → Z-ordering, compaction, vacuum
```

> O exemplo abaixo cria um dataset pequeno **só para praticar Delta Lake**. O dataset real do projeto (`churn_clientes`) é criado no Módulo 2.

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

schema = StructType([
    StructField("customer_id",     StringType(),  False),
    StructField("tenure_months",   IntegerType(), True),
    StructField("monthly_charges", DoubleType(),  True),
    StructField("churn",           IntegerType(), False),
])

dados = [
    ("C001", 12,  65.50, 1),
    ("C002", 24,  89.99, 0),
    ("C003",  3,  45.00, 1),
    ("C004", 60, 110.00, 0),
]

df = spark.createDataFrame(dados, schema)

TABELA = "workspace.default.demo_delta"   # nome de demonstração
df.write.format("delta").mode("overwrite").saveAsTable(TABELA)

display(spark.table(TABELA))
```

```python
# --- Adicionar dados: cria uma nova versão ---
novos = spark.createDataFrame([("C005", 8, 55.0, 1), ("C006", 36, 95.0, 0)], schema)
novos.write.format("delta").mode("append").saveAsTable(TABELA)

# --- Time Travel ---
display(spark.sql(f"DESCRIBE HISTORY {TABELA}"))    # lista todas as versões

df_v0 = spark.read.format("delta").option("versionAsOf", 0).table(TABELA)
print(f"Versão 0: {df_v0.count()} registros")
print(f"Versão atual: {spark.table(TABELA).count()} registros")

# Por timestamp, em vez de por versão:
# df_ts = spark.read.format("delta").option("timestampAsOf", "2024-01-01").table(TABELA)
```

```python
# --- Consultar com SQL (saveAsTable já registrou a tabela) ---
display(spark.sql(f"SELECT * FROM {TABELA} WHERE churn = 1"))
```

| Modo de escrita | Efeito |
|---|---|
| `"overwrite"` | Substitui todo o conteúdo da tabela |
| `"append"` | Adiciona linhas, preservando as existentes |
| `"errorifexists"` (default) | Falha se a tabela já existir |
| `"ignore"` | Não faz nada se a tabela já existir |

---

## 1.7 dbutils

```python
# dbutils.fs       → operações de arquivo (visto na seção 1.4)
# dbutils.widgets  → parâmetros interativos
# dbutils.secrets  → credenciais com segurança
# dbutils.notebook → chamar outros notebooks
# dbutils.data     → summarize (visto no Módulo 3)

# --- Widgets: parâmetros interativos ---
dbutils.widgets.text("modelo", "RandomForest", "Tipo de Modelo")
dbutils.widgets.dropdown("n_trees", "100", ["50", "100", "200"], "Nº de Árvores")
dbutils.widgets.slider("test_size", 0.2, 0.1, 0.4, 0.05, "Tamanho do Teste")

# get() SEMPRE retorna string — converta o tipo você mesmo
modelo  = dbutils.widgets.get("modelo")
n_trees = int(dbutils.widgets.get("n_trees"))
print(f"Modelo: {modelo} | Árvores: {n_trees}")

dbutils.widgets.removeAll()
```

```python
# --- Secrets: nunca coloque token no código ---
# token = dbutils.secrets.get(scope="meu-scope", key="api-token")

# --- Chamar outro notebook e receber o retorno ---
# resultado = dbutils.notebook.run("../outro_notebook", timeout_seconds=600,
#                                  arguments={"param1": "valor"})

# --- Ver a documentação de qualquer submódulo ---
# dbutils.fs.help()
```

> **Pegadinha:** `dbutils.widgets.get()` retorna **sempre uma string**, mesmo em `slider` e `dropdown` numéricos. Esquecer o `int()`/`float()` é fonte clássica de bug.

---

## Exercício prático

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

schema = StructType([
    StructField("produto",  StringType()),
    StructField("preco",    DoubleType()),
    StructField("estoque",  IntegerType()),
])

produtos = [
    ("Notebook", 3500.0, 10),
    ("Mouse",      80.0, 150),
    ("Teclado",   200.0, 75),
    ("Monitor",  1200.0, 30),
]

TABELA_EX = "workspace.default.exercicio_delta"

# 1. Criar e salvar como Delta
spark.createDataFrame(produtos, schema).write.format("delta").mode("overwrite").saveAsTable(TABELA_EX)

# 2. Adicionar um produto (nova versão)
spark.createDataFrame([("Headset", 350.0, 45)], schema) \
     .write.format("delta").mode("append").saveAsTable(TABELA_EX)

# 3. Ver o histórico
display(spark.sql(f"DESCRIBE HISTORY {TABELA_EX}"))

# 4. Comparar versões
df_v0 = spark.read.format("delta").option("versionAsOf", 0).table(TABELA_EX)
print(f"Versão 0: {df_v0.count()} produtos | Atual: {spark.table(TABELA_EX).count()} produtos")

# 5. Desafio: restaure a tabela para a versão 0
#    (dica: RESTORE TABLE ... TO VERSION AS OF 0)
```

---

## Pontos-chave para a prova

- **Driver** roda seu código e recebe `collect()`/`toPandas()`; **workers** processam partições
- **Compute** = hardware | **Runtime** = software instalado nele
- **Job Cluster** é criado e destruído automaticamente → mais barato para produção
- **All-Purpose** fica ligado → desenvolvimento interativo
- **Serverless** = Databricks gerencia tudo; único disponível na Free Edition
- **Runtime ML** = Runtime + MLflow, sklearn, XGBoost, LightGBM, PyTorch, TensorFlow
- Vantagens do Runtime ML: bibliotecas prontas, **versões validadas entre si**, start mais rápido, MLflow integrado, GPU pré-configurada, reprodutibilidade
- AutoML (classificação/regressão) exige compute **clássico com Runtime ML**
- **Unity Catalog** = `catalog.schema.objeto`; governa tabelas, volumes, modelos e funções
- **Volume** substitui `dbfs:/FileStore/` para arquivos crus
- **DBFS** persiste entre sessões (cai na prova; root público desabilitado na Free Edition)
- **Delta Lake** = ACID + time travel + schema enforcement/evolution
- `option("versionAsOf", N)` → lê a versão N | `option("timestampAsOf", "data")` → por data
- `DESCRIBE HISTORY` → lista todas as versões de uma tabela Delta
- `dbutils.widgets.get()` retorna **string** — converta o tipo

---

→ Próximo: [02_pyspark.md](02_pyspark.md)
