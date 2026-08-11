# Módulo 0 — Configuração do Ambiente

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Objetivo:** Ter o ambiente Databricks funcionando antes de iniciar os estudos.

---

## 0.1 Criar conta gratuita no Databricks (Free Edition)

O Databricks oferece uma conta gratuita chamada **Free Edition** (antiga Community Edition) — sem cartão de crédito e sem prazo de expiração.

**Passo a passo:**

1. Acesse: **https://www.databricks.com/learn/free-edition**
2. Clique em **"Sign up for free"**
3. Preencha nome, e-mail e senha
4. Verifique seu e-mail e clique no link de confirmação
5. Faça login em **https://login.databricks.com**

### O que dá e o que não dá para praticar na Free Edition

| Recurso | Free Edition | Onde isso aparece no guia |
|---|---|---|
| **Compute** | Somente **Serverless** — sem criar ou configurar cluster | Módulo 1 |
| **MLflow** (tracking + registry) | ✅ Completo | Módulo 4 |
| **Spark ML** | ✅ Completo | Módulos 5 e 6 |
| **Unity Catalog** (tabelas, volumes, modelos) | ✅ Completo (1 metastore) | Módulos 1, 4 e 9 |
| **Delta Lake** | ✅ Completo | Módulo 1 |
| **Feature Store** (offline, no UC) | ✅ Disponível | Módulo 9 |
| **Model Serving (REST)** | ✅ Com limites: poucos endpoints ativos, sem GPU, sem provisioned throughput | Módulo 10 |
| **AutoML classificação/regressão** | ❌ Exige compute clássico com Runtime ML | Módulo 8 — só teoria |
| **AutoML forecasting** | ⚠️ Suportado em serverless — vale testar na sua conta | Módulo 8 |
| **Hyperopt** | ⚠️ Não vem pré-instalado — `%pip install hyperopt` | Módulo 7 |
| **SparkTrials** | ❌ Depende de APIs de RDD, não suportadas no serverless | Módulo 7 — só teoria |
| **Online feature tables** | ❌ Não suportado | Módulo 9 — só teoria |
| **DBFS root público** | ❌ Desabilitado — use Unity Catalog | Módulo 1 |
| **R e Scala** | ❌ Não suportados (a prova é toda em Python) | — |

> **Nada disso te impede de passar na prova.** Os itens marcados como "só teoria" são cobrados de forma **conceitual** — a prova pergunta o que a ferramenta faz e quando usá-la, não pede para você escrever a chamada de cabeça. Onde não dá para executar, o guia sinaliza e explica o conceito.

> **Cota de uso:** a Free Edition tem cota diária de compute. Se você estourar, o workspace para até o dia seguinte. Estudar em sessões de 1–2h evita isso. Verificar sua identidade com o LinkedIn aumenta alguns limites.

---

## 0.2 Compute: por que não existe "Create Cluster" aqui

Diferente de contas pagas, o Free Edition **não permite criar nem configurar clusters**. Todo o compute é **Serverless**: o Databricks aloca e gerencia as máquinas automaticamente por trás dos panos, sem tela de "Create Cluster", sem escolher tamanho de máquina, sem esperar o cluster subir.

Isso significa:
- Notebooks conectam a **Serverless** automaticamente — não existe passo de criar cluster.
- O SQL Editor/Dashboards usam o **"Serverless Starter Warehouse"**, o único SQL Warehouse disponível na conta gratuita.
- Você não escolhe versão de Databricks Runtime — o ambiente serverless já vem com Python, Spark, MLflow, sklearn, XGBoost etc. pré-instalados e atualizados pelo Databricks.

> **Para a prova:** o conceito de cluster (All-Purpose vs Job Cluster, Databricks Runtime, Runtime ML) continua sendo cobrado — veja o Módulo 1. Aqui você só não vai *praticar* a criação manual, porque o Free Edition abstrai isso.

---

## 0.3 Criar o primeiro notebook

1. Menu esquerdo → **"Workspace"** → **"Users"** → sua pasta
2. Botão direito → **"Create"** → **"Notebook"**
3. Configure:
   - **Name:** `00_verificacao_ambiente`
   - **Language:** Python
4. Clique em **"Create"**
5. No canto superior esquerdo do notebook, confirme que o seletor de compute mostra **"Serverless"** (conecta sozinho ao rodar a primeira célula — não precisa fazer nada)

**Cole e execute este código de verificação:**

```python
import pyspark, mlflow, sklearn

print(f"PySpark : {pyspark.__version__}")
print(f"MLflow  : {mlflow.__version__}")
print(f"Sklearn : {sklearn.__version__}")
print(f"Usuário : {spark.sql('SELECT current_user()').first()[0]}")
print(f"Catalog : {spark.sql('SELECT current_catalog()').first()[0]}")
print("\nSpark ativo:", spark)
```

Se tudo aparecer sem erro, o ambiente está pronto.

```python
# Verificação opcional — pacotes que alguns módulos usam.
# Se algum acusar "NAO instalado", instale só quando chegar no módulo correspondente.
for pacote, modulo in [
    ("databricks-feature-engineering", "databricks.feature_engineering"),  # Módulo 9
    ("hyperopt", "hyperopt"),                                              # Módulo 7
]:
    try:
        __import__(modulo)
        print(f"OK           {pacote}")
    except ImportError:
        print(f"NAO instalado {pacote}  →  %pip install {pacote}")
```

### Se algo falhar

| Erro | O que fazer |
|---|---|
| `ModuleNotFoundError` | Rode `%pip install <pacote>` na primeira célula, depois `dbutils.library.restartPython()` |
| Célula fica "Waiting" por muito tempo | O serverless está iniciando. É normal na primeira execução do dia |
| `DBFS_DISABLED` | Esperado na Free Edition. Use Unity Catalog — explicado no [Módulo 1](01_plataforma.md#15-dbfs--teoria-de-prova) |
| Compute indisponível / quota | Você atingiu a cota diária da Free Edition. Ela reseta no dia seguinte |

---

## 0.4 Atalhos essenciais do notebook

| Atalho | Ação |
|---|---|
| `Shift + Enter` | Executar célula e avançar |
| `Ctrl + Enter` | Executar célula e ficar |
| `Esc + A` | Nova célula acima |
| `Esc + B` | Nova célula abaixo |
| `%sql` (início da célula) | Escrever SQL |
| `%md` (início da célula) | Escrever Markdown |
| `display(df)` | Visualização rica de DataFrame |

---

## Próximo módulo

→ [01_plataforma.md](01_plataforma.md) — Clusters, Unity Catalog & Volumes, Delta Lake, dbutils
