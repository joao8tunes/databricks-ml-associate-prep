# Databricks Certified Machine Learning Associate — Guia de Estudos

> Guia prático e incremental para passar na certificação **Databricks Certified Machine Learning Associate** partindo do zero.
> Escrito para quem **não trabalha com Databricks no dia a dia** e precisa passar mesmo assim.
> Cada módulo tem teoria mínima + código que roda de verdade na conta gratuita.
> Um projeto de **previsão de churn de clientes** conecta todos os módulos do início ao fim.

---

## ⚠️ Aviso importante — leia antes de começar

**Este é um material experimental, não oficial e sem qualquer vínculo com a Databricks.**

- **Não é material oficial.** Não foi produzido, revisado nem endossado pela Databricks. "Databricks" e os nomes das certificações são marcas de seus respectivos donos.
- **Não há garantia de aprovação.** Seguir este guia não garante que você vai passar na prova. Nenhum material de terceiros pode garantir isso.
- **Está em testes práticos neste momento.** O conteúdo está sendo validado na prática, à medida que o autor estuda e executa os exemplos. É possível que existam erros, imprecisões e trechos que não rodem na sua conta.
- **Está em evolução constante.** A plataforma Databricks e o exam guide oficial mudam com frequência. Partes deste material podem ficar desatualizadas sem aviso.
- **Use como complemento, não como fonte única.** A fonte da verdade é sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate) e a [documentação da Databricks](https://docs.databricks.com). Este repositório serve para organizar o estudo, não para substituí-los.

Encontrou um erro? [Abra uma issue](../../issues) — correções são bem-vindas e ajudam quem vier depois.

---

## Sobre a Certificação

Dados extraídos do [exam guide oficial](https://www.databricks.com/sites/default/files/2025-02/databricks-certified-machine-learning-associate-exam-guide-1-mar-2025.pdf) (versão vigente desde 1º de março de 2025) e da [página oficial da certificação](https://www.databricks.com/learn/certification/machine-learning-associate).

| Item | Valor |
|---|---|
| **Questões** | 48 questões pontuadas (múltipla escolha **e múltipla seleção**) |
| **Duração** | 90 minutos |
| **Custo** | US$ 200 |
| **Formato** | Online proctored (com fiscal remoto) |
| **Idiomas** | **Português (BR)**, Inglês, Japonês, Coreano |
| **Material de consulta** | Não permitido |
| **Pré-requisito formal** | Nenhum |
| **Validade** | 2 anos |

> **A prova tem versão em Português do Brasil.** Você escolhe o idioma no momento da inscrição. Mesmo assim, estude os nomes de funções e parâmetros em inglês — o código nas questões nunca é traduzido.

> **Sobre a nota de corte:** o Databricks **não publica** a nota mínima de aprovação deste exame. O número "70%" circula em blogs e fóruns, mas não é oficial. Neste guia usamos 70% apenas como meta de segurança nos simulados.

> **Questões de múltipla seleção existem.** Algumas questões pedem mais de uma alternativa ("Selecione DUAS"). O enunciado sempre avisa. Não existe crédito parcial: ou você marca exatamente o conjunto certo, ou erra a questão.

### As 4 seções oficiais e seus pesos

| # | Seção (nome no exam guide) | Peso | Módulos deste guia |
|---|---|---|---|
| 1 | **Databricks Machine Learning** — MLflow, Unity Catalog, Feature Store, AutoML, MLOps | **38%** | 01, 04, 08, 09, 11 |
| 2 | **Data Processing** — estatísticas, outliers, visualização, imputação, encoding, transformações | **19%** | 02, 03 |
| 3 | **Model Development** — algoritmos, pipelines, tuning, cross-validation, métricas | **31%** | 05, 06, 07 |
| 4 | **Model Deployment** — batch, streaming, real-time serving | **12%** | 10 |

> A página da certificação lista a Seção 2 com o nome antigo "ML Workflows"; o exam guide em PDF, que é a fonte autoritativa do conteúdo, chama de **"Data Processing"**. O conteúdo cobrado é o do PDF.

> **Onde investir o tempo:** Seções 1 e 3 somam **69% da prova**. Se o tempo apertar, garanta MLflow (Módulo 04), Feature Store (Módulo 09) e Model Development (Módulos 05–07) antes de qualquer outra coisa.

📄 **[Mapa completo: cada objetivo oficial → onde está neste guia](GUIA_DA_PROVA.md)**

---

## O Projeto Fio Condutor

Todos os módulos usam o mesmo problema:

> **Previsão de Churn de Clientes de Telecom**
> Uma empresa de telefonia quer prever quais clientes cancelarão o serviço para que o time comercial aja preventivamente.

O dataset é criado programaticamente no Módulo 02 e reutilizado em todos os módulos seguintes. Você nunca precisa baixar nada.

---

## Como usar este guia

1. **Faça o Módulo 00 primeiro.** Sem o ambiente funcionando, o resto não faz sentido.
2. **Abra um notebook no Databricks e rode o código.** Ler não fixa; rodar e quebrar fixa.
3. **Ao final de cada módulo, faça o simulado só daquele módulo.** Feedback imediato mostra o que não entrou.
4. **Não pule os blocos "Pegadinhas da prova".** É onde a maioria dos candidatos perde ponto.
5. **Nos últimos 15 dias, só simulado + revisão dos erros.**

Cada módulo indica no topo a qual seção oficial ele corresponde e quais objetivos do exam guide ele cobre.

---

## Módulos

| # | Arquivo | Conteúdo | Seção oficial | Tempo |
|---|---|---|---|---|
| 00 | [Configuração](modulos/00_configuracao.md) | Conta gratuita, compute serverless, primeiro notebook | — | 30 min |
| 01 | [Plataforma Databricks](modulos/01_plataforma.md) | Compute, Runtime ML, Unity Catalog, Delta Lake, dbutils | 1 | 1h30 |
| 02 | [PySpark para ML](modulos/02_pyspark.md) | DataFrames, transformações, lazy evaluation, split | 2 | 2h |
| 03 | [EDA e Preparação de Dados](modulos/03_eda_preparacao.md) | Estatísticas, outliers (IQR/desvio), visualização, imputação, encoding, log transform | 2 | 2h30 |
| 04 | [MLflow](modulos/04_mlflow.md) | Tracking, autolog, MLflow Client API, Registry no Unity Catalog, aliases | 1 | 3h |
| 05 | [Feature Engineering](modulos/05_feature_engineering.md) | Estimators vs transformers, Pipeline, encoders, scalers | 3 | 2h |
| 06 | [Treinamento e Avaliação](modulos/06_treinamento_modelos.md) | Escolha de algoritmo, desbalanceamento, métricas, CV, bias-variance | 3 | 3h |
| 07 | [Tuning de Hiperparâmetros](modulos/07_hyperopt.md) | Hyperopt `fmin`, grid/random/bayesiana, paralelização | 3 | 1h30 |
| 08 | [AutoML](modulos/08_automl.md) | O que automatiza, quando usar, output, seleção de features | 1 | 1h |
| 09 | [Feature Store](modulos/09_feature_store.md) | `FeatureEngineeringClient`, feature tables no UC, online vs offline, `score_batch` | 1 | 2h |
| 10 | [Deployment e Serving](modulos/10_deployment.md) | Batch, streaming (DLT), real-time, traffic split, modelo customizado | 4 | 2h |
| 11 | [MLOps e Governança](modulos/11_mlops.md) | Boas práticas, promover código vs promover modelo, champion/challenger | 1 | 1h30 |
| 12 | [Revisão Final](modulos/12_revisao_final.md) | Data leakage, glossário, armadilhas, checklist completo | Todas | 2h |

---

## Simulados

O sistema de simulados está em [`simulados/`](simulados/).

**Demo online:** https://databricks-ml-ass-prep.streamlit.app/

![Demo do simulado](demo.png)

Para rodar localmente:

```bash
pip install -r simulados/requirements.txt
```

```bash
streamlit run simulados/quiz.py
```

**Funcionalidades:**
- **176 questões autorais** mapeadas aos objetivos do exam guide oficial
- **Modo Prova:** 48 questões, 90 minutos, distribuição por seção igual à da prova real (18/9/15/6)
- **Modo Estudo:** filtro por seção, módulo, dificuldade e quantidade, com feedback imediato
- Questões de **múltipla seleção**, como na prova real
- Alternativas embaralhadas a cada sessão
- Relatório final com desempenho por seção oficial e por módulo, mais a lista de revisão

> As questões são **autorais**, escritas no estilo do exame a partir dos objetivos oficiais. **Não são questões reais da prova** — reproduzir conteúdo do exame violaria o acordo de confidencialidade da certificação.

---

## Plano de Estudos — 12 semanas, ~4h por semana

Ritmo pensado para quem estuda **depois do trabalho**, sem largar tudo. São ~4h semanais: duas sessões de 2h, ou quatro de 1h. Se você tiver mais tempo, veja a trilha acelerada abaixo.

| Semana | Foco | Meta concreta |
|---|---|---|
| **1** | Módulos 00 + 01 | Ambiente rodando, primeira tabela Delta criada no Unity Catalog |
| **2** | Módulo 02 | Dataset de churn criado, EDA básica feita em PySpark |
| **3** | Módulo 03 (parte 1) | Estatísticas descritivas, detecção e remoção de outliers |
| **4** | Módulo 03 (parte 2) | Imputação, encoding, log transform — simulado da Seção 2 |
| **5** | Módulo 04 (parte 1) | Primeiro experimento MLflow com params, métricas e modelo logados |
| **6** | Módulo 04 (parte 2) | Modelo registrado no Unity Catalog com alias `champion` |
| **7** | Módulo 05 | Pipeline Spark ML completo, do DataFrame bruto às features |
| **8** | Módulo 06 | Modelo treinado e avaliado, entendendo cada métrica |
| **9** | Módulo 07 | Tuning com Hyperopt registrado no MLflow |
| **10** | Módulos 08 + 09 | AutoML entendido, feature table criada no Unity Catalog |
| **11** | Módulos 10 + 11 | Batch inference rodando, conceitos de MLOps fechados |
| **12** | Módulo 12 + simulados | Simulado completo ≥ 80%, revisão dos erros |

> **Só marque a semana como concluída quando o código tiver rodado na sua conta.** Ler o módulo não conta.

### Se você atrasar (e você vai)

Isso é normal e está previsto. Duas regras:

1. **Nunca pule um módulo para "recuperar o atraso".** Prefira estender o cronograma. Os módulos dependem uns dos outros.
2. **Se perder duas semanas seguidas, volte um módulo** e refaça o simulado dele antes de seguir. É mais rápido do que avançar com base frágil.

### Trilha acelerada — 6 semanas, ~10h por semana

Para quem já tem experiência com Python, pandas e scikit-learn e pode dedicar mais tempo:

| Semana | Módulos |
|---|---|
| **1** | 00 + 01 + 02 |
| **2** | 03 + 05 |
| **3** | 04 |
| **4** | 06 + 07 |
| **5** | 08 + 09 + 10 + 11 |
| **6** | 12 + simulados diários |

### Trilha mínima — se a prova é semana que vem

Nesta ordem, e só isto: **04 (MLflow) → 09 (Feature Store) → 06 (Treinamento) → 03 (Data Processing) → 12 (Revisão)**. Juntos cobrem a maior parte dos pontos. Depois, simulados até acertar 80% consistentemente.

---

## Perguntas frequentes

**Preciso pagar alguma coisa para estudar?**
Não. A Databricks Free Edition é gratuita, sem cartão de crédito e sem prazo de expiração. Só a prova custa (US$ 200).

**Consigo praticar tudo na conta gratuita?**
Quase tudo. As exceções estão sinalizadas nos módulos — a principal é o AutoML de classificação/regressão, que exige compute clássico. Nesses casos o guia explica o conceito e o código, que é o que a prova cobra.

**Preciso saber Spark de verdade?**
Não. A prova cobra a API do Spark ML e do PySpark em nível de uso, não internals de otimização. O Módulo 02 dá o suficiente.

**Preciso saber Scala ou SQL?**
Scala não — todo código de ML da prova é Python. SQL pode aparecer em tarefas não relacionadas a ML, em nível básico.

**Quantos simulados devo fazer antes de marcar a prova?**
Acerte ≥ 80% em pelo menos três simulados completos em Modo Prova, em dias diferentes. Um resultado bom isolado costuma ser sorte na amostra de questões.

---

## Recursos Oficiais

| Recurso | Link |
|---|---|
| Página da certificação | https://www.databricks.com/learn/certification/machine-learning-associate |
| Exam Guide (PDF, versão vigente) | https://www.databricks.com/sites/default/files/2025-02/databricks-certified-machine-learning-associate-exam-guide-1-mar-2025.pdf |
| Databricks Academy (cursos gratuitos) | https://academy.databricks.com |
| Free Edition (conta grátis) | https://www.databricks.com/learn/free-edition |
| Limitações da Free Edition | https://docs.databricks.com/aws/en/getting-started/free-edition-limitations |
| Documentação MLflow | https://mlflow.org/docs/latest |
| Documentação Spark ML | https://spark.apache.org/docs/latest/ml-guide.html |
| Feature Engineering no Unity Catalog | https://docs.databricks.com/aws/en/machine-learning/feature-store/ |

> ⚠️ **Confira o exam guide oficial duas semanas antes da sua prova.** O Databricks atualiza o conteúdo periodicamente e este guia pode ficar defasado. A data da versão usada aqui está no topo desta seção.

---

## Contribuindo

Achou um erro, uma explicação confusa ou algo que mudou na plataforma? Abra uma issue ou um pull request. Este material foi feito para ser corrigido por quem usa.
