# Módulo 11 — MLOps e Governança

> ⚠️ Material **experimental e não oficial**, sem vínculo com a Databricks e **sem garantia de aprovação**. Em constante evolução — confira sempre o [exam guide oficial](https://www.databricks.com/learn/certification/machine-learning-associate). [Aviso completo](../README.md#️-aviso-importante--leia-antes-de-começar).


> **Seção oficial:** 1 — Databricks Machine Learning (**38% da prova**)
>
> **Notebook sugerido:** nenhum — este módulo é conceitual.

Este módulo não tem quase código, e mesmo assim vale pontos: dois objetivos oficiais da prova são puramente conceituais de MLOps, e são exatamente o tipo de questão que quem só estuda API erra.

**Objetivos oficiais cobertos neste módulo:**
- Identificar as boas práticas de uma estratégia de MLOps
- Identificar cenários em que promover **código** é preferível a promover **modelos**, e vice-versa

---

## 11.1 O que é MLOps na prática

MLOps é DevOps aplicado a machine learning, com três complicações que software tradicional não tem:

```
Software tradicional versiona:     código
Machine learning precisa versionar: código + DADOS + MODELO
```

```
As três complicações do ML:

1. O dado muda sozinho
   O código não mudou, o modelo não mudou — mas o mundo mudou.
   Resultado: o modelo degrada em produção sem ninguém tocar em nada.
   → chamado de "data drift" (a distribuição dos dados de entrada mudou)
     e "concept drift" (a relação entre features e target mudou)

2. Reproduzir um resultado exige 3 coisas
   Mesmo código + mesmos dados + mesmos hiperparâmetros.
   Faltando qualquer um, você não reproduz o modelo de 3 meses atrás.

3. "Funciona" não é binário
   Software tradicional: passa no teste ou não passa.
   ML: o modelo tem 0.87 de AUC. Isso é bom? Depende do baseline,
   do custo do erro e do que estava em produção antes.
```

### As boas práticas que a prova cobra

| Prática | O que significa |
|---|---|
| **Versionar código, dados e modelos juntos** | Git para código, Delta Lake (time travel) para dados, MLflow Registry para modelos |
| **Rastrear todo experimento** | Todo treino gera um MLflow run com params, métricas e artefatos — nada de "treinei no meu notebook" |
| **Separar ambientes** | Dev, staging e produção isolados, com dados e permissões próprios |
| **Automatizar re-treino** | Jobs agendados, não um humano rodando notebook |
| **Validar antes de promover** | Um modelo só vira champion depois de passar por testes automáticos de qualidade |
| **Monitorar em produção** | Acompanhar métricas do modelo e drift dos dados depois do deploy |
| **Governança centralizada** | Unity Catalog para controlar quem acessa quais dados, features e modelos |
| **CI/CD** | Testar e implantar automaticamente, como em qualquer software |

> **Pegadinha da prova:** questões de MLOps quase sempre têm como alternativa errada algo do tipo "treinar manualmente no notebook e copiar o arquivo do modelo para produção". Qualquer coisa que dependa de intervenção manual ou de um artefato solto fora do Registry está errada.

---

## 11.2 Os três ambientes

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│     DEV      │ ───→ │   STAGING    │ ───→ │  PRODUCTION  │
├──────────────┤      ├──────────────┤      ├──────────────┤
│ Exploração   │      │ Testes       │      │ Serve o      │
│ Experimentos │      │ automáticos  │      │ negócio      │
│ Dados de     │      │ Dados        │      │ Dados reais  │
│ amostra      │      │ realistas    │      │ completos    │
└──────────────┘      └──────────────┘      └──────────────┘
```

No Databricks isso normalmente vira **três catalogs no Unity Catalog** (`dev`, `staging`, `prod`), cada um com suas permissões. O mesmo código roda nos três, apontando para o catalog do ambiente.

> Na Free Edition existe só um workspace e um metastore, então você não consegue montar essa separação. O conceito continua caindo na prova.

---

## 11.3 Promover código ou promover modelo

Este é um **objetivo oficial explícito** e uma das questões conceituais mais características desta prova. Existem duas estratégias para levar um modelo de dev até produção:

### Deploy code (promover código) — o padrão recomendado

```
DEV: escreve o código de treino
       ↓ (promove o CÓDIGO via Git/CI)
STAGING: o código roda e treina com dados de staging
       ↓ (promove o CÓDIGO)
PROD: o código roda e TREINA O MODELO com os dados de produção
       ↓
     modelo registrado no Registry de produção
```

**O modelo que serve produção foi treinado em produção, com dados de produção.**

### Deploy model (promover modelo)

```
DEV: treina o modelo com dados de dev/produção
       ↓ (promove o ARTEFATO do modelo)
STAGING: valida o mesmo binário
       ↓ (promove o ARTEFATO)
PROD: serve exatamente aquele binário
```

**O modelo é treinado uma vez e o artefato viaja entre ambientes.**

### Quando usar cada um

| | **Deploy code** | **Deploy model** |
|---|---|---|
| **Recomendado pelo Databricks como padrão** | ✅ Sim | Só em casos específicos |
| **O modelo é treinado em** | Produção | Dev/staging |
| **Precisa de acesso a dados de produção no ambiente de dev?** | Não | Sim (ou dados equivalentes) |
| **Re-treino automático e frequente** | Fácil — o código já está lá | Difícil — exige repetir a promoção a cada re-treino |
| **CI/CD tradicional (testa código)** | Encaixa naturalmente | Encaixa mal |
| **Treino muito caro** (dias de GPU) | Ruim — retreinaria em produção | ✅ Bom — treina uma vez, promove o artefato |
| **Ambiente de produção sem capacidade de treino** | Inviável | ✅ Único caminho |
| **Restrição regulatória: o binário validado é o que vai para produção** | Ruim | ✅ Bom |
| **Auditoria: garantir que o modelo aprovado é exatamente o que serve** | Mais difícil | ✅ Mais fácil |

**Resumo para a prova:**

> **Promova código por padrão.** Promova o modelo quando o treino for proibitivamente caro, quando o ambiente de produção não puder treinar, ou quando houver exigência regulatória/de auditoria de que o binário validado seja exatamente o que vai para produção.

---

## 11.4 Champion e challenger

O padrão que o Databricks usa para trocar modelos em produção sem risco:

```
champion   → o modelo que está servindo produção agora
challenger → o candidato que quer tomar o lugar dele
```

O fluxo:

```
1. Treina um novo modelo               → nova versão no Registry
2. Aponta o alias "challenger" para ela
3. Compara challenger vs champion nos mesmos dados de validação
4. Se o challenger for melhor:
     move o alias "champion" para a versão do challenger
5. O código de inferência NÃO muda — ele sempre carrega
     models:/catalog.schema.modelo@champion
```

```python
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

MODEL_NAME = "workspace.default.churn_predictor"

# A versão nova entra como challenger
client.set_registered_model_alias(name=MODEL_NAME, alias="challenger", version=2)

# ... aqui rodam as validações comparando as duas versões ...

# Aprovado: challenger vira champion. O código de inferência não muda uma linha.
client.set_registered_model_alias(name=MODEL_NAME, alias="champion", version=2)
client.delete_registered_model_alias(name=MODEL_NAME, alias="challenger")
```

> **O ponto que a prova cobra:** o valor do alias é o **desacoplamento**. Quem consome o modelo referencia `@champion`, nunca um número de versão. Trocar o modelo em produção vira uma operação de metadados — sem redeploy, sem mudar código, e reversível instantaneamente (basta apontar o alias de volta).

Para testar o challenger com tráfego real antes de promover, veja [traffic split entre endpoints](10_deployment.md#107-dividir-tráfego-entre-modelos-ab-testing).

---

## 11.5 Unity Catalog como camada de governança

O Unity Catalog é o que amarra dados, features e modelos sob a mesma governança:

```
Unity Catalog governa:
├── Tabelas       → catalog.schema.tabela
├── Volumes       → arquivos crus
├── Feature tables→ catalog.schema.features
├── Modelos       → catalog.schema.modelo
└── Funções       → UDFs registradas
```

| Benefício | Por quê importa |
|---|---|
| **Permissões granulares** | `GRANT SELECT ON MODEL ...` — controle por objeto, não por workspace |
| **Compartilhamento entre workspaces** | Um modelo registrado no metastore é visível em todos os workspaces da conta |
| **Linhagem automática** | Você consegue rastrear qual tabela gerou qual feature, que treinou qual modelo |
| **Namespace único** | Mesmo padrão de 3 níveis para dados, features e modelos |
| **Auditoria** | Log de quem acessou o quê |

---

## 11.6 Monitoramento e drift

Depois do deploy, duas coisas podem dar errado sem que ninguém mude uma linha de código:

| Tipo | O que mudou | Exemplo |
|---|---|---|
| **Data drift** | A distribuição das **features de entrada** mudou | A empresa começou a vender para um público mais jovem; a média de idade caiu |
| **Concept drift** | A **relação entre features e target** mudou | Antes, contrato mensal previa churn. Depois de uma mudança de política, não prevê mais |

O que monitorar em produção:

```
├── Distribuição das features de entrada  → detecta data drift
├── Distribuição das predições            → detecta mudança de comportamento
├── Métricas do modelo (quando o rótulo real chega) → detecta degradação
├── Latência e taxa de erro do endpoint   → saúde da infraestrutura
└── Volume de requisições                 → uso real
```

> Se a resposta a "o modelo degradou em produção" for necessária: **re-treinar com dados recentes** é a ação padrão. Se a degradação foi por concept drift, pode ser necessário revisar as próprias features.

---

## Pontos-chave para a prova

- MLOps versiona **código + dados + modelo** — os três, não só código
- Git para código, **Delta Lake (time travel)** para dados, **MLflow Registry** para modelos
- Qualquer resposta que dependa de passo manual ou de arquivo solto fora do Registry está errada
- **Deploy code é o padrão**: promove-se o código e o modelo é treinado no ambiente de produção
- **Deploy model** só quando: treino é caro demais, produção não pode treinar, ou há exigência de auditoria/regulatória sobre o binário
- **Champion** = modelo servindo agora | **Challenger** = candidato em avaliação
- O código de inferência aponta para `@champion`, nunca para um número de versão
- Promover = mover o alias. Reverter = mover o alias de volta. Sem redeploy
- **Data drift** = distribuição das features mudou | **Concept drift** = relação feature→target mudou
- Unity Catalog governa dados, features e modelos no mesmo namespace de 3 níveis

---

→ Próximo: [12_revisao_final.md](12_revisao_final.md)
