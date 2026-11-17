# Regressão Linear Múltipla — Método dos Mínimos Quadrados

Projeto de regressão linear múltipla resolvida pelo **Método dos Mínimos Quadrados (MMQ)** por meio das equações normais:

$$(X^T X)\,c = X^T y$$

O programa lê os dados de arquivos CSV, monta a matriz de projeto `X` (com uma coluna de 1s para o termo constante), resolve o sistema, imprime as verificações numéricas, as métricas de ajuste, as matrizes em LaTeX para o relatório e salva os gráficos.

## Problemas

| # | Título | Dados | Variáveis explicativas |
|---|--------|-------|------------------------|
| 1 | Publicidade | `csv_files/publicidade.csv` | investimento |
| 2 | Consumo de combustível | `csv_files/consumo.csv` | peso, potência, cilindrada |
| 3 | Tempo de entrega | `csv_files/entregas.csv` | distância, peso, paradas |

## Estrutura do projeto

```
.
├── main.py                 # Configuração dos problemas, saída e gráficos
├── mmq.py                  # Funções do método dos mínimos quadrados
├── csv_files/              # Dados de entrada (CSV com cabeçalho)
│   ├── publicidade.csv
│   ├── consumo.csv
│   └── entregas.csv
├── graphics/               # Gráficos gerados (.png)
├── requirements.txt        # Dependências
└── .gitignore
```

## Instalação

Requer Python 3.8 ou superior.

```bash
pip install -r requirements.txt
```

## Execução

Execute na raiz do repositório:

```bash
python main.py
```

Para cada problema o programa exibe:

- as matrizes `A = XᵀX` e `b = Xᵀy`;
- os coeficientes estimados;
- verificações (simetria de `A`, comparação com `np.linalg.lstsq` e número de condição);
- a previsão para um novo ponto;
- métricas: SSE, SST, R², R² ajustado, RMSE e resíduos;
- as matrizes `A` e `b` em formato `bmatrix` do LaTeX.

Os gráficos são salvos em `graphics/` (`publicidade.png`, `consumo.png`, `entregas.png`).

## Formato dos CSVs

Cada arquivo possui uma linha de cabeçalho seguida pelos dados, uma coluna por variável (a última é a variável resposta). O `main.py` valida o formato esperado de cada tabela antes de resolver o sistema.

## Implementação

- `carregar_dados`: leitura e validação do CSV.
- `montar_X`: acrescenta a coluna de 1s do termo constante.
- `resolver_mmq`: monta e resolve as equações normais.
- `verificar`: checagens numéricas do resultado.
- `prever`: calcula a resposta para um novo ponto.
- `metricas`: resíduos, SSE, SST, R², R² ajustado e RMSE.
- `para_latex`: converte matrizes/vetores para `bmatrix`.
