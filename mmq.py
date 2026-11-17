"""Regressão linear múltipla pelo Método dos Mínimos Quadrados (equações normais).

Modelo:  y ≈ c1*x1 + c2*x2 + ... + c_{n-1}*x_{n-1} + c_n   (última coluna de X = 1)
Função objetivo:  s(c) = ||Xc - y||²
Equações normais: (XᵀX) c = Xᵀy
"""
import numpy as np
import pandas as pd


def carregar_dados(caminho, n_linhas, n_colunas):
    """Lê o CSV (com cabeçalho) e confere se o formato bate com a tabela do enunciado."""
    dados = pd.read_csv(caminho).to_numpy(dtype=float)
    if dados.shape != (n_linhas, n_colunas):
        raise ValueError(
            f"{caminho}: formato {dados.shape}, esperado {(n_linhas, n_colunas)}"
        )
    return dados


def montar_X(*colunas):
    """Matriz de dados: uma coluna por variável explicativa + coluna de 1s (termo constante)."""
    n = len(colunas[0])
    return np.column_stack([*colunas, np.ones(n)])


def resolver_mmq(X, y):
    """Monta e resolve as equações normais. Retorna (coeficientes, A, b)."""
    A = X.T @ X
    b = X.T @ y
    coef = np.linalg.solve(A, b)
    return coef, A, b


def verificar(X, y, coef, A):
    """Conferências numéricas do resultado."""
    ref = np.linalg.lstsq(X, y, rcond=None)[0]
    return {
        "A simétrica": bool(np.allclose(A, A.T)),
        "coeficientes batem com lstsq": bool(np.allclose(coef, ref)),
        "número de condição de A": float(np.linalg.cond(A)),
    }


def prever(coef, *valores):
    """Previsão para um novo ponto. Acrescenta o 1 do termo constante."""
    x_novo = np.append(np.array(valores, dtype=float), 1.0)
    return float(x_novo @ coef)


def metricas(X, y, coef):
    """Resíduos, SSE, SST, R², R² ajustado e RMSE."""
    m, n = X.shape
    residuos = y - X @ coef
    sse = float(residuos @ residuos)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst
    r2_aj = 1 - (sse / (m - n)) / (sst / (m - 1))
    rmse = float(np.sqrt(sse / m))
    return {"residuos": residuos, "SSE": sse, "SST": sst,
            "R2": r2, "R2 ajustado": r2_aj, "RMSE": rmse}


def para_latex(M, casas=4):
    """Converte matriz ou vetor em ambiente bmatrix, pronto para colar no relatório."""
    M = np.asarray(M, dtype=float)
    if M.ndim == 1:                      # vetor vira coluna
        M = M.reshape(-1, 1)
    linhas = [" & ".join(f"{v:.{casas}f}" for v in linha) for linha in M]
    corpo = " \\\\\n".join(linhas)
    return f"\\begin{{bmatrix}}\n{corpo}\n\\end{{bmatrix}}"