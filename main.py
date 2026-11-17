import os

import matplotlib.pyplot as plt
import numpy as np

from mmq import (carregar_dados, montar_X, resolver_mmq, verificar,
                 prever, metricas, para_latex)

np.set_printoptions(precision=4, suppress=True, linewidth=120)
os.makedirs("graphics", exist_ok=True)

# Ordem dos coeficientes = ordem das colunas de X = ordem dos "termos".
PROBLEMAS = [
    {
        "titulo": "Problema 1: Publicidade",
        "slug": "publicidade",
        "arquivo": "csv_files/publicidade.csv",
        "n_linhas": 10,
        "n_colunas": 2,
        "termos": ["investimento", "constante"],
        "novo_ponto": [13.0],
        "unidade": "mil unidades",
    },
    {
        "titulo": "Problema 2: Consumo de combustível",
        "slug": "consumo",
        "arquivo": "csv_files/consumo.csv",
        "n_linhas": 15,
        "n_colunas": 4,
        "termos": ["peso", "potência", "cilindrada", "constante"],
        "novo_ponto": [1400.0, 108.0, 1.8],
        "unidade": "km/L",
    },
    {
        "titulo": "Problema 3: Tempo de entrega",
        "slug": "entregas",
        "arquivo": "csv_files/entregas.csv",
        "n_linhas": 18,
        "n_colunas": 4,
        "termos": ["distância", "peso", "paradas", "constante"],
        "novo_ponto": [26.0, 8.0, 5.0],
        "unidade": "h",
    },
]


def grafico(p, explicativas, y, X, coef):
    """P1: dispersão + reta + previsão. P2 e P3: observado × previsto."""
    y_prev = X @ coef
    fig, ax = plt.subplots(figsize=(6, 4.5))

    if len(explicativas) == 1:
        x = explicativas[0]
        x_novo = p["novo_ponto"][0]
        xs = np.linspace(min(x.min(), x_novo), max(x.max(), x_novo), 100)
        ax.scatter(x, y, color="blue", label="Dados observados")
        ax.plot(xs, coef[0] * xs + coef[1], color="red", label="Reta ajustada")
        ax.scatter([x_novo], [prever(coef, x_novo)], color="green",
                   marker="*", s=150, label="Previsão")
        ax.set_xlabel("Investimento (mil R$)")
        ax.set_ylabel(f"Vendas ({p['unidade']})")
    else:
        lim = [min(y.min(), y_prev.min()), max(y.max(), y_prev.max())]
        ax.scatter(y, y_prev, color="blue", label="Observações")
        ax.plot(lim, lim, "r--", label="y = x (ajuste perfeito)")
        ax.set_xlabel(f"Observado ({p['unidade']})")
        ax.set_ylabel(f"Previsto ({p['unidade']})")

    ax.set_title(p["titulo"])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(f"graphics/{p['slug']}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def resolver_problema(p):
    dados = carregar_dados(p["arquivo"], p["n_linhas"], p["n_colunas"])
    *explicativas, y = dados.T

    X = montar_X(*explicativas)
    coef, A, b = resolver_mmq(X, y)

    print(f"\n{'=' * 60}\n{p['titulo']}\n{'=' * 60}")
    print("Matriz A = XᵀX:\n", A)
    print("Vetor b = Xᵀy:\n", b)

    print("\nCoeficientes:")
    for nome, c in zip(p["termos"], coef):
        print(f"  {nome:>12}: {c:.6f}")

    print("\nVerificações:")
    for nome, valor in verificar(X, y, coef, A).items():
        print(f"  {nome}: {valor}")

    previsao = prever(coef, *p["novo_ponto"])
    print(f"\nPrevisão para {p['novo_ponto']}: {previsao:.4f} {p['unidade']}")

    m = metricas(X, y, coef)
    print("\nAjuste:")
    for nome in ["SSE", "SST", "R2", "R2 ajustado", "RMSE"]:
        print(f"  {nome}: {m[nome]:.6f}")
    print("  Resíduos:", m["residuos"])

    print("\nLaTeX de A:\n" + para_latex(A))
    print("LaTeX de b:\n" + para_latex(b))

    grafico(p, explicativas, y, X, coef)


if __name__ == "__main__":
    for problema in PROBLEMAS:
        resolver_problema(problema)