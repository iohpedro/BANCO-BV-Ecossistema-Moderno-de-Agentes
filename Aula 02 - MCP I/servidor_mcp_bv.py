"""
Servidor MCP – Banco BV Demo
Ferramentas de simulação financeira expostas via protocolo MCP (HTTP).
Inicie com: python servidor_mcp_bv.py
"""
from mcp.server.fastmcp import FastMCP

# host/port configurados diretamente no construtor (via Settings)
mcp = FastMCP(
    "BancoVotorantimMCP",
    host="127.0.0.1",
    port=8765,
)


@mcp.tool()
def simular_credito(valor: float, parcelas: int) -> dict:
    """
    Simula uma oferta de crédito pessoal do Banco BV.

    Args:
        valor: Valor solicitado em reais (ex: 10000.0)
        parcelas: Número de parcelas mensais (ex: 12)

    Returns:
        Dicionário com valor da parcela, total pago e juros totais.
    """
    taxa_mensal = 0.0199  # 1.99% a.m. (taxa ilustrativa)
    fator = (1 + taxa_mensal) ** parcelas
    parcela = (valor * taxa_mensal * fator) / (fator - 1)
    total = round(parcela * parcelas, 2)
    return {
        "valor_solicitado": valor,
        "parcelas": parcelas,
        "valor_parcela": round(parcela, 2),
        "total_pago": total,
        "juros_totais": round(total - valor, 2),
        "taxa_mensal": "1.99%",
    }


@mcp.tool()
def calcular_iof(valor: float, dias: int) -> dict:
    """
    Calcula o IOF de uma operação de crédito pessoal.

    Args:
        valor: Valor do empréstimo em reais
        dias: Prazo da operação em dias

    Returns:
        Dicionário com o IOF calculado e detalhes das alíquotas.
    """
    taxa_diaria = 0.000082   # 0.0082% ao dia (Decreto 6.306/2007)
    taxa_adicional = 0.0038  # 0.38% sobre o principal
    iof_diario = valor * taxa_diaria * dias
    iof_adicional = valor * taxa_adicional
    iof_total = round(iof_diario + iof_adicional, 2)
    return {
        "valor_emprestado": valor,
        "dias": dias,
        "iof_diario": round(iof_diario, 2),
        "iof_adicional": round(iof_adicional, 2),
        "iof_total": iof_total,
    }


@mcp.tool()
def calcular_cet(valor: float, parcelas: int, taxa_mensal: float = 0.0199) -> dict:
    """
    Calcula o Custo Efetivo Total (CET) aproximado de uma operação de crédito.

    Args:
        valor: Valor do empréstimo em reais
        parcelas: Número de parcelas mensais
        taxa_mensal: Taxa de juros mensal em decimal (padrão 0.0199 = 1.99%)

    Returns:
        CET anual e mensal aproximados.
    """
    fator = (1 + taxa_mensal) ** parcelas
    parcela = (valor * taxa_mensal * fator) / (fator - 1)
    total = parcela * parcelas
    cet_mensal = (total / valor) ** (1 / parcelas) - 1
    cet_anual = (1 + cet_mensal) ** 12 - 1
    return {
        "valor": valor,
        "parcelas": parcelas,
        "taxa_mensal_base": f"{taxa_mensal*100:.2f}%",
        "cet_mensal": f"{cet_mensal*100:.4f}%",
        "cet_anual": f"{cet_anual*100:.2f}%",
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
