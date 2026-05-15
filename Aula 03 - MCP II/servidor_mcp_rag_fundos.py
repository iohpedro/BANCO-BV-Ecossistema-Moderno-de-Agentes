"""
servidor_mcp_rag_fundos.py

Servidor MCP local que encapsula um workflow RAG sobre regulamentos financeiros.

Inicie com:
    python servidor_mcp_rag_fundos.py

Endpoint:
    http://127.0.0.1:8766/mcp
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from rag_core import RAGRegulamentoFundos


mcp = FastMCP(
    "BVInteligenciaDocumentalMCP",
    host="127.0.0.1",
    port=8766,
)

_rag: Optional[RAGRegulamentoFundos] = None


def get_rag() -> RAGRegulamentoFundos:
    """
    Inicialização lazy do RAG.

    O servidor sobe rápido, e o índice é carregado/criado apenas quando uma tool precisa dele.
    """

    global _rag

    if _rag is None:
        _rag = RAGRegulamentoFundos(
            pdf_path="data/KNCR_Regulamento_05-2025.pdf",
            persist_directory="vectorstore/chroma_kncr",
            collection_name="regulamento_kncr",
            embedding_model="text-embedding-3-small",
            chat_model="gpt-4o-mini",
            k=5,
            recriar_indice=False,
        )

    return _rag


@mcp.tool()
def consultar_regulamento(pergunta: str, k: int = 5) -> dict:
    """
    Consulta o regulamento financeiro usando RAG.

    Use esta ferramenta quando a pergunta envolver:
    - regras do regulamento
    - política de investimento
    - taxas e remuneração
    - obrigações de administrador, gestor ou cotistas
    - informações que exigem fonte documental

    Args:
        pergunta: pergunta em linguagem natural sobre o regulamento.
        k: quantidade de trechos recuperados no vector store.

    Returns:
        Resposta estruturada com resposta, fontes, páginas e confiança.
    """

    return get_rag().consultar(pergunta=pergunta, k=k)


@mcp.tool()
def buscar_trechos_regulamento(pergunta: str, k: int = 5) -> dict:
    """
    Busca trechos relevantes no regulamento sem chamar o LLM.

    Esta ferramenta é útil para depurar a qualidade do retriever.

    Args:
        pergunta: consulta semântica.
        k: quantidade de trechos recuperados.

    Returns:
        Trechos recuperados, páginas e metadados.
    """

    return get_rag().buscar_trechos(pergunta=pergunta, k=k)


@mcp.tool()
def extrair_taxas_regulamento() -> dict:
    """
    Extrai taxas, remunerações e custos relevantes previstos no regulamento.

    Returns:
        Resposta estruturada com taxas encontradas e fontes.
    """

    return get_rag().extrair_taxas()


@mcp.tool()
def status_rag() -> dict:
    """
    Mostra o status operacional do RAG.

    Returns:
        Informações sobre PDF, índice vetorial, modelos e quantidade de chunks.
    """

    return get_rag().status()


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
