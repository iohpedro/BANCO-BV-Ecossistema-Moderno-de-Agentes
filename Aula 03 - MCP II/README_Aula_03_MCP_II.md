# Aula 02 - MCP II

Notebook gerado para a aula:

**Encapsulando um Workflow RAG em um Servidor MCP Local**

Arquivo principal:

- `BV_AI_EXPERT_02_MCP_II_RAG_SERVER.ipynb`

## Como usar

1. Crie uma pasta `data/`.
2. Coloque o PDF `KNCR_Regulamento_05-2025.pdf` dentro dela.
3. Crie um arquivo `.env` com:

```text
OPENAI_API_KEY=sua-chave-aqui
```

4. Abra o notebook e execute as células em ordem.

O notebook gera, durante a execução:

- `rag_core.py`
- `servidor_mcp_rag_fundos.py`
- `requirements.txt`
- `vectorstore/chroma_kncr/`
- `logs/`

## Observação didática

O material foi estruturado para mostrar primeiro o RAG no notebook e depois promover esse workflow para um servidor MCP.
