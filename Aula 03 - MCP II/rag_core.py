"""
rag_core.py

Capacidade especialista de RAG para consulta de regulamentos financeiros.

Este módulo é usado pelo servidor MCP da Aula 02 - MCP II.

Arquitetura:
    PDF -> chunks -> embeddings -> ChromaDB -> retriever -> LLM -> resposta estruturada
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class FonteConsulta(BaseModel):
    """Fonte textual recuperada pelo RAG."""

    documento: Optional[str] = Field(default=None, description="Nome/caminho do documento consultado")
    pagina: Optional[int] = Field(default=None, description="Página em numeração humana, começando em 1")
    trecho: str = Field(description="Trecho recuperado do documento")


class RespostaRAG(BaseModel):
    """Resposta estruturada devolvida pelo workflow RAG."""

    pergunta: str
    resposta: str
    fontes: list[FonteConsulta]
    paginas_consultadas: list[int]
    confianca: Literal["alta", "media", "baixa"]
    observacao: str


class RAGRegulamentoFundos:
    """
    Workflow especialista de RAG para regulamentos financeiros.

    Esta classe encapsula tudo que o agente consumidor NÃO precisa saber:
    carregamento de PDF, chunking, embeddings, vector store, prompt e LLM.

    O agente chamará apenas ferramentas MCP.
    O servidor MCP chamará esta classe.
    """

    def __init__(
        self,
        pdf_path: str = "data/KNCR_Regulamento_05-2025.pdf",
        persist_directory: str = "vectorstore/chroma_kncr",
        collection_name: str = "regulamento_kncr",
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        k: int = 5,
        recriar_indice: bool = False,
    ):
        load_dotenv()

        self.pdf_path = Path(pdf_path)
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.chat_model_name = chat_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.recriar_indice = recriar_indice

        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY não encontrada. Defina a variável no ambiente ou em um arquivo .env."
            )

        self.persist_directory.parent.mkdir(parents=True, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(model=self.embedding_model_name)
        self.llm = ChatOpenAI(model=self.chat_model_name, temperature=0)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
Você é um especialista em análise de regulamentos financeiros.

Responda usando APENAS o contexto fornecido.
Se a resposta não estiver no contexto, diga claramente que não encontrou a informação no documento.
Sempre que possível, cite fonte, página e/ou seção.
Não invente números, cláusulas, taxas ou regras.

Contexto:
{context}
"""),
            ("human", "{question}")
        ])

        self.vector_store = self._preparar_vector_store()

    def _preparar_vector_store(self) -> Chroma:
        """
        Cria ou carrega o índice vetorial.

        Em produção, o índice normalmente seria criado por um pipeline separado.
        Nesta aula, deixamos a criação dentro da classe para fins didáticos.
        """

        if self.recriar_indice and self.persist_directory.exists():
            shutil.rmtree(self.persist_directory)

        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_directory),
        )

        try:
            quantidade = vector_store._collection.count()
        except Exception:
            quantidade = 0

        if quantidade == 0:
            if not self.pdf_path.exists():
                raise FileNotFoundError(
                    f"PDF não encontrado: {self.pdf_path}. "
                    "Coloque o arquivo em data/ ou ajuste pdf_path."
                )

            paginas = self._carregar_pdf()
            chunks = self._dividir_documentos(paginas)

            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
            )

        return vector_store

    def _carregar_pdf(self):
        """Carrega o PDF em objetos Document do LangChain."""

        loader = PyPDFLoader(str(self.pdf_path))
        return loader.load()

    def _dividir_documentos(self, paginas):
        """Divide páginas em chunks menores para busca semântica."""

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(paginas)

    @staticmethod
    def _pagina_humana(doc) -> Optional[int]:
        """
        Converte a página zero-based do PyPDFLoader para numeração humana.

        PyPDFLoader costuma retornar page=0 para a primeira página.
        Para o usuário final, mostramos página 1.
        """

        pagina = doc.metadata.get("page")
        if isinstance(pagina, int):
            return pagina + 1
        return None

    def _formatar_documentos(self, docs) -> str:
        """Formata documentos recuperados para entrada no prompt."""

        partes = []
        for doc in docs:
            pagina = self._pagina_humana(doc)
            fonte = doc.metadata.get("source", "documento")
            partes.append(
                f"[Fonte: {fonte} | Página: {pagina}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(partes)

    def _fontes(self, docs) -> tuple[list[FonteConsulta], list[int]]:
        """Extrai fontes e páginas a partir dos documentos recuperados."""

        fontes: list[FonteConsulta] = []
        paginas: list[int] = []

        for doc in docs:
            pagina = self._pagina_humana(doc)
            if pagina is not None:
                paginas.append(pagina)

            fontes.append(FonteConsulta(
                documento=doc.metadata.get("source"),
                pagina=pagina,
                trecho=doc.page_content[:700],
            ))

        return fontes, sorted(set(paginas))

    @staticmethod
    def _inferir_confianca(docs) -> Literal["alta", "media", "baixa"]:
        """
        Heurística didática de confiança.

        Em produção, esta métrica deveria ser substituída por avaliação mais robusta:
        scores do retriever, checagem de citação, avaliação automática, ou validação humana.
        """

        if len(docs) >= 4:
            return "alta"
        if len(docs) >= 2:
            return "media"
        return "baixa"

    def buscar_trechos(self, pergunta: str, k: Optional[int] = None) -> dict:
        """
        Retorna apenas os trechos mais relevantes, sem chamar o LLM.

        Útil para debug do retriever.
        """

        k_final = k or self.k
        docs = self.vector_store.similarity_search(pergunta, k=k_final)
        fontes, paginas = self._fontes(docs)

        return {
            "pergunta": pergunta,
            "k": k_final,
            "paginas_consultadas": paginas,
            "fontes": [fonte.model_dump() for fonte in fontes],
        }

    def consultar(self, pergunta: str, k: Optional[int] = None) -> dict:
        """
        Consulta o regulamento usando RAG e retorna resposta estruturada.

        Args:
            pergunta: pergunta em linguagem natural.
            k: quantidade de chunks recuperados.

        Returns:
            dict compatível com JSON, contendo resposta e fontes.
        """

        k_final = k or self.k
        docs = self.vector_store.similarity_search(pergunta, k=k_final)
        contexto = self._formatar_documentos(docs)

        resposta_texto = (self.prompt | self.llm | StrOutputParser()).invoke({
            "context": contexto,
            "question": pergunta,
        })

        fontes, paginas = self._fontes(docs)

        resposta = RespostaRAG(
            pergunta=pergunta,
            resposta=resposta_texto,
            fontes=fontes,
            paginas_consultadas=paginas,
            confianca=self._inferir_confianca(docs),
            observacao="Resposta gerada apenas com base nos trechos recuperados pelo RAG.",
        )

        return resposta.model_dump()

    def extrair_taxas(self) -> dict:
        """
        Consulta orientada para extração de taxas, remunerações e custos.

        Esta tool reaproveita o mesmo workflow, mas fixa uma pergunta especialista.
        """

        pergunta = (
            "Liste as taxas, remunerações e custos relevantes previstos no regulamento. "
            "Inclua percentuais, responsáveis pelo pagamento e páginas de referência quando possível."
        )
        return self.consultar(pergunta=pergunta, k=max(self.k, 6))

    def status(self) -> dict:
        """Retorna informações operacionais do índice e dos modelos usados."""

        try:
            quantidade_chunks = self.vector_store._collection.count()
        except Exception:
            quantidade_chunks = None

        return {
            "pdf_path": str(self.pdf_path),
            "pdf_existe": self.pdf_path.exists(),
            "persist_directory": str(self.persist_directory),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "chat_model": self.chat_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "k_default": self.k,
            "quantidade_chunks_indexados": quantidade_chunks,
        }
