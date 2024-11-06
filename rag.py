from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_teddynote.prompts import load_prompt
from langchain.schema import Document
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_teddynote.retrievers import (
    EnsembleRetriever,
    EnsembleMethod,
)


def create_pdf_loader(file_path):
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    return docs


def create_news_loader(url):
    # 뉴스기사 내용을 로드하고, 청크로 나누고, 인덱싱합니다.
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                "div",
                attrs={
                    "class": ["newsct_article _article_body", "media_end_head_title"]
                },
            )
        ),
    )
    return loader.load()


def create_retriever(file_path):
    docs = create_pdf_loader(file_path=file_path)

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    return retriever


def create_tankerwire_retriever(sections):
    # print(sections)
    docs = []
    for key, value in sections.items():
        # Create a Document object
        doc = Document(
            page_content=f"<document><title>{key}</title><content>{value}</content></document>"
        )
        # print(doc)
        docs.append(doc)

    # docs = [Document(page_content=sections)]
    # Put the Document object into a list
    # docs = [doc]

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    faiss = vectorstore.as_retriever(search_kwargs={"k": 10})

    # KiwiBM25Retriever 생성(한글 형태소 분석기 + BM25 알고리즘)
    bm25 = KiwiBM25Retriever.from_documents(
        documents=split_documents, embedding=embeddings
    )
    bm25.k = 10

    # RRF 방식의 EnsembleRetriever (기본값으로 RRF 가 설정되어 있음)
    rrf_ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss, bm25],
        method=EnsembleMethod.RRF,
        weights=[0.3, 0.7],
    )

    # CC 방식의 EnsembleRetriever
    cc_ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss, bm25],
        method=EnsembleMethod.CC,  # method 지정: CC
        weights=[0.3, 0.7],
    )

    return cc_ensemble_retriever


def create_news_retriever(url):
    docs = create_news_loader(url)

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return retriever


def get_news(url):
    docs = create_news_loader(url)
    return docs


def create_rag_chain(retriever, prompt="pdf-rag-v2", model_name="gpt-4o-mini"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt(f"prompts/{prompt}.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def create_summary_chain(prompt="news-summary", model_name="gpt-4o-mini"):
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    prompt = load_prompt(f"prompts/{prompt}.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = prompt | llm | StrOutputParser()
    return chain
