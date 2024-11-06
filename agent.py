# 필요한 모듈 import
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


def create_search_agent(include_domains=["google.com"], k=10):
    ########## 1. 도구를 정의합니다 ##########
    ### 1-1. Search 도구 ###
    # TavilySearchResults 클래스의 인스턴스를 생성합니다
    # k=5은 검색 결과를 5개까지 가져오겠다는 의미입니다
    search = TavilySearchResults(
        max_results=k,
        include_answer=True,
        include_raw_content=True,
        # include_images=True,
        # search_depth="advanced", # or "basic"
        include_domains=include_domains,
        # exclude_domains = []
    )

    ### 1-3. tools 리스트에 도구 목록을 추가합니다 ###
    # tools 리스트에 search와 retriever_tool을 추가합니다.
    tools = [search]

    ########## 2. LLM 을 정의합니다 ##########
    # LLM 모델을 생성합니다.
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

    ########## 3. Prompt 를 정의합니다 ##########

    # hub에서 prompt를 가져옵니다 - 이 부분을 수정할 수 있습니다!
    prompt = hub.pull("hwchase17/openai-functions-agent")

    ########## 4. Agent 를 정의합니다 ##########

    # OpenAI 함수 기반 에이전트를 생성합니다.
    # llm, tools, prompt를 인자로 사용합니다.
    agent = create_openai_functions_agent(llm, tools, prompt)

    ########## 5. AgentExecutor 를 정의합니다 ##########

    # AgentExecutor 클래스를 사용하여 agent와 tools를 설정하고, 상세한 로그를 출력하도록 verbose를 True로 설정합니다.
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    ########## 6. 채팅 기록을 수행하는 메모리를 추가합니다. ##########

    # 채팅 메시지 기록을 관리하는 객체를 생성합니다.
    message_history = ChatMessageHistory()

    # 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        # 대부분의 실제 시나리오에서 세션 ID가 필요하기 때문에 이것이 필요합니다
        # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
        lambda session_id: message_history,
        # 프롬프트의 질문이 입력되는 key: "input"
        input_messages_key="input",
        # 프롬프트의 메시지가 입력되는 key: "chat_history"
        history_messages_key="chat_history",
    )
    return agent_with_chat_history