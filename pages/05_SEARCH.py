import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote import logging
from agent import create_search_agent


# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] PDF RAG")


# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "search_domains" not in st.session_state:
    st.session_state["search_domains"] = ["google.com", "naver.com"]


def print_urls():
    for url in st.session_state["search_domains"]:
        st.markdown(f"- {url}")


def add_url(url):
    if url not in st.session_state["search_domains"]:
        st.session_state["search_domains"].append(url)


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    st.markdown("----\n**검색 도메인**")

    search_domain = st.text_input("검색 도메인 추가")
    add_btn = st.button("추가")
    if add_btn:
        add_url(search_domain)

    selected_search_domain = st.multiselect(
        "검색할 도메인을 설정",
        st.session_state["search_domains"],
        ["google.com", "naver.com"],
    )


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []


# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 검색해 보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = create_search_agent(include_domains=selected_search_domain)

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        # 질의에 대한 답변을 출력합니다.
        response = chain.stream(
            {"input": user_input},
            # 세션 ID를 설정합니다.
            # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
            config={"configurable": {"session_id": "test123"}},
        )
        # print(f"답변: {response['output']}")

        with st.chat_message("assistant"):
            with st.status("웹 검색을 수행하는 중입니다...", expanded=True) as status:
                for steps in response:
                    if "actions" in steps:
                        action = steps["actions"].pop(0)
                        if action.tool == "tavily_search_results_json":
                            st.markdown(action.log)
                    if "steps" in steps:
                        action = steps["steps"].pop(0)
                        observation = action.observation
                        observation_string = [
                            f"<search_result><content>{doc['content']}</content><url>{doc['url']}</url></search_result>"
                            for doc in observation
                        ]
                        searched_string = [
                            f"\n- {doc['content']}\n(출처: {doc['url']}"
                            for doc in observation
                        ]
                        search_result = "\n".join(searched_string)
                        st.markdown(search_result)

                    if "output" in steps:
                        ai_answer = steps["output"]

                status.update(label="검색 완료!", state="complete", expanded=False)

        st.chat_message("assistant").markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("오류가 있습니다.")

########## 7. 질의-응답 테스트를 수행합니다. ##########