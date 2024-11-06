import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote import logging
from dotenv import load_dotenv
from rag import get_news, create_summary_chain
import os

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("GS-Project-NEWS")

if "news_summary_chain" not in st.session_state:
    st.session_state["news_summary_chain"] = None

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

with st.sidebar:
    news_url = st.text_input("URL을 입력하세요.")
    summary_btn = st.button("뉴스 요약 생성")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이전 대화 기록 출력
print_messages()

if summary_btn:
    # 뉴스기사 처리(GPT)
    docs = get_news(url=news_url)
    chain = create_summary_chain()

    # 스트리밍 호출
    response = chain.stream({"context": docs})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화기록을 저장한다.
    add_message("assistant", ai_answer)
