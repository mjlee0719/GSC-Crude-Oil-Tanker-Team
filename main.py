import streamlit as st
from langchain_core.prompts import load_prompt
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_teddynote import logging
import os

load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("GS-Project")

st.title("나만의 챗GPT")

# messages 라는 이름의 채팅 저장소를 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    # 모델 선택 메뉴추가
    selected_model = st.selectbox(
        "모델 선택", ["gpt-4o-mini", "gpt-4o", "o1-preview"], index=0
    )

    instruction_file = st.selectbox(
        "지시사항 선택",
        ["default.yaml", "email.yaml", "sns.yaml", "summary.yaml"],
        index=0,
    )


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 메시지 출력
print_messages()


# Chain 생성
def create_chain(instruction="default.yaml", model="gpt-4o-mini"):
    # Chain 생성
    # prompt
    # prompt = PromptTemplate.from_template(prompt_template)

    prompt_file = os.path.join("prompts", instruction)
    prompt = load_prompt(prompt_file, encoding="utf-8")

    # llm
    llm = ChatOpenAI(model=model, temperature=1)
    # output_parser
    output_parser = StrOutputParser()
    # chain 연결
    chain = prompt | llm | output_parser
    return chain


# 채팅창
user_input = st.chat_input("궁금한 내용을 입력해 주세요")

if user_input:
    # user 가 메시지를 입력한다면..
    # chain 생성
    chain = create_chain(instruction=instruction_file, model=selected_model)

    st.chat_message("user").markdown(user_input)

    result = chain.invoke({"question": user_input})

    ai_message = ""

    with st.chat_message("ai"):
        # chat_container = st.empty()
        # st.write_stream(result)
        st.markdown(result)

    ai_message = result

    # for token in result:
    #     ai_message += token
    #     chat_container.markdown(ai_message)

    # st.chat_message("ai").markdown(ai_message)

    # 메시지 추가
    add_message("user", user_input)
    add_message("ai", ai_message)