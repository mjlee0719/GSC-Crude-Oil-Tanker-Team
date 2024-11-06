import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote import logging
from dotenv import load_dotenv
from rag import create_retriever, create_tankerwire_retriever, create_rag_chain
import os
import PyPDF2
import re

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("[Project] PDF RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("PDF 기반 QA💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"], index=0
    )


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
# def embed_file(file):
#     # 업로드한 파일을 캐시 디렉토리에 저장합니다.
#     file_content = file.read()
#     file_path = f"./.cache/files/{file.name}"
#     with open(file_path, "wb") as f:
#         f.write(file_content)

#     retriever = create_retriever(file_path)
#     return retriever


# 체인 생성
@st.cache_resource
def create_chain(_retriever, model_name="gpt-4o-mini"):
    chain = create_rag_chain(retriever, model_name=model_name)
    return chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

        keyword_pairs = [
            ("Now available digitally", "All rights reserved."),
            ("East of Suez Dirty Tankers\nVLCC (PGT page 2980)", "<AASLC00>"),
            ("Suezmax Iran Loading Premium\n", "$/mt figure."),
            ("Contact Client Services:", "All rights reserved."),
            ("East of Suez Dirty Tankers\nVLCC (PGT page 2980)", "ACGHP00"),
            ("West of Suez dirty fuel oil barges ($/mt) (PGT page 1980)", "TDMMA00"),
            ("Time charter equivalents\nAframax TCE 0.5", "bunker overhang in tanks"),
            ("S&P Global, the S&P Global logo", "TDUGP00"),
            ("Carbon Emission Charges (PGT page 4030)", "ANEUC00"),
            ("West of Suez Aframax 10-Day Rolling Average (PGT page 2633)","Grade Loading dates Delivery dates Delivery Port Loading Port Size (barrels) Seller  Buyer"),
            ("Americas\n", "Ship name Size  Type Date  Route Rate Charterer"),
        ]

    for start_keyword, end_keyword in keyword_pairs:
        pattern = re.escape(start_keyword) + r".*?" + re.escape(end_keyword)
        text = re.sub(pattern, "", text, flags=re.DOTALL)

    patterns = [
        (
            r"Platts East of Suez Dirty Tanker Daily \nCommentary(.*?)Dirty Persian Gulf China 270kt VLCC \nAssessment Rationale & Exclusions",
            "Platts East of Suez Dirty Tanker Daily Commentary",
        ),
        (
            r"Dirty Persian Gulf China 270kt VLCC \nAssessment Rationale & Exclusions(.*?)Platts Daily Dirty Tanker PG-China Bids, \nOffers, Trades",
            "Dirty Persian Gulf China 270kt VLCC Assessment Rationale & Exclusions",
        ),
        (
            r"Platts Daily Dirty Tanker PG-China Bids, \nOffers, Trades(.*?)Platts West of Suez Dirty Tanker Daily \nCommentary",
            "Platts Daily Dirty Tanker PG-China Bids, Offers, Trades",
        ),
        (
            r"Platts West of Suez Dirty Tanker Daily \nCommentary(.*?)Platts Dirty Tanker UKC-UKC Aframax Daily \nRationale",
            "Platts West of Suez Dirty Tanker Daily Commentary",
        ),
        (
            r"Platts Dirty Tanker UKC-UKC Aframax Daily \nRationale(.*?)Platts Dirty Tanker UKC-UKC Aframax Daily \nBids, Offers, Trades",
            "Platts Dirty Tanker UKC-UKC Aframax Daily Rationale",
        ),
        (
            r"Platts Dirty Tanker UKC-UKC Aframax Daily \nBids, Offers, Trades(.*?)Platts Americas Dirty Tanker Daily \nCommentary",
            "Platts Dirty Tanker UKC-UKC Aframax Daily Bids, Offers, Trades",
        ),
        (
            r"Platts Americas Dirty Tanker Daily \nCommentary(.*?)Platts USGC-China",
            "Platts Americas Dirty Tanker Daily Commentary",
        ),
        (
            r"Platts USGC-China VLCC(.*?)Platts Dirty Tanker USGC-China VLCC Bids, \nOffers, Trades",
            "Platts USGC-China VLCC $/mt Daily Rationale & Exclusions",
        ),
        (
            r"Platts Dirty Tanker USGC-China VLCC Bids, \nOffers, Trades(.*?)Platts USGC-UKC VLCC",
            "Platts Dirty Tanker USGC-China VLCC Bids, Offers, Trades",
        ),
        (
            r"Platts USGC-UKC VLCC(.*?)Platts Dirty Tanker USGC-UKC VLCC Bids, \nOffers, Trades",
            "Platts USGC-UKC VLCC $/mt Daily Rationale & Exclusions",
        ),
        (
            r"Platts Dirty Tanker USGC-UKC VLCC Bids, \nOffers, Trades(.*?)Platts USGC UKC Aframax Dirty Tanker Daily \nRationale & Exclusions",
            "Platts Dirty Tanker USGC-UKC VLCC Bids, Offers, Trades",
        ),
        (
            r"Platts USGC UKC Aframax Dirty Tanker Daily \nRationale & Exclusions(.*?)Platts USGC UKC Aframax Dirty Tanker \nBids, Offers, Trades",
            "Platts USGC UKC Aframax Dirty Tanker Daily Rationale & Exclusions",
        ),
        (
            r"Platts USGC UKC Aframax Dirty Tanker \nBids, Offers, Trades(.*?)News\n",
            "Platts USGC UKC Aframax Dirty Tanker \nBids, Offers, Trades",
        ),
        (r"News\n(.*?)Subscriber Notes\n", "News"),
    ]

    # Extract sections
    sections = {}
    for pattern, section_name in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[section_name] = match.group(1).strip()
        else:
            sections[section_name] = (
                f"No text found between the specified delimiters for section {section_name}"
            )

    retriever = create_tankerwire_retriever(sections)
    # retriever = create_tankerwire_retriever(text)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

col1, col2, col3, col4 = st.columns(4)

with col1:
    q1 = st.button("VLCC East of Suez")
    q2 = st.button("VLCC West of Suez")
    q3 = st.button("VLCC Americas")

with col2:
    q5 = st.button("Suezmax East of Suez")
    q6 = st.button("Suezmax West of Suez")
    q7 = st.button("Suezmax Americas")
with col3:
    q8 = st.button("Aframax East of Suez")
    q9 = st.button("Aframax West of Suez")
    q10 = st.button("Aframax Americas")
with col4:
    q4 = st.button('News')
    q11 = st.button('PG-China Transactions')
    q12 = st.button('USGC-China VLCC Transactions')
    q13 = st.button('USGC-UKC VLCC Transactions')

if q1:
    user_input = "First, an update on the East of Suez VLCC market. Second, give me the prospects of the East of Suez VLCC market. Third, give me a list of East of Suez VLCC fixtures."
if q2:
    user_input = "First, an update on the West of Suez VLCC market. Second, give me the prospects of the West of Suez VLCC market. Third, give me a list of West of Suez VLCC fixtures."
if q3:
    user_input = "First, an update on the Americas VLCC market. Second, give me the prospects of the Americas VLCC market. Third, give me a list of Americas VLCC fixtures."
if q4:
    user_input='First, find at least 2 headlines under ''News Section''. Second, a 5 sentence summary of each headline under ''News Section'''
if q5:
    user_input = "First, an update on the East of Suez Suezmax market. Second, give me the prospects of the East of Suez Suezmax market. Third, give me a list of East of Suez Suezmax fixtures."
if q6:
    user_input = "First, an update on the West of Suez Suezmax market. Second, give me the prospects of the West of Suez Suezmax market. Third, give me a list of West of Suez Suezmax fixtures."
if q7:
    user_input = "First, an update on the Americas Suezmax market. Second, give me the prospects of the Americas Suezmax market. Third, give me a list of Americas Suezmax fixtures."
if q8:
    user_input = "First, an update on the East of Suez Aframax market. Second, give me the prospects of the East of Suez Aframax market. Third, give me a list of East of Suez Aframax fixtures."
if q9:
    user_input = "First, an update on the West of Suez Aframax market. Second, give me the prospects of the West of Suez Aframax market. Third, give me a list of West of Suez Aframax fixtures."
if q10:
    user_input = "First, an update on the Americas Aframax market. Second, give me the prospects of the Americas Aframax market. Third, give me a list of Americas Aframax fixtures."
if q11:
    user_input = "Give me the Daily Dirty Tanker PG-China Bids, Offers, Trades."
if q12:
    user_input = "Dirty Tanker USGC-China VLCC Bids, Offers, Trades"
if q13:
    user_input = "Give me the Dirty Tanker USGC-UKC VLCC Bids, Offers, Trades"
    
    


# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")

with st.expander("원문"):
    st.write(sections)

