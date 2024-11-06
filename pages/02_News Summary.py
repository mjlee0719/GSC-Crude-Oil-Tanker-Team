import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote import logging
from dotenv import load_dotenv
from rag import get_news, create_summary_chain
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from datetime import datetime
from googletrans import Translator


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

if "df" not in st.session_state:
    st.session_state["df"] = []
if 'radio1_selection' not in st.session_state:
    st.session_state.radio1_selection = None
if 'radio2_selection' not in st.session_state:
    st.session_state.radio2_selection = None

with st.sidebar:
    keywords = st.text_input("1단어 키워드를 입력하세요", value='vlcc,suezmax,aframax,fuel,baltic,carbon,weeklyshipping-market,tanker,iran,israel,houthi')
    crawl_btn = st.button('뉴스 크롤링')
    summary_btn = st.button("뉴스 요약 생성")
    

# 탭을 생성
main_tab1, main_tab2 = st.tabs(["이미지", "대화내용"])

keywords.replace(' ','')
keywords = keywords.split(',')


#####################################################################################################################################
dict_refined={}
if crawl_btn:
    driver = webdriver.Chrome()
    driver.maximize_window()

    url_hel_topst = "https://www.hellenicshippingnews.com/tag/top-stories/"
    url_hel_shipnews = "https://www.hellenicshippingnews.com/category/shipping-news/hellenic-shipping-news/"
    url_hel_intshipnews = "https://www.hellenicshippingnews.com/category/shipping-news/international-shipping-news/"

    url_trw_tankers = "https://www.tradewindsnews.com/tankers" #//*[@id="app"]/div/div[2]/div[3]/div[1]/div[2]/div/div/div[1]/div/div/div/div[2]/div/div[2]/div/div[1]/h2/a
    url_trw_shipyards = 'https://www.tradewindsnews.com/shipyards' #//*[@id="app"]/div/div[2]/div[3]/div[1]/div[2]/div/div/div[1]/div/div/div/div[2]/div/div[2]/div/div[1]/h2/a
    url_trw_finance = 'https://www.tradewindsnews.com/finance' #//*[@id="app"]/div/div[2]/div[3]/div[1]/div[2]/div/div/div[1]/div/div/div/div[2]/div/div[2]/div/div[1]/h2/a
    url_trw_casualties = 'https://www.tradewindsnews.com/casualties' #//*[@id="app"]/div/div[2]/div[3]/div[1]/div[2]/div/div/div[1]/div/div/div/div[2]/div/div[2]/div/div[1]/h2/a

    url_list_hel = [url_hel_topst, url_hel_shipnews, url_hel_intshipnews]
    url_list_trw = [url_trw_tankers, url_trw_shipyards, url_trw_finance, url_trw_casualties]

    df = pd.DataFrame(columns=['title', 'date', 'url'])
    dict = {}

    for i in url_list_hel : 
        driver.get(i)
        if i==url_hel_topst :
            for j in range (14) : 
                element_urltitle = driver.find_element(By.XPATH,f'//*[@id="main-content"]/div[1]/div/div[3]/article[{j+1}]/h2/a')
                element_date = pd.to_datetime(driver.find_element(By.XPATH, f'//*[@id="main-content"]/div[1]/div/div[3]/article[{j+1}]/p/span[1]').text, format='%d/%m/%Y')
                url = element_urltitle.get_attribute('href')
                title = element_urltitle.text
                df.loc[len(df)] = [title, element_date, url]
        else :
            for j in range(14) : 
                element_urltitle = driver.find_element(By.XPATH,f'//*[@id="main-content"]/div[1]/div/div[4]/article[{j+1}]/h2/a')
                element_date = pd.to_datetime(driver.find_element(By.XPATH, f'//*[@id="main-content"]/div[1]/div/div[4]/article[{j+1}]/p/span[1]').text, format='%d/%m/%Y')
                url = element_urltitle.get_attribute('href')
                title = element_urltitle.text
                df.loc[len(df)] = [title, element_date, url]

    driver.get('https://www.tradewindsnews.com/auth/user/login') 

    # CSS selector path
    css_selector_id = "#app > div.auth > div > div.form-wrapper > div > div > form > div:nth-child(1) > span:nth-child(1) > div > div.input-field-wrapper.d-flex > input"
    css_selector_pw = "#app > div.auth > div > div.form-wrapper > div > div > form > div:nth-child(1) > span:nth-child(2) > div > div.input-field-wrapper.d-flex > input"


    # Method 1: Using send_keys
    element_id = driver.find_element(By.CSS_SELECTOR, css_selector_id)
    element_id.clear()
    element_id.send_keys("ctco@gscaltex.com")

    element_pw = driver.find_element(By.CSS_SELECTOR, css_selector_pw)
    element_pw.clear()
    element_pw.send_keys("marine05")


    # Method 2: Using JavaScript Executor
    driver.execute_script(f"document.querySelector('{css_selector_id}').value = 'ctco@gscaltex.com';")
    driver.implicitly_wait(3)
    driver.execute_script(f"document.querySelector('{css_selector_pw}').value = 'marine05';")
    driver.implicitly_wait(3)

    def click_element(driver, x_path, sleep_time=3):                                # sleep_time: 요소를 기다리는 최대 시간 (기본값 3초)
        try:
            click_icon = WebDriverWait(driver, sleep_time).until(                   # WebDriverWait: 지정된 시간 동안 조건이 만족될 때까지 기다림
                EC.element_to_be_clickable((By.XPATH,  x_path))                     # EC.element_to_be_clickable: 요소가 클릭 가능한 상태가 될 때까지 기다림
            )                                                                       # By.XPATH: XPath를 사용하여 요소를 찾음
            click_icon.click()                                                      # mail_icon.click(): 찾은 요소를 클릭
            print("클릭 성공")                                                       # 클릭 성공 시 메시지 출력
        except Exception as e:                                                      # 요소를 찾지 못하거나 클릭할 수 없는 경우 예외 처리
            print(f"실패: {e}")  

    click_element(driver, '//*[@id="app"]/div[3]/div/div[3]/div/div/form/div[2]/div[1]/div/div[1]')
    driver.implicitly_wait(10)
    click_element(driver, '//*[@id="onetrust-button-group"]/div')

    for i in url_list_trw : 
        driver.get(i)
        for j in range (4) : 
            element_urltitle = driver.find_element(By.XPATH,f'//*[@id="app"]/div/div[2]/div[3]/div[1]/div[2]/div/div/div[{j+1}]/div/div/div/div[2]/div/div[2]/div/div[1]/h2/a')
            element_date = pd.to_datetime((driver.find_element(By.XPATH, f'//*[@id="app"]/div/div[2]/div[3]/div[1]/div[2]/div/div/div[{j+1}]/div/div/div/div[2]/div/div[2]/div/div[2]/span[1]').text).replace("Published", "").strip(), format='%d %B %Y %H:%M %Z')
            url = element_urltitle.get_attribute('href')
            title = element_urltitle.text
            df.loc[len(df)] = [title, element_date, url]
        for j in range (4) : 
            element_urltitle = driver.find_element(By.XPATH,f'//*[@id="app"]/div/div[2]/div[3]/div[4]/div/div[{j+1}]/div/div/div/div[2]/div/div[2]/div/div[1]/h2/a')
            element_date = pd.to_datetime((driver.find_element(By.XPATH, f'//*[@id="app"]/div/div[2]/div[3]/div[4]/div/div[{j+1}]/div/div/div/div[2]/div/div[2]/div/div[2]/span[1]').text).replace("Published", "").strip(), format='%d %B %Y %H:%M %Z')
            url = element_urltitle.get_attribute('href')
            title = element_urltitle.text
            df.loc[len(df)] = [title, element_date, url]
        for j in range (4) : 
            element_urltitle = driver.find_element(By.XPATH,f'//*[@id="app"]/div/div[2]/div[3]/div[6]/div/div[{j+1}]/div/div/div/div[2]/div/div[2]/div/div[1]/h2/a')
            element_date = pd.to_datetime((driver.find_element(By.XPATH, f'//*[@id="app"]/div/div[2]/div[3]/div[6]/div/div[{j+1}]/div/div/div/div[2]/div/div[2]/div/div[2]/span[1]').text).replace("Published", "").strip(), format='%d %B %Y %H:%M %Z')
            url = element_urltitle.get_attribute('href')
            title = element_urltitle.text
            df.loc[len(df)] = [title, element_date, url]
        for j in range (4) : 
            element_urltitle = driver.find_element(By.XPATH,f'//*[@id="app"]/div/div[2]/div[3]/div[8]/div/div[{j+1}]/div/div/div/div[2]/div/div[2]/div/div[1]/h2/a')
            element_date = pd.to_datetime((driver.find_element(By.XPATH, f'//*[@id="app"]/div/div[2]/div[3]/div[8]/div/div[{j+1}]/div/div/div/div[2]/div/div[2]/div/div[2]/span[1]').text).replace("Published", "").strip(), format='%d %B %Y %H:%M %Z')
            url = element_urltitle.get_attribute('href')
            title = element_urltitle.text
            df.loc[len(df)] = [title, element_date, url]

    df = df.drop_duplicates()
    df['date'] = pd.to_datetime(df['date'], errors='coerce').fillna(pd.Timestamp('1900-01-01'))
    df = df[~df.astype(str).apply(lambda x: x.str.contains('\\*{4}', na=False)).any(axis=1)]
    st.session_state["df"]=df



#####################################################################################################################################


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        main_tab2.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이전 대화 기록 출력
print_messages()


df = st.session_state['df']

with main_tab1 : 
    if st.button('Reset Selections'):
        st.session_state.radio1_selection = None
        st.session_state.radio2_selection = None
        st.session_state.radio3_selection = None

    # Initialize selected_title and selected_url
    selected_title = ''
    selected_url = ''

    keywords1 = st.text_input("키워드(소문자)", "vlcc,suezmax,aframax,tanker,baltic")
    keywords1_list = keywords1.split(',')
    if len(df)>0 and 'title' in df.columns:
        filtered_df = df[df['title'].str.contains('|'.join(keywords1_list), case=False)]
        filtered_df['display_date'] = filtered_df['date'].dt.strftime('%Y/%m/%d')
        radio_options = [f"**{row['title']}** - Published: {row['display_date']}" for _, row in filtered_df.iterrows()]

        selected_option1 = st.radio("요약할 기사를 선택하세요:", radio_options, index=None, key='radio1_selection',  format_func=lambda x: x)

        if selected_option1 is not None:
            selected_title = filtered_df[filtered_df['title'].apply(lambda x: f"**{x}**" in selected_option1)]['title'].iloc[0]
            selected_url = filtered_df[filtered_df['title'] == selected_title]['url'].iloc[0]

    keywords2 = st.text_input("키워드(소문자)", "iran,israel,houthi,attack")
    keywords2_list = keywords2.split(',')
    if len(df)>0 and 'title' in df.columns:
        filtered_df = df[df['title'].str.contains('|'.join(keywords2_list), case=False)]
        filtered_df['display_date'] = filtered_df['date'].dt.strftime('%Y/%m/%d')
        radio_options = [f"**{row['title']}** - Published: {row['display_date']}" for _, row in filtered_df.iterrows()]

        selected_option2 = st.radio("요약할 기사를 선택하세요:", radio_options, index=None, key='radio2_selection',  format_func=lambda x: x)

        if selected_option2 is not None:
            # Only update selected_title and selected_url if option2 is selected
            selected_title = filtered_df[filtered_df['title'].apply(lambda x: f"**{x}**" in selected_option2)]['title'].iloc[0]
            selected_url = filtered_df[filtered_df['title'] == selected_title]['url'].iloc[0]


    keywords3 = st.text_input("키워드(소문자)")
    keywords3_list = keywords3.split(',')
    if len(df)>0 and 'title' in df.columns:
        filtered_df = df[df['title'].str.contains('|'.join(keywords3_list), case=False)]
        filtered_df['display_date'] = filtered_df['date'].dt.strftime('%Y/%m/%d')
        radio_options = [f"**{row['title']}** - Published: {row['display_date']}" for _, row in filtered_df.iterrows()]

        selected_option3 = st.radio("요약할 기사를 선택하세요:", radio_options, index=None, key='radio3_selection',  format_func=lambda x: x)

        if selected_option3 is not None:
            # Only update selected_title and selected_url if option2 is selected
            selected_title = filtered_df[filtered_df['title'].apply(lambda x: f"**{x}**" in selected_option3)]['title'].iloc[0]
            selected_url = filtered_df[filtered_df['title'] == selected_title]['url'].iloc[0]

if summary_btn:
    if [selected_option1, selected_option2, selected_option3].count(None) != 2:
        st.error(
            """
            ⚠️ Error: 요약할 기사를 하나만 고르세요 
            """
        )
    # 뉴스기사 처리(GPT)
    docs = get_news(url=selected_url)
    chain = create_summary_chain()

    # 스트리밍 호출
    response = chain.stream({"context": docs})
    with main_tab2.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화기록을 저장한다.
    add_message("assistant", ai_answer)
