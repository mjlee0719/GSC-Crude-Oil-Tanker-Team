import streamlit as st
import PyPDF2
import re
import os
import openpyxl
import pandas as pd
from typing import List, Union
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_teddynote.messages import AgentStreamParser, AgentCallbacks
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dataanalysis import DataAnalysisAgent
from typing import List, Union
import win32com.client
from datetime import datetime, date
from PyPDF2 import PdfReader
import pythoncom


with st.sidebar:
    csv=st.file_uploader("Template 업로드", type=["csv"], key="uploader1")

    # Get today's date
    today = date.today()
    # Create date input fields
    start_date = st.date_input(
        "Start Date",
        value=today,
        min_value=date(2000, 1, 1),
        max_value=today,
        help="Select the start date"
    )
    end_date = st.date_input(
        "End Date",
        value=today,
        min_value=start_date,
        max_value=today,
        help="Select the end date"
    )
    
    st.write("Selected date range:", start_date, "to", end_date)
    if start_date > end_date:
        st.error("Error: End date must be after start date")
    
    if st.button('Index 추출'):
        #######이메일 자동 다운로드#######
        # Initialize COM
        pythoncom.CoInitialize()
        
        try:
            # Initialize Outlook
            outlook = win32com.client.Dispatch("Outlook.Application")
            namespace = outlook.GetNamespace("MAPI")
            inbox = namespace.GetDefaultFolder(6)

            # Set the download path
            download_path = r"D:\Application_Temp\Index Update\Download"

            # Create the directory if it doesn't exist
            if not os.path.exists(download_path):
                os.makedirs(download_path)

            # Filter messages
            messages = inbox.Items
            messages.Sort("[ReceivedTime]", True)
            messages = messages.Restrict("[ReceivedTime] >= '" + start_date.strftime('%m/%d/%Y') + 
                                    "' AND [ReceivedTime] <= '" + end_date.strftime('%m/%d/%Y') + "'")

            for message in messages:
                try:
                    subject = message.Subject
                    print(f"Processing email with subject: {subject}")
                    
                    if "Please find attached BDTI, BCTI, BTI, BLPG, BITR" in subject:
                        date_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', subject)
                        if date_match:
                            date_str = date_match.group(1)
                            date_obj = datetime.strptime(date_str, '%d.%m.%Y')
                            formatted_date = date_obj.strftime('%Y%m%d')
                            
                            for attachment in message.Attachments:
                                if "Baltic Dirty Tanker Index" in attachment.FileName:
                                    new_filename = f"BDTI_{formatted_date}.pdf"
                                    full_path = os.path.join(download_path, new_filename)
                                    print(f"Saving file: {full_path}")
                                    attachment.SaveAsFile(full_path)
                    
                    elif "Platts Dirty Tankerwire" in subject:
                        for attachment in message.Attachments:
                            if "DK" in attachment.FileName:
                                full_path = os.path.join(download_path, attachment.FileName)
                                print(f"Saving file: {full_path}")
                                attachment.SaveAsFile(full_path)
                                
                except Exception as e:
                    print(f"Error processing message: {str(e)}")

        finally:
            # Clean up
            outlook = None
            pythoncom.CoUninitialize()

        def get_date_from_filename(filename):
            # Extract the date part (last 6 characters before the extension)
            date_str = filename[-10:-4]  # Gets "241030" from "BDTI_241030.pdf"
            # Convert to datetime
            return datetime.strptime(f"20{date_str}", "%Y%m%d")

        def get_type_from_filename(filename):
            # Split by underscore and take the first part
            return filename.split('_')[0]

        def extract_pdf_text(file_path):
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                return ""

        def process_pdfs(folder_path, start_date, end_date):
            data = []
            
            # Convert dates to string format
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            # Convert string dates to datetime objects for comparison
            start_datetime = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_date_str, "%Y-%m-%d")
            
            # List all PDF files in the directory
            for filename in os.listdir(folder_path):
                if filename.endswith('.pdf'):
                    try:
                        # Get file date
                        file_date = get_date_from_filename(filename)
                        
                        # Check if file is within date range
                        if start_datetime <= file_date <= end_datetime:
                            file_path = os.path.join(folder_path, filename)
                            file_type = get_type_from_filename(filename)
                            
                            # Extract text
                            text = extract_pdf_text(file_path)
                            
                            # Add to data list
                            data.append({
                                'type': file_type,
                                'date': file_date,
                                'text': text
                            })
                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")
                        continue
            
            # Create DataFrame
            df = pd.DataFrame(data)
            return df

        # Usage
        folder_path = r"D:\Application_Temp\Index Update\Download"
        df = process_pdfs(folder_path, start_date, end_date)

        # Sort DataFrame by date
        df = df.sort_values('date')

######################################################################################################################################

##Tankerwire##

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
text_list_platts = df[(df['date'] >= start_date) & (df['date'] <= end_date) & (df['type'] == 'DK')]['text'].tolist()
text_list_bitr = df[(df['date'] >= start_date) & (df['date'] <= end_date) & (df['type'] == 'BDTI')]['text'].tolist()

def find_float_after_code(data, code):
    # Pattern to match: code + two spaces + number (with optional comma, decimal point, or negative sign)
    pattern = re.escape(code) + r'  (-?[\d,]+(?:\.\d{2})?)'
    
    # Find the first match
    match = re.search(pattern, data)
    
    if match:
        # Extract the number string
        number_str = match.group(1)
        
        # Remove commas and convert to float
        number = float(number_str.replace(',', ''))
        
        return number
    
    # If no number is found, return None
    return None


def extract_codes(data):
    # Pattern to match ABCDE00 or ABCDEFG
    pattern = r'\b([A-Z]{5}\d{2}|[A-Z]{7})\b'
    
    # Find all matches
    matches = re.findall(pattern, data)
    
    # Remove duplicates by converting to a set and back to a list
    unique_codes = list(set(matches))
    
    return unique_codes

code_list_platts = []
for i in text_list_platts:
    code_list_platts.append(extract_codes(i))

index_list_platts = []
index_dict_platts = {}

for idx, codes in enumerate(code_list_platts):
    for code in codes:
        value = find_float_after_code(text_list_platts[idx], code)
        index_dict_platts[code] = value 
    index_list_platts.append(index_dict_platts.copy())
    index_dict_platts.clear()





def find_float_after_code2(data, input_code):
    lines = data.split('\n')
    filtered_lines = [line for line in lines if input_code in line]
    if not filtered_lines:
        return []
    line = filtered_lines[0]
    line_without_parentheses = re.sub(r'\([^)]*\)', '', line)
    numbers = re.findall(r'-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?', line_without_parentheses)
    converted_numbers = [float(num.replace(',', '')) for num in numbers]
    return converted_numbers[2]


code_list_bitr=['TD20', 'TD18', 'TD6', 'TD22-TCE', 'TD19', 'TD23-TCE', 'TD8', 'TD22', 'TD9', 'TD2-TCE', 'TD21', 'TD23', 'TD25', 'TD15', 'TD7', 'TD25-TCE', 'TD15-TCE', 'TD18-TCE', 'TD6-TCE', 'TD14', 'TD19-TCE', 'TD8-TCE', 'TD26-TCE', 'TD7-TCE', 'TD27-TCE', 'TD27', 'TD21-TCE', 'TD14-TCE', 'TD9-TCE', 'TD2', 'TD26', 'TD20-TCE', 'TD3C', 'TD3C-TCE']

index_dict_bitr = {}
index_list_bitr = []

for text in text_list_bitr:
    for code in code_list_bitr:
        try:
            value = find_float_after_code2(text, code)
            index_dict_bitr[code] = value
        except:
            index_dict_bitr[code] = None
    index_list_bitr.append(index_dict_bitr.copy())
    index_dict_bitr.clear()






combined_list = []
for dict_A, dict_B in zip(index_list_bitr, index_list_platts):
    new_row = {}
    # Selectively add or rename keys
    new_row['Date'] = None
    new_row['TD3'] = None
    new_row['TD3C'] = dict_A['TD3C']
    new_row['TD15'] = dict_A['TD15']
    new_row['TD22'] = dict_A['TD22']
    new_row['TD20'] = dict_A['TD20']
    new_row['TD8'] = dict_A['TD8']
    new_row['TD14'] = dict_A['TD14']
    new_row['Caribbean ⇒ China'] = dict_B['TDAFL00']
    new_row['USGC ⇒ China'] = dict_B['TDUCB00']
    new_row['H.Point ⇒ F.East'] = dict_B['TDDHQ00']
    new_row['AG ⇒ East'] = dict_B['PFAGK10']
    new_row['WAF ⇒ F.East'] = dict_B['PFAHZ10']
    new_row['USGC ⇒ Spore'] = dict_B['TDUGC00']
    new_row['B.Sea ⇒ Med Suez'] = dict_B['TDADQ00']
    new_row['B.Sea-F.East'] = dict_B['DBSFB00']
    new_row['US STS Suez'] = None
    new_row['USGC ⇒ UKC Suez'] = None
    new_row['AG ⇒ East Afra'] = dict_B['PFAJD10']
    new_row['Indonesia ⇒ Korea'] = dict_B['PFALO10']
    new_row['Aus. ⇒ N.Asia'] = dict_B['TDAFA00']
    new_row['B.Sea ⇒ Med Afra'] = dict_B['TDADT00']
    new_row['USGC ⇒ UKC Afra'] = dict_B['TDUCG00']
    new_row['Kozmino/KR'] = None
    new_row['US STS Afra'] = None
    new_row['']=''
    new_row['터키해협(N)'] = dict_B['AAWIK00']
    new_row['터키해협(S)'] = dict_B['AAWIL00']
    combined_list.append(new_row)

for i, index in enumerate(combined_list):
    index['Date'] = (start_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d')




df = pd.DataFrame(combined_list)

df


# def extract_numbers_sigma(text, pattern, char_range, index, multiplier=1):
#     start_pos = text.find(pattern)
#     if start_pos == -1:
#         return None
    
#     substring = text[start_pos + len(pattern):start_pos + len(pattern) + char_range]
#     numbers = re.findall(r'\d+(?:\.\d+)?', substring)
    
#     if len(numbers) <= index:
#         return None
    
#     return float(numbers[index]) * multiplier

# def process_text_sigma(text):
#     result = {}

#     # Extract and process 'BSS\xa03\xa0DAYS' for USSTSAfra (first number)
#     ussts_afra = extract_numbers_sigma(text, 'BSS\xa03\xa0DAYS', 12, 0, 1000)
#     if ussts_afra is not None:
#         result['USSTSAfra'] = ussts_afra

#     # Extract and process 'BSS\xa03\xa0DAYS' for USSTSSuez (second number)
#     ussts_suez = extract_numbers_sigma(text, 'BSS\xa03\xa0DAYS', 12, 1, 1000)
#     if ussts_suez is not None:
#         result['USSTSSuez'] = ussts_suez

#     # Extract and process 'KOZMINO/KOREA' for Kozminokr (third number)
#     kozmino_value = extract_numbers_sigma(text, 'KOZMINO/KOREA', 20, 2, 1000)
#     if kozmino_value is not None:
#         result['Kozminokr'] = kozmino_value

#     return result