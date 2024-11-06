from langchain.tools import tool
from typing import List, Dict, Annotated, Optional
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import pandas as pd

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class ColumnMappingGenerator:
    def __init__(self):
        self.PREFIX_PROMPT = "[IMPORTANT] You should follow `Column Mapping Guidelines`. DO NOT ignore it.\n"
        self.prompt_template = """You are an experienced data analyst tasked with generating column mapping guidelines for a given dataframe. Your goal is to create a clear mapping between user-friendly terms (potentially in Korean) and the actual column names in the dataframe.

Here is the user's request for column names:

<user_request>
{user_request}
</user_request>

Here are the first 5 rows of the dataframe:

<dataframe_head>
{dataframe_head}
</dataframe_head>

Please follow these steps to generate the column mapping guidelines:

1. Analyze the dataframe and user request:
   Examine the dataframe head and user request provided above. In <data_exploration> tags:
   - List the column names you observe in the dataframe.
   - Categorize each column (e.g., numerical, categorical, date/time).
   - Identify potential user terms from the user request.
   - Note any insights about the data types or content of these columns.
   - Identify any potential data quality issues or anomalies in the sample data.
   - Consider possible variations or synonyms of user terms.

2. Match user terms with column names:
   Based on your analysis, attempt to match user terms with actual column names. In <term_matching> tags:
   - List each user term and its potential corresponding column name.
   - Note any difficulties or ambiguities in making these matches.
   - Consider data types, content of columns, and potential variations when making these mappings.

3. Create column mappings:
   Based on your term matching, create a mapping between the user-friendly terms and the actual column names. Include both Korean terms (if provided) and English translations.

4. Generate the column mapping guidelines:
   Present your findings in the following format:

<column_mapping_guidelines>
Column Mapping Guidelines:

When users refer to columns using alternative names or terms, use the following mapping to find the appropriate column:

- [User Term] ([English Translation]): Find columns including '[ActualColumnName]' in the name
- [User Term] ([English Translation]): Find columns including '[ActualColumnName]' in the name
[...continue for all identified mappings...]

[Include instructions for handling unmapped columns or user terms]

</column_mapping_guidelines>

5. Final check:
   In <final_check> tags, review your output to ensure:
   - All user terms are addressed
   - All dataframe columns are accounted for
   - The guidelines are clear and easy to follow
   - Any data quality issues or anomalies are appropriately addressed in the guidelines

Remember to maintain a professional tone and provide clear, concise instructions in your output. If you're unsure about any mappings, state your assumptions clearly."""

        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["user_request", "dataframe_head"],
        )
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.column_guideline_chain = (
            self.prompt | self.llm | StrOutputParser() | self.extract_column_mapping
        )

    def extract_column_mapping(self, text):
        start_tag = "<column_mapping_guidelines>"
        end_tag = "</column_mapping_guidelines>"

        start_index = text.find(start_tag)
        end_index = text.find(end_tag)

        if start_index != -1 and end_index != -1:
            extracted_content = text[start_index + len(start_tag) : end_index].strip()
            return self.PREFIX_PROMPT + extracted_content + "\n\n###"
        else:
            return ""

    def generate_column_mapping(self, user_request, dataframe_head):
        return self.column_guideline_chain.invoke(
            {"user_request": user_request, "dataframe_head": dataframe_head}
        )


class DataAnalysisAgent:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        model_name: str = "gpt-4o-mini",
        prefix_prompt: Optional[str] = None,
        postfix_prompt: Optional[str] = None,
        column_guideline: Optional[str] = None,
    ):
        self.df = dataframe
        self.model_name = model_name
        self.prefix_prompt = prefix_prompt
        self.postfix_prompt = postfix_prompt
        if column_guideline is not None and column_guideline.strip() != "":
            self.update_column_guideline(column_guideline)
        else:
            self.column_guideline = ""
        self.tools = [self.create_python_repl_tool()]
        self.store = {}
        self.setup_agent()

    def update_column_guideline(self, user_request):
        column_mapping_generator = ColumnMappingGenerator()
        self.column_guideline = column_mapping_generator.generate_column_mapping(
            user_request, self.df.head()
        )

    def create_python_repl_tool(self):
        @tool
        def python_repl_tool(
            code: Annotated[str, "실행할 파이썬 코드 (차트 생성용)"],
        ):
            """파이썬, 판다스 쿼리, matplotlib, seaborn 코드를 실행하는 데 사용합니다."""
            try:
                python_tool = PythonAstREPLTool(
                    locals={"df": self.df, "sns": sns, "plt": plt}
                )
                return python_tool.invoke(code)
            except Exception as e:
                return f"실행 실패. 오류: {repr(e)}"

        return python_repl_tool

    def build_system_prompt(self):
        system_prompt = f"""<CACHE_PROMPT>You are an advanced Data Analysis Agent, expertly skilled in using Pandas, matplotlib, and seaborn for data manipulation and visualization tasks. Your primary function is to generate code-only responses to user queries, without providing any explanations or comments.

Here's the head of the DataFrame you'll be working with:

<dataframe_head>
{self.df.head()}
</dataframe_head>

###

Important Guidelines:

1. Data Analysis Tasks:
   - Use Pandas DataFrame operations exclusively.
   - Do not create or overwrite the `df` variable in your code.
   - Generate only the Pandas query, nothing else.

2. Visualization Tasks:
   - Use either matplotlib or seaborn (preferred) for generating visualization code.
   - Include `plt.show()` at the end of your visualization code.
   - Apply the following styling preferences:
     * Use 'English' for visualization titles and labels.
     * Set the color palette to 'muted'.
     * Use a white background.
     * Remove grid lines.
     * Set the `cmap` or `palette` parameter for seaborn plots when applicable.

3. Response Format:
   - For data analysis tasks: Generate only the Pandas query code.
   - For visualization tasks: Generate only the matplotlib or seaborn code.

4. Language:
   - Although the final output should be code-only, if any text is required, use Korean.
   - BUT, titles and labels in visualization should be in English.

Before generating your code response, provide your analysis inside <analysis> tags:

1. Categorize the user's query as either a data analysis or visualization task.
2. Identify the specific operations or visualizations required.
3. List the relevant columns from the dataframe head for this task, using the column mapping guidelines if necessary.
4. Consider any potential data quality issues or edge cases, such as missing values or data type conflicts.
5. For data analysis tasks:
   - Outline the sequence of Pandas operations needed to accomplish the task
   - Consider any necessary data transformations or aggregations
6. For visualization tasks:
   - List potential plot types suitable for the task
   - Justify your choice of the most appropriate plot type
   - Outline any data preparation steps needed before plotting
7. Plan the structure of your code response, ensuring it adheres to all guidelines mentioned above.
8. Review your planned code to ensure it's the most efficient and effective solution for the given task.

Remember: 
- Your final output should be an answer with a tone of a professional data analyst. Keep it concise and professional.
- Use your expertise to generate the most efficient and effective answer for the given task.
- Be sure to use the `df` variable to access the dataframe.
- Use function call(`python_repl_tool`) to return the result.

###

{self.column_guideline}

Now, respond to the user's query accordingly.</CACHE_PROMPT>"""

        if self.prefix_prompt:
            system_prompt = f"{self.prefix_prompt}\n\n{system_prompt}"

        if self.postfix_prompt:
            system_prompt = f"{system_prompt}\n\n{self.postfix_prompt}"

        return system_prompt

    def setup_agent(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.build_system_prompt(),
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        llm = ChatOpenAI(model=self.model_name, temperature=0)
        agent = create_tool_calling_agent(llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=20,
            max_execution_time=60,
            handle_parsing_errors=True,
        )

    def get_session_history(self, session_id):
        return self.store.setdefault(session_id, ChatMessageHistory())

    def get_agent_with_chat_history(self):
        return RunnableWithMessageHistory(
            self.agent_executor,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def stream(self, input_query, session_id="abc123"):
        agent_with_chat_history = self.get_agent_with_chat_history()
        response = agent_with_chat_history.stream(
            {"input": input_query},
            config={"configurable": {"session_id": session_id}},
        )
        return response

    # def stream(self, user_input, session_id="abc123"):
    #     agent_with_chat_history = self.get_agent_with_chat_history()
    #     response = agent_with_chat_history.stream(
    #         {"input": user_input},
    #         config={"configurable": {"session_id": session_id}},
    #     )
    #     return response
