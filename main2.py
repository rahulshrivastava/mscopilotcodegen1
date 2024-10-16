# This is the version of the data transformation that works with chatGPT


from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Any, Dict, List, Union
from dotenv import load_dotenv
import os
import autopep8
import pandas as pd

class datachat():
    def __init__(self,file_path):
        load_dotenv()
        OPENAI_API_TYPE = os.environ['OPENAI_API_TYPE']
        OPENAI_API_BASE = os.environ['OPENAI_API_BASE']
        OPENAI_API_VERSION = os.environ['OPENAI_API_VERSION']
        OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
        DEPLOYEMENT_NAME = os.environ['DEPLOYEMENT_NAME']

        self.llm = AzureChatOpenAI(deployment_name=DEPLOYEMENT_NAME,
                    temperature=0.0,
                    max_tokens=4000,
                    )


        self.instruction = """
        As a python coder create a pythonic response for the query with reference to the columns in my pandas dataframe{columns}.
        Instruction:
        Do not write the whole script just give me a pythonic response for this query and do not extend more than asked. Assume a dataframe variable df_temp. Always store the target dataframe on df_temp.
        Enclose the generated code in Markdown code embedding format. Do not generate sample output. Answer the question and provide a one-line explanation and stop.

        example:
        ```python
        df = df['region'].unique()
        ```
                
        question: {input}

        answer:

        """
        self.file_path=file_path

    def extract_code(self,response):
        start = 0
        q = ""
        temp_block=""
        for line in response.splitlines(): 
            if '```python' in line and start==0:
                start=1
            if '```' == line.strip() and start==1:
                start =0
                break
            if start ==1 and '```' not in line:
                q=q+'\n'+line
        q=q+'\n' + "df_temp.to_csv('./data/output4mLLM/output.csv',index=False)"
        return q


    def data_ops(self,query):
        if os.path.isfile('./data/output4mLLM/output.csv'):
            df=pd.read_csv('./data/output4mLLM/output.csv',low_memory=False) 
        else:
            df=pd.read_csv(self.file_path,low_memory=False) 
        query = query 
        columns=df.columns.tolist()
        prompt = PromptTemplate.from_template(self.instruction)
        agent = LLMChain(llm=self.llm,prompt=prompt)
        response = agent.invoke(input={"columns":columns,"input":query})
        response = self.extract_code(response['text'])
        gencode=autopep8.fix_code(response)
        df_temp=df.copy()
        print(gencode)
        exec(gencode)
        #df_temp.to_csv('./data/output4mLLM/output.csv',index=False)
        df_temp=pd.read_csv('./data/output4mLLM/output.csv',low_memory=False)
        return df_temp.head(200)


"""

remove leading and trailing spaces from the column Item type
give me the count of records for Item Type as Household and Sales Channel as Offline from the country United Kingdom
add a column for discount if units sold < 5000 discount = 10% of total profit else 20% of total profit
mask the column total profit as *********
"""