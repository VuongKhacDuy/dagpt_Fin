
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import Literal
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.logger.base import BaseLogger
from src.models.llms import load_llm
from openai import OpenAI

logger = BaseLogger()
def execute_plt_code(code_string, df=None):
    try:
        # Clean the code string - remove markdown code block markers
        code_string = code_string.replace('```python', '').replace('```', '').strip()
        
        # Clear any existing plots
        plt.close('all')
        
        # Create a clean namespace
        local_namespace = {}
        global_namespace = {
            'pd': pd,
            'plt': plt,
            'df': df,
        }
        
        # Execute the code as a whole block
        exec(code_string, global_namespace, local_namespace)
        
        # Get the current figure
        fig = plt.gcf()
        plt.close('all')  # Clean up after getting the figure
        
        return fig
        
    except Exception as e:
        st.error(f"Error executing plot code: {str(e)}")
        logger.error(f"Plot execution error: {str(e)}\nCode:\n{code_string}")
        plt.close('all')  # Clean up on error
        return None
