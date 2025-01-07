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
from src.Logger.base import BaseLogger
from src.models.llms import load_llm
from openai import OpenAI


# Load environment variables
load_dotenv()
openApi_key = os.getenv("OPENAPI_KEY_API")

logger = BaseLogger()
# MODEL_NAME = "gpt-3.5-turbo-instruct"
MODEL_NAME = "gemini-1.5-flash"

# def process_query(query, da_agent):
#     # Process query
#     if query == "show data":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show columns":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show shape":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show info":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show missing values":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data types":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show unique values":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show summary statistics":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show correlation":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)
#         st.write(response)
#     elif query == "show data distribution":
#         response = da_agent(query)

        
def process_query(da_agent, query):
    try:
        response = da_agent(query)
        
        # If response is a string, display it directly
        if isinstance(response, str):
            st.write(response)
            st.session_state.history.append((query, response))
            return
            
        # If response is a dictionary with intermediate steps
        if isinstance(response, dict) and "intermediate_steps" in response:
            # Check if there are any intermediate steps
            if response["intermediate_steps"] and len(response["intermediate_steps"]) > 0:
                last_step = response["intermediate_steps"][-1]
                
                # Check if the last step contains plotting code
                if isinstance(last_step, tuple) and len(last_step) > 0:
                    action = last_step[0].tool_input.get("query", "")
                    
                    if "plt" in str(action):
                        st.write(response.get("output", ""))
                        fig = execute_plt_code(action, df=st.session_state.df)
                        if fig:
                            st.pyplot(fig)
                        st.write("*Executed code:*")
                        st.code(action)
                        to_display_string = f"{response.get('output', '')}\n```python\n{action}\n```"
                        st.session_state.history.append((query, to_display_string))
                        return
            
            # If no plotting is involved, display the output
            output = response.get("output", str(response))
            st.write(output)
            st.session_state.history.append((query, output))
        else:
            # Fallback for any other type of response
            st.write(str(response))
            st.session_state.history.append((query, str(response)))
            
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        logger.error(f"Error in process_query: {str(e)}")
        st.session_state.history.append((query, f"Error: {str(e)}"))
        
def display_chat_history():
    st.markdown("## Chat History: ")
    for i, (q, r) in enumerate(st.session_state.history):
        st.markdown(f"*Query: {i+1}:* {q}")
        st.markdown(f"*Response: {i+1}:* {r}")
        st.markdown("---")


def main():
    # Setup streamlit interface
    st.set_page_config(
        page_title="Smart Data Analysis Tool",
        page_icon="📊",
        layout="centered",
    )
    st.header("📊 Smart Data Analysis Tool")
    st.write("### Welcome to the Smart Data Analysis Tool, Upload your data and ask questions to get insights")

    # Load llm model
    llm = load_llm(model_name=MODEL_NAME)
    logger.info(f"Model loaded successfully {MODEL_NAME} ! ##############################")

    # Upload csv file
    with st.sidebar:
        st.title("Upload your data")
        uploaded_file = st.file_uploader("Choose a file", type='csv')
        # if uploaded_file is not None:
        #     data = pd.read_csv(uploaded_file)
        #     data.encode('utf-8').strip()
        #     st.write(data)
        #     st.write("Data uploaded successfully")
            
    # Initial chat history and DataFrame
    if "history" not in st.session_state:
        st.session_state.history = []
    if "df" not in st.session_state:
        st.session_state.df = None

    # Read csv file
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("### Your upload data: ", st.session_state.df)

    # Create data analysis agent only if DataFrame exists
    if st.session_state.df is not None:
        da_agent = create_pandas_dataframe_agent(
                llm=ChatGoogleGenerativeAI(
                    # model="gemini-1.5-pro",
                    model= MODEL_NAME,
                    temperature=0,
                    google_api_key=os.getenv("GEMINI_KEY_API")
                ),
                df=st.session_state.df,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                allow_dangerous_code=True,
                verbose=True,
                return_intermediate_steps=True,
            )
        logger.info(f"Data analysis agent created successfully ##############################")

        # Input query and process query
        query = st.text_input("Ask your question", "")
        if st.button("Ask"):
            with st.spinner("Processing..."):
                process_query(da_agent, query)
                logger.info(f"Query processed successfully ##############################")
    else:
        st.warning("Please upload a CSV file first.")

    # Display chat history
    display_chat_history()

if __name__ == "__main__":
    main()