import os
from langchain_openai import ChatOpenAI
#https://api.openai.com/v1/chat/completions?model=gpt-3.5-turbo
import google.generativeai as genai

# openApi_key = os.getenv("OPENAPI_KEY_API")
geminiApi_key = os.getenv("GEMINI_KEY_API")


# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def load_llm(model_name):

    """ Load Language Model
    Args:
        model_name (str): Name of the model to load
    Raises:
        ValueError: If the model name is not recognized
    
    Returns:
    _type_: Language Model
    
    """

    if model_name == "gpt-3.5-turbo":
        return ChatOpenAI(
            model=model_name, 
            api_key=openApi_key,
            temperature=0, 
            max_tokens=1000
        )
    elif model_name == "gpt-3.5-turbo-davinci":
        return ChatOpenAI(
            model=model_name, 
            api_key=openApi_key,
            temperature=0, 
            max_tokens=1000
        )
    elif model_name == "gpt-4":
        return ChatOpenAI(
            model=model_name, 
            api_key=openApi_key,
            temperature=0, 
            max_tokens=1000
        )
    
    elif model_name == "gemini-1.5-flash":
        model = genai.GenerativeModel(model_name)
        return model
    
    elif model_name == "gemini-1.5-pro":
        model = genai.GenerativeModel(model_name)
        return model
    
    else:
        raise ValueError(
            "Unknown model.\
                Please choose from: gpt-3.5-turbo, gpt-3.5-turbo-davinci, gpt-4, gemini-pro, etc."
        )