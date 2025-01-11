import streamlit as st
import pygwalker as pyg
import pandas as pd

def main():
    #set up streamlit interface
    st.set_page_config(
        page_title="Interactive Visualization Tool", 
        page_icon="",layout="wide"
    )

    st.header("Interactive Visualization Tool")
    st.write("### Welcome to interactive visualization tool!")
    
    #Render pygwalker
    if st.session_state.get("df") is not None:
        # Convert to pandas DataFrame if it's not already
        if not isinstance(st.session_state.df, pd.DataFrame):
            df = st.session_state.df.to_pandas()
        else:
            df = st.session_state.df
            
        # Render the visualization using the current API
        pyg_html = pyg.to_html(df)
        st.components.v1.html(pyg_html, height=1000)

    else:
        st.info("Please upload a data set to begin using")

if __name__ == "__main__":
    main()