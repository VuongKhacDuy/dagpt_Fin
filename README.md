# Data Analysis Gemini

## Description:
    ### Application Name: Data Analysis Tool

    ### Purpose:
        Engage in chat-based Q&A sessions related to data provided in CSV files.
        Analyze datasets from uploaded CSV files to extract meaningful insights.
        Visualize data effectively through charts and graphs.

    ### Technologies Used:
        gemini-1.5-flash: Advanced language model for intelligent and conversational responses.
        langchain_experimental: Framework for building custom applications using language models.
        llms (Large Language Models): Powering natural language processing for intuitive interaction.
        pandas: For data manipulation and analysis.
        matplotlib: For creating detailed data visualizations.
        streamlit: Interactive and user-friendly interface for smooth data interaction.
        This tool simplifies complex data analysis processes, making it accessible for both technical and non-technical users.

## Setup
    Install cookiecutter-data-science (template project structure):
        pip install cookiecutter-data-science

    Create virtual environment:
        python -m venv .venv

    Activate virtual environment:
        source .venv/bin/activate

    Install dependencies:
    pip install -r requirements.txt

Create project
cookiecutter https://github.com/drivendata/cookiecutter-data-science


## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------