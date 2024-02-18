# Ask Your Data
Ask any questions about your PDF or CSV documents powered by [OpenAI](https://platform.openai.com/docs/overview) LLMs and [PandasAI](https://github.com/Sinaptik-AI/pandas-ai).
<p float="left">
  <img src="media/AppScreenshot.png?raw=true" width="95%"/> 
<p/>

## About
This project is created to understand the working of LLMs using RAG on uploaded data. I started this project with [AutoGen](https://github.com/microsoft/autogen.git) library using agent-based architecture to create a data query wrapper for OpenAI's ChatGPT. It was quite fascinating how the agents worked and received excellent outputs in a few areas. Unfortunately, many times the model would hallucinate or lose context resulting in inaccuracy while using agents (Also the library was very new at the time). Then I discovered PandasAI library which was a good wrapper around OpenAI's ChatGPT and Langchain. I explored the library and hosted the project on Streamlit.
  
## Setup
Clone this repository from the terminal
```
git clone https://github.com/IrshadG/Ask-Your-Data.git
cd 'Ask-Your-Data'
```
or
```
gh repo clone IrshadG/Ask-Your-Data
cd 'Ask-Your-Data'
```


Install the following dependencies required to run the project
```
pip install streamlit
pip install langchain
pip install PyPDF2
```

## Usage
**[Note]**
Please make sure to enter your own `OPENAI_API_KEY` into *`keys.py`* file before use.

To run the application, enter this command into your terminal:
```
streamlit run app.py
```

## Acknowledgment
This project is based on [PandasAI](https://github.com/Sinaptik-AI/pandas-ai) library and hosted via [Streamlit](https://streamlit.io/)
