# import streamlit as st
# from rag import SurveyRAG
# import pandas as pd

# st.set_page_config(page_title="Survey RAG App", layout="wide")

# # Initialize RAG handler
# if "rag" not in st.session_state:
#     st.session_state["rag"] = SurveyRAG()
# if "summary" not in st.session_state:
#     st.session_state["summary"] = ""

# st.title("📊 Survey Feedback Summarizer and Q&A")

# uploaded_file = st.file_uploader("Upload survey file (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])

# if uploaded_file is not None:
#     st.success("File uploaded successfully!")
#     if st.button("Generate Summary"):
#         with st.spinner("Analyzing survey responses..."):
#             df = st.session_state["rag"].load_file(uploaded_file)
#             st.session_state["rag"].build_db(df)
#             st.session_state["summary"] = st.session_state["rag"].summarize()
#         st.success("Summary generated!")

# if st.session_state["summary"]:
#     st.subheader("Summary")
#     st.write(st.session_state["summary"])

#     st.subheader("Chat with your survey data")
#     question = st.text_input("Ask a question about the responses:")
#     if st.button("Get Answer") and question:
#         with st.spinner("Thinking..."):
#             answer = st.session_state["rag"].query(question)
#         st.markdown(f"**Answer:** {answer}")


import streamlit as st
from rag import SurveyRAG
import pandas as pd

st.set_page_config(page_title="Survey RAG App", layout="wide")

# Model selection
model_choice = st.selectbox(
    "Choose Ollama model:",
    options=["llama3.2", "llama3.2:1b"],
    index=0
)

# Initialize RAG handler
if "rag" not in st.session_state or st.session_state.get("model_choice") != model_choice:
    st.session_state["rag"] = SurveyRAG(model_name=model_choice)
    st.session_state["model_choice"] = model_choice

if "summary" not in st.session_state:
    st.session_state["summary"] = ""

st.title("📊 Survey Feedback Summarizer and Q&A")

uploaded_file = st.file_uploader("Upload survey file (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    if st.button("Generate Summary"):
        with st.spinner("Analyzing survey responses..."):
            df = st.session_state["rag"].load_file(uploaded_file)
            st.session_state["rag"].build_db(df)
            st.session_state["summary"] = st.session_state["rag"].summarize()
        st.success("Summary generated!")

if st.session_state["summary"]:
    st.subheader("Summary")
    st.write(st.session_state["summary"])

    st.subheader("Chat with your survey data")
    question = st.text_input("Ask a question about the responses:")
    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            answer = st.session_state["rag"].query(question)
        st.markdown(f"**Answer:** {answer}")
