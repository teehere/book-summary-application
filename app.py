from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
google_key = os.getenv('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = google_key

st.title('Book Summary Application')

input_text = st.text_input("Search the book you want: ")

@st.cache_resource
def load_llm(): 
    return ChatGoogleGenerativeAI(temperature=0.8, model='gemini-3-flash-preview', max_retries=3)
llm = load_llm()

#1 - book summary
first_prompt = PromptTemplate(
    input_variables=['book_name'],
    template="Provide a concise summary (100-150 words) of the book '{book_name}'."
)
parser = StrOutputParser()
first_chain = first_prompt | llm | parser

#2 - publication date
second_prompt = PromptTemplate(
    input_variables=['book_summary'],
    template="When was the book with the following summary published: '{book_summary}'?"
)
second_chain = second_prompt | llm | parser

#3 - author info
third_prompt = PromptTemplate(
    input_variables=['book_summary'],
    template="Please tell me about the authors and provide a brief biography (2–3 sentences) of the book with the following summary: '{book_summary}'."
)
third_chain = third_prompt | llm | parser

parallel_tasks = RunnableParallel(
    publication_date=second_chain,
    author_info=third_chain
)
full_chain = (
    first_chain 
    | RunnableLambda(
        lambda summary: {
            "summary": summary,
            "results": parallel_tasks.invoke(
                {"book_summary": summary}
            )
        }
    )
)

if input_text and st.button("Generate Summary"):
    try: 
        output = full_chain.invoke({"book_name": input_text})

        st.subheader("Book Summary")
        st.write(output["summary"])

        st.subheader("Publication Date")
        st.write(output["results"]["publication_date"])

        st.subheader("Author Info")
        st.write(output["results"]["author_info"])
    except Exception as e:
        st.error(f"Error: {e}") 