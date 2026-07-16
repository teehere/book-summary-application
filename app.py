from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.agents import create_agent
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
google_key = os.getenv('google_key')
os.environ['GOOGLE_API_KEY'] = google_key

st.title('Book Summary Application')

input_book = st.text_input("Search the book you want: ")

option = st.radio(
    "Choose an action", ("Generate Summary", "Search by Wikipedia")
)

@st.cache_resource
def load_llm(): 
    return ChatGoogleGenerativeAI(temperature=0.8, model='gemini-3-flash-preview', max_retries=3)
llm = load_llm()
parser = StrOutputParser()

#1 - book summary
first_prompt = PromptTemplate(
    input_variables=['book_name'],
    template="Provide a concise summary (100-150 words) of the book '{book_name}'."
)
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

wiki = WikipediaQueryRun(
    api_wrapper = WikipediaAPIWrapper() 
)

agent = create_agent(
    model=llm,
    tools=[wiki]
)

if input_book and st.button("Run"):

    if option == "Generate Summary":

        try:
            with st.spinner("Generating..."):
                output = full_chain.invoke({"book_name": input_book})

                st.subheader("Book Summary")
                st.write(output["summary"])

                st.subheader("Publication Date")
                st.write(output["results"]["publication_date"])

                st.subheader("Author Info")
                st.write(output["results"]["author_info"])
        except Exception as e:
            st.error(f"Error: {e}") 
    
    else:
        try:
            with st.spinner("Searching Wikipedia..."):

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Tell me about the book {input_book} using Wikipedia."
                            }
                        ]
                    }
                )

            st.subheader("Wikipedia Result")
            st.write(response["messages"][-1].content)

        except Exception as e:
            st.error(f"Error: {e}") 
                