import requests
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

def get_llm_response(query, chat_history, context):
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation and the context below:

    Chat history : {chat_history}

    Context:
    {context}

    User question : {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGroq(model="llama-3.1-70b-versatile")

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "context": context,
        "user_question": query
    })

def main():
    st.set_page_config(page_title='Streaming Chatbot', page_icon='🤖')

    st.header("Streaming Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content='Hello, I am a Bot. How can i help you? ')
        ]

    context = None
    if 'context' not in st.session_state:
        url = 'https://raw.githubusercontent.com/M00N69/nconfgroq/main/IFS_Food_v8_audit_checklist_guideline_v1_EN_1706090430.txt'
        response = requests.get(url)
        context = response.text
        st.session_state.context = context
    else:
        context = st.session_state.context

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('Human'):
                st.write(message.content)

    user_input = st.chat_input('Type your message here...')

    if user_input is not None and user_input != "":
        st.session_state.chat_history.append(HumanMessage(content=user_input))

        with st.chat_message("Human"):
            st.markdown(user_input)

        with st.chat_message("AI"):
            response = st.write_stream(get_llm_response(user_input, st.session_state.chat_history, context))

        st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()

if __name__=="__main__":
    main()
