from datetime import datetime

from decouple import config
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
#from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

# Adding History
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.globals import set_debug

set_debug(True)

st.title("INFORMACIÓN GENERAL DEL CURSO")
st.header("Lenguaje de Programación II (4691)")
st.subheader("4to Ciclo")

#with st.sidebar:
#    st.title("Ingrese su OpenAI API Key")
#    openai_key = st.text_input("OpenAI API Key", type="password")

#if not openai_key:
#    st.info("Ingrese su OpenAI API Key para continuar ..")
#    st.stop()
    
    

openai_key = st.secrets["OPENAI_KEY"]

#llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)

loader = PyPDFLoader("./documento_LP2.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
vector_store = Chroma.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever()

contextualize_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Configura un retriever que considera el historial de chat y utiliza el prompt de reformulación.
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_prompt
)

# Obtener la fecha actual
current_date = datetime.now().strftime("%d/%m/%Y")

system_prompt = (
    "Eres un asistente para tareas de preguntas y respuestas. "
    "Usa las siguientes piezas de contexto recuperado para responder "
    "la pregunta. Si no sabes la respuesta, di que no lo sabes. "
    "Usa un máximo de tres oraciones y mantén la respuesta concisa. "
    f"Si te preguntan acerca de fechas o preguntas del tipo '¿cuánto falta para?', "
    f"puedes usar la diferencia entre esa fecha y la fecha actual, que es {current_date}."
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

history = StreamlitChatMessageHistory()

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)



for message in st.session_state["langchain_messages"]:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

question = st.chat_input("Your Question: ")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    answer_chain = conversational_rag_chain.pick("answer")
    response = answer_chain.stream(
    #response = conversational_rag_chain.stream(
        {"input": question}, config={"configurable": {"session_id": "any"}}
    )
    with st.chat_message("assistant"):
        st.write_stream(response)
        
        
# El método pick() filtra la salida devuelta y solo devuelve el contenido de la clave que pasó como argumento. 
# De esta manera, el objeto de respuesta ahora es un flujo que devuelve solo la respuesta
