# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from pinecone import Pinecone, ServerlessSpec
import json, ast, time, argparse, datetime
import pandas as pd
import random
import streamlit as st
from streamlit_utils import sidebar
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Pinecone as lang_pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as lang_pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import warnings
warnings.filterwarnings("ignore")
# from langchain_community.embeddings import OpenAIEmbeddings

MODEL = 'text-embedding-ada-002'
parser = argparse.ArgumentParser(description='Know Me through LLM')
parser.add_argument('--query', type=str, help='Enter the user query:')
parser.add_argument('--task', type=str, help='Enter the task you would like to: (1) Process, (2) Infer')

SYSTEM_PROMPT = (
    """ ### Who are You? ### \n
    You are a resume evaluation expert. \n 
    ### What is your job? ### \n
    You answer queries from data science recruiters regarding a candidate profile using the context give below. \n
    ### Context: ### {context} \n
    ### INSTRUCTIONS: \n
    Do Not blindly use the context as the answer. Please frame the answer in a user readable format and make sure it perfectly answers the input. \n
    Feel free to rephrase the answer based on the input and the context as evidence. \n
    If you do not know any question, and if there is no context provided, simply say that you do not have enough information. \n
    ### \n
    """
)


def create_embeddings(client, texts, MODEL = 'text-embedding-ada-002'):
    embeddings_list = []
    for text in texts:
        res = client.embeddings.create(input=[text], model=MODEL)
        embeddings_list.append(res.data[0].embedding)
    print(embeddings_list)
    return embeddings_list

def process_pdf(filepath):
    pdf = PyPDFLoader(filepath)
    data = pdf.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    statements = [str(doc) for doc in docs]
    statements_clean = []
    for statement in statements:
        statement = statement.replace("\\n●", " ")
        statement = statement.replace("\\n", " ")
        statement = statement.replace("•", " ")
        statements_clean.append(statement)

    with open('statements.json', 'w') as f:
        json.dump(statements_clean, f, indent=4)

    return statements_clean

def definePcIndex(pc):
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc.delete_index('pinecode-index')
    pc.create_index(name="pinecode-index",
                           dimension= 1536,
                           metric="cosine",
                           spec=spec
                           )
    return

def get_bot_response(input):
    return input

def response_generator(pc, prompt):
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    if prompt is not None:
        response = query_LLM(pc, prompt)
    for word in response.split():
        yield word + " "
        time.sleep(0.07)

def process_embeddings(pc):
    definePcIndex(pc)
    index = pc.Index('pinecode-index')
    print(index.describe_index_stats())
    llm_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    text = process_pdf("TanmayiBallaResume.pdf")
    embedding_df = pd.read_csv('embeddings.csv')
    embeddings = None
    print(embedding_df.shape[0])
    if embedding_df.shape[0]<=5:
        embeddings = create_embeddings(llm_client, text)
        print("Hello")
        embedding_df = pd.DataFrame({
            "vectors":embeddings,
            "text": text
        })
        embedding_df.to_csv('embeddings.csv')
    else:
        embeddings = embedding_df["vectors"].apply(lambda x: list(map(float, ast.literal_eval(x)))).to_list()

    vector_list = []
    for i in range(len(text)):
        vector_list.append({
            "id": str(i),
            "values": embeddings[i],
            "metadata": {"text": text[i]}
        })
    # print(vector_list)

    ## Upsert index in vector_list
    upsert_result = index.upsert(vectors = vector_list, namespace='knowMenamespace')
    print(upsert_result)
    time.sleep(30)
    print(index.describe_index_stats())
    print("Data upserted into Pinecode")

def query_LLM(pc, user_query):
    index = pc.Index('pinecode-index')
    # print(index.describe_index_stats())
    openai_embed = OpenAIEmbeddings(model=MODEL, api_key=st.secrets["OPENAI_API_KEY"])
    vector_store = lang_pinecone(index, openai_embed.embed_query, "text", namespace='knowMenamespace')
    # print(vector_store.similarity_search(user_query, k = 3))
    llm = ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        model_name = 'gpt-3.5-turbo-0125',
        temperature = 0.0
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    response = rag_chain.invoke({"input": user_query})
    print(response["answer"])
    return response["answer"]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    st.title("Get to know Tanmayi Balla :)")
    sidebar()
    # user_name = st.sidebar.text_input("Name", "Tanmayi Balla")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # args = parser.parse_args()
    # user_query = args.query
    # user_query = prompt
    # print("User Query:", user_query)
    # task = args.task
    task = "infer"
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    if task == "process":
        process_embeddings(pc)

    if prompt := st.chat_input("What is up?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(pc, prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
