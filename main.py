from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
api_key='AIzaSyBQaNMD5HNWDIeTZbS7gzFQhHg8g8Ln-0g'

from langchain.llms import GooglePalm
llm = GooglePalm(google_api_key=api_key, temperature=0)

instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path="false_index"

def create_vector_db():
    loader = CSVLoader(file_path='/content/langchain/3_project_codebasics_q_and_a/codebasics_faqs.csv', source_column='prompt', encoding='cp1252')
    docs = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template = """Given

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT})
    return chain

if __name__ == "__main__":
    chain = get_qa_chain()

    print()

st.title("Codebasics QA")
btn = st.button("Create Knowledgebase")
if btn:
    pass
question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer: ")
    st.write(response["result"])