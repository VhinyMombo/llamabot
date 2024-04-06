# the .py where credentials are stored
import config


from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter



from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#from langchain_community.document_loaders import WebBaseLoader
#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


from langchain_openai import OpenAIEmbeddings

# openai models
embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
# embeddings = OllamaEmbeddings()
llm = ChatOpenAI(api_key= config.OPENAI_API_KEY, temperature= 0.8)

# 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class intelligent intellectual specialized in Thomas Sowell."),
    ("user", "{input}")
])

# chain = prompt | llm

# parser for llm ouput
output_parser = StrOutputParser()

# chain
chain = prompt | llm | output_parser

#calliig the model
chain.invoke({"input": "how can langsmith help with testing?"})

# reading pdf to store in the vector store
loader = PyPDFDirectoryLoader("files")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)



# urls = [
#     "https://wesportfr.com/pronostic-nba/",
#     "https://wesportfr.com/pronostic-foot/",
#     "https://www.sofascore.com",
#     "https://wesportfr.com/pronostic-tennis/"

# ]

# loader = SeleniumURLLoader(urls=urls)

# data = loader.load()

# text_splitter = RecursiveCharacterTextSplitter()
# documents = text_splitter.split_documents(data)

# storing the documents in the vector store
vector = FAISS.from_documents(documents, embeddings)



prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context and give their title:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# ##########

# metadata_field_info = [
#     AttributeInfo(
#         name="source",
#         description = "The title of the document",
#         type= "string",

#     ), 
#     AttributeInfo(
#         name="page",
#         description = "The page number of the document",
#         type= "string",

#     ),
# ]

# ##########




retriever = vector.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "among the text which one talk about cnn?"})
print(response["answer"])


# retriever = SelfQueryRetriever.from_llm(llm, 
#                                         vectorstore=vector, 
#                                         metadata_field_info= metadata_field_info)

# # This example only specifies a filter
# retriever.invoke("I want to watch a movie rated higher than 8.5")