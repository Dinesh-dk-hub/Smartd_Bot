import azure.functions as func
import logging
from openai import AzureOpenAI
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import AzureChatOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import tempfile
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import PyPDF2 as pdf
import json
import re
import os
import numpy as np

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
azureDeployment = "text-embedding-ada-002"
azureEndpoint = "https://azure-test-embedding-01.openai.azure.com/"
azureOpenAIKey = "a42e7f95f8c744e68191b733ad0bd4f9"
azureAPIVersion = "2024-02-01"
    #Creating Method to generate Embeddings
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
        azure_deployment=azureDeployment,
        azure_endpoint=azureEndpoint,
        api_key= azureOpenAIKey,
        api_version=azureAPIVersion,
        chunk_size=1
    )
    #Creating AzureOpenAI connection
client_AzureOpenAI = AzureOpenAI(
        api_key=azureOpenAIKey,
        api_version=azureAPIVersion,
        azure_endpoint=azureEndpoint
    )
llm = AzureChatOpenAI(
        azure_deployment="gpt-35",
        azure_endpoint=azureEndpoint,
        api_key= azureOpenAIKey,
        api_version= "2024-02-01"
    )
  

@app.route(route="smartdBot")

def smartdBot(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    query = req.params.get('query')

    chat_history=[]
    openai_credential = DefaultAzureCredential()
    scope_url = "https://cognitiveservices.azure.com/.default"
    token_provider = get_bearer_token_provider(openai_credential,  scope_url)
    conn_string = "DefaultEndpointsProtocol=https;AccountName=azurestorage2904;AccountKey=ybUJQVdTdq1vZurrXNrm9324QImtTNTxK9bXVN45SShwCBPEFAw4QaGP15yPyo1cOjOk478LruX++ASt0kRp4A==;EndpointSuffix=core.windows.net"
    container = "blobcontainer03"
    blob_name = "Workplace-Policies.pdf"
    blob_server_client = BlobServiceClient.from_connection_string(conn_string)
    container_client = blob_server_client.get_container_client(container)
    blob_client = container_client.get_blob_client(blob_name)
    
    blob_data = blob_client.download_blob()
    blob_bytes = blob_data.readall()
    # sample_pdf = open(blob_name, mode='rb')
    # pdfdoctor = pdf.PdfReader(sample_pdf)
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(blob_bytes)
        temp_file_path = temp_file.name
    pdfdoctor = pdf.PdfReader(temp_file_path)
    extractedPages = []
    textPages =''
    for x in range(0, len(pdfdoctor.pages)):
    
        page1 = pdfdoctor.pages[x]
        textPages+= page1.extract_text()
        extractedPages.append(page1.extract_text())
    
    azure_search_key :str = "FTGizmAruHGf6cMVHphl8d39ZtY1WSSguzbPUOhPlrAzSeDAHb03"
    azure_search_endpoint = "https://azurevectorsearch-demo.search.windows.net"
    AZURE_COGNITIVE_SEARCH_CREDENTIAL = AzureKeyCredential(azure_search_key)
    index = "idx-vector-azuresearch"
   
    client_AzureVectorIndex = SearchClient(endpoint=azure_search_endpoint, index_name= index, credential=AZURE_COGNITIVE_SEARCH_CREDENTIAL)
    query_embeddings = client_AzureOpenAI.embeddings.create(input = [query], model = "text-embedding-ada-002" ).data[0].embedding
    vector_query = VectorizedQuery(vector = query_embeddings, k_nearest_neighbors=3, fields = "vector")
    result = client_AzureVectorIndex.search(search_text=query, vector_queries=[vector_query], select=["title", "chunk", "vector"], top = 1)
    content = ""
    for item in result:
        content = item['chunk']
    db = FAISS.from_texts(extractedPages, embeddings)
    def get_conversation_chain(vector_store):  
        retriever = vector_store.as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                               retriever=retriever,
                                               memory=memory,
                                               return_source_documents=True,
                                               return_generated_question=True,
                                               chain_type= "stuff" )
        return qa
    
    qa = get_conversation_chain(db)
    result = qa({"question": query, "chat_history": chat_history})
  
    if not query:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            query = req_body.get('query')
    if query:
        return func.HttpResponse(result['answer'])
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
    
