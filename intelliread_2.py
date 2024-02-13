import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import time
import torch
import pinecone
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
def main():

    load_dotenv()  # take environment variables from .env.
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    st.set_page_config(page_title="Intelliread 🗨️")
    st.header("INTELLIREAD 🗨️ ")

    st.subheader("Illuminating PDFs with Intelligent Answers")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    #text extraction
    if pdf is not None:
        progress_bar = st.progress(0)
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            progress_bar.progress(25)
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=250,
            chunk_overlap=75,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        progress_bar.progress(50)
       
        # convert chunks into embeddings
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        embeddings = []
        for chunk in chunks:
            input_ids = tokenizer.encode(chunk, return_tensors='pt')
            with torch.no_grad():
                output = model(input_ids)
                embedding = output.last_hidden_state[:,0,:].numpy()
                
                embeddings.append(embedding.flatten().tolist())     
        progress_bar.progress(75)
   
        #pinecone setup
        pinecone.init(api_key='107badc9-220c-4a3e-b558-8223259ca8a6', environment='gcp-starter')
        index_name = "intelliread"
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name=index_name, metric="cosine", shards=1)
        indexer = pinecone.Index(index_name=index_name)

        vectors =[]
        id=[]
        for j in range(0,len(embeddings)):
            id.append(str(j))
        for i in range(0, len(embeddings)):
           tp={'id':id[i],'values':embeddings[i]}
           vectors.append(tp)
        indexer.upsert(vectors)
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        #user query input
        user_query = st.text_area("Enter Your Query")
        if user_query:
            with st.spinner('Processing your query...'):
                query_embeddings = []
                query_input_ids = tokenizer.encode(user_query, return_tensors='pt')
                with torch.no_grad():
                    output = model(query_input_ids)
                    query_embedding = output.last_hidden_state[:,0,:].numpy()
                    query_embeddings.append(query_embedding.flatten().tolist())
                search_results = indexer.query(query_embeddings, top_k=5)
                ids=[]
                for result in search_results['matches']:
                    ids.append(f"{result['id']}")
                print(int(ids[0]))
                selected_chunk=[]
                for j in range(0,5):
                    selected_chunk.append(chunks[int(ids[j])])
            
                llm = OpenAI(model="gpt-3.5-turbo-instruct")
                prompt = PromptTemplate(
                    input_variables=["query", "database"],
                    template="answer this {query}, Use knowledge from this text {database} to generate appropriate answer",
                )
                chain = LLMChain(llm=llm, prompt=prompt)
                with get_openai_callback() as cb:
                    response = chain.run({
                        'query': user_query,
                        'database': selected_chunk
                    })
                    print(cb)
                st.write(response)
    else:
        st.write("Please upload a PDF file")

if __name__ == '__main__':
    main()
