from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
import streamlit as st
import os
import time

# Klasörlerin oluşturulması
if not os.path.exists("files"):
    os.mkdir("files")

if not os.path.exists("data"):
    os.mkdir("data")

# Prompt Template tanımı
TEMPLATE = """
Sana kullanıcının sorusunu ve ilgili metin alıntılarını sağlayacağım.
Yanıtı oluştururken şu kurallara dikkat et:
- Sağlanan metin alıntısında açıkça yer alan bilgileri kullan.
- Metin alıntısında açıkça bulunmayan cevapları tahmin etmeye veya uydurmaya çalışma.
- "Sen kimsin" sorusuna "Ben İzmir Liman Başkanlığı asistan botuyum! Size nasıl yardımcı olabilirim?" şeklinde yanıt ver.
Eğer hazırsan, sana kullanıcının sorusunu ve ilgili metin alıntısını sağlıyorum.

Context: {context}

User: {question}
Chatbot:
"""

# Streamlit başlık
st.title("İzmir Liman Başkanlığı")

# Dosya yükleme
uploaded_file = st.file_uploader("PDF dosyalarını yükleyiniz", type="pdf")

if uploaded_file is not None:
    # Dosyanın kaydedilmesi
    file_path = os.path.join("files", uploaded_file.name)
    if not os.path.isfile(file_path):
        with st.status("Döküman analiz ediliyor..."):
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            # PDF dosyasını yükleme
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Metin parçalama
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            all_splits = text_splitter.split_documents(documents)

            # Vektör deposu oluşturma
            embedding_function = OllamaEmbeddings(model="mistral")
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_function, persist_directory="data")
            vectorstore.persist()

            # Retriever tanımlama
            retriever = vectorstore.as_retriever()

            # LLM ayarları
            llm = Ollama(
                base_url="http://localhost:11434",
                model="mistral",
                verbose=True,
                callback_manager=CallbackManager([]),
            )

            # QA zinciri oluşturma
            prompt = PromptTemplate(input_variables=["context", "question"], template=TEMPLATE)
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt},
            )

            st.session_state.qa_chain = qa_chain

    # Kullanıcıdan giriş alınması
    if "qa_chain" in st.session_state:
        user_input = st.text_input("Sorunuzu yazın:", "")
        if user_input:
            with st.spinner("Yanıt hazırlanıyor..."):
                response = st.session_state.qa_chain.run(user_input)
                st.write("### Yanıt:")
                st.write(response)
else:
    st.write("Lütfen bir PDF dosyası yükleyiniz.")