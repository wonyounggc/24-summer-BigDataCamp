import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 환경변수를 불러옵니다.
load_dotenv()

# 환경변수를 가져옵니다.
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

os.environ["OPENAI_API_KEY"] = openai_api_key

# txt파일 로드
loader = TextLoader("recipe2.txt")
docs = loader.load()

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, separators=["page_content"], chunk_overlap=15)
split_documents = text_splitter.split_text(docs[0].page_content)

# 임베딩 생성
embeddings = OpenAIEmbeddings()

# 벡터 DB 생성 및 저장
vectorstore = FAISS.from_texts(split_documents, embedding=embeddings)
vectorstore.save_local('./db/faiss')
