# https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/
import os 
from openai import OpenAI 
import chromadb 
from chromadb.config import Settings 
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 환경 변수 Load해서 api_key 가져오고 OpenAI 클라이언트(객체) 초기화
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 매 실행 시 DB 폴더를 삭제 후 새로 생성
def init_db(db_path="./chroma_db"):
    dbclient = chromadb.PersistentClient(path=db_path)
    collection = dbclient.create_collection(name="rag_collection", get_or_create=True)
    return dbclient, collection

# 텍스트 로딩 함수
def load_text_files(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        
        file_path = os.path.join(folder_path, filename)
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append((filename, text))
    return docs
# OpenAI Embeddings 생성 함수 
# def get_embedding(text, model="text-embedding-3-large"):
#     response = client.embeddings.create(input=[text], model=model)
#     embedding = response.data[0].embedding
#     return embedding
    

# 문서 청크 단위로 나누기
# def chunk_text(text, chunk_size=400, chunk_overlap=50):
#     chunks= []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end] #적당히 자름
#         chunks.append(chunk)
#         start = end - chunk_overlap

#         if start < 0:
#             start = 0
#         if start >= len(text): # 종료 시그널
#             break

#     return chunks

embedder = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

# 문서로드 -> 청크 나누고 -> 임베딩 생성 후 DB 삽입
if __name__ == "__main__":
    # db 초기화
    dbclient, collection = init_db("./chroma_db")

    folder_path = "./source_data" # 데이터 가져다 쓸 경로 지정
    docs = load_text_files(folder_path) # 처리할 문서 데이터 메모리로 불러오기

    doc_id = 0
    for filename, text in docs: 
        # chunks = chunk_text(text, chunk_size=400, chunk_overlap=50) # chunking
        chunks = text_splitter.split_text(text)
        for idx, chunk in enumerate(chunks): # 각 청크와 해당 청크의 인덱스 가져옴
            doc_id += 1 # 인덱스 하나씩 증가 시키면서
            embedding = embedder.embed_query(chunk) # 각 청크 임베딩 벡터 생성
            # vectorDB에 다음 정보 추가
            collection.add(
                documents=[chunk], # 실제 청크 text
                embeddings=[embedding], # 생성된 임베딩 벡터
                metadatas=[{"filename": filename, "chunk_index": idx}], # 파일 이름과 청크 인덱스를 포함하는 메타데이터
                ids=[str(doc_id)] # 각 청크의 Unique한 id 저장
                # 이 고유 id를 통해 db에서 업데이트, 삭제등의 작업 가능 
            )
    
    print("모든 문서 벡터DB에 저장 완료")