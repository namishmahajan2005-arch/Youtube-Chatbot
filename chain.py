from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Creating Transcript of the Video
video_id="WszrfSwMz58"
try:
    ytt_api=YouTubeTranscriptApi()
    transcript_list=ytt_api.fetch(video_id,languages=["en"])

    transcript="".join(chunk.text for chunk in transcript_list)
    #print(transcript_list)
    # print(len(transcript))
except Exception as e:
    print("Error type: ", type(e).__name__)
    print("Details: ",e)

# Splitting Text into Chunks
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.create_documents([transcript])
# print(len(chunks))
#print(chunks[0])

# Generating Embeddings of Chunks
embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=api_key, output_dimensionality=768)
vector_store=FAISS.from_documents(chunks,embeddings)
# print(vector_store.index_to_docstore_id)

# Creating the Retriever
retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})
# print(retriever.invoke("What is maximum path sum"))

# Creating LLM and Prompt Template
llm=ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.2)

prompt=PromptTemplate(
    template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
    """,
    input_variables=['context','question']
)

question="What is maximun path sum?"

def format_docs(retrieved_docs):
    context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain=RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# print(parallel_chain.invoke(question))

parser=StrOutputParser()

main_chain= parallel_chain | prompt | llm | parser

print(main_chain.invoke(question))