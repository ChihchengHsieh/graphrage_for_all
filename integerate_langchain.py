import getpass
import os
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from llm.send import Messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from llm.create import get_send_fn, get_text_emb_send_fn
from secret import OPENAI_API_KEY
from retreivers.radiowiki import RadioWikiRetriever
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


class TestChat(Runnable):
    def __init__(self) -> None:
        self.send_fn = get_send_fn(source="openai", model_name="gpt-3.5-turbo")

    def invoke(self, input, config):
        messages = self.to_messages(input)
        output = self.send_fn(messages, {}).output
        print(config)

        return HumanMessage(
            content=output,
        )

    def to_messages(
        self,
        invoke_input,
    ) -> Messages:
        messages = []
        for m in invoke_input.messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, AIMessage):
                role = "assistant"
            else:
                raise NotImplementedError("")
            messages.append(
                {
                    "role": role,
                    "content": m.content,
                }
            )
        return messages


class TestEmb:
    def __init__(self) -> None:
        self.text_emb_send_fn = get_text_emb_send_fn(
            source="openai", model_name="text-embedding-3-small"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """
        return self.text_emb_send_fn(texts, {})

    def embed_query(self, text: str) -> list[float]:

        return self.text_emb_send_fn([text], {})[0]


# what's needed for the llm and OpenAIEmbeddings.
# llm = ChatOpenAI(model="gpt-4o-mini")
# llm.invoke()
llm = TestChat()

doc_query = "atelectasis"
# Load, chunk and index the contents of the blog.
doc_retriever = RadioWikiRetriever(
    saving_dir="./documents",
)

docs = doc_retriever.request(
    query=doc_query,
    top_k=1,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=TestEmb())
# db3 = Chroma(persist_directory="./chroma_db", mbedding=OpenAIEmbeddings())


# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = rag_chain.invoke("what can cause atelectasis?")
print(res)
