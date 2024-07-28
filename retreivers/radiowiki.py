import os
from langchain.document_loaders import WikipediaLoader
from retreivers.source import RadioWebLoader
from typing import Optional
from utils.doc import clean_docs
from utils.pickle import load_pickle, save_pickle


class RadioWikiRetriever:
    def __init__(self, saving_dir="./documents") -> None:
        self.saving_dir = saving_dir

    def request(self, query: str, top_k: Optional[int] = None):

        # check if answer of the queries has been saved beforeF

        query = query.lower()
        wiki_path = os.path.join(self.saving_dir, f"wiki_{query}.pk")
        radio_path = os.path.join(self.saving_dir, f"radio_{query}.pk")

        # check if the searching has been done before.

        wiki_exist = os.path.isfile(os.path.join(self.saving_dir, f"wiki_{query}.pk"))
        radio_exist = os.path.isfile(os.path.join(self.saving_dir, f"radio_{query}.pk"))

        if wiki_exist:
            wiki_docs = load_pickle(wiki_path)
        else:
            wiki_docs = WikipediaLoader(query=query).load()
            save_pickle(wiki_docs, wiki_path)

        wiki_docs = clean_docs(wiki_docs)

        if radio_exist:
            radio_docs = load_pickle(radio_path)
        else:
            radio_docs = RadioWebLoader(query=query).load()
            save_pickle(radio_docs, radio_path)

        radio_docs = clean_docs(radio_docs)

        if top_k is not None:
            wiki_docs = wiki_docs[:top_k]
            radio_docs = radio_docs[:top_k]

        # documents = self.text_splitter.split_documents(radio_docs + wiki_docs)
        return radio_docs + wiki_docs
