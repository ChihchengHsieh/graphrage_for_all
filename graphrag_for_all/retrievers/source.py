import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader


class RadioWebLoader(WebBaseLoader):
    def __init__(
        self,
        query,
    ) -> None:
        URL = f"https://radiopaedia.org/search?lang=gb&q={query}&scope=articles"
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        radio_address = "https://radiopaedia.org"
        all_address = [
            f"{radio_address}{e['href']}"
            for e in soup.find_all("a", class_="search-result search-result-article")
        ]
        self.all_address = all_address
        super().__init__(all_address)
