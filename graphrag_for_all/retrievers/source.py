import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader


class RadioWebLoader(WebBaseLoader):
    def __init__(
        self,
        query,
    ) -> None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        URL = f"https://radiopaedia.org/search?lang=gb&q={query}&scope=articles"
        page = requests.get(URL, headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")
        radio_address = "https://radiopaedia.org"
        all_address = [
            f"{radio_address}{e['href']}"
            for e in soup.find_all("a", class_="search-result search-result-article")
        ]
        self.all_address = all_address
        super().__init__(all_address)
