from setuptools import setup, find_packages

setup(
    name="graphrag_for_all",
    version="0.1.0",
    description="Graphrag and vectorstore for all LLMs",
    url="https://github.com/ChihchengHsieh/rag-aug",
    author="Chihcheng Hsieh",
    author_email="chihcheng.hsieh.82@gmail.com",
    license="MIT",
    # packages=["graphrag_for_all"],
    packages=find_packages(),
    zip_safe=False,
)
