from setuptools import setup, find_packages

# python setup.py sdist bdist_wheel
# twine upload --skip-existing dist/*
# pip3 install graphrag_for_all -U
# pip3 install graphrag_for_all --force-reinstall -U

setup(
    # name="graphrag_for_all",
    version="0.1.12",
    description="Graphrag and vectorstore for all LLMs",
    url="https://github.com/ChihchengHsieh/rag-aug",
    author="Chihcheng Hsieh",
    author_email="chihcheng.hsieh.82@gmail.com",
    license="MIT",
    # packages=["graphrag_for_all"],
    packages=find_packages(),
    zip_safe=False,
)
