from setuptools import setup, find_packages

setup(
    name="ragtools",
    version="0.1.7",
    packages=find_packages(),  # This is crucial for proper package discovery
    install_requires=[
        "streamlit",
        "psutil",
        "langchain",
        "langchain-community",
        "langchain-ollama",
        "chromadb",
        "langchain-chroma",
        "pypdf",
    ],
    author="Wido Jansen",
    author_email="widojansen@gmail.com",
    description="Tools for RAG applications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/widojansen/ragtools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    entry_points={
        'console_scripts': [
            'ragtools=ragtools.streamlit_ui:launch_streamlit_ui',
        ],
    },
)