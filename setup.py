from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def read_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name="PyRAG",
    version="1.7.0",
    author="Sulaiman Shamasna",
    author_email="suleiman.shamasneh@gmail.com",
    description="A Python library for RAG systems - A deep dive into Retrieval-Augmented Generation.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/sulaiman-shamasna/RAG-python-library",
    packages=find_packages(exclude=["tests*"]),
    entry_points={
        "console_scripts": [
            "adaptive_retrieval=pyrag.techniques.adaptive_retrieval:main",
            "fusion_retrieval=pyrag.techniques.fusion_retrieval:main",
            "reliable_rag=pyrag.techniques.reliable_rag:main",
            "rag_with_feedback_loop=pyrag.techniques.rag_with_feedback_loop:main",
            "graph_rag=pyrag.techniques.graph_rag:main",
            "raptor=pyrag.techniques.raptor:main",
            "self_rag=pyrag.techniques.self_rag:main",
        ]
    },
    install_requires=read_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.8",
)
