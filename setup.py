from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def read_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()

setup(
    name="PyRAG",
    version="1.0.0",
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
