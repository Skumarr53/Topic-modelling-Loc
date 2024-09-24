# setup.py

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# Read requirements files
def read_requirements_file(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="centralized_nlp_package",
    version="0.1.0",
    author="Santhosh Kumar",
    author_email="santhosh.kumar3@voya.com",
    description="A centralized, modular Python package for NLP pipelines on Databricks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/centralized_nlp_package",
    packages=find_packages(),
    install_requires=read_requirements_file('requirements.txt'),
    extras_require={
        "dev": read_requirements_file('requirements_dev.txt')
    },
    python_requires='>=3.8',
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
