"""
Setup script for Gemini Batch Processing Framework
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="gemini-batch",
    version="0.2.0",
    author="Sean Brar",
    author_email="hello@seanbrar.com",
    description="A framework for batch processing with Google's Gemini API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/seanbrar/gemini-batch-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.13",
    install_requires=requirements,
)
