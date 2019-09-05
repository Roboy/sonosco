from setuptools import setup, find_packages

with open("Long.md", "r") as fh:
    long_description = fh.read()

required = []
with open("requirements.txt", "r") as freq:
    for line in freq.read().split():
        required.append(line)

setup(
    name="sonosco",
    version="0.1.0",
    author="Roboy",
    description="Framework for training deep automatic speech recognition models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Roboy/sonosco/tree/demo",
    packages=find_packages(),
    include_package_data=True,
    dependency_links=[],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
    ],
    python_requires='>=3.6',
    install_requires=required
)
