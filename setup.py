import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="asdfy",
    version="0.1.3",
    author="Congyue Cui",
    author_email="ccui@princeton.edu",
    description="A parallel processing library for pyasdf.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/icui/asdfy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)