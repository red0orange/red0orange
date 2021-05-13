import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="red0orange",
    version="0.0.1",
    author="red0orange",
    author_email="1031957961@qq.com",
    description="red0orange First Pip Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/red0orange/red0orange",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

