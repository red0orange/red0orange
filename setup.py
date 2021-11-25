import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="red0orange", # Replace with your own username
    version="1.0.0",
    author="red0orange",
    author_email="huangdehao919@gmail.com",
    description="red0orange first pypi package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/red0orange/red0orange",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)