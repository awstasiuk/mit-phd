import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mit-phd-awstasiuk",  # Replace with your own username
    version="0.1.0",
    author="Andrew Stasiuk",
    author_email="astasiuk@mit.edu",
    description="PhD Python stuff",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awstasiuk/mit-phd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
