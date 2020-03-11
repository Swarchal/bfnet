import setuptools


def read_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()


setuptools.setup(
    name="bfnet",
    version="0.1",
    author="Scott Warchal",
    author_email="scott_w@fastmail.com",
    license="BSD",
    packages=["bfnet"],
    python_requires=">=3.6",
    install_requires=read_requirements(),
    zip_safe=True
)
