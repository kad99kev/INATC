from setuptools import setup, find_packages


def load_requirements():
    with open("requirements.txt", "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    requirements = []
    for line in lines:
        if line:
            requirements.append(line)

    return requirements


setup(
    name="inatc",
    licence="MIT",
    version="0.1",
    url="https://github.com/kad99kev/INATC",
    author="Kevlyn Kadamala",
    author_email="k.kadamala1@nuigalway.ie",
    description="Investigating a Neuro-evolutionary Approach to Text Classification",
    packages=find_packages(),
    install_requires=load_requirements(),
)
