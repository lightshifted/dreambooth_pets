from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

docs_packages = [
    "mkdocs==1.4.2",
    "mkdocstrings==0.19.1",
]

style_packages = [
    "black==22.12.0",
    "flake8==6.0.0",
    "isort==5.11.4",
]

test_packages = [
    "pytest==7.2.0",
    "pytest-cov==4.0.0",
]

setup(
    name="dreambooth_pets",
    version=0.1,
    description="Generate DreamBooth animals from your images.",
    author="Jason Wheeler",
    author_email="jason.wheeler86@gmail.com",
    python_requires=">=3.8",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "dev": docs_packages + style_packages + test_packages + ["pre-commit==2.21.0"],
        "docs": docs_packages,
        "test": test_packages,
    },
)
