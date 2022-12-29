from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# setup.py
setup(
    name="dreambooth_pets",
    version=0.1,
    description="Generate DreamBooth animals from your images.",
    author="Jason Wheeler",
    author_email="jason.wheeler86@gmail.com",
    python_requires=">=3.8",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
)