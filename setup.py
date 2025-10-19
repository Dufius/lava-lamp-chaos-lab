"""Setup configuration for Lava Lamp Chaos Lab."""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lava-lamp-chaos-lab",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Testing AI prediction capabilities on chaotic physical systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dufius/lava-lamp-chaos-lab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lavalamp-train=src.train:main",
        ],
    },
)
