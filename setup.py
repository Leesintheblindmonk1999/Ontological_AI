# ============================================================================
# setup.py
# ============================================================================

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ontological-ai",
    version="0.1.0",
    author="Gonzalo Emir Durante",
    author_email="your.email@example.com",  # Add your email
    description="A library for measuring semantic stability in AI systems through geometric information theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gonzalodurante/ontological-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "transformers>=4.30.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    keywords="ai, machine-learning, semantic-stability, model-collapse, alignment, information-geometry",
    project_urls={
        "Bug Reports": "https://github.com/gonzalodurante/ontological-ai/issues",
        "Source": "https://github.com/gonzalodurante/ontological-ai",
        "Documentation": "https://ontological-ai.readthedocs.io",
        "Paper": "https://zenodo.org/records/17967232",
    },
)

