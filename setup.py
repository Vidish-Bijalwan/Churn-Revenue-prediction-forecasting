"""
Setup script for ForeTel.AI - Professional Telecom Analytics Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README_NEW.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="foretel-ai",
    version="2.0.0",
    author="ForeTel.AI Team",
    author_email="contact@foretel-ai.com",
    description="Professional Telecom Analytics Platform with Advanced ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/foretel-ai",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/foretel-ai/issues",
        "Documentation": "https://github.com/your-username/foretel-ai/docs",
        "Source Code": "https://github.com/your-username/foretel-ai",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "foretel-ai=foretel_ai.main:main",
            "foretel-train=foretel_ai.cli.train:main",
            "foretel-predict=foretel_ai.cli.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "foretel_ai": [
            "config/*.yaml",
            "assets/*",
            "templates/*",
        ],
    },
    keywords=[
        "machine learning",
        "telecom",
        "churn prediction", 
        "revenue forecasting",
        "analytics",
        "streamlit",
        "xgboost",
        "ensemble methods",
        "data science",
        "business intelligence"
    ],
    zip_safe=False,
)
