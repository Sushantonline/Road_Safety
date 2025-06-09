from setuptools import setup, find_packages

setup(
    name="road-accident-analysis",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered road accident data analysis system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/road-accident-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "PyPDF2>=3.0.0",
        "pdfplumber>=0.9.0",
        "langchain>=0.0.350",
        "sentence-transformers>=2.2.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "road-accident-app=streamlit_app.app:main",
        ],
    },
)
