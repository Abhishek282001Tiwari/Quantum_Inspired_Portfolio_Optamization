from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="quantum-portfolio-optimization",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Quantum-Inspired Portfolio Optimization System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/quantum-portfolio-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "gpu": ["cupy", "cudf"],
        "notebooks": ["jupyter", "ipykernel", "notebook"],
    },
    entry_points={
        "console_scripts": [
            "quantum-portfolio=quantum_portfolio.cli:main",
        ],
    },
)