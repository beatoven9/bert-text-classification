from setuptools import setup, find_packages

setup(
    name="Bert Classifier",
    version="0.1.0",
    description="A binary classification model using bert embeddings for text.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=2.2.0",
        "pandas>=2.2.3",
        "scikit-learn>=1.6.0",
        "torch>=2.5.1",
        "transformers>=4.47.0",
    ],
    python_requires=">=3.12.7",
)

