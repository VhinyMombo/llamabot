# Retrieval-Augmented Generation (RAG) -- Thomas Sowell Case


This repository contains a Jupyter notebook `rag.ipynb` that demonstrates the use of the `langchain` library to implement a Retrieval-Augmented Generation model for natural language processing tasks. The project integrates functionalities like vector storage with FAISS, document loading, and text splitting, alongside the usage of OpenAI embeddings and language models.


Integrating powerful tools like `langchain`, `FAISS`, `OpenAI`, `Ollama`, and `ChromaDB` for state-of-the-art data processing and machine learning.

[![langchain](https://img.shields.io/badge/langchain-integration-blue.svg)](https://github.com/your-username/langchain-repo)
[![FAISS](https://img.shields.io/badge/FAISS-fast%20indexing-red.svg)](https://github.com/facebookresearch/faiss)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/api/)
[![Ollama](https://img.shields.io/badge/Ollama-service-purple.svg)](https://ollama.your-service-url.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-database-orange.svg)](https://chromadb.your-database-url.com)


As a quick presentation of our project's retrieval capabilities, we have implemented a model that specifically handles queries related to the works of Thomas Sowell. Through the incorporation of advanced NLP techniques and leveraging the knowledge base of Sowell's extensive writings, our system provides insightful responses and references, showcasing the practical applications of our integrated tools.

## Overview

The notebook provides an example of how to:

- Load and process documents using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- Store document embeddings with FAISS for efficient similarity searching.
- Use OpenAI's language model to create a retrieval chain that can fetch relevant information based on input queries.

## Prerequisites

Ensure you have the following prerequisites installed:

- Python 3.9+
- Jupyter Notebook or JupyterLab
- Required Python libraries as specified in `requirements.txt`

## Installation

To install the required Python libraries, run the following command:

```bash
pip install -r requirements.txt
```

## Running the Notebook

Start the Jupyter notebook server with:

```bash
jupyter notebook
```
Then navigate to the rag.ipynb file and open it.

## Usage

The `rag.ipynb` notebook is designed to be a self-contained demonstration of a Retrieval-Augmented Generation (RAG) model using various components from the `langchain` library. Follow these steps to interact with the notebook:

1. **Open the Notebook**: Launch Jupyter Notebook and open `rag.ipynb`.

2. **Set API Key**: Make sure to create a `config.py` file with your OpenAI API key. The notebook imports this configuration to authenticate with OpenAI services.

3. **Run the Cells**: Execute the cells sequentially to see the RAG model in action. The notebook includes:
   - Loading and preprocessing documents.
   - Embedding documents using OpenAI's embeddings and storing them in FAISS for quick retrieval.
   - Constructing retrieval chains that leverage language models to generate responses based on the input and context from the loaded documents.

4. **Interact with the Model**: Input your queries and observe how the RAG model retrieves relevant context from the documents and provides answers.

5. **Modify and Experiment**: Change the queries, context, or the retrieval model parameters to experiment with different configurations and observe the outcomes.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please:

1. Fork the repository.
2. Create a new branch for your feature.
3. Commit your changes.
4. Push to the branch.
5. Open a pull request.


## Future Roadmap

In the spirit of fostering an open and collaborative environment, our next steps involve transitioning to a completely open-source model. We are committed to achieving this by:

- Adopting **Ollama** for the language models and embeddings, providing a community-driven alternative to proprietary systems.
- Integrating **ChromaDB** or others databases as vector database, enhancing accessibility and enabling seamless scaling.
- Ensuring the entire toolchain, from data processing to retrieval, is open to contribution and use by the community.

Our goal is to democratize access to advanced machine learning models and tools, making them freely accessible to everyone. Stay tuned for updates as we make progress towards this exciting milestone!

## License

This project is open-sourced under the MIT License. See the `LICENSE` file in this repository for the full text.

## Acknowledgments

Special thanks to:

- The `langchain` community for the development tools.
- OpenAI for their API, which powers the underlying language models.
- All contributors who help to improve this project.

Feel free to reach out with questions or suggestions, and happy coding!
