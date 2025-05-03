# RAG Tutorial for IEEE AI Conference 2025

This repository contains the code and resources for the **Retrieval-Augmented Generation (RAG) Tutorial** presented at the IEEE AI Conference 2025. The tutorial demonstrates how to build intelligent systems that combine retrieval-based techniques with generative AI models.

## Author
This tutorial was created by Partha Pritam Deka

## Features
- Integration with LangChain and Ollama for advanced language model capabilities.
- PDF processing using PyMuPDF and PyPDF.
- Community-driven extensions for LangChain.

## Requirements
To set up the environment, install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rag_tutorial_ieee_ai_conf_2025.git
   cd rag_tutorial_ieee_ai_conf_2025
   ```

## Setting Up the Environment and running the RAG application
You can create the Python environment for this project using the provided `environment.yml` file:

1. Create the environment:
   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:
   ```bash
   conda activate rag_tutorial_env
   ```

3. Run the following scripts on your conda prompt under your create Python enviroment:
   ```
   ollama pull llama3

   ollama run llama3

   ollama pull nomic-embed-text
   ```

4. Run the following scripts on your conda prompt under your create Python enviroment:

   ```
   streatmlit run rag_nomic_ollama_app.py
   ```
## Repository Structure


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or feedback, please contact [partha.pritamdeka@gmail.com].
