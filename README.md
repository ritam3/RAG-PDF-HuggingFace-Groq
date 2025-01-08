# RAG-PDF QA Method
This repository implements the RAG method to perform question answering on PDF documents. The application relies on HuggingFace embeddings in addition to the Groq API for efficient and accurate information retrieval. Session management ensures embeddings are computed once per document upload to optimize performance by avoiding redundant computations. In this illustration `Llama3-8b-8192` has been used



https://github.com/user-attachments/assets/73b26607-a2bb-4889-91ba-d55f2b2507fd



## Features
* Parsing of PDF Document: Supports parsing and preprocessing of PDF documents for QA.
* HuggingFace Embeddings: Uses HuggingFace models to generate embeddings for document text. 
* Groq API Integration: Uses Groq API for embedding retrieval and similarity search. 
* Session Management: Maintains session state so redundant calculations of embeddings can be avoided; hence, it will be more efficient. 
* Interactive QA: It allows users to ask queries after uploading documents. It retrieves relevant information from the processed PDFs.
