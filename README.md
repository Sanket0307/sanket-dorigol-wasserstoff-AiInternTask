# Document Research & Theme Identification Chatbot

An advanced document analysis system that combines AI-powered theme identification with intelligent document processing. This project leverages Google Gemini API and Streamlit to provide comprehensive document research capabilities with precise citation tracking.

## Overview

This chatbot system is designed to handle large document collections, extract meaningful insights, and provide accurate responses to complex queries. It features advanced document processing, semantic search, and AI-driven theme analysis for research-grade document analysis.

## Key Features

- Multi-format document ingestion supporting PDF, DOCX, TXT, and image files
- Advanced OCR capabilities for scanned documents and images
- Intelligent document segmentation for handling long and complex documents
- Semantic vector search using ChromaDB for efficient information retrieval
- AI-powered theme identification and synthesis using Google Gemini
- Interactive chat interface for natural language document querying
- Individual document analysis with detailed citation mapping
- Table extraction and structured data processing
- Smart chunking strategies for optimal document processing

## Technical Architecture

- **Frontend**: Streamlit web interface with custom CSS styling
- **AI Engine**: Google Gemini API for natural language processing and theme analysis
- **Vector Database**: ChromaDB for semantic search and document retrieval
- **Document Processing**: PyMuPDF, PyPDF2, and pytesseract for multi-format support
- **OCR Engine**: Tesseract with OpenCV preprocessing for enhanced accuracy

## Installation

1. Clone the repository:
git clone https://github.com/Sanket0307/sanket-dorigol-wasserstoff-AiInternTask.git
cd sanket-dorigol-wasserstoff-AiInternTask

text

2. Create a virtual environment:
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

text

3. Install dependencies:
pip install -r requirements.txt

text

4. Configure environment variables:
Create .env file
GOOGLE_API_KEY=your_gemini_api_key_here

text

## Usage

1. Start the application:
streamlit run app.py

text

2. Access the web interface at `http://localhost:8501`

3. Upload documents using the sidebar interface

4. Choose processing options:
- Basic Upload: Standard document processing
- Advanced Upload: Enhanced processing with OCR and table extraction

5. Interact with your documents through:
- Chat interface for general questions
- Individual document analysis for detailed citations
- Theme analysis dashboard for comprehensive insights

## Project Structure

document_research_chatbot/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── .env.example # Environment variables template
├── components/
│ ├── document_processor.py # Document parsing and OCR
│ ├── vector_store.py # ChromaDB integration
│ ├── theme_analyzer.py # AI theme identification
│ ├── gemini_client.py # Google Gemini API client
│ ├── document_segmenter.py # Intelligent document segmentation
│ └── citation_manager.py # Citation tracking and formatting
├── utils/
│ └── helpers.py # Utility functions
├── static/
│ └── style.css # Custom styling
└── database/ # Vector database storage

text

## Core Dependencies

- **streamlit**: Web application framework
- **google-genai**: Google Gemini API integration
- **chromadb**: Vector database for semantic search
- **PyMuPDF**: Advanced PDF processing
- **pytesseract**: OCR engine for text extraction
- **opencv-python**: Image preprocessing
- **sentence-transformers**: Text embedding generation
- **pandas**: Data manipulation and analysis
- **plotly**: Interactive visualizations

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

GOOGLE_API_KEY=your_gemini_api_key_here

text

### Processing Options

The system supports two processing modes:

1. **Basic Processing**: Standard text extraction and chunking
2. **Advanced Processing**: Enhanced with OCR, table extraction, and smart segmentation

### Chunking Strategies

- **Fixed Size**: Traditional word-based chunking with overlap
- **Sentence Boundary**: Preserves sentence structure for better context
- **Smart Segmentation**: AI-driven logical document segmentation

## API Integration

This project integrates with Google Gemini API for:
- Natural language understanding
- Theme identification and analysis
- Document summarization
- Query response generation

Ensure your API key has appropriate permissions for Gemini model access.

## Performance Considerations

- Large documents are processed using intelligent segmentation
- Vector embeddings are cached for improved query performance
- OCR processing can be resource-intensive for image-heavy documents
- ChromaDB provides efficient similarity search for large document collections

## Contributing

This project was developed as part of the Wasserstoff AI Internship program. For contributions or improvements, please follow standard Git workflow practices.

## License

This project is developed for educational and research purposes as part of the Wasserstoff AI Internship program.

## Acknowledgments

- Google Gemini API for advanced language processing capabilities
- Streamlit for the intuitive web interface framework
- ChromaDB for efficient vector storage and retrieval
- The open-source community for the various libraries and tools used
