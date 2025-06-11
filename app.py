import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from components.document_processor import DocumentProcessor
from components.vector_store import VectorStore
from components.theme_analyzer import ThemeAnalyzer
from components.gemini_client import GeminiClient
from components.document_segmenter import IntelligentSegmenter
from utils.helpers import load_css, format_timestamp
import json
import re

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document Research & Theme Identification Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css("static/style.css")

class DocumentResearchChatbot:
    def __init__(self):
        # Initialize components
        self.gemini_client = GeminiClient()
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.theme_analyzer = ThemeAnalyzer(self.gemini_client)
        self.segmenter = IntelligentSegmenter(self.gemini_client)

        # Initialize session state
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'document_themes' not in st.session_state:
            st.session_state.document_themes = {}
        if 'parsed_content' not in st.session_state:
            st.session_state.parsed_content = {}
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def render_header(self):
        """Render the application header"""
        st.markdown("""
        <div class="header-container">
            <h1 class="main-title">📚 Document Research & Theme Identification Chatbot</h1>
            <p class="subtitle">Advanced RAG System with Multi-Modal Document Processing</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render the sidebar with document management"""
        with st.sidebar:
            st.markdown('<div class="sidebar-header">📁 Document Management</div>', unsafe_allow_html=True)

            # Processing mode selection
            processing_mode = st.radio(
                "Processing Mode:",
                ["Basic Upload", "Advanced Upload"],
                help="Choose between basic or advanced document processing"
            )

            if processing_mode == "Basic Upload":
                self.render_basic_upload_interface()
            else:
                self.render_enhanced_upload_interface()

            # Document statistics
            if st.session_state.documents:
                st.markdown("### 📊 Document Statistics")
                total_docs = len(st.session_state.documents)
                st.metric("Total Documents", total_docs)

                total_themes = sum(len(themes) for themes in st.session_state.document_themes.values())
                st.metric("Total Themes Identified", total_themes)

                # Show document list
                st.markdown("### 📄 Processed Documents")
                for doc in st.session_state.documents:
                    with st.expander(f"📄 {doc['name']}"):
                        st.write(f"**Type:** {doc['type'].upper()}")
                        st.write(f"**Size:** {doc['size']} bytes")
                        st.write(f"**Words:** {doc['metadata']['word_count']}")
                        st.write(f"**Processing Mode:** {doc['metadata'].get('processing_mode', 'basic')}")
                        st.write(f"**Themes:** {len(st.session_state.document_themes.get(doc['id'], []))}")

                        # Show themes for this document
                        doc_themes = st.session_state.document_themes.get(doc['id'], [])
                        if doc_themes:
                            st.write("**Identified Themes:**")
                            for theme in doc_themes:
                                st.write(f"• {theme['name']}")

    def render_basic_upload_interface(self):
        """Basic upload interface"""
        st.markdown("### Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files (PDF, DOCX, TXT, Images)",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg'],
            help="Upload documents for basic parsing and theme analysis"
        )

        if uploaded_files:
            if st.button("🚀 Process & Analyze Documents", type="primary"):
                self.process_and_analyze_documents(uploaded_files)

    def render_enhanced_upload_interface(self):
        """Enhanced upload interface for long documents"""
        st.markdown("### 📁 Advanced Document Upload")

        uploaded_files = st.file_uploader(
            "Upload documents (supports large PDFs)",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg']
        )

        if uploaded_files:
            # Processing options
            col1, col2 = st.columns(2)

            with col1:
                ocr_enabled = st.checkbox("Enable OCR for images/scanned text", value=True)
                table_extraction = st.checkbox("Extract tables", value=True)

            with col2:
                chunking_strategy = st.selectbox(
                    "Chunking Strategy",
                    ["Smart Segmentation", "Fixed Size", "Sentence Boundary"]
                )
                vector_store_type = st.selectbox(
                    "Vector Store",
                    ["ChromaDB Only", "Enhanced Processing"]
                )

            if st.button("🚀 Process Documents (Advanced)", type="primary"):
                self.process_documents_advanced(
                    uploaded_files,
                    ocr_enabled,
                    table_extraction,
                    chunking_strategy,
                    vector_store_type
                )

    def process_documents_advanced(self, uploaded_files, ocr_enabled, table_extraction, chunking_strategy,
                                   vector_store_type):
        """Advanced document processing with progress tracking"""
        total_steps = len(uploaded_files) * 4  # OCR, Chunking, Embedding, Storage
        progress_bar = st.progress(0)
        status_container = st.container()

        for i, uploaded_file in enumerate(uploaded_files):
            with status_container:
                st.write(f"📄 Processing: {uploaded_file.name}")

                # Step 1: Text extraction with OCR
                progress_bar.progress((i * 4 + 1) / total_steps)
                st.write("🔍 Extracting text and applying OCR...")

                doc_data = self.doc_processor.process_file(
                    uploaded_file,
                    use_advanced_processing=True,
                    ocr_enabled=ocr_enabled,
                    table_extraction=table_extraction
                )

                if not doc_data:
                    st.error(f"❌ Failed to process {uploaded_file.name}")
                    continue

                # Step 2: Intelligent chunking
                progress_bar.progress((i * 4 + 2) / total_steps)
                st.write("✂️ Applying intelligent chunking...")

                if chunking_strategy == "Smart Segmentation":
                    try:
                        segments = self.segmenter.segment_long_document(doc_data['content'], doc_data['name'])
                        if segments:
                            # Convert segments to chunk format
                            doc_data['chunks'] = []
                            for j, segment in enumerate(segments):
                                doc_data['chunks'].append({
                                    'content': segment['content'],
                                    'chunk_id': j,
                                    'start_index': segment.get('start_pos', 0),
                                    'end_index': segment.get('end_pos', len(segment['content'])),
                                    'section_type': segment.get('section_type', 'unknown'),
                                    'importance_score': segment.get('importance_score', 0.5)
                                })
                            st.write(f"✅ Created {len(segments)} intelligent segments")
                    except Exception as e:
                        st.warning(f"Smart segmentation failed, using default chunking: {str(e)}")

                # Step 3: Generate embeddings
                progress_bar.progress((i * 4 + 3) / total_steps)
                st.write("🧠 Generating embeddings...")

                # Step 4: Store in vector database
                progress_bar.progress((i * 4 + 4) / total_steps)
                st.write("💾 Storing in vector database...")

                # Store in vector database
                vector_success = self.vector_store.add_document(doc_data)

                # Store in session state
                st.session_state.documents.append(doc_data)
                st.session_state.parsed_content[doc_data['id']] = doc_data['content']

                # Generate themes
                st.write("🎯 Analyzing themes...")
                themes = self.theme_analyzer.analyze_document_themes(doc_data['content'], doc_data['name'])
                st.session_state.document_themes[doc_data['id']] = themes

                st.success(
                    f"✅ Completed: {uploaded_file.name} - {len(doc_data['chunks'])} chunks, {len(themes)} themes")


        st.success(f" Successfully processed {len(uploaded_files)} documents with advanced features!")

    def process_and_analyze_documents(self, uploaded_files):
        """Basic document processing and theme analysis"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_files = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")

            # Step 1: Parse document
            doc_data = self.doc_processor.process_file(uploaded_file)

            if doc_data:
                # Step 2: Store in vector database
                vector_success = self.vector_store.add_document(doc_data)

                # Step 3: Store parsed content for Gemini (CRITICAL STEP)
                st.session_state.parsed_content[doc_data['id']] = doc_data['content']

                # Debug: Verify content was stored
                stored_content = st.session_state.parsed_content.get(doc_data['id'], "")
                st.write(f"📄 Stored {len(stored_content)} characters for {uploaded_file.name}")

                # Step 4: Immediate theme analysis with Gemini
                status_text.text(f"Analyzing themes in {uploaded_file.name}...")
                themes = self.theme_analyzer.analyze_document_themes(
                    doc_data['content'],
                    doc_data['name']
                )

                # Step 5: Store everything
                st.session_state.documents.append(doc_data)
                st.session_state.document_themes[doc_data['id']] = themes

                st.success(
                    f"✅ Processed {uploaded_file.name} - Content: {len(doc_data['content'])} chars, Themes: {len(themes)}")
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")

            progress_bar.progress((i + 1) / total_files)

        # Final verification
        st.write(f"🎯 **Final Status:**")
        st.write(f"- Documents processed: {len(st.session_state.documents)}")
        st.write(f"- Content entries stored: {len(st.session_state.parsed_content)}")
        st.write(
            f"- Total characters stored: {sum(len(content) for content in st.session_state.parsed_content.values())}")

        status_text.text("✅ All documents processed and analyzed!")
        st.balloons()

    def render_main_interface(self):
        """Render the main interface"""
        if not st.session_state.documents:
            st.markdown("""
            <div class="welcome-container">
                <h2>Welcome! </h2>
                <p>Upload your documents to get started with instant parsing and AI-powered theme analysis.</p>
                <div class="features">
                    <h3>Features:</h3>
                    <ul>
                        <li>📄 Basic & Advanced document parsing (PDF, DOCX, TXT, Images)</li>
                        <li>🤖 AI-powered theme identification using Google Gemini</li>
                        <li>💬 Chat with your documents using parsed content</li>
                        <li>📊 Individual document analysis with citations</li>
                        <li>🧠 Smart document segmentation for long PDFs</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # Create tabs for different functionalities (removed Advanced Query Processing)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💬 Chat with Documents",
            "📊 Individual Analysis",
            "🔍 Theme Analysis",
            "📄 Document Overview",
            "🔧 Debug"
        ])

        with tab1:
            self.render_chat_interface()

        with tab2:
            self.render_enhanced_chat_interface()

        with tab3:
            self.render_theme_analysis()

        with tab4:
            self.render_document_overview()

        with tab5:
            self.verify_document_content()

    def render_chat_interface(self):
        """Chat interface using parsed document content"""
        st.markdown("### 💬 Chat with Your Documents")

        # Debug information
        if st.checkbox("🔍 Show Debug Info"):
            st.markdown("**Debug Information:**")
            st.write(f"Documents in session: {len(st.session_state.documents)}")
            st.write(f"Parsed content entries: {len(st.session_state.parsed_content)}")
            st.write(f"Document themes: {len(st.session_state.document_themes)}")

            if st.session_state.parsed_content:
                st.write("**Parsed Content Preview:**")
                for doc_id, content in st.session_state.parsed_content.items():
                    doc_name = next((doc['name'] for doc in st.session_state.documents if doc['id'] == doc_id),
                                    "Unknown")
                    st.write(f"- {doc_name}: {len(content)} characters")
                    with st.expander(f"Preview {doc_name}"):
                        st.text(content[:500] + "..." if len(content) > 500 else content)

        st.markdown(
            "Ask questions about your uploaded documents. The AI will use the parsed content to provide accurate answers.")

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("sources"):
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"📄 {source}")

        # Chat input
        if prompt := st.chat_input("Ask anything about your documents..."):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.write(prompt)

            # Generate AI response using parsed content
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your documents..."):
                    response, sources = self.generate_ai_response(prompt)
                    st.write(response)

                    if sources:
                        with st.expander("📚 Sources"):
                            for source in sources:
                                st.write(f"📄 {source}")

                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources
                    })

    def search_individual_documents(self, user_question):
        """Search each document individually and return answers with citations"""
        individual_answers = []

        for doc in st.session_state.documents:
            doc_id = doc['id']
            doc_name = doc['name']
            doc_content = st.session_state.parsed_content.get(doc_id, "")

            if not doc_content:
                continue

            # Search within this specific document
            doc_chunks = doc['chunks']
            relevant_chunks = []

            # Find relevant chunks within this document
            for chunk in doc_chunks:
                # Simple relevance scoring (you can enhance this)
                chunk_text = chunk['content'].lower()
                question_words = user_question.lower().split()
                relevance_score = sum(1 for word in question_words if word in chunk_text) / len(question_words)

                if relevance_score > 0.1:  # Threshold for relevance
                    relevant_chunks.append({
                        'content': chunk['content'],
                        'chunk_id': chunk['chunk_id'],
                        'start_index': chunk['start_index'],
                        'relevance': relevance_score
                    })

            if relevant_chunks:
                # Get the most relevant chunk
                best_chunk = max(relevant_chunks, key=lambda x: x['relevance'])

                # Generate answer for this specific document
                answer_data = self.generate_individual_document_answer(
                    user_question,
                    doc_name,
                    doc_id,
                    best_chunk['content'],
                    best_chunk['chunk_id']
                )

                if answer_data:
                    individual_answers.append(answer_data)

        return individual_answers

    def generate_individual_document_answer(self, question, doc_name, doc_id, content, chunk_id):
        """Generate answer from individual document with precise citations"""

        prompt = f"""
        Based ONLY on the following document content, answer the question: "{question}"

        Document: {doc_name}
        Content: {content}

        If the document contains relevant information, provide:
        1. A clear, specific answer
        2. The exact page/section where this information is found
        3. The specific paragraph or sentence

        If the document doesn't contain relevant information, respond with "No relevant information found."

        Format your response as JSON:
        {{
            "has_answer": true/false,
            "answer": "Your specific answer here",
            "citation": {{
                "page": "page_number_if_available",
                "paragraph": "paragraph_number",
                "exact_text": "exact quote or reference"
            }},
            "relevance_score": 0.0-1.0
        }}
        """

        try:
            response = self.gemini_client.generate_response(prompt)

            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())

                if result.get('has_answer', False):
                    return {
                        'document_id': doc_id,
                        'document_name': doc_name,
                        'answer': result.get('answer', ''),
                        'citation': result.get('citation', {}),
                        'relevance_score': result.get('relevance_score', 0.5),
                        'chunk_id': chunk_id
                    }
        except Exception as e:
            st.error(f"Error processing {doc_name}: {str(e)}")

        return None

    def display_individual_document_answers(self, user_question):
        """Display individual document answers in tabular format"""

        st.markdown("### 📊 Individual Document Answers")
        st.markdown(f"**Question:** {user_question}")

        # Get individual answers from each document
        individual_answers = self.search_individual_documents(user_question)

        if not individual_answers:
            st.warning("No relevant answers found in any documents.")
            return individual_answers

        # Create the table data
        table_data = []
        for answer in individual_answers:
            citation_text = ""
            citation = answer.get('citation', {})

            # Format citation
            citation_parts = []
            if citation.get('page'):
                citation_parts.append(f"Page {citation['page']}")
            if citation.get('paragraph'):
                citation_parts.append(f"Para {citation['paragraph']}")

            citation_text = ", ".join(citation_parts) if citation_parts else "N/A"

            table_data.append({
                'Document ID': answer['document_id'][:8] + "...",  # Shortened ID
                'Document Name': answer['document_name'],
                'Extracted Answer': answer['answer'],
                'Citation': citation_text,
                'Relevance': f"{answer['relevance_score']:.2f}"
            })

        # Display as DataFrame
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

        # Detailed expandable view
        st.markdown("### 🔍 Detailed Document Analysis")
        for answer in individual_answers:
            with st.expander(f"📄 {answer['document_name']} (Relevance: {answer['relevance_score']:.2f})"):

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("**Extracted Answer:**")
                    st.write(answer['answer'])

                    # Show exact quote if available
                    citation = answer.get('citation', {})
                    if citation.get('exact_text'):
                        st.markdown("**Exact Quote:**")
                        st.markdown(f'> "{citation["exact_text"]}"')

                with col2:
                    st.markdown("**Citation Details:**")
                    st.write(f"📄 Document: {answer['document_name']}")
                    st.write(f"🆔 Doc ID: {answer['document_id']}")

                    if citation.get('page'):
                        st.write(f"📖 Page: {citation['page']}")
                    if citation.get('paragraph'):
                        st.write(f"📝 Paragraph: {citation['paragraph']}")

                    st.write(f"🎯 Relevance: {answer['relevance_score']:.2f}")

        return individual_answers

    def generate_synthesized_theme_answer(self, user_question, individual_answers):
        """Generate synthesized answer with theme identification"""

        if not individual_answers:
            return "No relevant information found across documents."

        # Prepare context for theme synthesis
        context = {
            'question': user_question,
            'individual_answers': [
                {
                    'document': ans['document_name'],
                    'answer': ans['answer'],
                    'citation': ans['citation']
                }
                for ans in individual_answers
            ]
        }

        prompt = f"""
        Based on the individual document answers below, provide a synthesized response that identifies themes and patterns.

        Question: "{user_question}"

        Individual Document Answers:
        {json.dumps(context['individual_answers'], indent=2)}

        Provide a response in this format:

        **Synthesized Answer:**
        [Comprehensive answer combining all relevant information]

        **Identified Themes:**

        Theme 1 - [Theme Name]:
        [Documents that support this theme with citations]

        Theme 2 - [Theme Name]:
        [Documents that support this theme with citations]

        **Key Insights:**
        [Overall insights and conclusions]
        """

        try:
            response = self.gemini_client.generate_response(prompt)
            return response
        except Exception as e:
            st.error(f"Error generating synthesized response: {str(e)}")
            return "Unable to generate synthesized response."

    def render_enhanced_chat_interface(self):
        """Enhanced chat interface with individual document analysis"""
        st.markdown("### 📊 Individual Document Analysis")
        st.markdown(
            "This interface searches each document individually and provides detailed citations for each answer.")

        # Question input
        user_question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the main findings about climate change?",
            key="individual_analysis_input"
        )

        if st.button("🔍 Analyze All Documents", type="primary"):
            if user_question:
                with st.spinner("Analyzing each document individually..."):

                    # Get individual document answers
                    individual_answers = self.display_individual_document_answers(user_question)

                    # Generate and display synthesized theme answer
                    if individual_answers:
                        st.markdown("---")
                        st.markdown("### 🎯 Synthesized Theme Analysis")

                        synthesized_answer = self.generate_synthesized_theme_answer(user_question, individual_answers)
                        st.markdown(synthesized_answer)
            else:
                st.warning("Please enter a question first.")

    def generate_ai_response(self, user_question):
        """Generate AI response using parsed document content"""

        # Debug: Check if we have parsed content
        if not st.session_state.parsed_content:
            return "No documents have been processed yet. Please upload and process documents first.", []

        # Get relevant document chunks from vector search
        search_results = self.vector_store.search(user_question, max_results=5)

        # If no search results, use all documents
        if not search_results:
            # Use all available documents
            context_parts = []
            sources = []

            for doc in st.session_state.documents:
                doc_id = doc['id']
                doc_name = doc['name']

                # Get full parsed content
                full_content = st.session_state.parsed_content.get(doc_id, "")

                if full_content:  # Only include if content exists
                    # Get themes for this document
                    doc_themes = st.session_state.document_themes.get(doc_id, [])
                    theme_names = [theme['name'] for theme in doc_themes]

                    context_parts.append({
                        'document_name': doc_name,
                        'content': full_content[:3000],  # Limit content length for API
                        'themes': theme_names,
                        'word_count': doc['metadata']['word_count']
                    })

                    sources.append(doc_name)
        else:
            # Use search results
            context_parts = []
            sources = []

            for result in search_results:
                doc_id = result['document_id']
                doc_name = result['document_name']

                # Get full parsed content for this document
                full_content = st.session_state.parsed_content.get(doc_id, "")

                if full_content:  # Only include if content exists
                    # Get themes for this document
                    doc_themes = st.session_state.document_themes.get(doc_id, [])
                    theme_names = [theme['name'] for theme in doc_themes]

                    context_parts.append({
                        'document_name': doc_name,
                        'content': full_content[:3000],  # Use more content
                        'themes': theme_names,
                        'relevance_score': result['similarity'],
                        'chunk_content': result['content']  # Also include the specific chunk
                    })

                    sources.append(doc_name)

        # Check if we have any content to work with
        if not context_parts:
            return "I don't have access to any document content. Please ensure documents are properly uploaded and processed.", []

        # Create comprehensive prompt for Gemini
        prompt = f"""
You are an AI assistant that analyzes documents and answers questions based on their content.

USER QUESTION: "{user_question}"

AVAILABLE DOCUMENT CONTENT AND THEMES:

"""

        # Add each document's content to the prompt
        for i, doc_context in enumerate(context_parts, 1):
            prompt += f"""
DOCUMENT {i}: {doc_context['document_name']}
IDENTIFIED THEMES: {', '.join(doc_context['themes']) if doc_context['themes'] else 'No themes identified'}
CONTENT:
{doc_context['content']}

---

"""

        prompt += f"""
INSTRUCTIONS:
1. Answer the user's question "{user_question}" using ONLY the document content provided above
2. Reference specific documents by name when making claims
3. If asked for a summary, provide a comprehensive summary of ALL the document content
4. Include relevant themes in your response
5. Use specific quotes or references when possible
6. If the question cannot be answered from the provided content, say so clearly
7. Be comprehensive and detailed in your response

RESPONSE:
"""

        # Get response from Gemini
        try:
            response = self.gemini_client.generate_response(prompt)
            return response, list(set(sources))
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}", sources

    def verify_document_content(self):
        """Verify that document content is properly stored"""
        st.markdown("### 🔍 Content Verification")

        if not st.session_state.documents:
            st.warning("No documents uploaded yet.")
            return

        for doc in st.session_state.documents:
            doc_id = doc['id']
            doc_name = doc['name']

            # Check if content exists
            content = st.session_state.parsed_content.get(doc_id, "")
            themes = st.session_state.document_themes.get(doc_id, [])

            with st.expander(f"📄 {doc_name}"):
                st.write(f"**Document ID:** {doc_id}")
                st.write(f"**Content Length:** {len(content)} characters")
                st.write(f"**Themes Count:** {len(themes)}")
                st.write(f"**Processing Mode:** {doc['metadata'].get('processing_mode', 'basic')}")

                if content:
                    st.write("**Content Preview:**")
                    st.text_area("", content[:1000] + "..." if len(content) > 1000 else content, height=200,
                                 key=f"content_{doc_id}")
                else:
                    st.error("❌ No content found for this document!")

                if themes:
                    st.write("**Themes:**")
                    for theme in themes:
                        st.write(f"• {theme['name']}: {theme['description']}")
                else:
                    st.warning("⚠️ No themes identified for this document")

    def render_theme_analysis(self):
        """Display comprehensive theme analysis"""
        st.markdown("### 🔍 Theme Analysis Dashboard")

        if not st.session_state.document_themes:
            st.info("No themes identified yet. Upload documents to see theme analysis.")
            return

        # Aggregate all themes
        all_themes = []
        for doc_id, themes in st.session_state.document_themes.items():
            doc_name = next((doc['name'] for doc in st.session_state.documents if doc['id'] == doc_id), "Unknown")
            for theme in themes:
                theme['document'] = doc_name
                theme['document_id'] = doc_id
                all_themes.append(theme)

        # Theme statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Themes", len(all_themes))

        with col2:
            unique_themes = len(set(theme['name'] for theme in all_themes))
            st.metric("Unique Themes", unique_themes)

        with col3:
            avg_confidence = sum(theme.get('confidence', 0.5) for theme in all_themes) / len(
                all_themes) if all_themes else 0
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

        # Theme details
        st.markdown("### 📋 Detailed Theme Analysis")

        # Group themes by name
        theme_groups = {}
        for theme in all_themes:
            name = theme['name']
            if name not in theme_groups:
                theme_groups[name] = []
            theme_groups[name].append(theme)

        for theme_name, theme_instances in theme_groups.items():
            with st.expander(f"🏷️ {theme_name} (appears in {len(theme_instances)} documents)"):

                # Show theme description
                if theme_instances[0].get('description'):
                    st.write(f"**Description:** {theme_instances[0]['description']}")

                # Show documents containing this theme
                st.write("**Found in documents:**")
                for instance in theme_instances:
                    confidence = instance.get('confidence', 0.5)
                    st.write(f"• 📄 {instance['document']} (Confidence: {confidence:.2f})")

                # Show key points if available
                if theme_instances[0].get('key_points'):
                    st.write("**Key Points:**")
                    for point in theme_instances[0]['key_points']:
                        st.write(f"• {point}")

    def render_document_overview(self):
        """Display document overview and statistics"""
        st.markdown("### 📊 Document Overview")

        if not st.session_state.documents:
            st.info("No documents uploaded yet.")
            return

        # Create overview table
        overview_data = []
        for doc in st.session_state.documents:
            doc_themes = st.session_state.document_themes.get(doc['id'], [])
            overview_data.append({
                'Document': doc['name'],
                'Type': doc['type'].upper(),
                'Size (KB)': f"{doc['size'] / 1024:.1f}",
                'Words': doc['metadata']['word_count'],
                'Themes': len(doc_themes),
                'Processing': doc['metadata'].get('processing_mode', 'basic'),
                'Upload Date': format_timestamp(doc['upload_date'])
            })

        df = pd.DataFrame(overview_data)
        st.dataframe(df, use_container_width=True)

        # Document content preview
        st.markdown("### 📄 Document Content Preview")

        selected_doc = st.selectbox(
            "Select document to preview:",
            options=[doc['name'] for doc in st.session_state.documents]
        )

        if selected_doc:
            doc = next((d for d in st.session_state.documents if d['name'] == selected_doc), None)
            if doc:
                # Show document info
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Type:** {doc['type'].upper()}")
                    st.write(f"**Size:** {doc['size']} bytes")
                    st.write(f"**Words:** {doc['metadata']['word_count']}")
                    st.write(f"**Processing Mode:** {doc['metadata'].get('processing_mode', 'basic')}")

                with col2:
                    doc_themes = st.session_state.document_themes.get(doc['id'], [])
                    st.write(f"**Themes:** {len(doc_themes)}")
                    st.write(f"**Chunks:** {len(doc['chunks'])}")
                    if doc_themes:
                        theme_names = [theme['name'] for theme in doc_themes]
                        st.write(f"**Theme List:** {', '.join(theme_names)}")

                # Show content preview
                st.markdown("**Content Preview:**")
                content = st.session_state.parsed_content.get(doc['id'], "No content available")
                st.text_area("", content[:2000] + "..." if len(content) > 2000 else content, height=300)

    def run(self):
        """Main application runner"""
        self.render_header()
        self.render_sidebar()
        self.render_main_interface()


# Initialize and run the application
if __name__ == "__main__":
    app = DocumentResearchChatbot()
    app.run()
