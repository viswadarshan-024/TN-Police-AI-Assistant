import os
import streamlit as st
import tempfile
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.utilities import GoogleSearchAPIWrapper

class DocumentSearchEngine:
    """Comprehensive document search across multiple sources"""
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.pre_indexed_vector_store = None
        self.uploaded_vector_store = None
        
    def load_pre_indexed_documents(self, index_path='./index'):
        """
        Load pre-indexed documents from FAISS index with robust error handling
        """
        try:
            # Check if index directory exists
            if not os.listdir(index_path):
                # st.warning(f"Index directory {index_path} does not exist. No pre-indexed documents will be loaded.")
                return False

            # Check if necessary index files exist
            required_files = ['index.faiss', 'index.pkl']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(index_path, f))]
            
            if missing_files:
                st.warning(f"Missing index files: {', '.join(missing_files)}. Skipping pre-indexed document loading.")
                return False

            # Attempt to load existing FAISS index
            self.pre_indexed_vector_store = FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            st.success("Pre-indexed documents loaded successfully!")
            return True
        
        except Exception as e:
            st.error(f"Error loading pre-indexed documents: {e}")
            logging.error(f"Pre-indexed document load error: {e}")
            return False
    
    def load_uploaded_documents(self, document_folder: str):
        """
        Load and process uploaded documents into vector store
        """
        documents = []
        
        for filename in os.listdir(document_folder):
            filepath = os.path.join(document_folder, filename)
            
            # Select appropriate loader
            if filename.endswith('.pdf'):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(('.txt', '.md')):
                loader = TextLoader(filepath, encoding='utf-8')
            else:
                continue
            
            # Load and split documents
            doc_parts = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            documents.extend(text_splitter.split_documents(doc_parts))
        
        # Create FAISS vector store for uploaded documents
        if documents:
            self.uploaded_vector_store = FAISS.from_documents(
                documents, 
                self.embeddings
            )
            st.success("Uploaded documents processed successfully!")
    
    def perform_vector_search(self, query: str, top_k: int = 3):
        """
        Perform vector search across pre-indexed and uploaded documents
        """
        results = []
        
        # Search pre-indexed documents
        if self.pre_indexed_vector_store:
            pre_indexed_results = self.pre_indexed_vector_store.similarity_search(
                query, k=top_k
            )
            results.extend([
                {
                    'source': 'Pre-Indexed Document',
                    'content': doc.page_content,
                    'metadata': doc.metadata
                } for doc in pre_indexed_results
            ])
        
        # Search uploaded documents
        if self.uploaded_vector_store:
            uploaded_results = self.uploaded_vector_store.similarity_search(
                query, k=top_k
            )
            results.extend([
                {
                    'source': 'Uploaded Document',
                    'content': doc.page_content,
                    'metadata': doc.metadata
                } for doc in uploaded_results
            ])
        
        return results

class TamilNaduPoliceLegalAssistant:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # API Configurations
        self.google_search_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        self.groq_api_key = os.getenv('GROQ_API_KEY')

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )

        # Initialize embeddings with multilingual support
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Initialize document search engine
        self.document_search = DocumentSearchEngine(self.embeddings)
        
        # Load pre-indexed documents
        self.document_search.load_pre_indexed_documents()

        # Initialize web search
        self.web_search = GoogleSearchAPIWrapper(
            google_api_key=self.google_search_api_key,
            google_cse_id=self.google_cse_id
        )
        
        # Initialize Language Models for different languages
        self.english_llm = ChatGroq(
            api_key=self.groq_api_key,
            model="llama-3.3-70b-versatile"
        )
        
        self.tamil_llm = ChatGroq(
            api_key=self.groq_api_key,
            model="llama-3.3-70b-versatile"
        )

    def generate_comprehensive_response(self, query: str, language: str = 'English'):
        """
        Generate a comprehensive response by combining multiple search strategies
        """
        # Vector search results
        vector_results = self.document_search.perform_vector_search(query)
        
        # Web search results
        web_results = self.web_search.results(query, num_results=3)
        
        # Prepare comprehensive sources
        comprehensive_sources = []
        
        # Add vector search results
        comprehensive_sources.extend([
            {
                'type': 'Document Search',
                'source': result['source'],
                'content': result['content'],
                'explanation': 'Highly relevant document found through semantic search'
            } for result in vector_results
        ])
        
        # Add web search results
        comprehensive_sources.extend([
            {
                'type': 'Web Search',
                'title': result['title'],
                'link': result['link'],
                'snippet': result['snippet'],
                'explanation': 'Additional context from web sources'
            } for result in web_results
        ])
        
        # Select language-specific prompt and LLM
        if language == 'Tamil':
            response_prompt = PromptTemplate(
                input_variables=['query', 'sources'],
                template="""
            நீங்கள் தமிழ்நாடு காவல்துறை துணை செயலாளர் ஆவீர்கள்.
            தரப்பட்ட கேள்விக்கு ஒரு மிகவும் விரிவான மற்றும் அதிகாரப்பூர்வ பதிலை வழங்கவும்.

            கேள்வி: {query}

            வழிகாட்டிக்கு மஞ்சள்:
            1. வெக்டர் தரவுத்தளத்திலிருந்து தகவலைப் பெறுங்கள்
            2. வலை தேடல் முடிவுகளுடன் பூரகப்படுத்துங்கள்
            3. தெளிவான, விரிவான மற்றும் சட்ட ரீதியாக சரியான தகவலை வழங்குங்கள்
            4. மூல நிரல்கள் சேர்க்கப்பட்டுள்ளன
            5. பயனர் நட்பு வடிவமைப்பில் வடிவமைக்கப்பட்டுள்ளது
            6. மிகப்பெரிய துல்லியத்தை மற்றும் சட்ட துல்லியத்தை உறுதி செய்யுங்கள்

            பதிலளிக்கும் வடிவம்:
                - கேள்விக்கான நேரடி பதில்
                - முக்கியமான சட்டத் தகவல்கள்
                - மேற்கோளிடப்பட்ட தகவல்கள்
                - தொடர்புடைய செயல்முறை விவரங்கள்
                """
            )
            llm = self.tamil_llm
        else:
            response_prompt = PromptTemplate(
                input_variables=['query', 'sources'],
                template="""
            You are an AI Assistant named 'LexWay AI' for the Tamil Nadu Police Department. 
            Provide a comprehensive and authoritative response to the query.

            Query: {query}

            Guidelines for Response:
            1. Prioritize information from vector database
            2. Supplement with web search results
            3. Provide clear, detailed, and legally precise information
            4. Include source references
            5. Format in a user-friendly manner 
            6. Ensure highest accuracy and legal precision

            Response Format:
            - Direct answer to the query
            - Key legal insights
            - Sourced information
            - Relevant procedural details
                """
            )
            llm = self.english_llm
        
        # Format sources for LLM
        formatted_sources = "\n\n".join([
            f"Source Type: {src.get('type', 'Unknown')}\n"
            f"Source Details: {src.get('source', src.get('title', 'N/A'))}\n"
            f"Content/Snippet: {src.get('content', src.get('snippet', 'N/A'))}\n"
            f"Explanation: {src.get('explanation', 'No additional context')}"
            for src in comprehensive_sources
        ])
        
        try:
            response = llm.invoke(
                response_prompt.format(
                    query=query,
                    sources=formatted_sources
                )
            ).content
            
            return {
                'response': response,
                'sources': comprehensive_sources
            }
        
        except Exception as e:
            logging.error(f"Response generation error: {e}")
            return {
                'response': "Unable to generate a comprehensive response. Please consult official sources." if language == 'English' else "ஒரு விரிவான பதிலை உருவாக்க முடியவில்லை. தயவுசெய்து அதிகாரப்பூர்வ மூலங்களைக் கலந்தாலோசிக்கவும்.",
                'sources': []
            }

st.set_page_config(
    page_title="LexWay AI",
    page_icon="⚖️",
    layout="wide"
)


def create_streamlit_interface():
    """
    Enhanced Streamlit interface for Tamil Nadu Police Legal Assistant
    """
    
    # Custom CSS for improved UI
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    /* Dark Theme Base Styles */
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Markdown Styling */
    .stMarkdown {
        color: #e0e0e0;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4e73df;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4e73df !important;
        color: white !important;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #3a5fc4 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(78, 115, 223, 0.3);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        color: #e0e0e0;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #e0e0e0;
    }
    
    /* User Message */
    .stChatMessage.user {
        background-color: #2c3e50;
    }
    
    /* Assistant Message */
    .stChatMessage.assistant {
        background-color: #1e1e1e;
        border: 1px solid #4e73df;
    }
    
    /* Chat Input */
    .stChatInput {
        background-color: #1e1e1e;
        border-radius: 10px;
        border: 1px solid #4e73df;
        color: #e0e0e0;
    }
    
    /* Radio Button */
    .stRadio>div {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    
    /* File Uploader */
    .stFileUploader>div {
        background-color: #1e1e1e;
        color: #e0e0e0;
        border: 1px dashed #4e73df;
    }
    
    /* Spinner */
    .stSpinner>div {
        border-color: #4e73df transparent #4e73df transparent;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize the assistant
    assistant = TamilNaduPoliceLegalAssistant()

    # Sidebar for document upload and language selection
    with st.sidebar:
        st.title("LexWay AI")
        st.markdown("## Police Legal Assistant")
        st.markdown("### Comprehensive Legal Information Support")
        
        # Language Selection
        language = st.radio(
            "Select Response Language",
            ["English", "Tamil"],
            index=0
        )
        
        # Multilingual sidebar text
        if language == 'Tamil':
            st.markdown("### சட்ட ஆவண மேம்பாடு")
        else:
            st.markdown("### Legal Document Enhancement")
        
        # Document Upload
        uploaded_files = st.file_uploader(
            "Upload Additional Legal Documents" if language == 'English' else "கூடுதல் சட்ட ஆவணங்களைப் பதிவேற்றம் செய்யவும்", 
            type=['pdf', 'txt', 'md'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                assistant.document_search.load_uploaded_documents(temp_dir)

    # Main interface
    st.title("LexWay AI")
    st.markdown("## Tamil Nadu Police Legal Assistant" if language == 'English' else "## தமிழ்நாடு காவல்துறை சட்ட உதவி")
    st.markdown("Providing accurate and comprehensive legal information." if language == 'English' else "துல்லியமான மற்றும் விரிவான சட்ட தகவல்கள் வழங்குகிறது.")

    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input with multilingual placeholder
    prompt_placeholder = "Enter your legal query" if language == 'English' else "உங்கள் சட்ட வினாவை உள்ளிடவும்"
    if prompt := st.chat_input(prompt_placeholder):
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching for comprehensive information..." if language == 'English' else "விரிவான தகவல்களைத் தேடுகிறது..."):
                response_data = assistant.generate_comprehensive_response(prompt, language)
                
                # Display main response
                st.markdown(response_data['response'])

        # Add assistant response to messages
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_data['response']
        })


def create_about_page():
    """
    Create a comprehensive About page for the Tamil Nadu Police Legal Assistant
    """

    # Custom CSS for About page
    st.markdown("""
    <style>
    .about-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #1e1e1e;
        border-radius: 10px;
        color: #e0e0e0;
    }
    .section-header {
        color: #4e73df;
        border-bottom: 2px solid #4e73df;
        padding-bottom: 10px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("LexWay AI")
    st.title("## About Tamil Nadu Police Legal Assistant")
    
    # About section with markdown formatting
    st.markdown("""
    ## Application Overview 

    The **LexWay AI - Tamil Nadu Police Legal Assistant** is an advanced AI-powered platform designed to provide comprehensive legal information and support for:
    - Citizens
    - Legal professionals
    - Law enforcement personnel

    ## Key Features 

    ### 1. Multilingual Support 
    - Seamlessly switch between English and Tamil
    - Accurate translations and context preservation

    ### 2. Advanced Document Search
    - Semantic search across multiple document sources
    - Deep indexing of legal documents
    - High-precision information retrieval

    ### 3. Integrated Research Capabilities
    - Combines local document search with web search results
    - Comprehensive information gathering
    - Cross-referencing multiple sources

    ### 4. AI-Powered Intelligent Responses
    - Generate precise, comprehensive legal information
    - Contextual understanding
    - Quick and accurate query resolution

    ## Technology Architecture

    ### Core Technologies
    - **Programming Language**: Python
    - **Web Framework**: Streamlit
    - **AI Integration**: 
      * LangChain
      * HuggingFace Embeddings
      * Groq AI Language Models

    ### Search Mechanism
    1. Multilingual semantic embeddings
    2. Vector database indexing
    3. Intelligent document chunking
    4. Hybrid search strategy

    ## Privacy and Security Commitment

    We prioritize data protection through:
    - No persistent storage of user queries
    - Secure, encrypted API integrations
    - Strict adherence to data protection guidelines

    ## Usage Guidelines

    1. Select preferred language (English/Tamil)
    2. Optional: Upload additional legal documents
    3. Enter your legal query in the chat interface
    4. Receive comprehensive, sourced responses

    ## Disclaimer

    *This is an AI-assisted platform. While we strive for accuracy, always consult official sources and legal professionals for critical legal matters.*
    """)


def main():
    # Add a multipage configuration
    pages = {
        "Chat": create_streamlit_interface,
        "About": create_about_page
    }

    # Determine which page to show
    page = st.sidebar.radio("Navigate", list(pages.keys()))
    
    # Run the selected page
    pages[page]()

if __name__ == "__main__":
    main()
