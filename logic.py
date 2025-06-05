import re
import os
import gc   
import sys
import nltk
import types
import time
import subprocess
import spacy
import spacy.cli
import importlib.util
import streamlit as st
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util 
from streamlit_option_menu import option_menu
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
sys.modules['torch.classes'] = types.ModuleType("torch.classes")

# Check if the model is installed
def is_model_installed(model_name):
    return importlib.util.find_spec(model_name) is not None

if not is_model_installed("en_core_web_sm"):
    import spacy.cli
    spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

# Add delayed import for Torch-related modules
@st.cache_resource(show_spinner=True)
def load_summarization_model():
    try:
        # Isolate Torch imports
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        model_name = "google/flan-t5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        
        st.session_state["summarization_tokenizer"] = tokenizer
        st.session_state["summarization_model"] = model
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading summarization model: {e}")
        return None, None

@st.cache_resource(show_spinner=True)
def load_embedder():
    """Load sentence embedding model with proper caching"""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading embedder: {e}")
        return None

def cleanup_resources():
    import torch
    """Clear memory after processing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def run_ruby_script(query, max_retries=3):
    """Runs the Ruby script with retry logic until scraping succeeds."""
    for attempt in range(1, max_retries + 1):
        try:
            st.info(f"Attempt {attempt} to run the scraper...")
            process = subprocess.Popen(["ruby", "Scraper.rb", query])
            process.wait()

            if os.path.exists("data/search_results.txt"):
                with open("data/search_results.txt", "r", encoding="utf-8") as file:
                    content = file.read().strip()
                    if content:
                        return True  # Scraping successful

            st.warning(f"Attempt {attempt} failed. Retrying...")
            time.sleep(1.5)

        except subprocess.CalledProcessError as e:
            st.error(f"Error running Ruby script (attempt {attempt}): {e}")
    
    return False  # All attempts failed

# Add to logic.py
def initialize_session_state():
    """Initialize all required session state variables"""
    if 'summarization_model_loaded' not in st.session_state:
        st.session_state.summarization_model_loaded = False
    if 'qa_model_loaded' not in st.session_state:
        st.session_state.qa_model_loaded = False
    if 'chat_started' not in st.session_state:
        st.session_state.chat_started = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = ""
    if 'conversation_active' not in st.session_state:
        st.session_state.conversation_active = False
    if 'current_view' not in st.session_state:
        st.session_state.current_view = "search"
    if 'last_error' not in st.session_state:
        st.session_state.last_error = None
    if 'search_topic' not in st.session_state:
        st.session_state.search_topic = ""

def clean_text(text):
    """Enhanced text cleaning to improve quality"""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove citations
    text = re.sub(r'\[[^\]]*\]', '', text)
    # Remove repetitive patterns
    text = re.sub(r'(\w+)(\s+\1){2,}', r'\1', text)
    # Collapse multiple newlines
    text = re.sub(r'\n{2,}', '\n', text)
    # Reduce repeated characters
    text = re.sub(r'(\.\s*){3,}', '... ', text)
    # Remove extra spaces
    text = re.sub(r'\s{2,}', ' ', text)
    # Remove common web elements
    text = re.sub(r'(Footer|Header|Navigation|Search Box|Search Results)', '', text)
    # Replace bullet points with consistent format
    text = re.sub(r'\n\s*•', '\n- ', text)
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and unnecessary symbols
    text = re.sub(r'[^\w\s.,;:?!()\'"-]', ' ', text)
    
    # Remove citations
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\s+([.,;:?!)])', r'\1', text)
    text = re.sub(r'([([\'"])\s+', r'\1', text)
    text = re.sub(r'---+', '', text)  # Remove divider lines
    text = re.sub(r'(Scraped on:.*?[\r\n])', '', text)
    text = re.sub(r'(Link:.*?[\r\n])', '', text)
    return text.strip()

def clear_chatbot():
    import torch
    if "qa_bot" in st.session_state:
        del st.session_state["qa_bot"]
    if "qa_model_loaded" in st.session_state:
        st.session_state["qa_model_loaded"] = False
    gc.collect()
    torch.cuda.empty_cache()

def remove_duplicates(text):
    """Improved duplicate removal function with semantic similarity detection"""
    # Split by paragraphs for better deduplication
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    
    seen_sentences = set()
    
    for paragraph in paragraphs:
        sentences = paragraph.split('. ')
        cleaned_sentences = []
        
        for sentence in sentences:
            # Normalize sentence for comparison
            normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
            if normalized and len(normalized) > 10 and normalized not in seen_sentences:
                seen_sentences.add(normalized)
                cleaned_sentences.append(sentence)
        
        if cleaned_sentences:
            cleaned_paragraph = '. '.join(cleaned_sentences)
            if not cleaned_paragraph.endswith('.') and cleaned_paragraph:
                cleaned_paragraph += '.'
            cleaned_paragraphs.append(cleaned_paragraph)
    
    return '\n\n'.join(cleaned_paragraphs)

def remove_low_content_lines(text):
    """Removes lines with low information density"""
    lines = text.split("\n")
    # Keep lines with sufficient content or that appear to be headers
    filtered_lines = []
    for line in lines:
        line = line.strip()
        # Keep headers (shorter lines that end without punctuation) or substantive content
        if (len(line) > 5 and len(line) < 30 and not line[-1] in ".,:;?!") or len(line) > 30:
            filtered_lines.append(line)
    return "\n".join(filtered_lines)

@st.cache_resource
def get_qa_bot():
    """Initialize and cache QA bot instance"""
    # Use the same models that are already loaded for summarization
    tokenizer, model = load_summarization_model()
    embedder = load_embedder()
    
    qa_bot = EnhancedQABot(
        qa_model=model,
        qa_tokenizer=tokenizer,
        embedder=embedder
    )
    return qa_bot


def clear_summarizer():
    import torch
    if "summarization_model" in st.session_state:
        del st.session_state["summarization_model"]
    if "summarization_tokenizer" in st.session_state:
        del st.session_state["summarization_tokenizer"]

    st.session_state["summarization_model_loaded"] = False
    torch.cuda.empty_cache()
    gc.collect()


class EnhancedQABot:

    def __init__(self, qa_model=None, qa_tokenizer=None, embedder=None):

        print("Initializing QA Bot...")
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        print(f"Using device: {self.device}")
        
        # Use provided models or load new ones
        if qa_model is None or qa_tokenizer is None:
            model_name = "google/flan-t5-large"
            self.qa_tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            if self.device == "cpu":
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            self.qa_model = model.to(self.device)

        else:
            self.qa_model = qa_model
            self.qa_tokenizer = qa_tokenizer
        
        # Use provided embedder or load new one
        if embedder is None:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        else:
            self.embedder = embedder
        
        self.knowledge_base = []
        self.chunk_embeddings = None
        self.chunk_size = 400
        self.overlap = 100  # Added overlap for better context preservation
        
    def load_knowledge_base_from_text(self, text: str, chunk_size: int = 400, overlap: int = 100) -> None:
        """Load and process the knowledge base from text"""
        print(f"Loading knowledge base from text...")
        start_time = time.time()
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Clean and split text
        text = self._clean_text(text)
       # self.knowledge_base = self._split_text_into_chunks(text, chunk_size, overlap)
        word_count = len(text.split())

        # Dynamically decide chunk size
        if word_count < 500:
            chunk_size = word_count  # don't split at all
            overlap = 0
        elif word_count < 2000:
            chunk_size = 400
            overlap = 100
        else:
            chunk_size = 250
            overlap = 75

        self.knowledge_base = self._split_text_into_chunks(text, chunk_size, overlap)

        
        # Pre-compute embeddings for all chunks
        self.chunk_embeddings = self.embedder.encode(
            self.knowledge_base, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        elapsed = time.time() - start_time
        print(f"Knowledge base loaded: {len(self.knowledge_base)} chunks created in {elapsed:.2f} seconds")
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Fix common issues with scraped text
        text = re.sub(r'(\w)\.(\w)', r'\1. \2', text)  # Fix periods without spaces
        return text.strip()

    def _split_text_into_chunks(self, text: str, chunk_size: int = 250, overlap: int = 100) -> List[str]:
        """Split text into semantically coherent sentence-based chunks"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if current_length + word_count <= chunk_size:
                current_chunk.append(sentence)
                current_length += word_count
            else:
                # Push current chunk
                chunks.append(" ".join(current_chunk))
                # Start new chunk with overlap
                current_chunk = current_chunk[-(overlap // 10):] if overlap else []  # overlap in sentences
                current_length = sum(len(s.split()) for s in current_chunk)
                current_chunk.append(sentence)
                current_length += word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    
    # def _split_text_into_chunks(self, text: str, chunk_size: int = 250, overlap: int = 100) -> List[str]:
    #     """Split text into overlapping chunks for better context preservation"""
    #     words = text.split()
    #     chunks = []
        
    #     i = 0
    #     while i < len(words):
    #         # Take chunk_size words, but don't exceed the array bounds
    #         end_idx = min(i + chunk_size, len(words))
    #         chunk = ' '.join(words[i:end_idx])
    #         chunks.append(chunk)
            
    #         # Move the window, accounting for overlap
    #         i += chunk_size - overlap
        
    #     return chunks

    
    def _find_relevant_chunks(self, question: str, top_k: int = 6) -> List[Tuple[str, float]]:
        """Find the most relevant chunks to the question with similarity scores"""
        if not self.knowledge_base:
            raise ValueError("Knowledge base not loaded. Call load_knowledge_base() first.")
        
        # Encode the question
        question_embedding = self.embedder.encode(question, convert_to_tensor=True).to(self.device)
        
        # Calculate similarities
        similarities = util.cos_sim(question_embedding, self.chunk_embeddings)[0]
        
        # Get top-k chunks
        top_k = min(top_k, len(self.knowledge_base))
        top_indices = similarities.topk(top_k)
        indices = top_indices.indices.tolist()
        scores = top_indices.values.tolist()
        
        # Return chunks with their similarity scores
        return [(self.knowledge_base[i], scores[idx]) for idx, i in enumerate(indices)]
        
    def _generate_answer(self, context: str, question: str, max_length: int = 250) -> str:
        import torch

        """Generate an answer based on the context and question"""
        #prompt = f"Answer the question accurately based on the given context. If you cannot find the answer in the context, say 'I don't have enough information to answer this question.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        prompt = (
        "Answer the question **clearly and concisely** using only the information in the context. "
        "Avoid repeating unrelated facts. "
        "If the answer is not found in the context, respond with: 'I don't have enough information to answer this question.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        inputs = self.qa_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        
        with torch.no_grad():
            outputs = self.qa_model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=75,
                num_beams=3,
                early_stopping=True,
                temperature=0.7,
                do_sample=False,  # Set to True for more diverse answers
                no_repeat_ngram_size=3
            )
        
        return self.qa_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    
    def _format_context(self, relevant_chunks: List[Tuple[str, float]], threshold: float = 0.7) -> str:
        """Format the context from relevant chunks, filtering by relevance threshold"""
        filtered_chunks = [chunk for chunk, score in relevant_chunks if score >= threshold]
        
        if not filtered_chunks:
            # If no chunks pass the threshold, use the top chunk anyway
            filtered_chunks = [relevant_chunks[0][0]]
            
        return "\n\n".join(filtered_chunks)
        
    def answer_question(self, question: str, top_k: int = 6, threshold: float = 0.7) -> dict:
        """Process a question and return the answer with metadata"""
        start_time = time.time()
        
        # Find relevant content
        relevant_chunks = self._find_relevant_chunks(question, top_k=top_k)
        context = self._format_context(relevant_chunks, threshold=threshold)
        
        # Generate answer
        #answer = self._generate_answer(context, question)
        raw_answer = self._generate_answer(context, question)
        answer = post_process_summary(raw_answer)

        
        # Calculate confidence based on relevance scores
        confidence = sum(score for _, score in relevant_chunks) / len(relevant_chunks)
        
        elapsed = time.time() - start_time
        
        return {
            "answer": answer,
            "confidence": float(confidence),
            "sources": len(relevant_chunks),
            "processing_time": f"{elapsed:.2f}s"
        }

def split_into_batches(text, batch_size=500):
    """Split text into batches using sentence tokenization for better coherence"""
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except:
        # Fallback if NLTK fails
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    batches = []
    current_batch = ""
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())

        if current_word_count + sentence_word_count <= batch_size:
            current_batch += sentence + " "
            current_word_count += sentence_word_count
        else:
            if current_batch:
                batches.append(current_batch.strip())
            current_batch = sentence + " "
            current_word_count = sentence_word_count

    if current_batch:
        batches.append(current_batch.strip())

    return [b for b in batches if len(b.split()) > 5]  # Filter out very short batches

def generate_summary_response(prompt, max_length=400, min_length=100):
    """Generate responses using BART-CNN model with error handling"""
    try:
        tokenizer, model = load_summarization_model()

        inputs = tokenizer(
            prompt,
            max_length=2048,
            truncation=True,
            return_tensors="pt"
        )

        output_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    except Exception as e:
        return f"Sorry, I encountered an error during summarization: {str(e)}"

def summarize_batch(batch, topic=""):
    """Improved batch summarization with topic-aware prompting"""
    try:
        if not batch.strip():
            return ""
        
        # Create a more guided prompt based on the topic
        if topic:
            prompt = (
                f"Provide a comprehensive summary of the following text about {topic}. "
                "Include key information, maintain factual accuracy and cover all major points:\n\n"
                + batch
            )
        else:
            prompt = (
                "Provide a comprehensive summary of the following text. "
                "Include key information, maintain factual accuracy and cover all major points:\n\n"
                + batch
            )
        
        return generate_summary_response(prompt, max_length=250, min_length=40)
    
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return ""

def summarize_text(text, topic=""):
    """Two-stage summarization approach for better quality"""
    if not st.session_state.summarization_model_loaded:
        load_summarization_model()
        st.session_state.summarization_model_loaded = True

    batches = split_into_batches(text, 400)  # Smaller batches for better processing
    batch_summaries = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, batch in enumerate(batches):
        status_text.info(f"Processing batch {i+1}/{len(batches)}...")
        progress_bar.progress((i + 1) / (len(batches) + 1))  # Save space for final summary

        summary = summarize_batch(batch, topic)
        if summary:
            batch_summaries.append(summary)

        # Free memory
        del batch, summary
        cleanup_resources()

    # For multiple batches, create a final summary of summaries
    if len(batch_summaries) > 1:
        combined_summaries = " ".join(batch_summaries)
        status_text.info("Creating final consolidated summary...")
        progress_bar.progress(1.0)
        final_summary = combined_summaries
    elif len(batch_summaries) == 1:
        final_summary = batch_summaries[0]
    else:
        final_summary = "Could not generate a summary from the provided text."

    # ✅ Clear summarizer model from memory after use
    clear_summarizer()

    return final_summary

def cleanup_files():
    """Deletes temporary scraped data files after summarization."""
    files_to_delete = ["data/search_results.txt", "data/search_results.json"]
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)

def read_text_file(filepath):
    """Reads text from the Ruby-generated text file."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    return ""

def split_knowledge_base(text, chunk_size=1000):
    """Split knowledge base into semantically coherent passages"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def post_process_summary(summary):
    """Clean up the final summary for presentation"""
    # Remove reference markers
    summary = re.sub(r'\[\d+\]', '', summary)
    
    # Fix spacing issues
    summary = re.sub(r'\s+([.,;:?!])', r'\1', summary)
    
    # Ensure proper capitalization
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
    
    # Rejoin with proper spacing
    summary = ' '.join(sentences)
    
    return summary

def query_chatbot(qa_bot, user_input):
    """Process a question using the EnhancedQABot"""
    try:
        result = qa_bot.answer_question(user_input, top_k=3)
        return result["answer"]
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")
        return "Sorry, I couldn't process your question. Please try rephrasing it."
    
# --- Entity Extraction ---
def extract_entities(summary):
    """Extract people and musical work names from summary text"""
    doc = nlp(summary)
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ("PERSON", "WORK_OF_ART", "ORG")]
