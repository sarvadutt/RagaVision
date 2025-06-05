# ui.py
import streamlit as st
import time
import os
import sys
import traceback
from raga_predictor import predict_raga_from_audio
import matplotlib.pyplot as plt
import librosa.display
from io import BytesIO
import gc

st.set_page_config(
    page_title="Music Assistant",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)
from raga_predictor import predict_raga_from_audio

from streamlit_option_menu import option_menu
from logic import (
        run_ruby_script, read_text_file, clean_text, remove_duplicates,
        load_summarization_model, summarize_text, post_process_summary,
        get_qa_bot, cleanup_files, query_chatbot, extract_entities,
        cleanup_files, clear_summarizer, clear_chatbot
    )
# Check for Ruby script existence
if not os.path.exists("Scraper.rb"):
    st.error("Scraper.rb file is missing. Please ensure the file exists in the application directory.")
    # Create a dummy scraper for testing
    with open("Scraper.rb", "w") as f:
        f.write('puts "Dummy scraper created for testing"\n')
        f.write('File.open("data/search_results.txt", "w") do |f|\n')
        f.write('  f.write("Test search results for: " + ARGV[0])\n')
        f.write('end\n')

def warmup_models():
    if 'warmup_done' not in st.session_state:
        try:
            with st.spinner("Warming up models..."):
                tokenizer, model = load_summarization_model()
                qa_bot = get_qa_bot()
                st.session_state.warmup_done = True
                st.session_state.summarization_model_loaded = True
                st.session_state.qa_model_loaded = True
        except Exception as e:
            st.error(f"Failed to warm up models: {str(e)}")
            st.session_state.warmup_done = False
            st.session_state.summarization_model_loaded = False
            st.session_state.qa_model_loaded = False

# Initialize session state variables
def initialize_ui_state():
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
  

def handle_search(query):
    """Handle search with proper error handling"""
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update progress
        status_text.info("Running web scraper...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        if run_ruby_script(query):
            progress_bar.progress(40)
            status_text.info("Scraping completed! Reading results...")
            time.sleep(0.5)
            
            extracted_text = read_text_file("data/search_results.txt")
            
            if extracted_text:
                progress_bar.progress(60)
                status_text.info("Analyzing content...")
                time.sleep(0.5)
                # Clean up extracted text
                cleaned_text = clean_text(extracted_text)
                deduped_text = remove_duplicates(cleaned_text)
                
                st.markdown("### Preview of collected information:")
                with st.expander("Show raw text", expanded=False):
                    st.text_area("", deduped_text[:1000] + "...", height=150)
                
                progress_bar.progress(80)
                status_text.info("Generating summary... Please wait.")
                
                # Load model if not already loaded
                if not st.session_state.get("summarization_model_loaded", False):
                    tokenizer, model = load_summarization_model()
                    st.session_state.summarization_model_loaded = True
                
                final_summary = summarize_text(deduped_text, st.session_state.search_topic)
                final_summary = post_process_summary(final_summary)

                st.session_state.final_summary = final_summary
                st.session_state.cleaned_text = deduped_text
                
                progress_bar.progress(100)
                if final_summary.strip():
                    status_text.success("Summary generated!")
                else:
                    status_text.warning("Generated summary was empty. Showing raw text instead.")
                    final_summary = deduped_text[:1000] + "..."

                st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                st.markdown("### 📝 Summary:")
                st.write(final_summary)
                entities = extract_entities(final_summary)
                if entities:
                    st.markdown("#### 📌 Key Entities Found:")
                    for ent_text, ent_type in entities:
                        st.markdown(f"- **{ent_text}** ({ent_type})")

                st.download_button(
                    label="📥 Download Summary as Text",
                    data=final_summary,
                    file_name="summary.txt",
                    mime="text/plain",
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Store knowledge base and start chat
                st.session_state.knowledge_base = deduped_text  
                st.session_state.chat_started = True
                st.session_state.conversation_active = True
                
                # Add a button to start chatting with this data
                if st.button("🎵 Chat About This Music", key="start_chat"):
                    st.session_state.current_view = "chat"
                    # Preload the QA model
                    if not st.session_state.get("qa_model_loaded", False):
                        st.session_state.qa_bot = get_qa_bot()
                        st.session_state.qa_model_loaded = True
                        st.experimental_rerun()
            else:
                status_text.error("Failed to retrieve content. Please try a different search.")
        else:
            status_text.error("Failed to run the web scraper. Please try again later.")
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        st.error(traceback.format_exc())

import base64

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def load_css(view_name):
    
    background_map = {
    "search": "static/Background_explore.png",
    "chat": "static/Background_chat.png",
    "about": "static/Background_about.png",
    "raga_predictor": "static/Background_Ra_ui.png"
     }

    image_path = background_map.get(view_name, "static/Background_explore.png")
    encoded_image = get_base64_encoded_image(image_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load your existing styles.css
    with open("static/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_interface():
    # Apply CSS
 
    load_css(st.session_state.current_view)

    # Initialize UI state
    initialize_ui_state()
        
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("<div class='icon-container'><span class='icon-large'>🎶</span></div>", unsafe_allow_html=True)
        st.title("Music Assistant")    
        
        # Navigation menu with error handling
        try:
            selected = option_menu(
                "",
                ["Explore", "Music Chat", "About"],
                icons=["search", "chat-dots", "info-circle"],
                menu_icon="cast",
                default_index=0,
            )
            # Store the previous view
            previous_view = st.session_state.get("current_view", None)

            # Keep your logic unchanged
            if selected == "Explore" and st.session_state.current_view not in ["raga_predictor", "chat"]:
                st.session_state.current_view = "search"
            elif selected == "Music Chat" and st.session_state.chat_started:
                st.session_state.current_view = "chat"
            elif selected == "About":
                st.session_state.current_view = "about"

            # Rerun only if the view changed
            if st.session_state.current_view != previous_view:
                st.rerun()


        except Exception as e:
            # Fallback if option_menu fails
            st.error(f"Navigation menu error: {str(e)}")
            st.markdown("### Navigation")
            if st.button("📚 Research", use_container_width=True):
                st.session_state.current_view = "search"
            
            chat_button = st.button("💬 Chat", use_container_width=True, disabled=not st.session_state.chat_started)
            if chat_button and st.session_state.chat_started:
                st.session_state.current_view = "chat"
                
            if st.button("ℹ About", use_container_width=True):
                st.session_state.current_view = "about"
        
        # Show a "New Search" button if chat is active
        if st.session_state.chat_started and st.session_state.current_view != "about":
            if st.button("🎶 New Music Search", key="new_search"):
                cleanup_files()
                clear_chatbot()
                clear_summarizer()
                st.session_state.chat_started = False
                st.session_state.chat_history = []
                st.session_state.knowledge_base = ""
                st.session_state.conversation_active = False
                st.session_state.search_topic = ""
                st.session_state.current_view = "search"
                st.rerun()

    # Main content area with error handling for each view
    try:
        if st.session_state.current_view == "search":
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("<h1 style='text-align: center; color: #2e4057;'>Music Assistant Tool</h1>", unsafe_allow_html=True)
                st.markdown("<p style='text-align: center; color: #666;'>Search, summarize, and chat about any music topic</p>", unsafe_allow_html=True)
                
                if st.session_state.chat_started and "final_summary" in st.session_state:
                    st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                    st.markdown("### 🎶 Music Insights (Previous):")
                    st.write(st.session_state.final_summary)
                    st.download_button(
                        label="📥 Download Music Insights as Text",
                        data=st.session_state.final_summary,
                        file_name="summary.txt",
                        mime="text/plain",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                if not st.session_state.chat_started: 
                    with st.container():
                        st.markdown("<div class='search-container'>", unsafe_allow_html=True)

                        with st.container():
                            st.markdown("### 🎤 Predict Raga")
                            if st.button("🎵 Upload Audio for Raga Prediction"):
                                st.session_state.current_view = "raga_predictor"
                                st.rerun()

                        st.markdown("<br>", unsafe_allow_html=True)

                        with st.container():
                            st.markdown("### Enter Music Topic:")
                            query = st.text_input("", placeholder="e.g., Hindustani classical, Jazz improvisation...")
                            if st.button("🔍 Search Topic"):
                                if query.strip():
                                    st.session_state.search_topic = query.strip()
                                    handle_search(query)
                                else:
                                    st.warning("Please enter a topic.")
                                    

            with col2:
                st.markdown("""
                <div class='how-it-works-container'>
                    <div class='icon-container'><span class='icon-medium'>🤔</span></div>
                    <h3 class='how-title'>How it works</h3>
                    <div class='feature-card'>
                        <strong>1. Find Music or Upload Songs</strong><br>
                        <span>Search for music topics, ragas, or upload your own music tracks.</span>
                    </div>
                    <div class='feature-card'>
                        <strong>2. Analyze & Summarize</strong><br>
                        <span>The AI processes the content and creates a concise summary.</span>
                    </div>
                    <div class='feature-card'>
                        <strong>3. Chat & Learn</strong><br>
                        <span>Ask questions about the extracted information to dive deeper.</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    

        elif st.session_state.current_view == "chat" and st.session_state.chat_started:
            try:
                st.markdown("<h2 style='text-align: center;'>Chat About This Music</h2>", unsafe_allow_html=True)
                
                # Initialize QA bot with knowledge base if not done already
                if 'qa_bot' not in st.session_state:
                    with st.spinner("Initializing chat capabilities..."):
                        # Get the bot instance (will be cached)
                        st.session_state.qa_bot = get_qa_bot()
                        # Load the knowledge base from our text
                        if st.session_state.knowledge_base:
                            st.session_state.qa_bot.load_knowledge_base_from_text(st.session_state.knowledge_base)
                            st.session_state.qa_model_loaded = True
                            st.info("Knowledge base loaded. You can now ask questions about the content.")
                        else:
                            st.error("No knowledge base available. Please perform a search first.")
                            st.session_state.current_view = "search"
                            st.rerun()
                
                # Display chat container
                st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
                
                # Display chat messages
                for message in st.session_state.chat_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                        if message["role"] == "assistant" and "metadata" in message:
                            with st.expander("Answer metadata"):
                                st.write(f"Confidence: {message['metadata'].get('confidence', 'N/A')}")
                                st.write(f"Sources used: {message['metadata'].get('sources', 'N/A')}")
                                st.write(f"Processing time: {message['metadata'].get('processing_time', 'N/A')}")

                
                # User input for chat
                user_input = st.chat_input("Ask a question about this music topic:")

                if user_input:
                    st.session_state.chat_history.append({"role": "user", "content": user_input})

                    with st.chat_message("assistant"):
                        try:
                            result = st.session_state.qa_bot.answer_question(user_input)
                            response = result["answer"]
                            st.markdown(response)

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response,
                                "metadata": {
                                    "confidence": result.get("confidence", "N/A"),
                                    "sources": result.get("sources", "N/A"),
                                    "processing_time": result.get("processing_time", "N/A")
                                }
                            })
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": "I encountered an error. Please try again.",
                                "metadata": {
                                    "confidence": 0,
                                    "sources": "error",
                                    "processing_time": "error"
                                }
                            })

                    # Rerun to update UI
                    st.rerun()

                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show a message if chat history is empty
                if not st.session_state.chat_history:
                    st.info("Ask a question about the topic to start the conversation.")
            except Exception as e:
                st.error(f"Error in chat view: {str(e)}")
                st.error(traceback.format_exc())

        elif st.session_state.current_view == "raga_predictor":
            st.title("🎶 Raga Predictor")
            st.markdown("Upload a music clip to predict the **Raga** and explore more.")

            audio_file = st.file_uploader("🎵 Upload audio (.mp3 / .wav)", type=["mp3", "wav"])

            if audio_file:
                with st.spinner("Analyzing your audio file..."):
                    predicted_raga, confidence, errors = predict_raga_from_audio(audio_file)

                if errors:
                    st.error("Prediction failed: " + ", ".join(errors))
                else:
                    st.success(f"🎼 Predicted Raga: **{predicted_raga}** ({confidence:.1f}%)")

                    # Save for continuity
                    st.session_state["saved_raga"] = predicted_raga

                    # Show waveform
                    y, sr = librosa.load(BytesIO(audio_file.getvalue()), sr=None)
                    fig, ax = plt.subplots()
                    librosa.display.waveshow(y, sr=sr, ax=ax)
                    ax.set(title='Waveform')
                    st.pyplot(fig)

                    # Enhanced Proceed button

                    if st.button("🚀 Let's Explore This Raga"):
                        st.session_state.search_topic = predicted_raga
                        st.session_state.current_view = "search"

                        with st.spinner("Running web scraper for this raga..."):
                            handle_search(predicted_raga)

                        st.session_state.chat_started = True
                        st.session_state.conversation_active = True
                        st.session_state.current_view = "chat"
                        st.rerun()


            st.markdown("You can also [go back](#) to start a new search.")
            if st.button("🔙 Back"):
                st.session_state.current_view = "search"
                st.rerun() 
   
        elif st.session_state.current_view == "about":
            st.markdown("<h1 style='text-align: center; color:#4e8df5;'>🎵 About Music Assistant</h1>", unsafe_allow_html=True)

            # Light Hero Banner
            st.markdown("""
            <div style="background: linear-gradient(to right, #e0eaff, #ffffff); padding: 16px; border-radius: 12px; color: #222; text-align: center; margin-bottom: 30px;">
                <h2 style="margin: 0;">🌍 Discover the World of Music with AI</h2>
                <p style="margin-top: 5px; font-size: 16px;">Smart tools for musicians, learners, and creators.</p>
            </div>
            """, unsafe_allow_html=True)

            # What is Music Assistant?
            st.markdown("""
            ## 🎼 What is Music Assistant?

            **Music Assistant** is a powerful AI-driven platform to **explore**, **analyze**, and **learn** music from around the world.

            Whether you're studying Hindustani classical, jazz improvisation, or modern music theory, this tool helps you understand more deeply using web intelligence and document summarization.

            <div style="background-color: rgba(255,255,255,0.9); border-left: 4px solid #4e8df5; padding: 15px 20px; border-radius: 8px; margin: 20px 0; color: #222;">
                <h3 style="margin-bottom: 10px;">✨ Key Features</h3>
                <ul style="padding-left: 20px;">
                    <li>🔍 <strong>Music Research</strong>: Explore genres, artists, and instruments</li>
                    <li>📄 <strong>Document Analysis</strong>: Upload and analyze PDFs or lyrics</li>
                    <li>🎹 <strong>Composition Insights</strong>: Understand structure and theory</li>
                    <li>💬 <strong>Interactive Q&A</strong>: Ask questions and get smart answers</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            # How It Works – Light style boxes
            st.markdown("""
            ## 🎓 How It Works

            <div style="display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0;">
                <div style="background-color: rgba(255,255,255,0.9); color: #222; padding: 20px; border-radius: 10px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); min-width: 220px; flex: 1;">
                    <h4 style="color: #4e8df5;">1️⃣ Search</h4>
                    <p>Upload an audio file or type a musical topic to begin your journey.</p>
                </div>
                <div style="background-color: rgba(255,255,255,0.9); color: #222; padding: 20px; border-radius: 10px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); min-width: 220px; flex: 1;">
                    <h4 style="color: #4e8df5;">2️⃣ Summarize</h4>
                    <p>The system scrapes the web, cleans the content, and summarizes it for quick understanding.</p>
                </div>
                <div style="background-color: rgba(255,255,255,0.9); color: #222; padding: 20px; border-radius: 10px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); min-width: 220px; flex: 1;">
                    <h4 style="color: #4e8df5;">3️⃣ Chat</h4>
                    <p>Interact with an AI assistant to clarify doubts, dive deeper, or get quick facts.</p>
                </div>
            </div>

            ## ⚙️ Technologies Used

            - 🤖 HuggingFace Transformers (Flan-T5)
            - 🧠 Sentence Embeddings (MiniLM)
            - 🕸️ Ruby Web Scraper
            - 🎼 Librosa for audio processing
            - 🐍 Streamlit & Python

            ## 🔐 Your Privacy Matters

            All your audio and text files are processed locally. **Nothing is stored or shared externally**.

            <div style="text-align: center; margin-top: 30px; font-style: italic; color: #555;">
                <p>"Music gives a soul to the universe, wings to the mind, flight to the imagination, and life to everything." — Plato</p>
            </div>

            ---
            <p style="text-align: center; color: #888;">Made with ❤️ for musicians, educators, and creators.</p>
            """, unsafe_allow_html=True)


    except Exception as e:
        st.error(f"Error rendering interface: {str(e)}")
        st.error(traceback.format_exc())

    # Add a musical footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #666;'>🎵 Music Assistant Tool | AI-Powered Musical Research & Learning 🎼</p>", unsafe_allow_html=True)

# Run the app
def main():
    render_interface()

if __name__ == "__main__":
    main()
