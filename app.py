import streamlit as st
import requests
import base64
from PIL import Image
import io

# --- 1. Page Config ---
st.set_page_config(
    page_title="Intelligent Document Assistant",
    page_icon="📚",
    layout="wide"
)

# --- 2. Backend API Definition ---
BACKEND_URL = "http://127.0.0.1:8000"

# --- 3. API Client Functions ---
def api_get_indexed_files():
    try:
        response = requests.get(f"{BACKEND_URL}/indexed_files")
        response.raise_for_status()
        return response.json().get("files", [])
    except requests.exceptions.RequestException:
        st.sidebar.error("Connection to backend failed. Please ensure the backend service is running.")
        return []

def api_upload_file(uploaded_file):
    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    try:
        with st.status(f"Processing '{uploaded_file.name}'...") as status:
            status.update(label="Uploading file...")
            response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=600)
            response.raise_for_status()
            result = response.json()
            
            status.update(label="Analyzing and indexing content...")
            import time
            time.sleep(2)

            if result.get("status") == "exists":
                status.update(label=f"File '{result['filename']}' already exists!", state="complete")
                st.toast(f"ℹ️ File '{result['filename']}' is already in the knowledge base.")
            else:
                status.update(label="Processing complete!", state="complete")
                st.toast(f"✅ File '{result['filename']}' was added successfully!")
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"File upload failed: {e}")
        return None

def api_ask_question(question: str, image_file=None):
    data = {'question': question}
    files = {}
    if image_file:
        files['file'] = (image_file.name, image_file.getvalue(), image_file.type)
    
    try:
        response = requests.post(f"{BACKEND_URL}/query", data=data, files=files if image_file else None, timeout=600)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Query failed: {e}")
        return None

def api_clear_db():
    try:
        with st.spinner("Clearing knowledge base..."):
            response = requests.post(f"{BACKEND_URL}/clear_all")
            response.raise_for_status()
        st.toast("💥 Knowledge base cleared successfully!")
        st.sidebar.success("Knowledge base has been cleared on the backend.")
        st.sidebar.warning("Please restart the backend server to ensure all connections are reset before uploading new documents.")
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to clear knowledge base: {e}")
        return None

# --- 4. Session State Initialization ---
# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# Initialize list of indexed files
if 'indexed_files' not in st.session_state:
    st.session_state.indexed_files = api_get_indexed_files()
# Initialize staged image for query
if 'staged_image' not in st.session_state:
    st.session_state.staged_image = None
# Initialize upload key
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

# --- 5. Callback Functions for UI interactions ---
def handle_knowledge_upload():
    """Handle new document uploads."""
    if st.session_state.knowledge_uploader is not None:
        uploaded_file = st.session_state.knowledge_uploader
        api_upload_file(uploaded_file)
        st.session_state.indexed_files = api_get_indexed_files()

def handle_send_message():
    """Handle sending a new message (text + image)."""
    prompt_text = st.session_state.prompt
    staged_image = st.session_state.staged_image

    if prompt_text or staged_image:
        # Append user message to chat history
        user_message = {"role": "user", "content": prompt_text if prompt_text else "Please analyze this image."}
        if staged_image:
            user_message["image"] = staged_image
        st.session_state.chat_history.append(user_message)

        # Get assistant response
        response = api_ask_question(prompt_text, staged_image)
        if response:
            assistant_message = {
                "role": "assistant",
                "content": response.get("answer", "Sorry, I was unable to answer the question."),
                "sources": response.get("sources", [])
            }
            st.session_state.chat_history.append(assistant_message)
        else:
            st.session_state.chat_history.append({
                "role": "assistant", "content": "Could not retrieve an answer from the backend.", "sources": []
            })
        
        # Clear inputs for the next message
        st.session_state.prompt = ""
        st.session_state.staged_image = None
        
def handle_clear_db():
    """
    Handles the logic for clearing the database, including user guidance for the manual restart process.
    """
    try:
        st.toast("Sending request to clear knowledge base...")
        response = requests.post(f"{BACKEND_URL}/clear_all")
        
        # Success Case: The backend somehow managed to delete the files
        if response.status_code == 200:
            st.toast("✅ Backend reports knowledge base is cleared!")
            st.session_state.clear_db_requires_restart = True # Still recommend restart
        
        # File Lock Error Case: The backend reports it couldn't delete the files
        elif response.status_code == 409:
            st.toast("Backend process has locked the files, manual action required.")
            st.session_state.clear_db_requires_restart = True # Trigger the warning message
            
        else:
            # Other HTTP errors
            response.raise_for_status()

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to send 'clear' request to backend: {e}")

    # Reset frontend state regardless of backend outcome
    st.session_state.indexed_files = []
    st.session_state.chat_history = []

# --- 6. Sidebar ---
with st.sidebar:
    st.title("📚 Knowledge Base")
    
    st.header("Upload New Document", divider='rainbow')
    st.file_uploader(
        "Drag and drop your files here (PDF, DOCX, TXT...)",
        type=['pdf', 'docx', 'txt', 'md'],
        key="knowledge_uploader",
        on_change=handle_knowledge_upload # ✨ Use callback for robust handling
    )

    st.header("Indexed Documents", divider='rainbow')
    if not st.session_state.indexed_files:
        st.info("The knowledge base is empty.")
    else:
        for filename in st.session_state.indexed_files:
            st.markdown(f"📄 {filename}")
    
    if st.button("🔄 Refresh List"):
        st.session_state.clear_db_requires_restart = False # Acknowledge the warning on refresh
        st.rerun()

# --- 7. Main Chat Interface ---
st.title("🤖 Intelligent Document Assistant")
st.caption("Upload your documents to the knowledge base, then ask questions about their content using text or text + images.")

chat_container = st.container(border=True)
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user" and "query_image_b64" in message:
                img_data = base64.b64decode(message["query_image_b64"])
                img = Image.open(io.BytesIO(img_data))
                st.image(img, caption=message.get("query_image_name", "Uploaded image"), width=180)
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("View Cited Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**File:** `{source.get('source', 'N/A')}` | **Page:** `{source.get('page', 'N/A')}`")
                        if source.get('type') == 'image' and source.get('image_b64'):
                            try:
                                img_data = base64.b64decode(source['image_b64'])
                                img = Image.open(io.BytesIO(img_data))
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(img, use_column_width=True)
                                with col2:
                                    summary_text = source.get('summary', 'No summary available.')
                                    st.info(summary_text)
                            except Exception:
                                st.warning("Could not display source image.")
                        else:
                            summary_text = source.get('summary', 'No summary available.')
                            st.info(summary_text)
                        st.divider()

# --- 8. Integrated Chat Input ---
with st.form(key="main_input_form"):
    st.markdown("#### 📥 Enter your question and image (optional)")
    input_col1, input_col2 = st.columns([5, 2])
    with input_col1:
        prompt = st.text_area("Question/Description", key="main_chat_input", height=80, placeholder="Type your question or description...")
    with input_col2:
        query_image_file = st.file_uploader("Upload image (optional)", type=['png', 'jpg', 'jpeg'], key=f"main_query_uploader_{st.session_state.upload_key}")
        if query_image_file:
            st.image(query_image_file, caption="Your uploaded image", width=180)
    send_btn = st.form_submit_button("Send")

if send_btn and (prompt or query_image_file):
    if not prompt and not query_image_file:
        st.warning("Please enter a question or upload an image.")
    else:
        user_message = {"role": "user", "content": prompt}
        if query_image_file:
            image_bytes = query_image_file.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            user_message["query_image_b64"] = image_b64
            user_message["query_image_name"] = query_image_file.name
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            if query_image_file:
                st.image(query_image_file, width=180)
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = api_ask_question(prompt, query_image_file)
                if response:
                    answer = response.get("answer", "Sorry, I can't answer this question.")
                    sources = response.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("View sources"):
                            for source in sources:
                                st.markdown(f"**Source:** `{source.get('source', 'N/A')}` | **Page:** `{source.get('page', 'N/A')}`")
                                if source.get('type') == 'image' and source.get('image_b64'):
                                    try:
                                        img_data = base64.b64decode(source['image_b64'])
                                        img = Image.open(io.BytesIO(img_data))
                                        st.image(img, caption=f"Source image (Page: {source.get('page')})", use_column_width=True)
                                    except Exception as e:
                                        st.warning(f"Cannot display source image: {e}")
                                st.info(source.get('summary', 'No summary available.'))
                                st.divider()
                    st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
                else:
                    error_msg = "Failed to get response from backend."
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg, "sources": []})
        st.session_state.upload_key += 1
        st.rerun()