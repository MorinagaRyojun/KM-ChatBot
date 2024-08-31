import streamlit as st
import logging
import time
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Get the API key from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    st.error("GEMINI_API_KEY is not set. Please set it in your environment variables.")
    st.stop()

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Set up the Streamlit page
st.set_page_config(page_title="SMEGPT", page_icon="🤖")
st.title("SMEGPT")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message("Human" if isinstance(message, HumanMessage) else "AI"):
        st.markdown(message.content)

def is_related_question(new_question, chat_history, threshold=0.3):
    if not chat_history:
        return False
    
    # รวมคำถามก่อนหน้าทั้งหมดเป็นข้อความเดียว
    previous_questions = " ".join([msg.content for msg in chat_history if isinstance(msg, HumanMessage)])
    
    # ใช้ TF-IDF และ cosine similarity เพื่อคำนวณความเกี่ยวข้อง
    vectorizer = TfidfVectorizer().fit_transform([previous_questions, new_question])
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    
    return similarity > threshold

# Function to get general answer context from website using WebBaseLoader
def get_general_answer_context(query):
    url = "https://apex.oracle.com/pls/apex/ryo/km-api/km//"
    
    try:
        # Initialize the loader with the URL
        loader = WebBaseLoader(url)
        
        # Load data from the URL
        documents = loader.load()
        
        # Process the documents to extract relevant text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        context = ""
        
        for document in documents:
            # WebBaseLoader returns Document objects with a 'page_content' attribute
            text = document.page_content
            splits = text_splitter.split_text(text)
            context += " ".join(splits)
        
        # Ensure the context is within acceptable size limits
        context = context[:2000]  # Limit to first 2000 characters or adjust as needed
        
        return context
    except Exception as e:
        logging.error(f"Error fetching general answer context: {e}")
        return "Error fetching context. Please try again later."

# New function to get event information
def get_event_information():
    url = "https://apex.oracle.com/pls/apex/ryo/events/from_today"
    
    try:
        # Initialize the loader with the URL
        loader = WebBaseLoader(url)
        
        # Load data from the URL
        documents = loader.load()
        
        # Process the documents to extract relevant text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        event_info = ""
        
        for document in documents:
            # WebBaseLoader returns Document objects with a 'page_content' attribute
            text = document.page_content
            splits = text_splitter.split_text(text)
            event_info += " ".join(splits)
        
        # Ensure the event information is within acceptable size limits
        event_info = event_info[:2000]  # Limit to first 2000 characters or adjust as needed
        
        return event_info
    except Exception as e:
        logging.error(f"Error fetching event information: {e}")
        return "Error fetching event information. Please try again later."

    
# Improved function to check if the query is event-related
def is_event_query(query):
    query_lower = query.lower()
    
    event_terms = [
        "event", "exhibition", "conference", "seminar", "workshop", "meetup",
        "gathering", "festival", "concert", "show", "expo", "fair",
        "symposium", "convention", "forum", "ceremony", "gala",
        "งาน", "กิจกรรม", "นิทรรศการ", "การประชุม", "สัมมนา", "เวิร์คช็อป",
        "การรวมตัว", "เทศกาล", "คอนเสิร์ต", "การแสดง", "มหกรรม", "ออกร้าน",
        "ประชุมวิชาการ", "พิธี", "งานเลี้ยง", "มีตติ้ง", "อีเวนต์"
    ]
    
    # Calculate the TF-IDF scores for the query and event terms
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query_lower] + event_terms)
    
    # Calculate cosine similarity between the query and event terms
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    
    # If the maximum similarity score is above a threshold, consider it an event query
    threshold = 0.3  # Adjust this value as needed
    return similarity_scores.max() > threshold

# Function to get combined context
def get_combined_context(query):
    general_context = get_general_answer_context(query)
    event_context = get_event_information()
    
    return f"General Context: {general_context}\n\nEvent Context: {event_context}"

# Updated function to get response from LLM
def get_response(query, chat_history, context):
    template = """
    คุณเป็นผู้ช่วยลูกค้าที่มีประโยชน์ ช่วยตอบคำถามเกี่ยวกับ SME และกิจกรรมต่างๆ
    โปรดตอบคำถามต่อไปนี้โดยละเอียดโดยใช้บริบทและประวัติการแชทต่อไปนี้:
    
    บริบท: {context}
    
    ประวัติการแชท: {chat_history}
    
    คำถามของผู้ใช้: {user_question}
    
    หากคำถามเกี่ยวข้องกับกิจกรรมหรืองานเฉพาะ ให้ใช้ข้อมูลจาก Event Context
    หากเป็นคำถามทั่วไปเกี่ยวกับ SME ให้ใช้ข้อมูลจาก General Context
    ถ้าคำถามเกี่ยวข้องกับทั้งสองส่วน ให้ผสมผสานข้อมูลจากทั้งสองแหล่งเพื่อให้คำตอบที่ครอบคลุมที่สุด
    """
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=gemini_api_key)
        
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(context=context, chat_history=chat_history, user_question=query)
        
        return llm.stream(messages)
    except Exception as e:
        logging.error(f"Error in get_response: {e}")
        return iter([AIMessage(content="ขออภัย เกิดข้อผิดพลาดในการประมวลผลคำตอบ โปรดลองอีกครั้งในภายหลัง")])
    
    
# User input
user_query = st.chat_input("คำถามของคุณ")
if user_query:
    # ตรวจสอบว่าคำถามใหม่เกี่ยวข้องกับบทสนทนาก่อนหน้าหรือไม่
    if not is_related_question(user_query, st.session_state.chat_history):
        st.session_state.chat_history = []  # เริ่มการสนทนาใหม่
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        message_placeholder = st.empty()
        full_response = ""
        
        # ดึงบริบทรวม
        context_text = get_combined_context(user_query)
        
        start_time = time.time()
        try:
            for chunk in get_response(user_query, st.session_state.chat_history, context_text):
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")
                
                # ตรวจสอบการหมดเวลา
                if time.time() - start_time > 60:  # หมดเวลาหลัง 60 วินาที
                    raise TimeoutError("การสร้างคำตอบใช้เวลานานเกินไป")
                
            message_placeholder.markdown(full_response)
        except Exception as e:
            logging.error(f"เกิดข้อผิดพลาดระหว่างการสร้างคำตอบ: {e}")
            message_placeholder.markdown("ขออภัย เกิดข้อผิดพลาดในการสร้างคำตอบ โปรดลองอีกครั้งในภายหลัง")
        
        if full_response:
            st.session_state.chat_history.append(AIMessage(content=full_response))
        else:
            st.warning("ไม่ได้รับการตอบกลับจากระบบ โปรดลองอีกครั้ง")

# Add a button to clear chat history
if st.button("ล้างประวัติการสนทนา"):
    st.session_state.chat_history = []
    st.experimental_rerun()