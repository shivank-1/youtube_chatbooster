import os
import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
groqkey = os.getenv('GROQKEY')

# Streamlit App Configuration
st.set_page_config(page_title="YouTube Video Summarizer", page_icon="🎥")
st.title("🎥 YouTube Video Summarizer")

def extract_youtube_video_id(url):
    """
    Extract YouTube video ID from different URL formats
    """
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^?&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^?&]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_youtube_transcript(video_id):
    """
    Retrieve YouTube video transcript
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Combine transcript texts, limit to prevent overwhelming the model
        full_transcript = ' '.join([entry['text'] for entry in transcript])
        return full_transcript[:10000]  # Limit to first 10000 characters
    except Exception as e:
        st.error(f"Error retrieving transcript: {e}")
        return None

def main():
    # Initialize Groq LLM
    try:
        llm = ChatGroq(
            groq_api_key=groqkey, 
            model_name="llama-3.1-70b-versatile", 
            temperature=0.7
        )

        # Prompt Template
        prompt_template = """
        Provide a comprehensive and clear summary of the following YouTube video transcript. 
        Capture the main points, key insights, and overall message of the video. 
        Ensure the summary is concise, informative, and approximately 300-350 words.

        Transcript:{text}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

        # YouTube URL input
        youtube_url = st.text_input(
            "Enter YouTube Video URL", 
            placeholder="https://www.youtube.com/watch?v=example"
        )
        
        # Summarize button
        if st.button("Generate Summary"):
            # Validate URL input
            if not youtube_url:
                st.warning("Please enter a YouTube URL")
                return
            
            # Extract video ID
            video_id = extract_youtube_video_id(youtube_url)
            if not video_id:
                st.error("Invalid YouTube URL. Please check the link.")
                return
            
            # Show loading spinner
            with st.spinner('Extracting transcript and generating summary...'):
                try:
                    # Retrieve transcript
                    transcript = get_youtube_transcript(video_id)
                    
                    if not transcript:
                        st.error("Could not retrieve video transcript. The video might not have captions.")
                        return
                    
                    # Create a Document for Langchain
                    docs = [Document(page_content=transcript)]
                    
                    # Summarization Chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary = chain.run(docs)
                    
                    # Display Summary
                    st.success("Video Summary:")
                    st.write(output_summary)
                
                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")

    except Exception as init_error:
        st.error(f"Failed to initialize the application: {init_error}")

if __name__ == "__main__":
    main()