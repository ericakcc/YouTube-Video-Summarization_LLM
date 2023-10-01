import streamlit as st
from pytube import YouTube
from pathlib import Path
import shutil
import os
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import openai
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI


openai.api_key = os.environ["OPENAI_API_KEY"]


def save_video(url, video_filename):
    youtubeObject = YouTube(url)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download()
    except:
        print("An error has occureed")
    print("Download is completed successfully")

    return video_filename

def save_audio(url):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    file_name = base + '.mp3'
    try:
        os.rename(out_file, file_name)
    except WindowsError:
        os.remove(file_name)
        os.rename(out_file, file_name)
    audio_filename = Path(file_name).stem + '.mp3'
    video_filename = save_video(url, Path(file_name).stem + '.mp4')
    print(yt.title + "Has been successfully download!")
    return yt.title, audio_filename, video_filename

# load wihsper jax model
@st.cache_resource
def load_model():
    pipeline = FlaxWhisperPipline("openai/whisper-base")
    return pipeline

# get video transcription with whisper jax
def transcription(audio_file):
    model = load_model()
    outputs = model(audio_file, task="transcribe", return_timestamps=True)
    return outputs

# summarization with gpt-3.5-turbo
def summarization(video_transcript: str):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    prompt_template = '''
    Please analyze the following video transcript and produce a summary along with 5 key points. 

    Transcript: {video_transcript}

    ---

    Output:

    Summarization:
    

    5 Key Points:
    1. key_point_1:
    2. key_point_2:
    3. key_point_3:
    4. key_point_4:
    5. key_point_5:
    '''

    prompt = PromptTemplate(
        input_variables = ["video_transcript"],
        template = prompt_template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    results = chain.run(video_transcript=video_transcript)

    return results

st.set_page_config(layout="wide")

def main():
    
    st.markdown("<h2 style='text-align: center; color:green;'>Enter the clip URL👇</h2>", unsafe_allow_html=True)
    url =  st.text_input('Enter URL of YouTube video:')

    if url is not None:
        if st.button("Submit"):
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                st.subheader("Video Preview")
                video_title, audio_filename, video_filename = save_audio(url)
                st.video(video_filename)
            with col2:
                st.subheader("Transcript Below") 
                transcript_result = transcription(audio_filename)
                st.write(transcript_result['text'])
            with col3:
                st.subheader("Video Summarization") 
                transcript_text = transcript_result['text']
                result = summarization(transcript_text)
                st.write(result)


if __name__ == "__main__":
    main()