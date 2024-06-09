import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
from dotenv import load_dotenv
from streamlit_cookies_manager import EncryptedCookieManager

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
cookies = EncryptedCookieManager(
    prefix="ktosiek/streamlit-cookies-manager/",
    password=os.environ.get("COOKIES_PASSWORD", "My secret password"),
)
if not cookies.ready():
    # Wait for the component to load and send us current cookies.
    st.stop()

# get errors from cookies
errors = cookies.get("errors", []) # if no errors, return empty list

# display errors in the top left corner
st.markdown(f"""
    <style>
    .top-left {{
        position: fixed;
        top: 30px;
        left: 10px;
        z-index: 100;
    }}
    #err {{
        color: red;
    }}
    </style>
    <div class="top-left">
        <h4>Errors: <span id="err">{len(errors)}</span></h4>
    </div>
    """, unsafe_allow_html=True)

generated_image_url = 'https://veritusgroup.com/wp-content/uploads/2021/03/talking-big-2021-Mar29.jpg' # placeholder image for first message

def speech_to_text(user_audio_file_path):
    client = OpenAI(api_key=OPENAI_API_KEY)
    audio_file = open(user_audio_file_path, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text

def text_generation_ai(prompt): # image to text generation api
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [
        {"role": "system", "content": "Invent your own character as a chill language teacher through helpful conversation answer the questions and, correct any grammar mistakes if exists."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(model="gpt-3.5-turbo-0125", messages=messages)
    return response.choices[0].message.content  

def describe_image(image_url):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "What’s in this image? describe what is in the center and the backgroundd, what vibe the image has and what persons in the image looks like in only very few words."},
            {
            "type": "image_url",
            "image_url": {
                "url": image_url,
            },
            },
        ],
        }
    ],
    max_tokens=30,
    )
    return response.choices[0].message.content



def generate_image(prompt, image_description):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.images.generate(
        model="dall-e-2",
        prompt= "the image should show and visualize an appropriate image for the following chatgpt reply to add a visual experience: " + prompt + "the general image can contain the following style and vibe as a base: " + image_description,
        size="256x256",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    return image_url


def text_to_speech(ai_speech_file_path, api_response_text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.audio.speech.create(model="tts-1", voice="nova", input=api_response_text)
    response.stream_to_file(ai_speech_file_path)

st.title('Language Learning Game')

errors = []
# accesslocal variables from browser
# st.write(os.environ.get('OPENAI_API_KEY'))


"""
You can record your voice and await feedback from the AI to improve your language skills.
"""

audio_bytes = audio_recorder()
if audio_bytes:
    user_audio_file_path = 'audio.wav'
    with open(user_audio_file_path, 'wb') as f:
        f.write(audio_bytes)
    user_text = speech_to_text(user_audio_file_path)
    st.write(user_text) 
    # image_description = describe_image(generated_image_url)
    ai_response = text_generation_ai(user_text)
    # todo descibe what you see in teh last image and also feed into the model
    # generated_image_url = generate_image(ai_response, image_description)
    ai_speech_file_path = 'ai_speech.mp3'
    # text_to_speech(ai_speech_file_path, ai_response)
    st.write(ai_response)
    # st.audio(ai_speech_file_path)
    # st.image(generated_image_url)

"""
You are hearing an AI generated voice using OpenAI's TTS model.
"""