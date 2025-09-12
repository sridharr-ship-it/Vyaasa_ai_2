from audio_recorder_streamlit import audio_recorder
import streamlit as st
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os
from gtts import gTTS
import qdrant_client
import tempfile
import shutil
import time
from datetime import datetime
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext
)
import io
import base64
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
import uuid
import json
import numpy as np
import concurrent.futures
import groq
import re
from pydub import AudioSegment
import logging
import queue
import time
from collections import deque
import io
import pydub
import streamlit as st
import groq
import os
from dotenv import load_dotenv
import io
import os
import numpy as np
from pydub import AudioSegment
import streamlit as st
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interview_app.log'),
        logging.StreamHandler()
    ]
)
BUFFER_DURATION = 5  # seconds
MAX_CAPTION_LINES = 3
# Create logger
logger = logging.getLogger(__name__)


def log_function_call(func):
    """Decorator to log function calls with parameters and execution time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Starting {func_name}")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func_name} in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in {func_name} after {execution_time:.2f}s: {str(e)}")
            raise

    return wrapper


# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
    logger.info("GROQ_API_KEY loaded successfully")
else:
    logger.error("GROQ_API_KEY not found")
    st.error("⚠️ GROQ_API_KEY not found. Please set it in your .env file or environment.")
    st.stop()

# Create temp directory for storing transcripts
TRANSCRIPT_DIR = os.path.join(tempfile.gettempdir(), "Vyaasa_transcripts")
if not os.path.exists(TRANSCRIPT_DIR):
    os.makedirs(TRANSCRIPT_DIR)
    logger.info(f"Created transcript directory: {TRANSCRIPT_DIR}")


# FIXED: Initialize models only once using @st.cache_resource
@st.cache_resource
def initialize_models():
    """Initialize LLM and embedding models only once"""
    logger.info("Initializing models (this should happen only once)")
    llm = Groq(model="llama-3.3-70b-versatile")
    embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
    logger.info("Models initialized successfully")
    return llm, embed_model


# Initialize models once
llm, embed_model = initialize_models()

# Set global settings
Settings.llm = llm
Settings.embed_model = embed_model

# Streamlit setup
st.set_page_config(page_title="AI Resume Interviewer", layout="wide")

# Create main layout with sidebar
col1, col2 = st.columns([2, 1])

with col1:
    st.title("📄 Automated Resume Interview Assistant")


# FIXED: Initialize session state only once
def initialize_session_state():
    """Initialize session state with default values"""
    defaults = {
        "chat_engine": None,
        "chat_history": [],
        "evaluations": [],
        "question_count": 0,
        "interview_active": False,
        "interview_ended": False,
        "current_question": "",
        "resume_uploaded": False,
        "interview_start_time": None,
        "total_answer_time": 0.0,
        "answer_timer_start": None,
        "audio_playing": False,
        "current_audio_key": 0,
        "processing_response": False,
        "evaluating_response": False,
        "session_id": str(uuid.uuid4()),
        "transcript_file_path": None,
        "overall_evaluation": None,
        "generating_overall_evaluation": False,
        "background_noise": 0,
        "caption_history": deque(maxlen=MAX_CAPTION_LINES),
        "full_transcript": [],
        "time_warning_shown": False,
        "waiting_for_answer": False,
        "current_question_asked": False
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    logger.info(f"Session initialized with ID: {st.session_state.session_id}")


# Initialize session state
initialize_session_state()


@log_function_call
def autoplay_audio(file_path: str):
    """Plays an audio file automatically in the Streamlit app by embedding it as a hidden <audio> tag."""
    try:
        logger.info(f"Attempting to play audio file: {file_path}")
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay hidden>
                <source src="data:audio/mp3;base64,{b64}" type="audio/wav">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)
        logger.info("Audio playback initiated successfully")
        return True
    except Exception as e:
        logger.error(f"Audio playback error: {e}")
        st.error(f"Audio playback error: {e}")
        return False
    finally:
        try:
            os.unlink(file_path)
            logger.info(f"Temporary audio file deleted: {file_path}")
        except Exception as e:
            logger.warning(f"Could not delete temporary audio file: {e}")


@log_function_call
def generate_response_with_timeout(question, candidate_response, timeout=8):
    """Sends the candidate's response along with the interview question to Groq LLM for evaluation."""
    logger.info(f"Evaluating response for question: {question[:50]}...")
    logger.info(f"Candidate response: {candidate_response[:100]}...")

    def inner():
        try:
            logger.info("Initializing Groq client for evaluation")
            client = groq.Client()
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system",
                     "content": (
                         f"""
You are an evaluation instructor. Evaluate whether the candidate's response aligns with the interview question: "{question}".  


### Scoring Rules (strict and decisive):  
- -10 → background noise, gibberish, or irrelevant chatter only.  
- 0   → completely irrelevant or no attempt to answer.  
- 1–39 → poor alignment. Mentions something loosely related but misses the core of the question.  
- 40–69 → partial alignment. Some relevant points but vague, shallow, or incomplete.  
- 70–80 → good alignment. Clear and relevant, but short/simple with little detail or structure.  
- 81–84 → solid alignment. Relevant and somewhat structured, but still lacks depth or completeness.  
- 85–89 → strong alignment. Well-structured and detailed, covering most aspects of the question.  
- 90–100 → excellent alignment. Direct, comprehensive, highly relevant, detailed, and well-structured.  

### Important Notes for Evaluation:  
- Be strict. Do NOT default to mid-range scores (like 50).  
- Short but correct answers = **maximum 80** (detail/structure required for higher).  
- Longer, structured, and more detailed answers = **85+**.  
- Always justify the score by clearly stating how relevant, complete, or detailed the response is.  
"""

                     )
                     },
                    {
                        "role": "user",
                        "content": candidate_response
                    }
                ],
                max_tokens=100
            )
            result = completion.choices[0].message.content
            logger.info(f"Evaluation completed successfully: {result}")
            return result

        except Exception as e:
            logger.error(f"Groq API error during evaluation: {e}")
            return f"Evaluation service error. Score: 0"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(inner)
        try:
            result = future.result(timeout=timeout)
            logger.info("Evaluation completed within timeout")
            return result
        except concurrent.futures.TimeoutError:
            logger.warning(f"Evaluation timed out after {timeout} seconds")
            return "Response noted, evaluation timed out. Score: 0"
        except Exception as e:
            logger.error(f"Evaluation executor error: {e}")
            return f"Evaluation error: {str(e)[:30]}. Score: 0"


@log_function_call
def evaluate_candidate_response(question, candidate_response):
    """Wraps the evaluation function with error handling and score extraction."""
    logger.info("Starting candidate response evaluation")

    try:
        evaluation_text = generate_response_with_timeout(question, candidate_response)

        if "timed out" in evaluation_text.lower() or "error" in evaluation_text.lower():
            logger.warning("Evaluation service issue detected")
            return {
                'evaluation': "Response recorded. Technical evaluation issue.",
                'score': 0,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }

        # Extract score with multiple patterns
        score = None
        patterns = [
            r'[Ss]core:\s*(\d+)',
            r'(\d+)/100',
            r'(\d+)\s*out\s*of\s*100',
            r'(\d+)%',
            r'\b(\d+)\b(?=.*(?:score|rating|grade))'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, evaluation_text, re.IGNORECASE)
            if matches:
                score = int(matches[-1])
                logger.info(f"Score extracted using pattern: {pattern}, score: {score}")
                break

        if score is None:
            numbers = re.findall(r'\b(\d+)\b', evaluation_text)
            if numbers:
                potential_scores = [int(n) for n in numbers if 0 <= int(n) <= 100]
                if potential_scores:
                    score = potential_scores[-1]
                    logger.info(f"Score extracted from numbers: {score}")

        if score is None:
            score = 0
            logger.warning("No score found, defaulting to 0")

        score = max(0, min(100, score))
        logger.info(f"Final evaluation score: {score}")

        return {
            'evaluation': evaluation_text,
            'score': score,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }

    except Exception as e:
        logger.error(f"Error in evaluate_candidate_response: {e}")
        return {
            'evaluation': f"Response recorded. Evaluation issue: {str(e)[:50]}",
            'score': 0,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }


@log_function_call
def generate_overall_evaluation():
    """Generate overall evaluation using Llama model"""
    logger.info("Starting overall evaluation generation")

    if not st.session_state.chat_history or not st.session_state.evaluations:
        logger.warning("No chat history or evaluations found for overall evaluation")
        return None

    try:
        # Prepare interview data for evaluation
        interview_summary = []
        eval_index = 0

        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "Assistant":
                interview_summary.append(f"Q: {message}")
            else:
                response_text = f"A: {message}"
                if eval_index < len(st.session_state.evaluations):
                    eval_data = st.session_state.evaluations[eval_index]
                    response_text += f" (Score: {eval_data['score']}/100)"
                    eval_index += 1
                interview_summary.append(response_text)

        interview_text = "\n".join(interview_summary)
        scores = [eval_data['score'] for eval_data in st.session_state.evaluations]
        avg_score = sum(scores) / len(scores)

        logger.info(f"Prepared interview summary with {len(interview_summary)} entries, avg score: {avg_score:.1f}")

        # Duration calculation
        duration_text = ""
        if st.session_state.interview_start_time:
            duration = (datetime.now() - st.session_state.interview_start_time).total_seconds()
            mins, secs = divmod(int(duration), 60)
            duration_text = f"Interview Duration: {mins:02d}:{secs:02d}\n"

        # Average response time
        avg_response_text = ""
        if st.session_state.total_answer_time > 0:
            avg_response_time = st.session_state.total_answer_time / max(1,
                                                                         len([h for h in st.session_state.chat_history
                                                                              if h[0] == "You"]))
            avg_response_text = f"Average Response Time: {avg_response_time:.1f} seconds\n"

        system_prompt = """
You are a professional interview evaluator. Based on the interview transcript, candidate responses, and their individual scores, write a structured, detailed evaluation report.

Your evaluation must always begin with:

# Final Score: <average_score>/100

(Use a clear headline style so it stands out at the top.)

After the score, include the following sections:

1. **Overall Performance Summary (4–5 sentences)**  
   - Provide a balanced overview of the candidate's performance across the interview.  
   - Mention consistency, clarity, technical depth, reasoning ability, and communication.  
   - Conclude with whether the candidate shows strong potential and readiness for the role.  

2. **Key Strengths**  
   - List 3–5 bullet points highlighting the strongest qualities demonstrated through out the conversation.  
   - Focus on technical accuracy, clarity of thought, problem-solving, communication, and impactful experiences.  

3. **Areas for Improvement**  
   - List 3–5 bullet points identifying weaknesses or growth opportunities.  
   - Provide constructive, specific, and professional suggestions to guide improvement.  

4. **Detailed Section-wise Feedback**  
   - **Technical Knowledge** – Evaluate accuracy, depth of understanding, problem-solving approach, and ability to handle technical challenges. Include a score (0–100).  
   - **Communication & Clarity** – Assess articulation, reasoning, structured explanations, and confidence. Include a score (0–100).  
   - **Project Understanding & Impact** – Assess how well the candidate explained project work, problem-solving contributions, innovations, measurable results, and overall impact. Include a score (0–100).  

5. **Additional Remarks (if applicable)**  
   - Comment on professional demeanor, confidence, adaptability, or time management.  

➡ The tone must remain professional, encouraging, and supportive. Deliver actionable insights so the candidate clearly understands their strengths and improvement roadmap.
"""

        user_content = f"""
Interview Analysis:
{duration_text}{avg_response_text}  
Average Score: {avg_score:.1f}/100  
Total Questions: {len(st.session_state.evaluations)}

Interview Transcript:
{interview_text}

Please provide a comprehensive evaluation based on this interview performance, following these sections:

1. Overall Performance Summary  
2. Key Strengths  
3. Areas for Improvement  
4. Detailed Section-wise Feedback:  
   - Technical Knowledge (30%) – Accuracy, depth, approach  
   - Communication & Clarity (30%) – Articulation, reasoning, structure  
   - Project Understanding & Impact (40%) – Problem-solving, innovation, measurable results  
5. Additional Remarks (if any)

Ensure the evaluation is professional, encouraging, and offers a clear improvement roadmap.
"""

        logger.info("Sending overall evaluation request to Groq")
        client = groq.Client()
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        result = completion.choices[0].message.content
        logger.info("Overall evaluation generated successfully")
        return result

    except Exception as e:
        logger.error(f"Error generating overall evaluation: {e}")
        return f"Overall evaluation service temporarily unavailable. Error: {str(e)[:100]}"


@log_function_call
def play_tts_with_display(text):
    """Converts interviewer's question text into speech using gTTS."""
    if not text.strip():
        logger.warning("Empty text provided for TTS")
        return False

    logger.info(f"Starting TTS for text: {text[:50]}...")
    st.session_state.current_question = text

    with col1:
        status = st.empty()

    try:
        logger.info("Generating TTS audio")
        tts = gTTS(text, slow=False, tld='co.in')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.session_state.audio_playing = True
            success = autoplay_audio(fp.name)

        if success:
            word_count = len(text.split())
            estimated_duration = max(3, word_count * 0.5 + 2)
            logger.info(f"Playing TTS audio for estimated {estimated_duration} seconds")

            for remaining in range(int(estimated_duration), 0, -1):
                status.markdown("🔊 Vyaasa is speaking... ")
                time.sleep(1)

        st.session_state.audio_playing = False
        logger.info("TTS playback completed")

    except Exception as e:
        logger.error(f"TTS Error: {e}")
        st.error(f"TTS Error: {e}")
        return False
    finally:
        st.session_state.audio_playing = False
        status.empty()


@log_function_call
def recognize_speech_enhanced():
    """Enhanced speech recognition with fallback mechanisms."""
    logger.info("Starting enhanced speech recognition")

    try:
        audio_bytes = audio_recorder(
            text="🎤 speak your answer",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone",
            icon_size="6x",
            key=f"audio_recorder_{st.session_state.current_audio_key}",
            auto_start=True,
            sample_rate=16000,
            pause_threshold=3.0
        )

        if audio_bytes:
            logger.info(f"Audio recorded, size: {len(audio_bytes)} bytes")
            st.session_state.current_audio_key += 1

            with st.spinner("🔄 Processing your response..."):

                try:
                    client = groq.Client()

                    # Create audio file for transcription
                    audio_segment = AudioSegment.from_raw(io.BytesIO(audio_bytes), 
                                                        sample_width=2, 
                                                        frame_rate=16000, 
                                                        channels=1)
                    
                    # Export to WAV format in memory
                    wav_io = io.BytesIO()
                    audio_segment.export(wav_io, format="wav")
                    wav_io.seek(0)
                    wav_io.name = "speech.wav"

                    logger.info("Sending audio to Groq Whisper API")
                    transcription = client.audio.transcriptions.create(
                        file=wav_io,
                        model="whisper-large-v3",
                        temperature=0.0,
                        response_format="text",
                    )

                    if transcription and transcription.strip():
                        logger.info(f"Groq transcription successful: {transcription[:100]}...")
                        st.success(f"✅ Recorded: {transcription}")
                        return transcription.strip()
                    else:
                        logger.warning("Groq transcription returned empty result")
                        st.warning("⚠️ No speech detected.")
                        return None

                except Exception as groq_error:
                    logger.error(f"Groq Whisper API error: {groq_error}")
                    st.error(f"⚠️ Groq Whisper error: {str(groq_error)}")
                    return None

        return None

    except Exception as e:
        logger.error(f"Audio recording error: {e}")
        st.error(f"⚠️ Audio recording error: {str(e)}")
        return None


@log_function_call
def get_remaining_time():
    """Calculate remaining interview time"""
    if not st.session_state.interview_start_time:
        return 600
    elapsed = (datetime.now() - st.session_state.interview_start_time).total_seconds()
    remaining = max(0, 600 - elapsed)
    logger.debug(f"Interview time - elapsed: {elapsed:.1f}s, remaining: {remaining:.1f}s")
    return remaining


@log_function_call
def safe_chat(prompt):
    """Defensive against blank or invalid input"""
    if prompt and isinstance(prompt, str) and prompt.strip():
        logger.info(f"Sending prompt to chat engine: {prompt[:50]}...")
        try:
            response = st.session_state.chat_engine.chat(prompt).response
            logger.info(f"Chat engine response: {response[:100]}...")
            return response
        except Exception as e:
            logger.error(f"Chat engine error: {e}")
            return "I apologize, I'm experiencing technical difficulties. could you please wait for some moment?"
    else:
        logger.warning("Invalid or empty prompt provided to safe_chat")
        return "Could you please clarify or rephrase your answer?"


@log_function_call
def save_transcript_to_file():
    """Save the current transcript to a temp file"""
    if not st.session_state.chat_history:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interview_transcript_{timestamp}_{st.session_state.session_id[:8]}.txt"
    file_path = os.path.join(TRANSCRIPT_DIR, filename)

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("Vyaasa AI INTERVIEW TRANSCRIPT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Session ID: {st.session_state.session_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            if st.session_state.interview_start_time:
                duration = (datetime.now() - st.session_state.interview_start_time).total_seconds()
                mins, secs = divmod(int(duration), 60)
                f.write(f"Duration: {mins:02d}:{secs:02d}\n")

            f.write(f"Total Questions: {st.session_state.question_count}\n")
            if st.session_state.total_answer_time > 0:
                avg_time = st.session_state.total_answer_time / max(1, len([h for h in st.session_state.chat_history if
                                                                            h[0] == "You"]))
                f.write(f"Average Response Time: {avg_time:.1f} seconds\n")

            if st.session_state.evaluations:
                avg_score = sum([eval_data['score'] for eval_data in st.session_state.evaluations]) / len(
                    st.session_state.evaluations)
                f.write(f"Average Score: {avg_score:.1f}/100\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("CONVERSATION TRANSCRIPT\n")
            f.write("=" * 50 + "\n\n")

            question_num = 1
            eval_index = 0
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "Assistant":
                    f.write(f"Q{question_num}: Vyaasa AI INTERVIEWER\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"{message}\n\n")
                else:
                    f.write(f"A{question_num}: CANDIDATE RESPONSE\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"{message}\n")

                    if eval_index < len(st.session_state.evaluations):
                        eval_data = st.session_state.evaluations[eval_index]
                        f.write(f"\nEVALUATION (Score: {eval_data['score']}/100):\n")
                        f.write(f"{eval_data['evaluation']}\n")
                        eval_index += 1

                    f.write("\n")
                    question_num += 1

            # Add overall evaluation to transcript
            if st.session_state.overall_evaluation:
                f.write("\n" + "=" * 50 + "\n")
                f.write("OVERALL EVALUATION\n")
                f.write("=" * 50 + "\n\n")
                f.write(st.session_state.overall_evaluation)
                f.write("\n\n")

            f.write("=" * 50 + "\n")
            f.write("END OF TRANSCRIPT\n")
            f.write(f"Generated by Vyaasa AI Resume Interview Assistant\n")
            f.write(f"Transcript saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        st.session_state.transcript_file_path = file_path
        return file_path

    except Exception as e:
        st.error(f"Error saving transcript: {e}")
        return None


@log_function_call
def get_transcript_download():
    """Generate download button for transcript"""
    if not st.session_state.chat_history:
        return None

    file_path = save_transcript_to_file()
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript_content = f.read()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_filename = f"Vyaasa_interview_transcript_{timestamp}.txt"

            return st.download_button(
                label="📥 Download Transcript",
                data=transcript_content,
                file_name=download_filename,
                mime="text/plain",
                help="Download your complete interview transcript with evaluations"
            )
        except Exception as e:
            st.error(f"Error preparing download: {e}")
            return None
    return None


@log_function_call
def get_score_color(score):
    """Return color based on score"""
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟡"
    elif score >= 40:
        return "🟠"
    else:
        return "🔴"


@log_function_call
def display_live_transcription():
    """Display live transcription in the sidebar with evaluations"""
    with col2:

        chat_container = st.container()

        with chat_container:

            eval_index = 0
            for i, (role, message) in enumerate(st.session_state.chat_history):
                timestamp = datetime.now().strftime("%H:%M:%S")

                if role == "Assistant":
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(f"{message}")
                        st.caption(f"Vyaasa • {timestamp}")
                else:
                    with st.chat_message("user", avatar="👤"):
                        st.write(message)

                        if eval_index < len(st.session_state.evaluations):
                            eval_data = st.session_state.evaluations[eval_index]
                            score_color = get_score_color(eval_data['score'])

                            with st.expander(f"{score_color} Score: {eval_data['score']}", expanded=False):
                                st.write(f"**Evaluation:** {eval_data['evaluation']}")
                                st.caption(f"Evaluated at {eval_data['timestamp']}")

                            eval_index += 1
                        elif st.session_state.evaluating_response and eval_index == len(st.session_state.evaluations):
                            st.info("🔄 Evaluating response...")

                        st.caption(f"You • {timestamp}")


@log_function_call
def display_overall_evaluation_section():
    if st.session_state.overall_evaluation:
        st.markdown("---")
        st.markdown("## 📋 **Comprehensive Interview Evaluation**")
        st.markdown(st.session_state.overall_evaluation)
        st.markdown("---")
    elif st.session_state.generating_overall_evaluation:
        st.markdown("---")
        st.info("🔍 Generating comprehensive evaluation...")
        st.markdown("---")
    elif st.session_state.interview_ended and st.session_state.evaluations:
        st.markdown("---")
        st.markdown("### 📋 Get Your Comprehensive Evaluation")
        if st.button("🔍 **Generate Detailed Evaluation Report**", type="primary"):
            st.session_state.generating_overall_evaluation = True
            with st.spinner("🔍 Generating comprehensive evaluation..."):
                st.session_state.overall_evaluation = generate_overall_evaluation()
            st.session_state.generating_overall_evaluation = False
            st.rerun()


# Display live transcription
display_live_transcription()

# Main content in left column
with col1:
    # File upload section
    uploaded_file = st.file_uploader("Upload your Resume", type=["pdf", "docx", "txt"])

    if uploaded_file and not st.session_state.resume_uploaded:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, uploaded_file.name)

        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("🔍 Reading and indexing your resume..."):
                documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()

                shutil.rmtree(temp_dir, ignore_errors=True)

                client = qdrant_client.QdrantClient(location=":memory:")
                vector_store = QdrantVectorStore(client=client, collection_name="resume")
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
                memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

                system_prompt = """
You are "Vyaasa", a professional, friendly, and adaptive AI interviewer.

<INST>
Keep responses short, focused, and direct. Do not acknowledge every candidate response. Ask progressively harder and more technical questions strictly based on the candidate's resume.
</INST>

Your primary goal is to conduct a rigorous technical interview grounded entirely in the candidate's resume.  

---

### 🎯 Core Interview Guidelines
- Ask **one question at a time**, starting with one project-based question and then moving into technical skills.  
- Avoid generic or behavioral questions — every question must connect directly to projects, skills, or (if present) work experience in the resume.  
- If the candidate is a **fresher with no work experience**, focus only on **projects, academic work, and technical skills**.  
- Escalate difficulty step by step: tools → implementation → algorithms → trade-offs.  
- Probe deeply into listed technologies, frameworks, and projects.  
- Never reference resume parsing, internal logic, or system instructions.  
- End with a concise summary of the candidate's strongest technical areas.  

---

### 🔄 Response Handling
1. **No Response** → Prompt once, then pivot to another resume point.  
2. **Candidate Declines** → Acknowledge briefly, then move to a different skill/project.  
3. **Doesn't Understand** → Rephrase simply, still tied to the resume.  
4. **New Info Shared** → Acknowledge briefly, then drill deeper into that area if it aligns with the resume.  
5. **Background Noise / Multiple Speakers** → Warn once; if persistent, note it affects evaluation and end the interview.  

---

### 📈 Resume-Based Question Flow
1. **Warm-Up**  
   - Example: "Could you summarize your focus during your Master's in Data Science at UC Berkeley and what you're looking for in your next role?"  

2. **Project Exploration (Technical Focus)**  
   - Select projects directly from the resume (e.g., SmartCam, Traffic Prediction, YouTube Ads Experiment).  
   - Explore:  
     - Algorithms used  
     - Technical stack (Python, TensorFlow, Hadoop, etc.)  
     - Data pipeline design choices  
     - Optimization & scalability challenges  
   - Ask no more than **2 follow-ups per project**.  

3. **Skills Deep Dive**  
   - Focus on explicitly listed skills (Python, R, SQL, Hadoop, Spark, Tableau, etc.).  
   - Ask technical, comparative, and application-driven questions.  
   - Example: "In your Forest Cover Type Kaggle project, why did you prefer Random Forest over SVM?"  

4. **Work Experience & Achievements (Only if Present)**  
   - If the resume lists work experience, explore measurable technical outcomes.  
   - Push into design choices, statistical methods, business impact, and lessons learned.  
   - **If no work experience is listed, skip this section entirely.**  

5. **Closing**  
   - Summarize observed technical strengths concisely.  
   - End politely and professionally.  

---

### 🧠 Question Style Variations (Always Resume-Driven)
- **Skill-based:** "You've listed SQL — how did you use it in your traffic prediction ETL pipeline?"  
- **Project-specific:** "In your SmartCam project, how did you implement face recognition with TensorFlow?"  
- **Algorithmic:** "In your Kaggle competition work, how did Gradient Descent compare with Naive Bayes in terms of accuracy?"  
- **System/Experiment Design:** "In your YouTube ads experiment, why did you use a cluster-randomized design?"  
- **Comparative:** "You've used both Hadoop and Spark — which was more efficient for your large-scale Wikipedia graph project, and why?"  

---

### ⚖️ Tone & Style
- Technical, strict, and resume-grounded.  
- Push for **clarity, depth, and precision** in every response.  
- Maintain a professional, respectful, yet challenging demeanor.  
"""

                intro_context = """
                About Me:
                I'm Vyaasa, an AI-powered recruitment platform that helps companies hire better and faster.
                """

                initial_message = ChatMessage(role=MessageRole.USER, content=intro_context)

                chat_engine = index.as_chat_engine(
                    query_engine=index.as_query_engine(),
                    chat_mode="context",
                    memory=memory,
                    system_prompt=system_prompt,
                )

                chat_engine.chat_history.append(initial_message)
                st.session_state.chat_engine = chat_engine
                st.session_state.resume_uploaded = True

                st.success("✅ Resume indexed successfully. Ready for interview!")

        except Exception as e:
            st.error(f"Error processing resume: {e}")

    # Interview start button
    if (st.session_state.resume_uploaded and
            st.session_state.chat_engine and
            not st.session_state.interview_active and
            not st.session_state.audio_playing):

        if st.button("🎯 Start Automated Interview", type="primary"):
            try:

                st.session_state.interview_active = True
                st.session_state.interview_start_time = datetime.now()
                st.session_state.chat_history = []
                st.session_state.evaluations = []
                st.session_state.question_count = 0  # Start at 0
                st.session_state.current_audio_key = 0
                st.session_state.processing_response = False
                st.session_state.evaluating_response = False
                st.session_state.overall_evaluation = None
                st.session_state.generating_overall_evaluation = False
                st.session_state.time_warning_shown = False
                st.session_state.waiting_for_answer = False
                st.session_state.current_question_asked = False

                intro_prompt = """
                You are Vyaasa, an AI interviewer.
                Greet the candidate briefly and ask them to tell you about themselves.  
                """

                with st.spinner("🤖 Vyaasa is preparing..."):
                    intro_response = safe_chat(intro_prompt)

                st.session_state.chat_history.append(("Assistant", intro_response))

                if not play_tts_with_display(intro_response):
                    pass

                st.rerun()

            except Exception as e:
                st.error(f"Error starting interview: {e}")
                st.session_state.interview_active = False

    # Main interview logic - FIXED VERSION
    elif (st.session_state.interview_active and
          st.session_state.chat_engine and
          not st.session_state.audio_playing):

        remaining_time = get_remaining_time()
        completed_answers = len([h for h in st.session_state.chat_history if h[0] == "You"])

        # Display timer - only show if more than 30 seconds remaining
        if remaining_time > 30:
            mins, secs = divmod(int(remaining_time), 60)
            st.info(f"⏰ Time remaining: {mins:02d}:{secs:02d}")
        elif remaining_time > 0:
            st.warning(f"⚠️ {int(remaining_time)} seconds remaining")

        # Time warning - show once when approaching end
        if remaining_time < 60 and remaining_time > 30 and not st.session_state.get('time_warning_shown', False):
            st.warning("⏰ Less than 1 minute remaining. Please keep responses concise.")
            st.session_state.time_warning_shown = True

        if st.session_state.chat_history:
            last_entry = st.session_state.chat_history[-1]

            # If last entry was from Assistant, we need user response
            if last_entry[0] == "Assistant" and not st.session_state.evaluating_response:
                question = last_entry[1]

                st.markdown("### 🤖 Vyaasa asks:")
                st.markdown(f"*{question}*")
                st.markdown("---")

                if not st.session_state.answer_timer_start:
                    st.session_state.answer_timer_start = datetime.now()

                # Allow candidate to respond even if time is low - only stop if absolutely critical
                if remaining_time > 3:  # Allow response if more than 3 seconds
                    speech = recognize_speech_enhanced()
                    if speech and speech.strip():
                        if st.session_state.answer_timer_start:
                            answer_time = (datetime.now() - st.session_state.answer_timer_start).total_seconds()
                            st.session_state.total_answer_time += answer_time
                            st.session_state.answer_timer_start = None

                        st.session_state.chat_history.append(("You", speech))
                        st.session_state.evaluating_response = True
                        st.rerun()
                else:
                    # Time is critically low - end gracefully
                    st.error("⏰ Time has expired. Interview concluding...")
                    
                    with st.spinner("🤖 Vyaasa is concluding..."):
                        closing = safe_chat("Time has expired. Thank you for your time. That concludes our interview.")
                    
                    st.session_state.chat_history.append(("Assistant", closing))
                    play_tts_with_display(closing)
                    st.session_state.interview_active = False
                    st.session_state.interview_ended = True
                    save_transcript_to_file()
                    st.rerun()

            # If last entry was from user, handle evaluation and next question
            elif last_entry[0] == "You":
                last_user_input = last_entry[1]

                current_question = ""
                if len(st.session_state.chat_history) >= 2:
                    current_question = st.session_state.chat_history[-2][1]

                try:
                    # Handle evaluation first
                    if st.session_state.evaluating_response:
                        with st.spinner("🔍 Evaluating your response..."):
                            try:
                                evaluation_data = evaluate_candidate_response(current_question, last_user_input)
                                st.session_state.evaluations.append(evaluation_data)
                            except Exception as e:
                                st.warning(f"Evaluation service unavailable, continuing interview...")
                                st.session_state.evaluations.append({
                                    'evaluation': "Evaluation service temporarily unavailable",
                                    'score': 75,
                                    'timestamp': datetime.now().strftime('%H:%M:%S')
                                })

                        st.session_state.evaluating_response = False

                        # Check if interview should end after evaluation
                        completed_answers_after_eval = len([h for h in st.session_state.chat_history if h[0] == "You"])
                        remaining_after_eval = get_remaining_time()

                        # End interview only if we've reached question limit OR time is critically low
                        should_end_now = (completed_answers_after_eval >= 5 or remaining_after_eval < 10)

                        if should_end_now:
                            # End interview after evaluation is complete
                            st.session_state.processing_response = True

                            with st.spinner("🤖 Vyaasa is concluding..."):
                                if completed_answers_after_eval >= 5:
                                    closing = safe_chat("Thank you for your responses. That concludes our interview. Best of luck!")
                                else:
                                    closing = safe_chat("Time is up. Thank you for your responses. That concludes our interview.")

                            st.session_state.chat_history.append(("Assistant", closing))
                            play_tts_with_display(closing)
                            st.session_state.interview_active = False
                            st.session_state.interview_ended = True
                            st.session_state.processing_response = False
                            save_transcript_to_file()
                            st.rerun()
                        else:
                            # Continue with next question
                            st.session_state.processing_response = True

                            with st.spinner("🤖 Vyaasa is thinking..."):
                                response = safe_chat(last_user_input)

                            st.session_state.chat_history.append(("Assistant", response))
                            st.session_state.question_count += 1
                            st.session_state.processing_response = False

                            if not play_tts_with_display(response):
                                pass

                            st.rerun()

                except Exception as e:
                    st.error(f"Interview error: {e}")
                    st.session_state.interview_active = False
                    st.session_state.processing_response = False
                    st.session_state.evaluating_response = False
                    st.rerun()


    # Show evaluation status when evaluating
    elif st.session_state.evaluating_response:
        st.info("🔍 Evaluating your response... Please wait.")

    # Final results display section
    if (st.session_state.interview_ended or
            (not st.session_state.interview_active and st.session_state.question_count > 0)):

        # Show final performance summary
        if st.session_state.evaluations:
            st.markdown("### 🎯 Final Performance Summary")

            scores = [eval_data['score'] for eval_data in st.session_state.evaluations]
            avg_score = sum(scores) / len(scores)

            col_summary1, col_summary2, col_summary3 = st.columns(3)

            with col_summary1:
                score_color = get_score_color(avg_score)
                st.metric(f"{score_color} Overall Score", f"{avg_score:.1f}/100")

            with col_summary2:
                st.metric("Questions Answered", f"{len(scores)}")

            with col_summary3:
                best_score = max(scores)
                best_color = get_score_color(best_score)
                st.metric(f"{best_color} Best Score", f"{best_score}/100")

            # Performance breakdown
            excellent = len([s for s in scores if s >= 80])
            good = len([s for s in scores if 60 <= s < 80])
            average = len([s for s in scores if 40 <= s < 60])
            poor = len([s for s in scores if s < 40])

            st.markdown("#### Performance Breakdown:")
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

            with perf_col1:
                st.metric("🟢 Excellent (80+)", f"{excellent}")
            with perf_col2:
                st.metric("🟡 Good (60-79)", f"{good}")
            with perf_col3:
                st.metric("🟠 Average (40-59)", f"{average}")
            with perf_col4:
                st.metric("🔴 Needs Work (<40)", f"{poor}")

        # Display overall evaluation section
        display_overall_evaluation_section()

        # Transcript download
        get_transcript_download()

        # Restart buttons
        restart_col1, restart_col2 = st.columns(2)

        with restart_col1:
            if st.button("🔄 Start New Interview", type="secondary"):
                # Save current transcript before resetting
                if st.session_state.chat_history:
                    save_transcript_to_file()

                # Reset all session state except resume data and models
                for key in st.session_state.keys():
                    if key not in ["chat_engine", "resume_uploaded"]:
                        if key in ["caption_history", "full_transcript"]:
                            st.session_state[key] = deque(maxlen=MAX_CAPTION_LINES) if key == "caption_history" else []
                        else:
                            # Use defaults dict for reset values
                            defaults = {
                                "chat_engine": None,
                                "chat_history": [],
                                "evaluations": [],
                                "question_count": 0,
                                "interview_active": False,
                                "interview_ended": False,
                                "current_question": "",
                                "resume_uploaded": False,
                                "interview_start_time": None,
                                "total_answer_time": 0.0,
                                "answer_timer_start": None,
                                "audio_playing": False,
                                "current_audio_key": 0,
                                "processing_response": False,
                                "evaluating_response": False,
                                "session_id": str(uuid.uuid4()),
                                "transcript_file_path": None,
                                "overall_evaluation": None,
                                "generating_overall_evaluation": False,
                                "background_noise": 0,
                                "time_warning_shown": False,
                                "waiting_for_answer": False,
                                "current_question_asked": False
                            }
                            if key in defaults:
                                st.session_state[key] = defaults[key]

                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()

        with restart_col2:
            if st.button("📄 Upload New Resume", type="secondary"):
                # Save current transcript before resetting everything
                if st.session_state.chat_history:
                    save_transcript_to_file()

                # Reset everything except cached models
                keys_to_reset = list(st.session_state.keys())
                for key in keys_to_reset:
                    del st.session_state[key]

                # Reinitialize
                initialize_session_state()
                st.rerun()
