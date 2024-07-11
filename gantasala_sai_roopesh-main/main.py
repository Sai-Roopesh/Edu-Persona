import streamlit as st
from pdfminer.high_level import extract_text
import json
from dotenv import load_dotenv
import os
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Load environment variables
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"credentials.json"
project_id = "gemini-practice-sai"
vertexai.init(project=project_id, location="us-central1")

quiz_response_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question-number": {"type": "number"},
            "question": {"type": "string"},
            "options": {
                "type": "array",
                "items": {"type": "string"}
            },
            "answer": {"type": "string"},
            "difficulty": {"type": "string"}
        },
        "required": ["question-number", "question", "options", "answer", "difficulty"]
    }
}


def extract_text_from_pdf(pdf_file):
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    text = extract_text("temp.pdf")
    os.remove("temp.pdf")
    return text


def interact_with_gemini(model_id, prompt_text):
    model_instance = GenerativeModel(model_id)
    response = model_instance.generate_content(
        prompt_text,
        generation_config=GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048,
            top_p=0.8,
            top_k=40,
            response_mime_type="application/json",
            response_schema=quiz_response_schema
        ),
    )

    print("Raw response from Gemini:", response.text)  # Debug print

    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return response.text


def generate_quiz(content, num_questions=5):

    prompt = f"""Generate a quiz with {num_questions} questions based on the following content:

    {content}

    For each question, provide:
    1. The question text
    2. Four multiple-choice options (A, B, C, D)
    3. The correct answer (A, B, C, or D)

    the answer shouls follow the response_schema:

   

    """
    model_id = 'gemini-1.5-pro-001'
    response = interact_with_gemini(model_id, prompt)

    if isinstance(response, list):
        return response
    else:
        st.error("Failed to generate quiz. Please try again.")
        return []


def generate_simplified_content(content, incorrect_questions):
    prompt = f"""Based on the following content and the questions the user answered incorrectly, 
    provide a simplified explanation of the key concepts related to these questions:

    Content: {content}

    Incorrect questions:
    {incorrect_questions}

    Please provide a concise, easy-to-understand explanation of the relevant concepts.
    """
    model_id = 'gemini-1.5-pro-001'
    response = interact_with_gemini(model_id, prompt)
    return response


# Streamlit app
st.title("PDF Learning Assistant")

# Initialize the page in session state if it doesn't exist
if 'page' not in st.session_state:
    st.session_state['page'] = "PDF Upload"

# Sidebar for navigation
page = st.sidebar.radio(
    "Choose a feature",
    ["PDF Upload", "Generate Quiz", "Take Quiz"],
    key="sidebar",
    index=["PDF Upload", "Generate Quiz", "Take Quiz"].index(
        st.session_state['page'])
)

# Update the page in session state
st.session_state['page'] = page

if page == "PDF Upload":
    st.header("PDF Upload")
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")
    if pdf_file:
        pdf_text = extract_text_from_pdf(pdf_file)
        st.session_state['pdf_text'] = pdf_text
        st.success("PDF uploaded and processed successfully!")

        # Add navigation button to Generate Quiz
        if st.button("Go to Generate Quiz", key="go_to_generate_quiz"):
            st.session_state['page'] = "Generate Quiz"


elif page == "Generate Quiz":
    st.header("Generate Quiz from PDF Content")

    if 'pdf_text' not in st.session_state:
        st.warning(
            "No PDF uploaded yet. Please upload a PDF in the PDF Upload section first.")
    else:
        pdf_text = st.session_state['pdf_text']
        st.success("PDF content loaded. You can now generate a quiz.")

        num_questions = st.number_input(
            "Number of questions", min_value=1, max_value=10, value=5)
        if st.button("Generate Quiz", key="generate_quiz_button"):
            quiz_data = generate_quiz(pdf_text, num_questions)
            st.session_state['quiz_data'] = quiz_data
            if quiz_data:
                st.success(
                    "Quiz generated successfully! Go to 'Take Quiz' to start.")

                # Add navigation button to Take Quiz
                if st.button("Go to Take Quiz", key="go_to_take_quiz"):
                    st.session_state['page'] = "Take Quiz"


elif page == "Take Quiz":
    st.header("Take Quiz")

    if 'quiz_data' in st.session_state:
        quiz_data = st.session_state['quiz_data']
        user_answers = []

        for idx, q in enumerate(quiz_data):
            st.subheader(f"Question {idx + 1}: {q['question']}")
            options = q['options']
            user_answer = st.radio("Choose an answer:",
                                   options, key=f"q_{idx}")
            user_answers.append({
                "question": q['question'],
                "user_answer": user_answer,
                "correct_answer": q['correct_answer']
            })

        if st.button('Submit Answers', key="submit_answers"):
            score = sum(
                1 for ua in user_answers if ua['user_answer'][0] == ua['correct_answer'])
            st.write(f"You scored {score} out of {len(quiz_data)}!")

            incorrect_questions = []
            for ua in user_answers:
                st.write(f"Q: {ua['question']}")
                st.write(f"Your answer: {ua['user_answer']}")
                st.write(f"Correct answer: {ua['correct_answer']}")
                st.write("---")
                if ua['user_answer'][0] != ua['correct_answer']:
                    incorrect_questions.append(ua['question'])

            if incorrect_questions:

                simplified_content = generate_simplified_content(
                    st.session_state['pdf_text'], "\n".join(incorrect_questions))
                st.subheader(
                    "Simplified Content for Incorrectly Answered Questions")
                st.write(simplified_content)

            # Add navigation button to return to PDF Upload
            if st.button("Upload Another PDF", key="upload_another_pdf"):
                st.session_state['page'] = "PDF Upload"
    else:
        st.warning("No quiz available. Please generate a quiz first.")
