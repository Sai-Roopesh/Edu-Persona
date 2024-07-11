# streamlit_app.py
import streamlit as st
import json

# Load the generated quiz from JSON file


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


quiz_data = load_json('generated_quizzes.json')

st.title("Quiz App")

user_answers = []
for idx, q in enumerate(quiz_data):
    st.subheader(f"Question {q['question-number']}: {q['question']}")
    options = q['options']
    user_answer = st.radio("Choose an answer:", options, key=idx)
    user_answers.append({
        "question-number": q['question-number'],
        "question": q['question'],
        "answer": user_answer
    })

if st.button('Submit Answers'):
    st.write("Answers submitted successfully!")
    with open('user_answers.json', 'w') as f:
        json.dump(user_answers, f, indent=4)
