
import json

def validate_user_answers(quiz_file, user_answers_file, validation_results_file):
    quiz_data = load_json(quiz_file)
    user_answers = load_json(user_answers_file)

    results = []
    for question_data in quiz_data:
        question_number = question_data['question-number']
        question = question_data['question']
        correct_answer = question_data['answer']
        user_answer_data = next(
            (ua for ua in user_answers if ua['question-number'] == question_number), None)
        if user_answer_data:
            user_answer = user_answer_data['answer']
            result = {
                "question-number": question_number,
                "question": question,
                "correct_answer": correct_answer,
                "user_answer": user_answer,
                "is_correct": user_answer == correct_answer
            }
            results.append(result)

    with open(validation_results_file, 'w') as f:
        json.dump(results, f, indent=4)

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)