# pipeline.py

from utils import pdf2text, load_json, save_json
from generative_models import generate_quiz, simplify_content
from validation import validate_user_answers
import json
# Configuration
pdf_filepath = r"notes/SE-M1.pdf"
number_of_quiz_questions = 5
quiz_response_schema_file = 'quiz_generator_response-schema.json'
simplify_response_schema_file = 'content_simplification_response_schema.json'

# Load schemas
quiz_response_schema = load_json(quiz_response_schema_file)
simplify_response_schema = load_json(simplify_response_schema_file)

# Extract text from PDF
pdf_content = pdf2text(pdf_filepath)

# Generate quiz
quizzes = generate_quiz('gemini-1.5-pro-001', pdf_content,
                        number_of_quiz_questions, quiz_response_schema)
save_json(json.loads(quizzes), 'generated_quizzes.json')

# User answers validation
validate_user_answers('generated_quizzes.json',
                      'user_answers.json', 'validation_results.json')

# Load validation results
validation_results = load_json('validation_results.json')

# Simplify incorrect answers
simplified_contents = []
for incorrect in validation_results:
    if not incorrect['is_correct']:
        question_text = incorrect['question']
        simplified_content = simplify_content(
            'gemini-1.5-pro-001', pdf_content, question_text, simplify_response_schema)
        incorrect['simplified_content'] = simplified_content
        simplified_contents.append(incorrect)
save_json(simplified_contents, 'simplified_contents.json')
