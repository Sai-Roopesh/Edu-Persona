# generative_models.py
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from dotenv import load_dotenv

load_dotenv()
project_id = "gemini-practice-sai"
vertexai.init(project=project_id, location="us-central1")


def prompt(model_name, prompt_text, response_schema=None):
    model = GenerativeModel(model_name)
    generation_config = GenerationConfig(
        response_mime_type="application/json",
        response_schema=response_schema
    ) if response_schema else GenerationConfig(response_mime_type="application/json")
    response = model.generate_content(
        prompt_text, generation_config=generation_config)
    return response.text


def generate_quiz(model_name, text, number_of_questions, response_schema):
    prompt_text = f"Generate {number_of_questions} quiz questions for the given context: {text} with options and correct answers from the options generated. The question should start from being easy then medium and later hard difficulty."
    return prompt(model_name, prompt_text, response_schema)


def simplify_content(model_name, content, context, response_schema):
    prompt_text = f"Simplify the following content for better understanding:\n\nContent: {content}\n\nContext: {context}"
    return prompt(model_name, prompt_text, response_schema)
