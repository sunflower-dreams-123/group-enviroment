import torch
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Tuple
import uvicorn
from fastapi.templating import Jinja2Templates
import logging
from datetime import datetime
# Set the templates folder for website appearance
templates = Jinja2Templates(directory="templates")
# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("all-mpnet-base-v2-finetuned-NER", local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained("all-mpnet-base-v2-finetuned-NER", local_files_only=True)
 
# Initialize the FastAPI app
app = FastAPI()



# Configure logging
logging.basicConfig(filename='predictions.log', level=logging.INFO)

#Resposnse to show basic text input box
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("ner_template.html", {"request": request, "results": None})
 
#response when text is posted from website calling process text and logging
@app.post("/", response_class=HTMLResponse)
async def post_form(request: Request, text: str = Form(...)):
    results = process_text(text)
    # Log the input and output
    log_interaction(text, results)
    #logging.info(f"{datetime.now()} - Input: {results[0]}, Prediction: {results[1]}")
    return templates.TemplateResponse("ner_template.html", {"request": request, "results": results})
 
# Processes text using the model and tokenizer loaded
def process_text(text: str) -> List[Tuple[str, str]]:
    # Tokenize the input text and prepare input IDs
    inputs = tokenizer.encode(text, return_tensors="pt")
 
    # Predicting the tokens (words) classification
    with torch.no_grad():
        outputs = model(inputs)[0]
 
    # Decode predictions
    predictions = outputs.argmax(dim=2)
    label_mapping = {"LABEL_0": 'Others',"LABEL_1": 'Abbreviations',"LABEL_2": 'Longform'}
    # Convert ids to tokens and predictions to labels
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]
    converted_list = [label_mapping[item] for item in labels]
    # Pair tokens with their labels
    return list(zip(tokens[1:-1], converted_list[1:-1]))

# Records timestamp, user input, and results for each interaction
def log_interaction(user_input: str, results: List[Tuple[str, str]]):
    timestamp = datetime.now().isoformat()
    results_str = ", ".join([f"({word}, {entity})" for word, entity in results])
    log_entry = f"{timestamp} - Input: {user_input} - Predictions: {results_str}"
    logging.info(log_entry)

#if __name__ == "__main__":
#   uvicorn.run(app, port=80)