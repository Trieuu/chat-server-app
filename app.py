from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import requests
import json
import logging

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load FAISS + dữ liệu
model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = faiss.read_index("faiss_index.index")

with open("product_data.json", "r", encoding="utf-8") as f:
    product_data = json.load(f)
with open("product_texts.json", "r", encoding="utf-8") as f:
    product_texts = json.load(f)

def semantic_search(query, top_k=3):
    embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(embedding, top_k)
    results = [product_data[i] for i in indices[0]]
    return results

def user_prompt_handler(user_message: str, k: int = 1) -> str:
    
    # Thay vì dùng regex, dùng semantic search
    retrieved_products = semantic_search(user_message, k)
    context_text = "\n".join([json.dumps(p, ensure_ascii=False) for p in retrieved_products])

    prompt_with_context = f"""
You are an friendly assistant that answers user questions based strictly on the provided data.

[Shop Data]:
{context_text}

[User Question]:
"{user_message}"

[Instructions]:
1. Only use facts and wording that appear exactly in the [Shop Data].
2. Do not invent or assume anything not shown in the data.
3. Do not add extra punctuation, currency symbols (like "$"), or any formatting not present in the original data (like redundancy punctuation).
4. If the information is not found in the [Shop Data], respond exactly with: Data not found.
5. Rephrase the response naturally, as if you are talking to a customer.
6. List product names clearly, using '-' if there are multiple items 
7. Make SURE the products AVAILABLE in order to be printed, and make sure not to print the stock of the product.
8. Do NOT use JSON or code format in your response. 
9. If the you can add some description about the products (like price, description, etc), but make sure not includ the stock
    """

    
    return prompt_with_context

# Load environment variables from .env file
load_dotenv()

# Check if API key is set
api_key = os.getenv('OPENROUTER_API_KEY_LLMA')
if not api_key:
    print("Error: OPENROUTER_API_KEY environment variable not set")
    exit(1)
    
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "deepseek/deepseek-r1-distill-llama-70b:free",
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt_handler(user_message, 3)
                    }
                ],
            })
        )

        logger.info(f"OpenRouter response: {response.status_code} - {response.text}")
        if response.status_code == 200:
            try:
                data = response.json()
                return jsonify({"response": data['choices'][0]['message']['content']})
            except (KeyError, ValueError) as e:
                return jsonify({"error": f"Error parsing response: {e}"}), 500
        else:
            return jsonify({"error": f"API Error: {response.status_code} - {response.text}"}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)