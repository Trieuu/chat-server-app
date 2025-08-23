from flask import Flask, request, jsonify
from flask_cors import CORS

import logging

from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import json

import requests
import json
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

EMBEDDING_MODEL = "gemini-embedding-001"
EMB_CACHE = "product_embeddings.npy"

load_dotenv()

# Check if API key is set
api_key_llm = os.getenv('OPENROUTER_API_KEY_LLMA')
api_key_emd = os.getenv('GOOGLE_API_KEY')
if not api_key_llm:
    print("Error: OPENROUTER_API_KEY environment variable not set")
    exit(1)

client = genai.Client(api_key=api_key_emd)

with open("shop_data.json", "r", encoding="utf-8") as f:
    products = json.load(f)
    
products = products['products']

def build_embedding_text(p: dict) -> str:
    # Lấy danh sách options nếu có
    options = p.get('flower_details', {}).get('options', [])
    
    if options:
        prices = [opt.get('price') for opt in options if isinstance(opt.get('price'), (int, float))]
        if prices:
            min_price = min(prices)
            max_price = max(prices)
            price_text = f"Price range: from {min_price} to {max_price} (optional)."
        else:
            price_text = ""
    else:
        price_text = ""

    # Tạo mô tả embedding
    return f"""
ID: {p['product_id']}
Name: {p['name']}
Description: {p.get('description', '')}
Type: {p.get('flower_details', {}).get('flower_type', '')}
Color: {', '.join(p.get('flower_details', {}).get('color', []))}
Occasion: {', '.join(p.get('flower_details', {}).get('occasion', []))}
Options: {p.get('flower_details', {}).get('options', [])}
{price_text}
""".strip()

# Chọn text để encode (tùy chỉnh theo nhu cầu)
texts = [
    build_embedding_text(p) for p in products
]

def embed_texts_google(texts, batch_size=1):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        resp = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        vecs = [np.array(e.values, dtype=np.float32) for e in resp.embeddings]
        all_vecs.extend(vecs)
    mat = np.vstack(all_vecs)
    # L2-normalize for fast cosine similarity
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return mat

if os.path.exists(EMB_CACHE):
    product_embs = np.load(EMB_CACHE)
else:
    product_embs = embed_texts_google(texts)
    np.save(EMB_CACHE, product_embs)

def semantic_search(query: str, top_k: int = 3):
    # embed query
    q_resp = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    q = np.array(q_resp.embeddings[0].values, dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-12)

    sims = cosine_similarity(q.reshape(1, -1), product_embs)[0]  # shape (N,)
    idx = np.argsort(-sims)[:top_k]
    indices = [texts[i].splitlines()[0].split()[1] for i in idx]
    return [product for product in products if product.get('product_id','') in indices]    

def user_prompt_handler(user_message: str, k: int = 3) -> str:
    # Get relevant products
    retrieved_products = semantic_search(user_message, k)

    def normalize_product(p: dict):
        """Normalize both flower and vase products into a clean context."""
        if p.get("type") == "vase":
            return {
                "id": p.get("product_id"),
                "type": "vase",
                "name": p.get("name"),
                "description": p.get("description", ""),
                "price": p.get("price"),
                "available": bool(p.get("available", False)),
            }
        else:  # assume flower by default
            opts = []
            for o in p.get("flower_details", {}).get("options", []):
                opts.append({
                    "name": o.get("name"),
                    "price": o.get("price"),
                    "stems": o.get("stems"),
                    "available": bool(o.get("stock", 0) and o.get("stock", 0) > 0),
                })
            return {
                "id": p.get("product_id"),
                "type": "flower",
                "name": p.get("name"),
                "description": p.get("description", ""),
                "flower_type": p.get("flower_details", {}).get("flower_type", ""),
                "color": p.get("flower_details", {}).get("color", []),
                "occasion": p.get("flower_details", {}).get("occasion", []),
                "options": opts,
            }

    ctx_items = [normalize_product(p) for p in retrieved_products]
    context_text = json.dumps(ctx_items, ensure_ascii=False, indent=2)

    prompt = f"""
You are a friendly assistant helping customers choose products.  
Always start your answer with a short friendly introduction sentence before listing details.  
Answer based strictly on the [Shop Data] below.

[Shop Data]
{context_text}

[User Question]
{user_message}

[Instructions]
1. Start every response with a friendly, natural intro (e.g., "Here are some flowers you might like:" or "This vase could be a perfect match:").
2. Use facts ONLY from [Shop Data]. If missing, reply exactly: Data not found.
3. Prices:
   - Flowers: If a user asks about an option (original/deluxe/grand), provide the price if it exists.
   - Vases: Provide the single listed price if available.
4. Availability:
   - You may say "currently out of stock" if "available" is false.
   - Never mention or infer stock numbers.
5. Alternatives:
   - Flowers: If requested option is unavailable, suggest the nearest available tier (original ↔ deluxe ↔ grand).
   - Vases: If not available, simply say it’s out of stock.
6. Formatting:
   - Be concise, friendly, and natural.
   - List multiple products/options with "-" bullets.
   - Include product and option names with their prices.
   - Do NOT output JSON or code.
   - Do NOT include reasoning steps.

[Examples]
Q: What's the price of the Red Rose Bouquet grand version?  
A: Here’s what I found: The grand Red Rose Bouquet is 39.99, but it’s currently out of stock. A similar option is "deluxe" at 29.99.

Q: Do you have purple orchids for Women's Day?  
A: Here are some flowers that are suitable for Women’s Day:  
- Orchid Elegance — "original" 35.0, "deluxe" 55.0, "grand" 79.0.

Q: What’s the price of the Montecito Vase?  
A: This vase could be a perfect match: The Montecito Vase is available at 19.

Q: What if something isn’t in the data?  
A: Data not found.
"""
    return prompt


    
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
                "Authorization": f"Bearer {api_key_llm}",
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
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or default to 5000
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "0") == "1")