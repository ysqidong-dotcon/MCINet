# python chatglm_api.py

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel
import torch


print("Loading ChatGLM3...")
tokenizer = AutoTokenizer.from_pretrained('', trust_remote_code=True)
model = AutoModel.from_pretrained('', trust_remote_code=True).half().cuda()
model.eval()
print("Model loaded.")

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "Missing 'prompt' in request."}), 400

    response, _ = model.chat(tokenizer, prompt, history=[])
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
