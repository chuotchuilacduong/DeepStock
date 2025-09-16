# app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
from dotenv import load_dotenv 
import os 

# Tải các biến môi trường từ file .env
load_dotenv()

app = FastAPI(
    title="Stock Chatbot API",
    description="API for a chatbot that analyzes stocks using the SmolLM2-1.7B-Instruct model",
    version="1.0.0"
)

# Cấu hình CORS để cho phép truy cập từ mọi nguồn
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response ---
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    thinking: str | None = None 

# --- Model Configuration ---
MODEL_ID = "chuotchuilacduong/SmolLM2-1.7B-Instruct-Finetuned-Stock"
SYSTEM_PROMPT = (
    "You are an AI assistant designed to function as an expert reasoner. Your response must follow a strict structure composed of two parts: a thinking process and a final answer."

    "### Part 1: The <think> Block"
    "Inside the <think></think> tags, you will perform your internal, step-by-step reasoning. This is your private scratchpad. Break down the user's question, gather relevant information, evaluate different aspects, and formulate a plan for the final answer. This section is for your internal use only and will not be shown to the user."

    "### Part 2: The <answer> Block"
    "Inside the <answer></answer> tags, you will provide the complete, polished, and user-facing answer. This response must be comprehensive and self-contained, directly addressing the user's query by synthesizing the conclusions from your thinking process."

    "### Crucial Rules to Follow:"
    "1. **Standalone Answer:** The content within <answer> must make complete sense on its own, without needing the <think> block for context."
    "2. **No Self-Reference:** You MUST NOT use phrases that refer to your own reasoning process. For example, AVOID: 'Based on the points above,', 'As I reasoned in my thinking process,', 'In conclusion from my analysis,', etc."
    "3. **Strict Format:** Your entire output must only contain the <think>...</think> block immediately followed by the <answer>...</answer> block, with no other text before, between, or after these blocks."
)

# Biến toàn cục để giữ model và tokenizer sau khi load
chatbot_pipeline = None
tokenizer = None

# --- Hàm Parse tối ưu ---
def parse_response(model_text: str) -> dict:
    """
    Parse model output bằng cách tìm ranh giới giữa khối think và answer.
    Ưu tiên xử lý trường hợp dính liền "</think><answer>".
    """
    # 1. Ưu tiên cao nhất: Tìm điểm nối dính liền "</think><answer>"
    separator = "</think><answer>"
    if separator in model_text:
        parts = model_text.split(separator, 1)
        think_content = parts[0].replace("<think>", "").strip()
        answer_content = parts[1].replace("</answer>", "").strip()
        return {"thinking": think_content, "answer": answer_content}

    # 2. Ưu tiên thứ hai: Nếu không dính liền, tìm thẻ <answer> làm ranh giới
    separator = "<answer>"
    if separator in model_text:
        parts = model_text.split(separator, 1)
        think_content = parts[0].replace("<think>", "").replace("</think>", "").strip()
        answer_content = parts[1].replace("</answer>", "").strip()
        if not think_content:
            think_content = "No thinking process found."
        return {"thinking": think_content, "answer": answer_content}
        
    # 3. Trường hợp cuối: Không tìm thấy ranh giới, toàn bộ là câu trả lời
    return {"thinking": "No boundary found.", "answer": model_text.strip()}

# --- Sự kiện Startup: Load Model ---
@app.on_event("startup")
def load_model():
    """
    Load model và tokenizer khi ứng dụng khởi động.
    """
    global chatbot_pipeline, tokenizer
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("HUGGING_FACE_HUB_TOKEN not found in .env file. Please set it.")
        return
    try:
        print(f"Loading model from Hub: {MODEL_ID}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"  # Tự động sử dụng GPU nếu có
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        chatbot_pipeline = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer
        )
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        chatbot_pipeline = None 

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Stock Chatbot API is running."}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if not chatbot_pipeline or not tokenizer:
        raise HTTPException(status_code=503, detail="Model is not available or still loading.")

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.message.strip()},
        ]

        # Tạo prompt hoàn chỉnh bằng template của tokenizer
        final_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # Chạy pipeline
        results = chatbot_pipeline(
            final_prompt, 
            max_new_tokens=1024, 
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
        )
        
        # Trích xuất phần văn bản do model tạo ra một cách an toàn
        full_generated_text = results[0]['generated_text']
        # Tối ưu: Dùng slicing thay vì replace để tránh lỗi
        model_output_only = full_generated_text[len(final_prompt):].strip()

        # Parse output
        parsed_output = parse_response(model_output_only)
        return ChatResponse(
            answer=parsed_output["answer"], 
            thinking=parsed_output["thinking"]
        )
        
    except Exception as e:
        print(f"Model inference error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during model inference.")