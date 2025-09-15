
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re
from dotenv import load_dotenv 
import os 
load_dotenv()
app = FastAPI(
    title="Stock Chatbot API",
    description="API for chatbot analyzes stock using model SmolLM2-1.7B-Instruct",
    version="1.0.0"
)

origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    thinking: str | None = None 

# [Checklist] Load your model directly from the Hugging Face Hub

MODEL_ID = "chuotchuilacduong/SmolLM2-1.7B-Instruct-Finetuned-Stock"
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# Biến toàn cục để giữ model
chatbot_pipeline = None
tokenizer = None

def parse_model_output(model_text: str):
   
    think_content = re.search(r"<think>(.*?)</think>", model_text, re.DOTALL)
    answer_content = re.search(r"<answer>(.*?)</answer>", model_text, re.DOTALL)

    thinking = think_content.group(1).strip() if think_content else "No thinking process found."
    answer = answer_content.group(1).strip() if answer_content else model_text.strip()

    if answer_content:
        answer = answer_content.group(1).strip()
    else:
        answer = model_text.strip()

    return {"thinking": thinking, "answer": answer}

@app.on_event("startup")
def load_model():
    global chatbot_pipeline, tokenizer
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        print("Hugging Face token not found. Please set HUGGING_FACE_HUB_TOKEN in your .env file.")
        return
    try:
        print(f"Loading model from Hub: {MODEL_ID}...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="auto" # Tự động sử dụng GPU nếu có
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

@app.get("/")
def read_root():
    return {"status": "Stock Chatbot API is running."}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    if not chatbot_pipeline or not tokenizer:
        raise HTTPException(status_code=500, detail="Model is not available.")

    try:
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.message.strip()},
        ]

        # 2. Sử dụng tokenizer để tạo prompt cuối cùng
        final_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        results = chatbot_pipeline(
            final_prompt, 
            max_new_tokens=1024, 
            num_return_sequences=1,
            # Thêm các tham số khác nếu cần, ví dụ: temperature, top_p...
        )
        
        
        full_generated_text = results[0]['generated_text']
        model_output_only = full_generated_text.replace(final_prompt, "").strip()

        parsed_output = parse_model_output(model_output_only)
        return ChatResponse(
            answer=parsed_output["answer"], 
            thinking=parsed_output["thinking"]
        )
        
    except Exception as e:
        print(f"Model inference error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during model inference.")