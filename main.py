import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai

app = FastAPI()

# 환경 변수에서 Gemini API 키 가져오기
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "index.html 파일을 찾을 수 없습니다."

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    try:
        # 대화 내용을 Gemini 형식으로 변환
        contents = []
        for msg in request.messages:
            role = "user" if msg.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        # 최신 google-genai 라이브러리 호출
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
        )
        
        return {"reply": response.text}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
