import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

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
        contents = []
        for msg in request.messages:
            role = "user" if msg.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        # 시스템 인스트럭션으로 링크 포맷 강제화
        system_instruction = (
            "당신은 전문 AI 패션 스타일리스트 '옷미새'입니다. 사용자의 코디를 추천할 때, "
            "추천하는 주요 의류/아이템 이름에는 반드시 마크다운 링크 형식으로 "
            "네이버 쇼핑 검색 링크를 걸어주세요. "
            "형식 예시: [블랙 오버사이즈 블레이저](https://search.shopping.naver.com/search/all?query=블랙+오버사이즈+블레이저). "
            "반드시 이 링크 형식을 지켜서 답변하세요."
        )

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            )
        )
        
        return {"reply": response.text}
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
