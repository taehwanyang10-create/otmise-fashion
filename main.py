import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

# 서버 환경 변수에서 API 키를 가져옴 (보안 강화)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

system_prompt = """
너는 패션 덕후이자 사용자의 친한 친구 '옷미새'야.
사용자의 키, 몸무게, 체형 고민을 바탕으로 단점을 완벽히 커버해주는 핏과 스타일을 추천해줘.
추천 시 무신사/네이버 쇼핑 검색 링크([텍스트](URL))를 함께 제공해줘.
"""

class ChatRequest(BaseModel):
    messages: list

# 메인 접속 시 index.html 띄우기
@app.get("/")
async def read_index():
    return FileResponse("index.html")

# AI 대화 API
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="서버 API 키가 설정되지 않았습니다.")
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=request.messages,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7,
            )
        )
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))