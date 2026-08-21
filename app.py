from fastapi import FastAPI, File, UploadFile, HTTPException
from rapidocr_onnxruntime import RapidOCR
import numpy as np
import cv2

app = FastAPI(title="RapidOCR Service")

# 模組載入時初始化，常駐記憶體避免請求延遲
ocr = RapidOCR()

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="請上傳圖片檔案")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="圖片解析失敗")
        
    result, elapse = ocr(img)
    
    # 格式化輸出
    formatted_result = []
    if result:
        for item in result:
            formatted_result.append({
                "box": item[0],
                "text": item[1],
                "confidence": float(item[2])
            })
            
    return {
        "text_count": len(formatted_result),
        "elapse": elapse,
        "data": formatted_result
    }
