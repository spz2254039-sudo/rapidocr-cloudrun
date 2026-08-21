import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

app = FastAPI(title="RapidOCR Service")

# 載入自訂繁體模型與字典檔
rec_model_path = os.path.join("models", "chinese_cht_PP-OCRv3_rec.onnx")
rec_keys_path = os.path.join("models", "chinese_cht_dict.txt")

engine = RapidOCR(
    rec_model_path=rec_model_path,
    rec_keys_path=rec_keys_path
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ocr")
async def process_ocr(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result, elapse_list = engine(image)
        
        # 整理輸出格式
        formatted_result = []
        if result:
            for item in result:
                formatted_result.append({
                    "box": item[0],
                    "text": item[1],
                    "score": float(item[2])
                })
        
        return {
            "success": True,
            "data": formatted_result,
            "latency": elapse_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
