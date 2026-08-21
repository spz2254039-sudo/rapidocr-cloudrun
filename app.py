import os
from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR

app = FastAPI(title="RapidOCR Traditional Chinese Service")

# 直接指向繁體模型與字典
REC_MODEL = os.path.join("models", "chinese_cht_PP-OCRv3_rec.onnx")
REC_KEYS = os.path.join("models", "chinese_cht_dict.txt")

ocr = RapidOCR(
    rec_model_path=REC_MODEL,
    rec_keys_path=REC_KEYS,
)


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

  # det_limit_side_len=960 可避免大圖浪費推論時間
  result, elapse = ocr(img, det_limit_side_len=960)

  formatted_result = []
  if result:
    for item in result:
      formatted_result.append({
          "box": item[0],
          "text": item[1],
          "confidence": round(float(item[2]), 4),
      })

  return {
      "text_count": len(formatted_result),
      "elapse": elapse,
      "data": formatted_result,
  }
