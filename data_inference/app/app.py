from fastapi import FastAPI, File, UploadFile, Request,Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse,JSONResponse
from .inference.inference import predict_image
from pydantic import BaseModel
import numpy as np
import base64
import cv2, numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class Analyzer(BaseModel):
    filename: str
    img_dimension: str
    encoded: str

@app.get("/",response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html",{"request": request,"message":"Hello, World!"})

@app.post("/")
async def handle_form(file:UploadFile = File(...)):
    print()
    content = await file.read()
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    predicted_label,predicted_image = predict_image(img)
    _, encoded_img = cv2.imencode('.PNG', predicted_image)
    encoded_img = base64.b64encode(encoded_img).decode("utf-8")
    base64_str = f"data:image/{file.filename.split('.')[-1]};base64,{encoded_img}"
    return JSONResponse(content = {
        'predictedlabel': predicted_label,
        'encoded_img': base64_str
    })