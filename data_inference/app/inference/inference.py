"""
Performs inference for the loaded image
The inference is carried out on a onnx file
"""

import onnxruntime as ort
import numpy as np
import os
import cv2
from .inference_helper import (
    preprocess,
    non_max_suppression,
    construct_result
)

def predict_image(img:np.ndarray=None):
    """
    Makes a prediction on the image
    If the water mark is present then bounded box is
    appended to the image else image image is left as is
    Params:
        model_name: str = Name of the onnx model for inference
        img: on which preprocessing needs to be carried out
    """
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(current_file_path,'model.onnx')
    if not os.path.exists(model_name):
        raise AttributeError("Model Not found")
    session = ort.InferenceSession(model_name, None)
    input_name = session.get_inputs()[0].name
    img_copy = preprocess(img)
    outputs = session.run([],{input_name:[img_copy]})
    bounding_box_inference = non_max_suppression(outputs)
    if len(bounding_box_inference[0])==0:
        print("No bounding box found returning empty image")
        return "No Watermark Present", img
    predicted_img = construct_result(bounding_box_inference, img_copy, orig_image=img)
    return "watermark",predicted_img

if __name__ == "__main__":
    # test case
    img = cv2.imread("<image_name>")
    predict_image("model.onnx")

    


