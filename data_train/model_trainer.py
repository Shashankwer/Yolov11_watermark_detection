from ultralytics import YOLO

model = YOLO("yolov11n.pt")

results = model.train(data="dataset.yaml",epochs=100, batch=32,device='mps',lr0=0.1,)

model.export(format="onnx")