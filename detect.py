from ultralytics import YOLO

model = YOLO("my_custom_model.pt")

results = model.predict(source='data/videos/workervideo2.mp4',save=True, show=True, conf=0.7)