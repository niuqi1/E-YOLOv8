from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('ultralytics/models/v8/yolov8.yaml')  # build a new model from YAML
    model.info()

