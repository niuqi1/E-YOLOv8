from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    # model = YOLO('yolov5.yaml')#train
    model = YOLO('yolov8-CARAFE .yaml')#train
    # model = YOLO('yolov8-EMA.yaml')#train2
    # model = YOLO('yolov8-ECA.yaml')# train15
    # model = YOLO('yolov8-CBAM.yaml')  # train16
    # model = YOLO('yolov8-SE.yaml')  # train17
    # model = YOLO('yolov8-EMA-1.yaml')
    # model = YOLO('yolov8-EMA(P3).yaml')  # train18
    # model = YOLO('yolov8-EMA-SPD.yaml') #train24 new
    # model = YOLO('yolov8-EMA(Neck).yaml')  # train20
    # model = YOLO('yolov8-EMA(P5).yaml')  # train19
    # model = YOLO('yolov8-C2f_EMA.yaml')  # train20
    # model = YOLO('yolov8-EMA-SPD.yaml')#train3
    # model = YOLO('yolov8-EMA-SPD-CARAFE.yaml') #train4
    # model = YOLO('yolov8-EMA-SPD-p3,p4-CARAFE.yaml')#train5
    # model = YOLO('yolov8-EMA-SPD-p3,p4.yaml')#train6 #train7
    # model = YOLO('yolov8-EMA-SPD-p3,p4.yaml')
    # model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    # model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

    # Train the model
    model.train(data='data.yaml', epochs=100, imgsz=640, workers=2)

