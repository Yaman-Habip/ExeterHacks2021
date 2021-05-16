import cv2
import numpy as np
import image_processing.plot as pl
import time
import os
from configparser import ConfigParser


CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

config_path = "C:/Users/jryan/PycharmProjects/python video stream/image_processing/yolov3.cfg"
weights_path = "C:/Users/jryan/PycharmProjects/python video stream/image_processing/yolov3.weights/yolov3.weights"

labels = open("C:/Users/jryan/PycharmProjects/python video stream/image_processing/coco.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

#####################################
def parse_xy(xy):
    x, y = xy.split(',')
    return int(x), int(y)


def merge_xy(x, y):
    return ','.join([str(x),str(y)])

class Configuration:
    def __init__(self, configfile_path):
        self.configfile_path = configfile_path
        self.bl_x = 0
        self.bl_y = 0
        self.br_x = 0
        self.br_y = 0
        self.tr_x = 0
        self.tr_y = 0
        self.tl_x = 0
        self.tl_y = 0
        self.width = 0
        self.depth = 0
        self.video_source = 0

    def initialize(self):
        config = ConfigParser()
        config.read(self.configfile_path)
        self.bl_x, self.bl_y = parse_xy(config['ROI']['bottomleft'])
        self.br_x, self.br_y = parse_xy(config['ROI']['bottomright'])
        self.tl_x, self.tl_y = parse_xy(config['ROI']['topleft'])
        self.tr_x, self.tr_y = parse_xy(config['ROI']['topright'])
        self.width = int(config['DIMENSION']['width'])
        self.depth = int(config['DIMENSION']['depth'])
        self.video_source = config['VIDEO']['source']

    def save(self):
        config = ConfigParser()
        config['ROI'] = {}
        config['ROI']['bottomleft'] = merge_xy(self.bl_x, self.bl_y)
        config['ROI']['bottomright'] = merge_xy(self.br_x, self.br_y)
        config['ROI']['topleft'] = merge_xy(self.tl_x, self.tl_y)
        config['ROI']['topright'] = merge_xy(self.tr_x, self.tr_y)
        config['DIMENSION'] = {}
        config['DIMENSION']['width'] = str(self.width)
        config['DIMENSION']['depth'] = str(self.depth)
        config['VIDEO'] = {}
        config['VIDEO']['source'] = self.video_source
        with open(self.configfile_path, 'w') as configfile:
            config.write(configfile)




def go(file):
    # insert rasberi pi live feed
    path_name = "C:/Users/jryan/PycharmProjects/python video stream/test.jpg"
    image = cv2.imread(path_name)
    file_name = os.path.basename(path_name)
    filename, ext = file_name.split(".")

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    start = time.perf_counter()
    layer_outputs = net.forward(ln)


    ###################################
    def detect_people(image, config):
        # print("Start detecting ...")
        (H, W) = image.shape[:2]
        src = np.float32([[config.bl_x, config.bl_y], [config.br_x, config.br_y],
                          [config.tr_x, config.tr_y], [config.tl_x, config.tl_y]])
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        p_transform = cv2.getPerspectiveTransform(src, dst)

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # detecting people in the image
                if class_id == 0:

                    if confidence > CONFIDENCE:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (cx, cy, width, height) = box.astype("int")

                        x = int(cx - (width / 2))
                        y = int(cy - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, SCORE_THRESHOLD)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x, y, w, h = boxes[i]

        # print(len(boxes1))
        if len(boxes1) == 0:
            return image

        bottom_points = []
        for box in boxes1:
            pnts = np.array([[[int(box[0] + (box[2] * 0.5)), int(box[1] + box[3])]]], dtype="float32")
            bd_pnt = cv2.perspectiveTransform(pnts, p_transform)[0][0]
            pnt = [int(bd_pnt[0]), int(bd_pnt[1])]
            bottom_points.append(pnt)

        distances_mat = []
        bxs = []

        for i in range(len(bottom_points)):
            for j in range(len(bottom_points)):
                if i != j:
                    p1 = bottom_points[i]
                    p2 = bottom_points[j]
                    dis_w = float((abs(p2[0] - p1[0]) / W) * config.width)
                    dis_h = float((abs(p2[1] - p1[1]) / H) * config.depth)
                    dist = int(np.sqrt(((dis_h) ** 2) + ((dis_w) ** 2)))

                    if dist <= 150:
                        closeness = 0
                        distances_mat.append([bottom_points[i], bottom_points[j], closeness])
                        bxs.append([boxes1[i], boxes1[j], closeness])
                    elif dist > 150 and dist <= 180:
                        closeness = 1
                        distances_mat.append([bottom_points[i], bottom_points[j], closeness])
                        bxs.append([boxes1[i], boxes1[j], closeness])
                    else:
                        closeness = 2
                        distances_mat.append([bottom_points[i], bottom_points[j], closeness])
                        bxs.append([boxes1[i], boxes1[j], closeness])

        r = []
        g = []
        y = []

        for i in range(len(distances_mat)):

            if distances_mat[i][2] == 0:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                    r.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                    r.append(distances_mat[i][1])

        for i in range(len(distances_mat)):

            if distances_mat[i][2] == 1:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                    y.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                    y.append(distances_mat[i][1])

        for i in range(len(distances_mat)):

            if distances_mat[i][2] == 2:
                if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                    g.append(distances_mat[i][0])
                if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                    g.append(distances_mat[i][1])

        risk_count = (len(r), len(y), len(g))

        image_copy = np.copy(image)

        image = pl.social_distancing_view(image_copy, bxs, boxes1, risk_count)
        # image = plot.detection_view(image_copy, boxes1)

        return image

    # first function call
    configuration = Configuration('C:/Users/jryan/PycharmProjects/python video stream/image_processing/Configure.ini')


    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")

    font_scale = 1
    thickness = 1
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIDENCE:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    for i in range(len(boxes)):
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color = color, thickness = thickness)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale = font_scale, thickness = thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = (
        (text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color = color, thickness = cv2.FILLED)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = font_scale, color = (0, 0, 0), thickness = thickness)

    cv2.imwrite(filename + "_yolo3." + ext, image)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    ppl = []

    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            print("NONO")
            # if text == "person":
            print("YESTYES")
            image = detect_people(image, configuration)

            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    cv2.imwrite("C:/Users/jryan/PycharmProjects/python video stream/" + filename + "_yolo3." + ext, image)
    print("YOLO finished.")




