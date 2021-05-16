import cv2
import numpy as np
import plot
from ImageDetection import net, CONFIDENCE, SCORE_THRESHOLD, layer_outputs
from configparser import ConfigParser

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

def detect_people(image, config):
    # print("Start detecting ...")
    (H, W) = image.shape[:2]
    src = np.float32([[config.bl_x, config.bl_y], [config.br_x, config.br_y],
                      [config.tr_x, config.tr_y], [config.tl_x, config.tl_y]])
    dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
    p_transform = cv2.getPerspectiveTransform(src, dst)

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
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
        pnts = np.array([[[int(box[0] + (box[2] * 0.5)), int(box[1] + box[3])]]], dtype = "float32")
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

    image = plot.social_distancing_view(image_copy, bxs, boxes1, risk_count)
    # image = plot.detection_view(image_copy, boxes1)

    return image