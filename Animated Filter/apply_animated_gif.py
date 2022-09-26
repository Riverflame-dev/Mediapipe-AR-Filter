import mediapipe as mp
import cv2
import math
import numpy as np
import faceBlendCommon as fbc
import csv
from random import uniform
import time
from itertools import cycle

VISUALIZE_FACE_POINTS = False

angry_cat_config = {
    'frame0':
        [{'path': "gifs/angry-cat/frame_0.png",
          'anno_path': "gifs/angry-cat/angry-cat_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame1':
        [{'path': "gifs/angry-cat/frame_1.png",
          'anno_path': "gifs/angry-cat/angry-cat_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame2':
        [{'path': "gifs/angry-cat/frame_2.png",
          'anno_path': "gifs/angry-cat/angry-cat_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame3':
        [{'path': "gifs/angry-cat/frame_3.png",
          'anno_path': "gifs/angry-cat/angry-cat_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
}

thug_lyf_config = {
    'frame0':
        [{'path': "gifs/thug-lyf/giphy-0.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame1':
        [{'path': "gifs/thug-lyf/giphy-1.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame2':
        [{'path': "gifs/thug-lyf/giphy-2.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame3':
        [{'path': "gifs/thug-lyf/giphy-3.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame4':
        [{'path': "gifs/thug-lyf/giphy-4.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame5':
        [{'path': "gifs/thug-lyf/giphy-5.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame6':
        [{'path': "gifs/thug-lyf/giphy-6.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame7':
        [{'path': "gifs/thug-lyf/giphy-7.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame8':
        [{'path': "gifs/thug-lyf/giphy-8.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame9':
        [{'path': "gifs/thug-lyf/giphy-9.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame10':
        [{'path': "gifs/thug-lyf/giphy-10.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame11':
        [{'path': "gifs/thug-lyf/giphy-11.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame12':
        [{'path': "gifs/thug-lyf/giphy-12.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame13':
        [{'path': "gifs/thug-lyf/giphy-13.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame14':
        [{'path': "gifs/thug-lyf/giphy-14.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame15':
        [{'path': "gifs/thug-lyf/giphy-15.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame16':
        [{'path': "gifs/thug-lyf/giphy-16.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame17':
        [{'path': "gifs/thug-lyf/giphy-17.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame18':
        [{'path': "gifs/thug-lyf/giphy-18.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame19':
        [{'path': "gifs/thug-lyf/giphy-19.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame20':
        [{'path': "gifs/thug-lyf/giphy-20.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame21':
        [{'path': "gifs/thug-lyf/giphy-21.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame22':
        [{'path': "gifs/thug-lyf/giphy-22.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame23':
        [{'path': "gifs/thug-lyf/giphy-23.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame24':
        [{'path': "gifs/thug-lyf/giphy-24.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame25':
        [{'path': "gifs/thug-lyf/giphy-25.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame26':
        [{'path': "gifs/thug-lyf/giphy-26.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame27':
        [{'path': "gifs/thug-lyf/giphy-27.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame28':
        [{'path': "gifs/thug-lyf/giphy-28.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame29':
        [{'path': "gifs/thug-lyf/giphy-29.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame30':
        [{'path': "gifs/thug-lyf/giphy-30.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame31':
        [{'path': "gifs/thug-lyf/giphy-31.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame32':
        [{'path': "gifs/thug-lyf/giphy-32.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame33':
        [{'path': "gifs/thug-lyf/giphy-33.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame34':
        [{'path': "gifs/thug-lyf/giphy-34.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame35':
        [{'path': "gifs/thug-lyf/giphy-35.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame36':
        [{'path': "gifs/thug-lyf/giphy-36.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame37':
        [{'path': "gifs/thug-lyf/giphy-37.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame38':
        [{'path': "gifs/thug-lyf/giphy-38.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame39':
        [{'path': "gifs/thug-lyf/giphy-39.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame40':
        [{'path': "gifs/thug-lyf/giphy-40.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame41':
        [{'path': "gifs/thug-lyf/giphy-41.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame42':
        [{'path': "gifs/thug-lyf/giphy-42.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame43':
        [{'path': "gifs/thug-lyf/giphy-43.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame44':
        [{'path': "gifs/thug-lyf/giphy-44.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame45':
        [{'path': "gifs/thug-lyf/giphy-45.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame46':
        [{'path': "gifs/thug-lyf/giphy-46.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame47':
        [{'path': "gifs/thug-lyf/giphy-47.png",
          'anno_path': "gifs/thug-lyf/giphy_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
}

ugly_beard_config = {
    'frame0':
        [{'path': "gifs/ugly-beard/ugly-beard-0.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],

    'frame1':
        [{'path': "gifs/ugly-beard/ugly-beard-1.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame2':
        [{'path': "gifs/ugly-beard/ugly-beard-2.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame3':
        [{'path': "gifs/ugly-beard/ugly-beard-3.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame4':
        [{'path': "gifs/ugly-beard/ugly-beard-4.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame5':
        [{'path': "gifs/ugly-beard/ugly-beard-5.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame6':
        [{'path': "gifs/ugly-beard/ugly-beard-6.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame7':
        [{'path': "gifs/ugly-beard/ugly-beard-7.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame8':
        [{'path': "gifs/ugly-beard/ugly-beard-8.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame9':
        [{'path': "gifs/ugly-beard/ugly-beard-9.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame10':
        [{'path': "gifs/ugly-beard/ugly-beard-10.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame11':
        [{'path': "gifs/ugly-beard/ugly-beard-11.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame12':
        [{'path': "gifs/ugly-beard/ugly-beard-12.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame13':
        [{'path': "gifs/ugly-beard/ugly-beard-13.png",
          'anno_path': "gifs/ugly-beard/ugly-beard_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],

}

stars_config = {
    'frame0':
        [{'path': "gifs/stars/stars-0.png",
          'anno_path': "gifs/stars/stars.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame1':
        [{'path': "gifs/stars/stars-1.png",
          'anno_path': "gifs/stars/stars.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame2':
        [{'path': "gifs/stars/stars-2.png",
          'anno_path': "gifs/stars/stars.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame3':
        [{'path': "gifs/stars/stars-3.png",
          'anno_path': "gifs/stars/stars.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame4':
        [{'path': "gifs/stars/stars-4.png",
          'anno_path': "gifs/stars/stars.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame5':
        [{'path': "gifs/stars/stars-5.png",
          'anno_path': "gifs/stars/stars.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame6':
        [{'path': "gifs/stars/stars-6.png",
          'anno_path': "gifs/stars/stars.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame7':
        [{'path': "gifs/stars/stars-7.png",
          'anno_path': "gifs/stars/stars.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame8':
        [{'path': "gifs/stars/stars-8.png",
          'anno_path': "gifs/stars/stars.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
}

work_config = {
    'frame0':
        [{'path': "gifs/work/work-0.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame1':
        [{'path': "gifs/work/work-1.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame2':
        [{'path': "gifs/work/work-2.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame3':
        [{'path': "gifs/work/work-3.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame4':
        [{'path': "gifs/work/work-4.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame5':
        [{'path': "gifs/work/work-5.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame6':
        [{'path': "gifs/work/work-6.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame7':
        [{'path': "gifs/work/work-7.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame8':
        [{'path': "gifs/work/work-8.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame9':
        [{'path': "gifs/work/work-9.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame10':
        [{'path': "gifs/work/work-10.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame11':
        [{'path': "gifs/work/work-11.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame12':
        [{'path': "gifs/work/work-12.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame13':
        [{'path': "gifs/work/work-13.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame14':
        [{'path': "gifs/work/work-14.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame15':
        [{'path': "gifs/work/work-15.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame16':
        [{'path': "gifs/work/work-16.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame17':
        [{'path': "gifs/work/work-17.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame18':
        [{'path': "gifs/work/work-18.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'frame19':
        [{'path': "gifs/work/work-19.png",
          'anno_path': "gifs/work/work.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],

}
config_list = [thug_lyf_config, angry_cat_config, ugly_beard_config, stars_config, work_config]
iter_config_list = cycle(config_list)
filters_config = next(iter_config_list)
# detect facial landmarks in image
def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]

    height, width = img.shape[:-1]

    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0

        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))

            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y

            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []

            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0


def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))

    return img, alpha


def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points


def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex


def load_filter(filter_name="dog"):

    filters = filters_config[filter_name]

    multi_filter_runtime = []

    for filter in filters:
        temp_dict = {}

        img1, img1_alpha = load_filter_img(filter['path'], filter['has_alpha'])

        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha

        points = load_landmarks(filter['anno_path'])

        temp_dict['points'] = points

        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)

            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue

        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap

        multi_filter_runtime.append(temp_dict)

    return filters, multi_filter_runtime


# process input from webcam or video file
cap = cv2.VideoCapture(0)

# Some variables
count = 0
isFirstFrame = True
sigma = 50

iter_filter_keys = iter(filters_config.keys())
filters, multi_filter_runtime = load_filter(next(iter_filter_keys))

# The main loop
loop = 1
while True:

    ret, frame = cap.read()
    if not ret:
        break
    else:

        points2 = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # if face is partially detected
        if not points2 or (len(points2) != 75):
            continue

        ################ Optical Flow and Stabilization Code #####################
        img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if isFirstFrame:
            points2Prev = np.array(points2, np.float32)
            img2GrayPrev = np.copy(img2Gray)
            isFirstFrame = False

        lk_params = dict(winSize=(101, 101), maxLevel=15,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
        points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, points2Prev,
                                                        np.array(points2, np.float32),
                                                        **lk_params)

        # Final landmark points are a weighted average of detected landmarks and tracked landmarks

        for k in range(0, len(points2)):
            d = cv2.norm(np.array(points2[k]) - points2Next[k])
            alpha = math.exp(-d * d / sigma)
            points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
            points2[k] = fbc.constrainPoint(points2[k], frame.shape[1], frame.shape[0])
            points2[k] = (int(points2[k][0]), int(points2[k][1]))

        # Update variables for next pass
        points2Prev = np.array(points2, np.float32)
        img2GrayPrev = img2Gray
        ################ End of Optical Flow and Stabilization Code ###############

        if VISUALIZE_FACE_POINTS:
            for idx, point in enumerate(points2):
                cv2.circle(frame, point, 2, (255, 0, 0), -1)
                cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
            cv2.imshow("landmarks", frame)

        for idx, filter in enumerate(filters):

            filter_runtime = multi_filter_runtime[idx]
            img1 = filter_runtime['img']
            points1 = filter_runtime['points']
            img1_alpha = filter_runtime['img_a']

            if filter['morph']:

                hullIndex = filter_runtime['hullIndex']
                dt = filter_runtime['dt']
                hull1 = filter_runtime['hull']

                # create copy of frame
                warped_img = np.copy(frame)

                # Find convex hull
                hull2 = []
                for i in range(0, len(hullIndex)):
                    hull2.append(points2[hullIndex[i][0]])

                mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                mask1 = cv2.merge((mask1, mask1, mask1))
                img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                # Warp the triangles
                for i in range(0, len(dt)):
                    t1 = []
                    t2 = []

                    for j in range(0, 3):
                        t1.append(hull1[dt[i][j]])
                        t2.append(hull2[dt[i][j]])

                    fbc.warpTriangle(img1, warped_img, t1, t2)
                    fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                # Blur the mask before blending
                mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                mask2 = (255.0, 255.0, 255.0) - mask1

                # Perform alpha blending of the two images
                temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                output = temp1 + temp2
            else:
                dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                tform = fbc.similarityTransform(list(points1.values()), dst_points)
                # Apply similarity transform to input image
                trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                # Blur the mask before blending
                mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                mask2 = (255.0, 255.0, 255.0) - mask1

                # Perform alpha blending of the two images
                temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                output = temp1 + temp2

            frame = output = np.uint8(output)

        cv2.putText(frame, "Press F to change filters", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)

        # Create named window for resizing
        cv2.namedWindow('AR Filter', cv2.WINDOW_AUTOSIZE)



        # Display the frame
        cv2.imshow("AR Filter", frame)


        if loop == 1:
            try:
                filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
                print(filters)
            except:
                iter_filter_keys = iter(filters_config.keys())
                filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
            time.sleep(0.01)

        keypressed = cv2.waitKey(1) & 0xFF
        if keypressed == 27:
            break
        # Put next filter if 'f' is pressed
        elif keypressed == ord('f'):
            filters_config = next(iter_config_list)
            print(filters_config)

        count += 1

cap.release()
cv2.destroyAllWindows()
