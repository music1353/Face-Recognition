from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import cv2
import face_recognition

# @desc: 算眉毛中心點座標的函數
# @input: {list} 眉毛特徵座標
# @return: {tuple} 眉毛最左最右點的平均座標點
def middle_of_eyebrow(eyebrow_landmarks):
    first_x_point = eyebrow_landmarks[0][0]
    last_x_point = eyebrow_landmarks[-1][0]
    first_y_point = eyebrow_landmarks[0][1]
    last_y_point = eyebrow_landmarks[-1][1]
    avg_point = ((first_x_point+last_x_point)/2, (first_y_point+last_y_point)/2)
    return avg_point
    
    
# @desc: 將座標轉換成可以畫圓的座標
# @input: {tuple} 一組座標點
# @return: {tuple} 一組可以畫圓的座標點
def transform_2_point(tuple_point):
    l = list(tuple_point)
    l.append(l[0]+5)
    l.append(l[1]+5)
    t = tuple(l)
    return t


# @desc: 找到眼睛最高最低點的座標
# @input: {list} 眼睛特徵座標
# @return: {tuple, tuple} 眼睛特徵點的最高點及最低點座標
def find_eye_feacture_point(eye_landmarks):
    # 找到y軸最高的特徵點
    max_x = 0
    max_y = 0
    for x, y in eye_landmarks:
        if(y > max_y):
            max_y = y
            max_x = x

    # 找到y軸最低的特徵點
    min_x = 0
    min_y = max_y
    for x, y in eye_landmarks:
        if(y < min_y):
            min_y = y
            min_x = x

    return (max_x, max_y), (min_x, min_y)
    
    
# @desc: 找很多點中的四個角(top_left, top_right, bottom_left, bottom_right)
# @input: {list[tuple]} 特徵座標點
# @return: {list} [top_left, top_right, bottom_left, bottom_right]
def find_box_points(feacture_list):
    max_x = 0
    min_x = 5000
    max_y = 0
    min_y = 5000
        
    for point in feacture_list:
        # find max & min x-axis
        if point[0] > max_x:
            max_x = point[0]

        if point[0] < min_x:
            min_x = point[0]

        # find max & min y-axis
        if point[1] > max_y:
            max_y = point[1]

        if point[1] < min_y:
            min_y = point[1]
            top_point_y = point[1]
            top_point_x = point[0]

    top_left = (min_x, min_y)
    top_right = (max_x, min_y)
    bottom_left = (min_x, max_y)
    bottom_right = (max_x, max_y)
    box_points = [top_left, top_right, bottom_left, bottom_right]
        
    return box_points
    
    
# @desc: 辨識眉毛及眼睛座標
# @input: {string} 原始圖片位置
# @return: {tuple} 左眉座標, 右眉座標, 左眼低點座標, 右眼低點座標, {list} 左眉box, 右眉box
def recognize_eye_and_eyebrow_axis(IMG_URL):
    image = face_recognition.load_image_file(IMG_URL)
    face_locations  = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")
    image_landmarks = face_recognition.face_landmarks(image, face_locations=face_locations)

    for face_landmarks in image_landmarks:
        # 處理 left_eyebrow
        left_eyebrow_avg_point = middle_of_eyebrow(face_landmarks['left_eyebrow'])
        # 處理 right_eyebrow
        right_eyebrow_avg_point = middle_of_eyebrow(face_landmarks['right_eyebrow'])
        # 處理 left_eyebrow box
        left_eyebrow_box = find_box_points(face_landmarks['left_eyebrow'])
        # 處理 right_eyebrow box
        right_eyebrow_box = find_box_points(face_landmarks['right_eyebrow'])
        # 處理 left_eye
        left_eye_max_point, left_eye_min_point = find_eye_feacture_point(face_landmarks['left_eye'])
        # 處理 right_eye
        right_eye_max_point, right_eye_min_point = find_eye_feacture_point(face_landmarks['right_eye'])
        
    return left_eyebrow_avg_point, right_eyebrow_avg_point, left_eye_max_point, right_eye_max_point, left_eyebrow_box, right_eyebrow_box
    
    
# @desc: 辨識出圖片中眼睛的最高點座標(包含眼皮)
# @input: {string} 原始圖片位置, 左眉毛箱型座標, 右眉毛箱型座標
# @return: {list} 第一個為左眼最高點座標, 第二個為右眼最高點座標
def recognize_eye_top_axis(IMG_URL, left_eyebrow_box, right_eyebrow_box):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

    img = cv2.imread(IMG_URL)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eye_top_points = [] # 左眼、右眼的原圖片最高點座標
        
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 3, 5) # 右眼、左眼
        list_eyes = [eyes[0], eyes[1]]
            
        for idx, (ex, ey, ew, eh) in enumerate(list_eyes):
            # 如果包到眉毛, 則把眉毛底部當作眼睛的頂部
            # idx==0, 處理左眼
            # idx==1, 處理右眼
            print('ey:', ey)
            if idx == 0:
                bottom = left_eyebrow_box[2][1] - y
                print('bottom:', bottom)
                if bottom>=ey:
                    print('左眼眼睛包含眉毛')
                    ey = bottom + 5
            elif idx == 1:
                bottom = right_eyebrow_box[2][1] - y
                print('bottom:', bottom)
                if bottom>=ey:
                    print('右眼眼睛包含眉毛')
                    ey = bottom + 5
                
            crop_img = roi_color[ey:ey+eh, ex:ex+ew]
            eye_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

            # find edge
            edges = cv2.Canny(eye_gray, 100, 200)

            # dilation
            closed = cv2.dilate(edges, None, iterations=4)
            plt.imshow(closed)
            plt.show()

            # find counter
            _, contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # find eyelid counter's axis
            max_x = 0
            min_x = 5000
            max_y = 0
            min_y = 5000
            top_point_x = 0
            top_point_y = 0
            for point in contours[-1]:
                # find max & min x-axis
                if point[0][0] > max_x:
                    max_x = point[0][0]

                if point[0][0] < min_x:
                    min_x = point[0][0]

                # find max & min y-axis
                if point[0][1] > max_y:
                    max_y = point[0][1]

                if point[0][1] < min_y:
                    min_y = point[0][1]
                    top_point_y = point[0][1]
                    top_point_x = point[0][0]
                
            eye_top_points.append((x+ex+top_point_x, y+ey+top_point_y))
                
        return eye_top_points
    
    
# @desc: 將座標點畫點在圖片上
# @input: {string}圖片位置, {list}要畫的座標點 
# @return: {img} 繪圖好的圖片
def draw_point(IMG_URL, draw_points):
    img = cv2.imread(IMG_URL)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    d = ImageDraw.Draw(pil_image)
        
    for point in draw_points:
        d.ellipse(RecognitionUtils.transform_2_point(point), fill=(255, 0, 0))

    return pil_image