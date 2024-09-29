import cv2
import numpy as np
import math
import os


def get_largest_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # ノイズ除去
    kernel = np.ones((10, 10), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_max = 0
    largest_cnt = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > area_max:
            area_max = w * h
            largest_cnt = cnt
    rect = cv2.minAreaRect(largest_cnt)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    '''if np.linalg.norm(box[0]-box[1])>np.linalg.norm(box[1]-box[2]):
        box = np.roll(box,1)'''
    return largest_cnt, box




def angle_ratio_and_vector(pt0,pt1,pt2,pt3):
    vector1 = np.array(pt1) - np.array(pt0)
    vector2 = np.array(pt3) - np.array(pt2)
    length1 = np.linalg.norm(vector1)
    length2 = np.linalg.norm(vector2)
    cos_theta = np.dot(vector1, vector2) / (length1 * length2)
    sin_theta = np.cross(vector1, vector2) / (length1 * length2)
    angle = np.arctan2(sin_theta, cos_theta)
    ratio = length2 / length1
    vector = np.int32(np.array(pt2)-np.array(pt0))
    return angle, ratio, vector

def transform_points(points, scale, theta, trans, center=(960,840)):
    moved_points = points + np.array(trans)
    # 回転・拡大縮小行列の作成
    rotation_matrix = np.array([
        [scale * np.cos(theta), -scale * np.sin(theta)],
        [scale * np.sin(theta), scale * np.cos(theta)]
    ])
    # 中心座標を基準にポイントを移動
    centered_points = moved_points - np.array(center)

    # 回転・拡大縮小を適用
    transformed_pts = np.dot(centered_points, rotation_matrix.T)

    # ポイントを元の位置に戻す
    transformed_pts += np.array(center)

    return transformed_pts

def calculate_distances(cnt1, cnt2):
    distances = []
    for point in cnt1.reshape(-1, 2):
        point_tuple = (int(point[0]), int(point[1]))
        distance = cv2.pointPolygonTest(cnt2, point_tuple, True)
        distances.append(abs(distance))  # 距離は絶対値を取る
    return distances

def total_distance(scale, theta, trans, cnt1, cnt2, center=(960,840), origin=None,show = False):
    transformed_cnt1 = transform_points(cnt1.reshape(-1,2), scale, theta, trans, center)
    if origin is not None and show:
        origin2 = origin.copy()
        cv2.drawContours(origin2, [np.int32(transformed_cnt1)], 0, (0,105,255), 2)
        cv2.drawContours(origin2, [cnt2],0,(0,255,255),2)
        cv2.imshow('origin2', origin2)
        cv2.waitKey(0)
    distances = calculate_distances(np.int32(transformed_cnt1),cnt2)
    s = 0.5 #平均最大重み比
    value = s*np.mean(distances)+(1-s)*np.max(distances)
    return value/scale





def main(frame,origins,tags=None,qr_points =None):
    if tags is None:
        length = len(origins)
        tags = [str(i) for i in range(length)]
    # 最大の輪郭を取得
    largest_cnt,boundary = get_largest_contour(frame)
    cv2.drawContours(frame, [largest_cnt], 0, (255, 0, 0), 3)
    cv2.drawContours(frame, [boundary], 0, (0, 0, 255), 2)
    if largest_cnt is not None:

        best_origin = None
        min_min_dist = math.inf
        best_best_trans = [],[],[]
        tag=None

        for i in range(0,len(origins)):
            origin = origins[i]
            largest_origin_cnt,origin_boundary = get_largest_contour(origin)

            min_dist = math.inf
            cent = origin_boundary[0]
            best_trans = [],[],[]

            for n in range(0,4):
                ang,rat,vec = angle_ratio_and_vector(boundary[n-1],boundary[n],origin_boundary[0],origin_boundary[1])
                
                dist = total_distance(rat,ang,vec,largest_cnt,largest_origin_cnt,cent,origin)
                if dist < min_dist:
                    min_dist = dist
                    best_trans = rat,ang,vec
            if min_dist < min_min_dist:
                min_min_dist = min_dist
                best_best_trans = best_trans
                best_origin = origin
                tag = tags[i]
            #print(min_dist)

        origin=best_origin.copy()
        largest_origin_cnt,origin_boundary = get_largest_contour(origin)
        cv2.drawContours(origin, [largest_origin_cnt], 0, (255, 0, 0), 3)
        cv2.drawContours(origin, [origin_boundary], 0, (0, 0, 255), 2)
        cent = origin_boundary[0]
        best_trans_pt = transform_points(largest_cnt.reshape(-1,2),best_best_trans[0],best_best_trans[1],best_best_trans[2],cent)
        cv2.drawContours(origin, [np.int32(best_trans_pt)], 0, (0,55,255), 3)
        print(tag)

    points = None    
    if qr_points is None:
        # 緑の枠線の新しい位置を計算
        qr = cv2.QRCodeDetector()
        data, points, straight_qrcode = qr.detectAndDecode(frame)
        if points is not None:
            points = points.astype(np.int32)
        else:
            print("QRコードが検出されませんでした")
            return
    else:
        points = qr_points
    cv2.polylines(frame, [points], True, (0, 255, 0), thickness=10)
    for i, point in enumerate(points[0]):
        cv2.circle(frame, tuple(point), 10, (0, 0, 255), -1)
        cv2.putText(frame, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,0), 8)
    qr_coord = None
    qr_x_pixel = None
    if largest_cnt is not None:
        qr_trans = transform_points(points.reshape(-1,2),best_best_trans[0],best_best_trans[1],best_best_trans[2],cent)
        cv2.polylines(origin, [np.int32(qr_trans)], True, (0, 255, 0), thickness=10)
        for i, point in enumerate(np.int32(qr_trans)):
            cv2.circle(origin, tuple(point), 10, (0, 0, 255), -1)
            cv2.putText(origin, str(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,0), 8)
        qr_coord = qr_trans[0]
        qr_x_pixel = qr_trans[1]

    frame = cv2.resize(frame,(640,480))
    origin = cv2.resize(origin,(640,480))

    cv2.imshow('frame',frame)
    cv2.imshow('origin',origin)

    return qr_coord, qr_x_pixel


if __name__ == '__main__':

    image_path = r"T:\Goro\ComputerVision\CamBookRaw\IMG_0009.png"
    origins = []
    tags = []
    joint_name_values =["PS-10SH","PS-18SU","PS-24SU","PS-33SU","TH-10","TH-18","TH-24","TH-33"]
    joint_position = ["FRONT","BACK","SIDE"]
    for obj_name in joint_name_values:
        for pattern in joint_position:
            output_dir = os.path.join(r'T:\Goro\ComputerVision\joints_data\generate', obj_name, pattern)
            file_name = "000.png" 
            file_path = os.path.join(output_dir, file_name)
            origin = cv2.imread(file_path)
            tag = obj_name + "@" + pattern
            origins.append(origin)
            tags.append(tag)

    image = cv2.imread(image_path)
    frame = image
    main(frame,origins,tags)
    cv2.waitKey(0)
    cv2.destroyAllWindows()