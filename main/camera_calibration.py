import numpy as np
import trimesh
import pyrender
import cv2
import edge_generate as eg

def create_qrmat(size, scene,color=[1.0,1.0,1.0]):
    # 頂点の定義
    x = size[0]
    y = size[1]
    vertices = np.array([
        [-x / 2, -y / 2, 0],  # 左下
        [ x / 2, -y / 2, 0],  # 右下
        [ x / 2,  y / 2, 0],  # 右上
        [-x / 2,  y / 2, 0]   # 左上
    ])

    # 面の定義
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])
    # メッシュの色（白）を設定
    colors = np.array([color,color,color,color])
    # メッシュの作成
    square_mesh = trimesh.Trimesh(vertices=vertices, faces=faces,vertex_colors=colors)

    # pyrender ノードの作成
    mesh = pyrender.Mesh.from_trimesh(square_mesh)
    scene.add(mesh)

def render_img(qr_length,camera_height,y_fov,resolution=(1920,1680)):
    # シーンの作成
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])  # 背景を黒に設定
    create_qrmat(qr_length, scene)

    # ライトの追加
    eg.add_light(scene,2000000,700)

    # edge_generate モジュールからカメラとレンダリング関数を使用
    camera, camera_pose = eg.create_camera(camera_height,yfov=y_fov,aspect_ratio=resolution[0]/resolution[1])
    color = eg.render_scene(scene, camera, camera_pose, resolution=resolution)

    # color データを OpenCV 形式に変換
    color = np.array(color, dtype=np.uint8)
    return color

def rendering(qr_length,camera_height,y_fov):
    
    color = render_img(qr_length,camera_height,y_fov)

    gray = cv2.cvtColor(color, cv2.COLOR_RGBA2GRAY)  # グレースケールに変換


    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)

    # 輪郭を取得
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:

        area_max = 0
        largest_cnt = None

        for cnt in contours:
            # 単純な外接矩形を描画
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > area_max:
                area_max = w * h
                largest_cnt = cnt
        cv2.drawContours(color,[largest_cnt],0,[255,0,0],3)

    area_max
    #print(area_max)

    '''# エッジ検出結果を表示
    cv2.imshow('Edges', color)
    cv2.waitKey(1)'''
    return area_max

def find_optimal_yfov(target_area, qr_size, camera_height, initial_yfov=np.pi/3, tol=10, max_iter=100):
    low, high = 0.01, np.pi  # 初期の範囲を設定
    best_yfov = initial_yfov
    best_area = rendering(qr_size, camera_height, initial_yfov)

    for _ in range(max_iter):
        mid = (low + high) / 2
        area = rendering(qr_size, camera_height, mid)

        if abs(area - target_area) < tol:
            best_yfov = mid
            best_area = area
            break

        if area > target_area:
            low = mid
        elif area < target_area:
            high = mid
        else:
            best_yfov = mid
            best_area = area
            break

        if abs(area - target_area) < abs(best_area - target_area):
            best_yfov = mid
            best_area = area

    return best_yfov, best_area
if __name__ == '__main__':
    target_area = 34798.5
    qr_size = (42,42)
    camera_height = 320
    initial_yfov = np.pi/3
    optimal_yfov, optimal_area = find_optimal_yfov(target_area, qr_size, camera_height, initial_yfov)
    print(f"Optimal yfov: {optimal_yfov}, Area: {optimal_area}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
