import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_nearest_intersection_and_normal(mesh, origin, direction):
    """
    メッシュと直線の交点と、その交点での外向き法線ベクトルを返す関数。

    引数:
    mesh (trimesh.Trimesh): メッシュオブジェクト
    origin (numpy.array): 直線の始点座標 (3要素の配列)
    direction (numpy.array): 直線の方向ベクトル (3要素の配列)

    戻り値:
    tuple: 交点の座標 (numpy.array) と交点における外向き法線ベクトル (numpy.array)
    交点が見つからない場合は (None, None) を返す
    """
    # 交点を求める
    locations, index_ray, index_tri = mesh.ray.intersects_location([origin], [direction])

    # もし交点が見つかった場合
    if len(locations) > 0:
        # 交点までの距離を計算
        distances = np.linalg.norm(locations - origin, axis=1)
        
        # 最も近い交点のインデックスを取得
        nearest_index = np.argmin(distances)
        
        # 最も近い交点に対する三角形のインデックス
        nearest_triangle_index = index_tri[nearest_index]
        
        # 最も近い交点の法線ベクトルを取得
        normal_at_nearest_intersection = mesh.face_normals[nearest_triangle_index]
        
        # 交点の座標を取得
        nearest_intersection = locations[nearest_index]
        
        return nearest_intersection, normal_at_nearest_intersection
    else:
        # 交点が見つからなかった場合
        return None, None

def screen_to_world_ray(a, b, w, h, z, yfov):
    """
    スクリーン座標 (a, b) をワールド空間のレイに変換する関数。
    
    引数:
    a (float): スクリーン上のx座標
    b (float): スクリーン上のy座標
    w (float): 画面の幅
    h (float): 画面の高さ
    z (float): カメラからスクリーンまでの距離（z軸方向）
    yfov (float): 縦方向の視野角 (ラジアン)
    
    戻り値:
    tuple: レイの始点 (numpy.array) とレイの方向ベクトル (numpy.array)
    """
    # スクリーンのアスペクト比
    aspect_ratio = w / h

    # 縦方向の視野角からスクリーンの高さを計算
    screen_height = 2 * z * np.tan(yfov / 2)
    screen_width = screen_height * aspect_ratio

    # スクリーン空間の座標 (a, b) を正規化デバイス座標に変換
    ndc_x = (a / w) * 2 - 1  # x方向の正規化
    ndc_y = 1 - (b / h) * 2  # y方向の正規化 (上下が反転)

    # NDC座標をスクリーン座標系に変換
    screen_x = ndc_x * (screen_width / 2)
    screen_y = ndc_y * (screen_height / 2)

    # カメラの位置 (原点)
    ray_origin = np.array([0, 0, z])

    # スクリーン座標上の点へのベクトルを計算 (レイの方向)
    ray_direction = np.array([screen_x, screen_y, -z])

    # レイの方向ベクトルを正規化
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    return ray_origin, ray_direction

def get_mesh_intersection_from_image_coords(mesh, a, b, w, h, z, y_fov):
    """
    カメラの画像の座標 (a, b) からメッシュとの交点と交点における法線ベクトルを計算する関数。
    
    引数:
    mesh (trimesh.Trimesh): メッシュオブジェクト
    camera_position (numpy.array): カメラの位置 (3要素の配列)
    a (float): 画像上のピクセル座標の x 座標
    b (float): 画像上のピクセル座標の y 座標
    w (int): 画像の幅 (ピクセル)
    h (int): 画像の高さ (ピクセル)
    z (float): カメラから投影平面までの距離
    y_fov (float): 垂直方向の視野角 (degree)

    戻り値:
    tuple: メッシュとの交点の座標と法線ベクトル
    """
    # Step 1: ピクセル座標 (a, b) から方向ベクトルを計算
    _, direction = screen_to_world_ray(a, b, w, h, z, y_fov)
    
    # Step 2: カメラの位置からのレイとメッシュの交点を計算
    nearest_intersection, normal_at_intersection = get_nearest_intersection_and_normal(
        mesh, [0,0,z], direction
    )
    
    return nearest_intersection, normal_at_intersection



def calculate_qr_code_transform(mesh, qr_coord_pixel, qr_x_pixel, image_width, image_height, z, y_fov):
    """
    カメラ画像上で得られるQRコードの中心位置とx軸方向から、3D空間におけるQRコードの
    平行移動と回転を計算する関数。
    
    引数:
    mesh (trimesh.Trimesh): メッシュオブジェクト
    camera_position (numpy.array): カメラの位置 (3要素の配列)
    qr_coord_pixel (tuple): QRコードの画像上の中心位置 (x, y)
    qr_x_axis_pixel (tuple): QRコードのx軸方向に対応する画像上の位置 (x, y)
    image_width (int): 画像の幅
    image_height (int): 画像の高さ
    z (float): カメラから投影平面までの距離
    y_fov (float): 垂直方向の視野角 (degree)
    
    戻り値:
    tuple: 平行移動ベクトル (x, y, z) と 四元数 (a, b, c, d)
    """
    # QRコードの中心位置の3D座標を計算
    qr_coord_3d, normal_at_qr_coord = get_mesh_intersection_from_image_coords(
        mesh, qr_coord_pixel[0], qr_coord_pixel[1], 
        image_width, image_height, z, y_fov
    )

    
    if qr_coord_3d is None:
        print("QR code coordinate not on mesh")
        return None, None  # QRコードがメッシュ上に見つからない場合

    # QRコードの中心位置と法線ベクトルから平面を定義
    qr_plane = trimesh.creation.box(
        extents=(1000, 1000, 0.01), transform=trimesh.transformations.translation_matrix(qr_coord_3d)
    )
    
    # QRコードのx軸方向のピクセル位置に対応する3D方向ベクトルを計算
    qr_x_axis_3d, _ = get_mesh_intersection_from_image_coords(qr_plane
        , qr_x_pixel[0], qr_x_pixel[1], 
        image_width, image_height, z, y_fov
    )
    
    if qr_x_axis_3d is None:
        return None, None  # QRコードのx軸方向が見つからない場合
    
    # QRコードのx軸方向ベクトル (QRコード中心からx軸方向へのベクトル)
    qr_x_axis_direction = qr_x_axis_3d - qr_coord_3d
    qr_x_axis_direction = qr_x_axis_direction / np.linalg.norm(qr_x_axis_direction)  # 正規化
    # QRコードのz軸はメッシュの法線ベクトル
    qr_z_axis_direction = normal_at_qr_coord
    qr_z_axis_direction = qr_z_axis_direction / np.linalg.norm(qr_z_axis_direction)  # 正規化
    
    # QRコードのy軸はz軸とx軸の外積で計算
    qr_y_axis_direction = np.cross(qr_z_axis_direction, qr_x_axis_direction)
    qr_y_axis_direction = qr_y_axis_direction / np.linalg.norm(qr_y_axis_direction)  # 正規化
    
    # 回転行列を構成 (各列がQRコードの座標系のx, y, z軸に対応)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.column_stack((qr_x_axis_direction, qr_y_axis_direction, qr_z_axis_direction))
    rotation_inv = np.linalg.inv(rotation_matrix)

    # 3D座標を回転行列によって変換
    qr_coord_3d = rotation_inv[:3, :3] @ qr_coord_3d

    # 回転行列から四元数を計算
    rotation_quaternion = trimesh.transformations.quaternion_from_matrix(rotation_inv)  # 四元数 (x, y, z, w)

    # 原点から回転後のQRコード中心位置へのベクトルを計算
    translation_vector = - qr_coord_3d
    
    return {
        "translation": translation_vector,  # x, y, z
        "quaternion": rotation_quaternion     # a, b, c, d
    }

def transformation_dict_to_matrix(transformation):
    """
    平行移動ベクトルとクォータニオンから変換行列を計算する関数。
    
    引数:
    transformation (dict): 平行移動ベクトルとクォータニオンを格納した辞書
    transformation = {
        "translation": [x, y, z],
        "quaternion": [a, b, c, d]
    }

    戻り値:
    numpy.array: 4x4の変換行列
    """
    # 平行移動ベクトル
    translation = transformation["translation"]
    
    # クォータニオン
    quaternion = transformation["quaternion"]
    
    # クォータニオンから回転行列を計算
    rotation_matrix = trimesh.transformations.quaternion_matrix(quaternion)
    
    # 平行移動ベクトルを変換行列に追加
    rotation_matrix[:3, 3] = translation
    
    return rotation_matrix

def transformation_matrix_to_dict(transformation_matrix):
    """
    変換行列から平行移動ベクトルとクォータニオンを計算する関数。
    
    引数:
    transformation_matrix (numpy.array): 4x4の変換行列

    戻り値:
    dict: 平行移動ベクトルとクォータニオンを格納した辞書
    transformation = {
        "translation": [x, y, z],
        "quaternion": [a, b, c, d]
    }
    """
    # 平行移動ベクトル
    translation = transformation_matrix[:3, 3]
    
    # 回転行列からクォータニオンを計算
    quaternion = trimesh.transformations.quaternion_from_matrix(transformation_matrix)
    
    return {
        "translation": translation,
        "quaternion": quaternion
    }


def multiply_transformations(t1, t2):
    """
    2つの変換を掛け合わせて1つの変換にする関数。
    t1, t2: 辞書形式の変換 {'translation': [x, y, z], 'quaternion': [a, b, c, d]}
    """
    combined_transformation_matrix = transformation_dict_to_matrix(t2) @ transformation_dict_to_matrix(t1)

    combined_transformation_dict = transformation_matrix_to_dict(combined_transformation_matrix)

    return combined_transformation_dict

def load_mesh_position(obj_path,transform):
    '''
    objファイルを読み込んで、quaternionで指定した位置に配置する関数。

    引数：
    obj_path: .objファイルのファイルパス
    transform: 辞書形式{'translation':[x,y,z], 'quaternion':[a,b,c,d]} の座標変換表示

    戻り値：
    adjusted_mesh: 座標変換移動後のmesh
    '''
    mesh = trimesh.load(obj_path)
    # Y-up (左手系) を Z-up (右手系) に変換する行列
    transform_z_up = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    # 変換の適用
    # mesh.apply_transform(transform_z_up)
    quater = transform["quaternion"]
    trans = transform["translation"]
    rotate_mtx = trimesh.transformations.quaternion_matrix(quater)
    translation_mtx = trimesh.transformations.translation_matrix(trans)
    transformation_mtx = translation_mtx @ rotate_mtx
    mesh.apply_transform(transformation_mtx)
    return mesh

def print_transformation(transformation):
    x, y, z = transformation["translation"]
    a, b, c, d = transformation["quaternion"]
    print(f"Translation:\n{x},\n{y},\n{z}")
    print(f"Quaternion:\n{a},\n{b},\n{c},\n{d}")

def main(obj_path,transform_first,qr_coord_pixel,qr_x_pixel,image_width,image_height,z,y_fov):
    '''
    画像を処理して、3D空間の座標変換を行う関数。

    引数：
    obj_path: .objファイルのファイルパス
    transform_first: 辞書形式{'translation':[x,y,z], 'quaternion':[a,b,c,d]} 、objの初期座標変換表示
    qr_coord_pixel: QRコードの画像上の原点位置 (x, y)
    qr_x_pixel: QRコードのx軸方向に対応する画像上の位置 (x, y)
    image_width: 画像の幅
    image_height: 画像の高さ
    z: カメラから投影面までの距離
    y_fov: 垂直方向の視野角 (degree)

    戻り値：
    merged_transformation: 初期の変換と結合させた変換

    '''

    
    mesh = load_mesh_position(obj_path,transform_first)
    transform_second = calculate_qr_code_transform(mesh, qr_coord_pixel,qr_x_pixel,image_width,image_height,z,y_fov)
    if transform_second == (None,None):
        return None
    merged_transformation = multiply_transformations(transform_first,transform_second)
    return merged_transformation