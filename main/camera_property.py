import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import numpy as np
import math
from PIL import Image, ImageTk
import json
import camera_calibration
import os

current_frame = None

def get_camera_cap(camera_number, width):
    cap = cv2.VideoCapture(camera_number)
    default_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    default_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect_ratio = default_width / default_height
    height = int(width / aspect_ratio)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

# ボタンが押された時の動作
def submit_action(entry_height, entry_size, entry_camera):
    height = entry_height.get()
    size = entry_size.get()
    name = entry_camera.get()

    try:
        # 入力された値をfloatに変換し、メッセージで表示
        height = int(height)
        size = float(size)
        name = str(name)
        messagebox.showinfo("入力内容", f"カメラ名:{name}\n高さ: {height} mm, 大きさ: {size} mm")
        return height, size, name
    except ValueError:
        messagebox.showerror("エラー", "有効な数値を入力してください。")
        return None, None, None



def process(entry_height, entry_size, entry_camera):
    global root, current_frame, camera_height, qr_size,camera_tags
    height, size, name = submit_action(entry_height, entry_size, entry_camera)
    if height is not None and size is not None:
        if current_frame is not None:
            current_frame.destroy()
        select_camera(root)
        camera_height, qr_size, camera_tags = height, size, name
        

    

def on_image_click(event, index):
    global camera_number
    camera_number = index
    for label in camera_labels:
        label.config(borderwidth=0, relief="flat", highlightbackground="black", highlightcolor="black", highlightthickness=0)
    event.widget.config(borderwidth=2, relief="solid", highlightbackground="red", highlightcolor="red", highlightthickness=2)


def show_camera_images():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        elif not cap.isOpened():
            print(f"Camera index {index} out of range")
            continue
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img.thumbnail((400, 300))
            imgtk = ImageTk.PhotoImage(image=img)

            label = tk.Label(frame_container,image=imgtk, borderwidth=0)
            label.image = imgtk  # この行が重要です。参照を保持します。
            label.grid(row=0, column=index, padx=10, pady=10)
            
            label.bind("<Button-1>", lambda event, idx=index: on_image_click(event, idx))

            camera_labels.append(label)
        index += 1

        cap.release()



def on_confirm(root, frame):
    frame.destroy()
    root.quit()

def select_camera(root):
    global current_frame, camera_labels, frame_container
    # 新しいフレームを作成して配置
    current_frame = tk.Frame(root)
    current_frame.pack(expand=True, fill="both", pady=20, padx=20)

    camera_labels = []

    
    frame_container = tk.Frame(current_frame)
    frame_container.pack()
    label = tk.Label(current_frame, text="カメラを選択してください")
    label.pack(pady=10)

    show_camera_images()

    confirm_button = tk.Button(current_frame, text="決定", command=lambda: on_confirm(root, current_frame))
    confirm_button.pack(pady=10)




def input():
    global root, current_frame, camera_height, qr_size, camera_number, camera_tags

    camera_height = None
    qr_size = None
    camera_number = None

    # ウィンドウの設定
    root = tk.Tk()
    root.title("入力フォーム")

    # メインフレームの作成
    current_frame = tk.Frame(root)
    current_frame.pack(pady=20)

    # カメラ名の入力フィールドとラベルを一つのフレームにまとめる
    frame_camera = tk.Frame(current_frame)
    label_camera = tk.Label(frame_camera, text="カメラ名前:")   
    label_camera.pack(side=tk.LEFT)
    entry_camera = tk.Entry(frame_camera)
    entry_camera.pack(side=tk.LEFT)
    frame_camera.pack(pady=5)

    # 高さの入力フィールドと「mm」ラベルを一つのフレームにまとめる
    frame_height = tk.Frame(current_frame)
    label_height = tk.Label(frame_height, text="カメラ高さ:")
    label_height.pack(side=tk.LEFT)
    entry_height = tk.Entry(frame_height)
    entry_height.insert(0, "350")  # 初期値を350に設定
    entry_height.pack(side=tk.LEFT)
    label_height_unit = tk.Label(frame_height, text=" mm")
    label_height_unit.pack(side=tk.LEFT)
    frame_height.pack(pady=5)

    # 大きさの入力フィールドと「mm」ラベルを一つのフレームにまとめる
    frame_size = tk.Frame(current_frame)
    label_size = tk.Label(frame_size, text="QRコード1辺長さ:")
    label_size.pack(side=tk.LEFT)
    entry_size = tk.Entry(frame_size)
    entry_size.insert(0, "40.0")  # 初期値を40.0に設定
    entry_size.pack(side=tk.LEFT)
    label_size_unit = tk.Label(frame_size, text=" mm")
    label_size_unit.pack(side=tk.LEFT)
    frame_size.pack(pady=5)

    # 送信ボタン
    submit_button = tk.Button(current_frame, text="送信", command=lambda: process(entry_height, entry_size, entry_camera))
    submit_button.pack(pady=10)

    # ウィンドウを表示
    root.mainloop()


    return camera_height, qr_size, camera_number, camera_tags

def get_camera_info(camera_number, height, size):
    cap = get_camera_cap(camera_number, 5000)
    if not cap.isOpened():
        messagebox.showerror("エラー", "カメラが開けません")
        return None, None, None
    qr = cv2.QRCodeDetector()
    y_fov_list = []
    count = 0
    while True:
        ret, frame = cap.read()
        # Show camera output on screen
        if ret:

            # If QR code is not detected, show warning message on screen
            if not qr_code_detected(frame,qr):
                show_frame = cv2.resize(frame, (640, 480))
                cv2.putText(show_frame, "QR code UNDETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Camera Output", show_frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    return None
            # Perform QR code detection on the frame
            # If QR code is detected, break the loop and return the cap
            elif qr_code_detected(frame,qr):
                data, points, straight_qrcode = qr.detectAndDecode(frame)
                if points is not None and len(points) > 0:
                    points = points.astype(np.int32)
                    # QRコードを検出した位置に四角形を描画
                    cv2.polylines(frame, [points], True, (0, 255, 0), thickness=5)
                    show_frame = cv2.resize(frame, (640, 480))
                    cv2.putText(show_frame, "DETECTED!!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Camera Output", show_frame)
                    y_fov = calculate_y_fov(size, frame, height, points)
                    y_fov_list.append(y_fov)
                    count += 1
                    if count >= 10:
                        y_fov_median = np.median(y_fov_list)
                        return frame.shape[0], frame.shape[1], y_fov_median
        else:
            messagebox.showerror("エラー", "カメラが開けません")
        

def qr_code_detected(frame,qr):
    data, points, straight_qrcode = qr.detectAndDecode(frame)
    if points is not None and data is not None and data.strip() != "":
        print(data)
        return True
    else:
        return False

def calculate_y_fov(qrcode_width, frame, distance, points):
    
    
    camera_pixel_height = frame.shape[0]  # カメラ画像の縦解像度
    camera_pixel_width = frame.shape[1]   # カメラ画像の横解像度
    
    if points is not None:
        # QRコードの正方形のピクセル上での面積を計算
        qr_area = cv2.contourArea(points)

        # QRコードの縦横比を1（正方形）として、高さのピクセル数を計算
        qr_height_px = math.sqrt(qr_area)
        
        # 画像中のQRコードの高さ (qr_height_px) が、カメラの全縦ピクセル (camera_pixel_height) に対してどれだけの割合を占めるかを計算
        qr_height_ratio = qr_height_px / camera_pixel_height
        
        # カメラの視野の中でQRコードが占める高さを基に、物体の実際の高さとFOVの関係を計算
        # 縦方向のFOVを計算 (物体の物理的な高さとピクセルの割合を使用)
        y_fov = 2 * math.atan((qrcode_width / qr_height_ratio) / (2 * distance))
        
        
        
        return y_fov
    else:
        # QRコードが検出されなかった場合はNoneを返す
        messagebox.showerror("エラー", "QRコードが検出されませんでした")
        return None

def save_data(camera_name, camera_index, y_fov, camera_pixel_height, camera_pixel_width, camera_height):
    camera_data_path = filedialog.askdirectory(title="Select Output Directory")
    camera_data_file = os.path.join(camera_data_path, "camera_data.json")
    data = {
        "camera_tag": camera_name,
        "camera_number": camera_index,
        "y_fov": y_fov,
        "camera_pixel_height": camera_pixel_height,
        "camera_pixel_width": camera_pixel_width,
        "camera_height": camera_height
    }
    try:
        with open(camera_data_file, "r") as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        existing_data = {}
    
    tags = existing_data.get("tags", [])
    if camera_name not in tags:
        tags.append(camera_name)
    existing_data["tags"] = tags
    existing_data[camera_name] = data
    
    with open(camera_data_file, "w") as file:
        json.dump(existing_data, file)

def main():
    height,size,camera_index,name = input()
    if camera_index is not None:
        camera_pixel_height,camera_pixel_width,y_fov = get_camera_info(camera_index,height,size)
        y_fov_degrees = math.degrees(y_fov)
        messagebox.showinfo("処理結果", f"縦視野角: {y_fov_degrees} °\nカメラの画素数: {camera_pixel_width} x {camera_pixel_height}")
        color = camera_calibration.render_img([size,size],height,y_fov,resolution=(camera_pixel_width,camera_pixel_height))
        color = cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)
        resize = cv2.resize(color,(640,480))
        cv2.imshow("sample",resize)
        save_data(name, camera_index, y_fov, camera_pixel_height, camera_pixel_width, height)


if __name__ == "__main__":
    main()