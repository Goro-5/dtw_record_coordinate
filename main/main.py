import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import edge_generate
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
import transform
import image_to_coordinate as itc

# Function to load camera data from JSON
def load_camera_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def select_file():
    global output_folder_path
    output_folder_path = filedialog.askdirectory(title="Select Output Folder")
    return

def on_select_obj_files():
    global selected_obj_files
    obj_files = filedialog.askopenfilenames(title="Select OBJ Files", filetypes=[("OBJ Files", "*.obj")])
    selected_obj_files = obj_files

def on_image_click(event, index):
    global img_id
    img_id = index
    for label in camera_labels:
        label.config(borderwidth=0, relief="flat", highlightbackground="black", highlightcolor="black", highlightthickness=0)
    event.widget.config(borderwidth=2, relief="solid", highlightbackground="red", highlightcolor="red", highlightthickness=2)

def save_quaternion(quaternion_merged, id):
    output_dir = output_folder_path
    file_name = "transformation.json"
    file_name = os.path.join(output_dir, file_name)
    quaternion_data = {
        "id": id,
        "kanamono_type": object_name_list[img_id[0]],
        "position": quaternion_merged["translation"].tolist(),
        "rotation": quaternion_merged["quaternion"].tolist()
    }

    # 既存ファイルがあるかチェック
    if os.path.exists(file_name):
        # ファイルがある場合はデータを読み込む
        with open(file_name, "r") as json_file:
            data = json.load(json_file)
    else:
        # ファイルがない場合は空のリストを作成
        data = []

    # 同じIDが存在するか確認
    for i, entry in enumerate(data):
        if entry["id"] == id:
            # 同じIDがあれば上書き
            data[i] = quaternion_data
            break
    else:
        # 同じIDがなければ新しい項目を追加
        data.append(quaternion_data)

    # ファイルに書き込む
    with open(file_name, "w") as json_file:
        json.dump(data, json_file, indent=4)

def on_confirm2(current_frame):
    cv2.destroyAllWindows()
    save_quaternion(quaternion_merged,id)
    # Close the current frame
    current_frame.destroy()

    # タイトルを変更
    root.title("Next")

    # 新しいフレームの作成
    current_frame = tk.Frame(root)
    current_frame.pack(expand=True, fill="both", pady=20, padx=20)

    # ボタンの配置
    #　続けるボタン
    continue_button = tk.Button(current_frame, text="続ける", command=lambda: original_selection(current_frame))
    continue_button.pack(pady=10)

    # 終了ボタン
    exit_button = tk.Button(current_frame, text="終了", command=lambda: root.quit())
    exit_button.pack(pady=10)

def on_confirm(root, current_frame):
    global img_id

    root.attributes("-fullscreen", not root.attributes("-fullscreen"))

    if img_id is None:
        tk.messagebox.showerror("Error", "Please select an image.")
        return

    get_orientation(img_id)

    
    # Close the current frame
    current_frame.destroy()
    
    # タイトルを変更
    root.title("Finished?")

    # 新しいフレームの作成
    current_frame = tk.Frame(root)
    current_frame.pack(expand=True, fill="both", pady=20, padx=20)
    
    # これでよいかの確認ボタン
    confirm_button = tk.Button(current_frame, text="OK", command=lambda: on_confirm2(current_frame))
    confirm_button.pack(pady=10)

    # やり直しボタン
    redo_button = tk.Button(current_frame, text="処理やり直し", command=lambda: get_orientation(img_id))
    redo_button.pack(pady=10)

    # 画像選びなおしボタン
    select_image_button = tk.Button(current_frame, text="画像再選択", command=lambda: original_selection(current_frame))
    select_image_button.pack(pady=10)

def get_orientation(img_id):
    global quaternion_merged, id, cap
    quaternion_merged = None
    if cap is None:
        cv2.VideoCapture(camera_number)
        cap = cv2.VideoCapture(camera_number)
        # 画質を設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_pixel_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_pixel_height)
    
    qr_points_list = []
    qr_frame_list = []
    org_img = img_mtx[img_id[0]][img_id[1]]
    id_list = []
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('frame', cv2.resize(frame, (640, 480)))
            cv2.imshow('origin', cv2.resize(org_img, (640, 480)))
            cv2.waitKey(1)
            qr = cv2.QRCodeDetector()
            data, points, straight_qrcode = qr.detectAndDecode(frame)
            if points is not None and data is not None and data.strip() != "":
                id_list.append(data)
                points = points.astype(np.int32)
                cv2.polylines(frame, [points], True, (0, 255, 0), thickness=10)
                qr_frame_list.append(frame)
                qr_points_list.append(points)
                print(len(qr_points_list))

        if len(qr_points_list) > 0:
            areas = []
            for points in qr_points_list:
                # Calculate the area of the polygon formed by the points
                area = cv2.contourArea(points[0])
                areas.append(area)

            # Find the median area
            median_area = np.median(areas)

            # Extract the points and frames corresponding to the median area
            for i, area in enumerate(areas):
                if area == median_area:
                    qr_points=qr_points_list[i]
                    qr_frame=qr_frame_list[i]
                    break
            qr_coord,qr_x_pixel = transform.main(qr_frame,[org_img],qr_points=qr_points) #ＱＲコードの左上原点、右上ｘ軸正
            quaternion_first = quaterions_mtx[img_id[0]][img_id[1]] #オブジェクト座標系からカメラ座標系への変換行列
            quaternion_merged = itc.main(selected_obj_files[img_id[0]],quaternion_first,qr_coord,qr_x_pixel,camera_pixel_width,camera_pixel_height,camera_height,y_fov) #オブジェクト座標系からQR座標系への変換行列
            # id_listの中で最も多い要素を取得
            id = max(id_list, key=id_list.count)
            break


def on_execute_button(current_frame):
    # UIのフレームを破棄して、レンダリング中と表記
    
    current_frame.destroy()
    current_frame = tk.Frame(root)
    current_frame.pack(expand=True, fill="both", pady=20, padx=20)
    tk.Label(current_frame, text="Rendering...").pack()

    # Get the selected camera from the dropdown
    selected_camera = camera_var.get()
    if not selected_camera:
        print("No camera selected.")
        return

    # Retrieve the camera details
    camera_info = camera_data.get(selected_camera)
    if not camera_info:
        print(f"Camera {selected_camera} not found.")
        return

    # Get the selected OBJ files
    if not selected_obj_files:
        print("No OBJ files selected.")
        return
    
    if output_folder_path == "":
        print("No output folder selected.")
        return

    # Extract individual data from camera_info
    global camera_tag, camera_number, y_fov, camera_pixel_height, camera_pixel_width, camera_height, img_mtx, quaterions_mtx, object_name_list
    camera_tag = camera_info.get("camera_tag")
    camera_number = camera_info.get("camera_number")
    y_fov = camera_info.get("y_fov")
    camera_pixel_height = camera_info.get("camera_pixel_height")
    camera_pixel_width = camera_info.get("camera_pixel_width")
    camera_height = camera_info.get("camera_height")

    # Now you can use these variables in your logic
    print(f"Camera Tag: {camera_tag}")
    print(f"Y Field of View: {y_fov}")
    print(f"Camera Pixel Height: {camera_pixel_height}")
    print(f"Camera Pixel Width: {camera_pixel_width}")
    print(f"Camera Height: {camera_height}")

    img_mtx = []
    quaterions_mtx = []
    object_name_list = []

    # Process each selected OBJ file
    for obj_file in selected_obj_files:
        # Extract the object name from the file name (remove path and extension)
        object_name = os.path.splitext(os.path.basename(obj_file))[0]

        print(f"Processing {object_name}...")

        # Output the information for debugging purposes
        print(f"Rendering object '{object_name}' with the following camera data:")
        print(f"Camera Tag: {camera_tag}")

        img_list,quaterions_list = edge_generate.render_basic(obj_file,y_fov,camera_pixel_height,camera_pixel_width,camera_height)
        
        
        img_mtx.append(img_list)
        quaterions_mtx.append(quaterions_list)
        object_name_list.append(object_name)

        # Add rendering or processing logic here for each object
    original_selection(current_frame)

def original_selection(current_frame):
    current_frame.destroy()

    # タイトルを変更
    root.title("Select Image")

    # 新しいフレームの作成
    current_frame = tk.Frame(root)
    current_frame.pack(expand=True, fill="both", pady=20, padx=20)

    global camera_labels
    camera_labels = []

    # キャンバスとスクロールバーを作成
    canvas = tk.Canvas(current_frame)
    canvas.pack(side="left", fill="both", expand=True)

    # 縦スクロールバー
    v_scrollbar = tk.Scrollbar(current_frame, orient="vertical", command=canvas.yview)
    v_scrollbar.pack(side="right", fill="y")

    # 横スクロールバー
    h_scrollbar = tk.Scrollbar(current_frame, orient="horizontal", command=canvas.xview)
    h_scrollbar.pack(side="bottom", fill="x")

    canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

    # フレームコンテナをキャンバス内に作成
    frame_container = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame_container, anchor="nw")

    # 画像の配置
    for i, img_list in enumerate(img_mtx):
        for j, img in enumerate(img_list):
            cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2image = cv2.resize(cv2image, (240, 180))
            tkimg = Image.fromarray(cv2image)
            tkimg.thumbnail((240, 180))
            imgtk = ImageTk.PhotoImage(image=tkimg)

            label = tk.Label(frame_container, image=imgtk, borderwidth=0)
            label.image = imgtk  # 参照を保持するために必要
            label.grid(row=i, column=j, padx=10, pady=10)
            label.bind("<Button-1>", lambda event, idx=[i, j]: on_image_click(event, idx))
            camera_labels.append(label)

    # キャンバスのスクロール領域を更新
    frame_container.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox("all"))


    # 「決定」ボタンを画像の下に配置
    confirm_button = tk.Button(current_frame, text="決定", command=lambda: on_confirm(root, current_frame))
    confirm_button.pack(pady=10, side="bottom")
    root.attributes("-fullscreen", not root.attributes("-fullscreen"))

    print("Rendering complete.")
    cv2.destroyAllWindows()


def UI():
    global camera_data, camera_var, selected_obj_files, root, current_frame, output_folder_path, cap

    # Initialize the main application window
    root = tk.Tk()
    root.title("Edge Generation")

    output_folder_path = ""

    cap = None

    # Load camera data from JSON
    camera_data_path = filedialog.askopenfilename(title="Select Camera Data JSON", filetypes=[("JSON Files", "*.json")])
    if camera_data_path == "":
        print("No camera data selected.")
        return
    camera_data = load_camera_data(camera_data_path)

    current_frame = tk.Frame(root)
    current_frame.grid(row=0, column=0)
    # Create a UI dropdown to select the camera
    tk.Label(current_frame, text="Select Camera:").grid(row=0, column=0)
    camera_var = tk.StringVar()
    camera_options = list(camera_data.keys())
    camera_dropdown = ttk.Combobox(current_frame, textvariable=camera_var, values=camera_options, state="readonly")
    camera_dropdown.grid(row=0, column=1)

    # Fields to enter object name, pattern, and rotation
    tk.Label(current_frame, text="Object Name:").grid(row=1, column=0)
    

    # Button to select multiple obj files
    obj_files_button = tk.Button(current_frame, text="Select OBJ Files", command=on_select_obj_files)
    obj_files_button.grid(row=1, column=1)

    # Button to select output folder
    output_folder_button = tk.Button(current_frame, text="Select Output Folder", command=select_file)
    output_folder_button.grid(row=2, column=0, columnspan=2)

    # Button to trigger rendering
    render_button = tk.Button(current_frame, text="Confirm", command=lambda: on_execute_button(current_frame))
    render_button.grid(row=3, column=0, columnspan=2)

    

    # Start the GUI loop
    root.mainloop()

def main():
    global current_frame
    current_frame = None
    UI()

if __name__ == "__main__":
    main()
