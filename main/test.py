import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import trimesh
import pyrender
from PIL import Image, ImageTk
import numpy as np

class InteractiveViewer:
    def __init__(self, canvas, info_label):
        self.canvas = canvas
        self.info_label = info_label
        self.scene = pyrender.Scene()
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self.camera_pose = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, -300.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.camera_node = pyrender.Node(camera=self.camera, matrix=self.camera_pose, name='camera')
        self.scene.add_node(self.camera_node)
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.scene.add(self.light, pose=self.camera_pose)
        self.renderer = pyrender.OffscreenRenderer(800, 600)
        self.mesh_node = None
        self.last_x = 0
        self.last_y = 0
        self.canvas.bind("<B1-Motion>", self.on_rotate_drag)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<Button-1>", self.on_mouse_click)
        self.canvas.bind("<Button-2>", self.on_mouse_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.update_info_label()

    def move_mesh_tocenter(self, mesh):
        # Move mesh center point to the origin
        centroid = mesh.vertices.mean(axis=0)
        translation_vector = -centroid
        mesh.vertices += translation_vector
        print("Translated vertices:", mesh.vertices)
        
        # Reset camera to focus on the mesh
        self.reset_camera(mesh)
        self.render()

    def reset_camera(self, mesh):
        # Reset the camera position to focus on the mesh
        camera_distance = np.max(np.abs(mesh.vertices))
        self.camera_pose[:3, 3] = np.array([0.0, 0.0, -camera_distance * 2.0])
        self.scene.set_pose(self.camera_node, pose=self.camera_pose)

    def load_mesh(self, mesh):
        if self.mesh_node:
            self.scene.remove_node(self.mesh_node)
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.0, 0.0, 0.0, 1.0])  # Red color
        self.mesh_node = pyrender.Mesh.from_trimesh(mesh)
        self.scene.add(self.mesh_node)
        
        '''# meshのConvex Hullを描画
        convex_hull = mesh.convex_hull
        convex_hull_mesh = pyrender.Mesh.from_trimesh(convex_hull)
        self.scene.add(convex_hull_mesh)

        vertices = convex_hull.vertices
        faces = convex_hull.faces
        all_faces = []

        for face in faces:
            face_vertices = vertices[face]
            
            # 面の中心を計算
            center = face_vertices.mean(axis=0)
            
            # 面の法線ベクトルを計算
            normal = np.cross(face_vertices[1] - face_vertices[0], face_vertices[2] - face_vertices[0])
            normal = normal / np.linalg.norm(normal)
            
            # 面積を計算
            if face_vertices.shape[0] == 3:
                area = trimesh.triangles.area(face_vertices[np.newaxis, :, :]).sum()
            else:
                area = trimesh.triangles.area(face_vertices).sum()

            all_faces.append((center, normal, area))

        cyliners = self.faces_to_cylinders(all_faces)
        for cylinder in cyliners:
            cylinder_mesh = pyrender.Mesh.from_trimesh(cylinder)
            self.scene.add(cylinder_mesh)'''

        
        large_faces = self.find_large_base_faces(mesh, coefficient=0.2)
        cylinders = self.faces_to_cylinders(large_faces)
        
        # シリンダーをシーンに追加
        for cylinder in cylinders:
            cylinder_mesh = pyrender.Mesh.from_trimesh(cylinder)
            self.scene.add(cylinder_mesh)

        # Move the mesh to the center
        self.move_mesh_tocenter(mesh)
        
        self.render()
    
    def render(self):
        color, _ = self.renderer.render(self.scene)
        self.display_image(color)
        self.update_info_label()

    def display_image(self, image):
        image = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.image = imgtk  # Keep a reference to avoid garbage collection

    def on_mouse_click(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def on_rotate_drag(self, event):
        self.arcball_rotate(event.x, event.y)
        self.render()

    def on_pan_drag(self, event):
        dx = event.x - self.last_x
        dy = event.y - self.last_y
        self.last_x = event.x
        self.last_y = event.y
        self.pan_camera(-dx, -dy)
        self.render()

    def on_mouse_wheel(self, event):
        self.dolly_camera(event.delta)
        self.render()

    def arcball_rotate(self, x, y):
        def map_to_sphere(x, y):
            width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
            x = (2.0 * x - width) / width
            y = (height - 2.0 * y) / height
            z = 0.0
            length = x * x + y * y
            if length > 1.0:
                norm = 1.0 / np.sqrt(length)
                x *= norm
                y *= norm
            else:
                z = np.sqrt(1.0 - length)
            return np.array([x, y, z])

        last_pos = map_to_sphere(self.last_x, self.last_y)
        curr_pos = map_to_sphere(x, y)
        self.last_x = x
        self.last_y = y

        axis = np.cross(last_pos, curr_pos)
        angle = np.arccos(np.clip(np.dot(last_pos, curr_pos), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-6:
            return

        axis = axis / np.linalg.norm(axis)
        rotation = self.axis_angle_to_matrix(axis, angle)
        self.camera_pose[:3, :3] = rotation @ self.camera_pose[:3, :3]
        self.scene.set_pose(self.camera_node, pose=self.camera_pose)

    def axis_angle_to_matrix(self, axis, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c
        x, y, z = axis
        return np.array([
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]
        ])

    def pan_camera(self, dx, dy):
        right = self.camera_pose[:3, 0]
        up = self.camera_pose[:3, 1]
        translation = np.eye(4)
        translation[:3, 3] = right * dx * 0.05 - up * dy * 0.05  # Increase sensitivity and invert vertical movement
        self.camera_pose = translation @ self.camera_pose
        self.scene.set_pose(self.camera_node, pose=self.camera_pose)

    def dolly_camera(self, delta):
        forward = self.camera_pose[:3, 2]
        translation = np.eye(4)
        translation[:3, 3] = forward * delta * 0.01
        self.camera_pose = translation @ self.camera_pose
        self.scene.set_pose(self.camera_node, pose=self.camera_pose)

    def update_info_label(self):
        position = self.camera_pose[:3, 3]
        rotation = self.camera_pose[:3, :3]
        self.info_label.config(text=f"Position: {position}\nRotation:\n{rotation}")


    def generate_cylinder(self, center, normal, area):
        if area <= 0:
            raise ValueError("Area must be positive")
        # 円柱の半径は1とする
        radius = 1.0
        # 高さを面積の1/2乗で計算
        height = area ** 0.5
        # 円柱を生成 (Z軸方向に高さが伸びる円柱)
        cylinder = trimesh.creation.cylinder(radius=radius, height=height)
        
        # 円柱の法線を指定された法線に合わせて回転させる
        z_axis = np.array([0, 0, 1])
        rotation_matrix = trimesh.geometry.align_vectors(z_axis, normal)
        cylinder.apply_transform(rotation_matrix)
        
        # 円柱の中心が面の中心座標に一致するように平行移動
        # ここでは、円柱の底面を中心に合わせるため、高さの半分だけZ方向にシフト
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = center + normal * (height / 2.0)
        cylinder.apply_transform(translation_matrix)
        
        return cylinder
    
    def faces_to_cylinders(self, merged_faces):
        cylinders = []
        
        for i in range(len(merged_faces)):
            center, normal, area = merged_faces[i]
            print(f"Center: {center}, Normal: {normal}, Area: {area}")
            cylinders.append(self.generate_cylinder(center, normal, area))
        return cylinders


    def find_large_base_faces(self, mesh, coefficient=1.0):
        # 凸包の計算
        convex_hull = mesh.convex_hull
        # 凸包の体積を計算
        volume = convex_hull.volume
        
        # 体積の2/3乗に係数を掛けた面積の閾値を計算
        area_threshold = coefficient * (volume ** (2.0 / 3.0))
        print(f"閾値面積: {area_threshold}")
        
        merged_faces, faces_area, faces_normal = self.merge_faces_on_same_plane(mesh)
        large_faces = []
        for i in range(len(merged_faces)):
            if faces_area[i] > area_threshold:
                large_faces.append((merged_faces[i].mean(axis=0), faces_normal[i], faces_area[i]))
        
        return large_faces
    
    def are_faces_on_same_plane(self, center_i, normal_i, face_j_vertices):
        # face_jの各頂点からcenter_iへのベクトルv_ijを計算
        for vertex in face_j_vertices:
            v_ij = vertex - center_i
            
            # 内積を計算
            dot_product = np.dot(v_ij, normal_i)
            
            # 内積が0でない場合、同一平面上にない
            if not np.isclose(dot_product, 0):
                return False
        return True

    def merge_faces_on_same_plane(self, mesh):
        # 凸包の面法線、面、頂点を取得
        hull = mesh.convex_hull
        normals = hull.face_normals
        faces = hull.faces
        vertices = hull.vertices
        
        # 同一平面上の面を結合するためのリスト
        merged_faces = []
        faces_area = []
        faces_normal = []
        visited = np.zeros(len(faces), dtype=bool)
        
        # 各面についてループし、同一平面上の面を探す
        for i in range(len(faces)):
            if visited[i]:
                continue
            
            # face_iの中心を計算
            face_i_vertices = vertices[faces[i]]
            center_i = face_i_vertices.mean(axis=0)
            normal_i = normals[i]
            # 面積を計算
            if face_i_vertices.shape[0] == 3:
                face_area = trimesh.triangles.area(face_i_vertices[np.newaxis, :, :])
                face_area = face_area.sum()
            else:
                face_area = trimesh.triangles.area(face_i_vertices)
            
            # 同一平面上の面を保存するリスト
            group_faces = [i]
            visited[i] = True
            
            # 他の面と比較して、同一平面上にあるかを確認
            for j in range(i + 1, len(faces)):
                if not visited[j]:
                    face_j_vertices = vertices[faces[j]]
                    
                    # 内積がすべて0であれば同一平面上にあるとみなす
                    if self.are_faces_on_same_plane(center_i, normal_i, face_j_vertices):
                        if face_j_vertices.shape[0] == 3:
                            area_j = trimesh.triangles.area(face_j_vertices[np.newaxis, :, :])
                            face_area += area_j.sum()
                        else:
                            face_area += trimesh.triangles.area(face_j_vertices)
                        group_faces.append(j)
                        visited[j] = True
            
            # グループ内の面を結合
            combined_vertices = np.vstack([vertices[faces[f]] for f in group_faces])
            combined_vertices = np.unique(combined_vertices, axis=0) # 重複する頂点を削除
            faces_normal.append(normal_i)
            faces_area.append(face_area)
            merged_faces.append(combined_vertices)
        
        return merged_faces, faces_area, faces_normal
    
    def calculate_face_properties(self, merged_faces):
        faces_area = []
        faces_normal = []

        for face_vertices in merged_faces:
            # 頂点の数が3つ未満ならスキップ
            if len(face_vertices) < 3:
                print(f"Warning: face with vertices {face_vertices} has less than 3 vertices")
                continue

            # 面の三角形分割を行う
            face_trimesh = trimesh.Trimesh(vertices=face_vertices, process=True)

            # 面積の計算
            area = face_trimesh.area
            faces_area.append(area)

            # 法線ベクトルの計算
            if face_trimesh.face_normals.size > 0:
                normal = face_trimesh.face_normals.mean(axis=0)
                normal = normal / np.linalg.norm(normal)  # 正規化
                faces_normal.append(normal)
            else:
                print(f"Warning: face_trimesh.face_normals is empty for face with vertices {face_vertices}, area is {area}")

        return faces_area, faces_normal
    



def load_obj(file_path):
    return trimesh.load(file_path)

def on_load_button():
    file_path = filedialog.askopenfilename(filetypes=[("OBJ files", "*.obj")])
    if not file_path:
        messagebox.showerror("Error", "No OBJ files selected.")
        return

    mesh = load_obj(file_path)
    print(f"Loaded mesh from file: {file_path}")
    viewer.load_mesh(mesh)

def UI():
    global viewer
    root = tk.Tk()
    root.title("3D Model Viewer")

    load_button = ttk.Button(root, text="Load OBJ File", command=on_load_button)
    load_button.pack(pady=20)

    canvas = tk.Canvas(root, width=800, height=600)
    canvas.pack()

    info_label = tk.Label(root, text="", justify=tk.LEFT)
    info_label.pack(pady=10)

    viewer = InteractiveViewer(canvas, info_label)

    root.mainloop()

if __name__ == "__main__":
    UI()