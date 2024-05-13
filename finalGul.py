import cv2
import numpy as np
from customtkinter import CTk, CTkLabel, CTkButton, CTkSlider, CTkEntry, CTkFrame, CTkCheckBox
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tensorflow as tf
from rembg import remove

# Tải mô hình SVM đã được đào tạo
model = tf.keras.models.load_model('modelai.h5')
thresh = 50

class_names = [
    "Áo phông (T-shirt/top)",
    "Quần dài (Trouser)",
    "Áo len (Pullover)",
    "Đầm (Dress)",
    "Áo khoác (Coat)",
    "Sandal",
    "Áo sơ mi (Shirt)",
    "Giày thể thao (Sneaker)",
    "Túi xách (Bag)",
    "Ủng (Ankle boot)"
]


class ImageProcessor:
    def __init__(self, root):
        self.image_predict = None
        self.image_show = None
        self.root = root
        self.root.title("Canny Edge Detection")
        self.left_frame = CTkFrame(root, width=400, height=300)
        self.mid_frame = CTkFrame(root, width=400, height=300)
        self.right_frame = CTkFrame(root, width=400, height=300)

        self.left_frame.pack(side="left", fill="y")
        self.right_frame.pack(side="right", fill="y", expand=True)
        self.mid_frame.pack(side="right", fill='both', expand=True)

        self.variable_value = tk.DoubleVar()
        self.variable_value.set(50)
        self.variable_value_slider = CTkSlider(
            self.left_frame, from_=0, to=255, variable=self.variable_value)
        self.variable_value_slider.pack(pady=10)

        self.width_image_show_label = CTkLabel(self.left_frame, text="Width")
        self.width_image_show_label.pack(pady=4)
        self.width_image_show = CTkEntry(self.left_frame)
        self.width_image_show.insert(0, "100")
        self.width_image_show.pack(pady=10)

        self.height_image_show_label = CTkLabel(
            self.left_frame, text="Height")
        self.height_image_show_label.pack(pady=4)
        self.height_image_show = CTkEntry(self.left_frame)
        self.height_image_show.insert(0, "100")
        self.height_image_show.pack(pady=10)

        self.threshold1_canny_lable = CTkLabel(
            self.left_frame, text="ngưỡng dưới của canny")
        self.threshold1_canny_lable.pack(pady=4)
        self.threshold1_canny_show = CTkEntry(self.left_frame)
        self.threshold1_canny_show.insert(0, "30")
        self.threshold1_canny_show.pack(pady=10)

        self.threshold2_canny_lable = CTkLabel(
            self.left_frame, text="ngưỡng trên của canny")
        self.threshold2_canny_lable.pack(pady=4)
        self.threshold2_canny_lable_show = CTkEntry(self.left_frame)
        self.threshold2_canny_lable_show.insert(0, "100")
        self.threshold2_canny_lable_show.pack(pady=10)

        self.check_user_lib = tk.BooleanVar()
        self.check_button = CTkCheckBox(
            self.left_frame, text="Dùng thư viện để remove background", variable=self.check_user_lib)
        self.check_button.pack(pady=10)

        self.process_usecv_button = CTkButton(
            self.mid_frame, text="Thực hiện remove ảnh", command=self.use_cv)
        self.process_usecv_button.pack(pady=10)

        # Nút mở ảnh
        self.open_button = CTkButton(
            self.mid_frame, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10)

        # Nút xử lý ảnh
        self.process_button = CTkButton(
            self.mid_frame, text="Process Image", command=self.process_image)
        self.process_button.pack(pady=10)

        # Nút dự đoán ảnh
        self.predict_button = CTkButton(
            self.mid_frame, text="Predict", command=self.predict)
        self.predict_button.pack(padx=10)

        # Hiển thị ảnh gốc và sau xử lý
        self.original_image_label = CTkLabel(self.right_frame, text="")
        self.original_image_label.pack(pady=5)

        self.processed_image_label = CTkLabel(self.right_frame, text="")
        self.processed_image_label.pack(pady=5)

        self.result_label = CTkLabel(self.mid_frame, text="")
        self.result_label.pack(pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Đọc ảnh bằng thư viện PIL
            pil_image = Image.open(file_path)
            # Chuyển ảnh từ định dạng PIL Image sang NumPy array
            self.image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            self.display_image(self.image, self.original_image_label)

    def process_image(self):
        if hasattr(self, 'image'):
            # Chuyển đổi ảnh thành ảnh xám
            var = self.variable_value.get()
            gray = cv2.cvtColor(self.image_show, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            _, threshold_img = cv2.threshold(blur, var, 255, cv2.THRESH_BINARY)

            mask = 255 - threshold_img

            result = cv2.bitwise_and(self.image, self.image, mask=mask)
            self.display_image(result, self.processed_image_label)

    def display_image(self, image, label_widget):
        # Chuyển đổi mảng ảnh OpenCV thành đối tượng Image của Pillow

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (label_widget == self.processed_image_label):
            self.image_predict = image
        else:
            self.image_show = image
        width = int(self.width_image_show.get())
        height = int(self.height_image_show.get())
        image = cv2.resize(image, (width, height))
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # Hiển thị ảnh trên label
        label_widget.configure(image=image)
        label_widget.image = image

    def predict(self):
        if (self.check_user_lib.get() and self.image_show.all() != None):
            image_arr = np.array(self.image_show)
            img_removed_bg = remove(image_arr)
            # Áp dụng Gaussian Blur để làm mịn ảnh và làm giảm nhiễu
            img_removed_bg = cv2.GaussianBlur(img_removed_bg, (5, 5), 0)
            mask = img_removed_bg[:, :]
            # Thiết lập màu nền là màu đen
            black_background = np.zeros_like(
                img_removed_bg[:, :], dtype=np.uint8)

            # Gán vật thể lên màu đen sử dụng mask
            img_removed_bg = np.where(mask[:, :].astype(
                bool), img_removed_bg[:, :], black_background)
            gray_image = cv2.cvtColor(img_removed_bg, cv2.COLOR_BGR2GRAY)
            plt.subplot(121)
            plt.title("ảnh remove background")
            plt.imshow(gray_image)
            # Áp dụng Gaussian Blur để làm mịn ảnh và làm giảm nhiễu
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

            # Sử dụng phương pháp Canny để tìm biên của vật thể
            threshold1 = int(self.threshold1_canny_show.get())
            threshold2 = int(self.threshold1_canny_show.get())
            edges = cv2.Canny(blurred_image, threshold1, threshold2)

            # Tìm các đường viền trong ảnh
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Khởi tạo danh sách chứa tất cả các bounding box
            all_bboxes = []

            # Lặp qua các đường viền để tìm bounding box và thêm vào danh sách
            for contour in contours:
                # Tìm bounding box của vật thể
                x, y, w, h = cv2.boundingRect(contour)

                # Thêm bounding box vào danh sách
                all_bboxes.append((x, y, x + w, y + h))

            # Tìm tọa độ và kích thước của hình mới
            min_x = min(bbox[0] for bbox in all_bboxes)
            min_y = min(bbox[1] for bbox in all_bboxes)
            max_x = max(bbox[2] for bbox in all_bboxes)
            max_y = max(bbox[3] for bbox in all_bboxes)

            # Cắt ảnh ban đầu để tạo hình mới chứa tất cả các bounding box
            cropped_image = gray_image[min_y:max_y, min_x:max_x]
            height, width = cropped_image.shape
            m = max(height, width)
            final_image = np.zeros((m, m), dtype=np.uint8)
            start_x = (final_image.shape[1] - cropped_image.shape[1]) // 2
            start_y = (final_image.shape[0] - cropped_image.shape[0]) // 2
            final_image[start_y:start_y+cropped_image.shape[0],
                        start_x:start_x+cropped_image.shape[1]] = cropped_image

            new_image = cv2.resize(final_image, (28, 28),
                                   interpolation=cv2.INTER_AREA)
            plt.subplot(122)
            plt.imshow(new_image)
            plt.title("ảnh cuối xử lý")
            plt.show()
            image_data = tf.convert_to_tensor(new_image, dtype=tf.float32)
            image_data = tf.expand_dims(image_data, axis=0)

            prediction = model.predict(image_data)
            predicted_class = class_names[np.argmax(prediction)]
            self.result_label.configure(text=f"Dự đoán: {predicted_class}")
        else:
            Image = self.processed_image_label.cget("image")
            if Image:
                image_arr = np.array(self.image_predict)
                gray_image = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
                # Áp dụng Gaussian Blur để làm mịn ảnh và làm giảm nhiễu
                blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

                threshold1 = int(self.threshold1_canny_show.get())
                threshold2 = int(self.threshold1_canny_show.get())
                edges = cv2.Canny(blurred_image, threshold1, threshold2)

                # Tìm các đường viền trong ảnh
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Khởi tạo danh sách chứa tất cả các bounding box
                all_bboxes = []

                # Lặp qua các đường viền để tìm bounding box và thêm vào danh sách
                for contour in contours:
                    # Tìm bounding box của vật thể
                    x, y, w, h = cv2.boundingRect(contour)

                    # Thêm bounding box vào danh sách
                    all_bboxes.append((x, y, x + w, y + h))

                # Tìm tọa độ và kích thước của hình mới
                min_x = min(bbox[0] for bbox in all_bboxes)
                min_y = min(bbox[1] for bbox in all_bboxes)
                max_x = max(bbox[2] for bbox in all_bboxes)
                max_y = max(bbox[3] for bbox in all_bboxes)

                # Cắt ảnh ban đầu để tạo hình mới chứa tất cả các bounding box
                cropped_image = gray_image[min_y:max_y, min_x:max_x]
                height, width = cropped_image.shape
                m = max(height, width)
                final_image = np.zeros((m, m), dtype=np.uint8)
                start_x = (final_image.shape[1] - cropped_image.shape[1]) // 2
                start_y = (final_image.shape[0] - cropped_image.shape[0]) // 2
                final_image[start_y:start_y+cropped_image.shape[0],
                            start_x:start_x+cropped_image.shape[1]] = cropped_image

                new_image = cv2.resize(final_image, (28, 28),
                                       interpolation=cv2.INTER_AREA)
                plt.imshow(new_image)
                plt.title("ảnh cuối xử lý trước khi đưa và model")
                plt.show()
                image_data = tf.convert_to_tensor(new_image, dtype=tf.float32)
                image_data = tf.expand_dims(image_data, axis=0)

                prediction = model.predict(image_data)
                predicted_class = class_names[np.argmax(prediction)]
                self.result_label.configure(text=f"Dự đoán: {predicted_class}")

    def onTrackbarChange(self, value):
        global thresh
        thresh = value
        print("Variable value:", thresh)

    def valueScaling(self, value):
        min_value = 0
        max_value = 255
        new_min = 0
        new_max = 255
        scaled_value = (value - min_value) * (new_max - new_min) / \
            (max_value - min_value) + new_min
        return int(scaled_value)

    def use_cv(self):
        if (self.image_show.all() != None):
            image_arr = np.array(self.image_show)
            image = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
            desired_width = 400
            aspect_ratio = image.shape[1] / image.shape[0]
            desired_height = int(desired_width / aspect_ratio)
            resized_image = cv2.resize(image, (desired_width, desired_height))
            window_name = 'Background Removed'
            scaled_thresh = self.valueScaling(thresh)
            cv2.namedWindow(window_name)

            cv2.createTrackbar('thresh', window_name,
                               scaled_thresh, 255, self.onTrackbarChange)
            while True:
                gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

                blur = cv2.GaussianBlur(gray, (5, 5), 0)

                _, threshold_img = cv2.threshold(
                    blur, thresh, 255, cv2.THRESH_BINARY)

                mask = 255 - threshold_img

                result = cv2.bitwise_and(
                    resized_image, resized_image, mask=mask)

                cv2.imshow(window_name, result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                elif cv2.waitKey(1) & 0xFF == 13:
                    self.display_image(result, self.processed_image_label)
                    break
            cv2.destroyAllWindows()


if __name__ == "__main__":
    root = CTk()
    app = ImageProcessor(root)
    root.mainloop()
