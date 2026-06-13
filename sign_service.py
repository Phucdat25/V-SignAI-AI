# # Khởi tạo mô hình VSignExtractorAndPredictor4
# isolated_service = VSignExtractorAndPredictor4(
#     onnx_model_path="/content/ctrgcn_model.onnx",
#     gloss_to_id_path="/content/gloss_to_id.json"
# )

# # Chạy thử dự đoán trên video
# print("Đang dự đoán từ đơn...")
# result4 = isolated_service.predict_isolated_word("/content/Pro.mp4")
# print(result4['predicted_gloss'])
# # import pprint
# # pprint.pprint(result4)


import os
import json
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class VSignExtractorAndPredictor:
    def __init__(self, onnx_model_path: str, gloss_to_id_path: str, min_conf=0.5):
        # 1. Khởi tạo MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_conf,
            min_tracking_confidence=min_conf,
            smooth_landmarks=True
        )

        # Chỉ số các điểm cần lấy (54 điểm)
        self.pose_indices = [0, 2, 5, 7, 11, 12, 13, 14, 15, 16, 23, 24]
        self.hand_indices = list(range(501, 543))
        self.selected_indices = self.pose_indices + self.hand_indices

        # 2. Khởi tạo ONNX Runtime Session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 3. Tải từ điển Map ID sang Gloss
        with open(gloss_to_id_path, 'r', encoding='utf-8') as f:
            gloss_to_id = json.load(f)
        self.id2label = {v: k for k, v in gloss_to_id.items()}

        self.TARGET_FRAMES = 70
        self.NUM_POINTS = 54
        self.IN_CHANNELS = 2
        
        # Cấu hình cửa sổ trượt
        self.WINDOW_SIZE = 70
        self.STEP = 10

    def _extract_frame_landmarks(self, results):
        """Trích xuất 54 điểm"""
        full_landmarks = np.zeros((543, 2), dtype=np.float32)

        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                full_landmarks[idx] = [lm.x, lm.y]

        if results.left_hand_landmarks:
            for idx, lm in enumerate(results.left_hand_landmarks.landmark):
                full_landmarks[501 + idx] = [lm.x, lm.y]

        if results.right_hand_landmarks:
            for idx, lm in enumerate(results.right_hand_landmarks.landmark):
                full_landmarks[522 + idx] = [lm.x, lm.y]

        extracted_points = full_landmarks[self.selected_indices]
        return extracted_points

    def _trim_and_fill_skeleton(self, all_frames):
        """
        Tiền xử lý skeleton:
        1. Cắt bỏ frame đầu/cuối nếu không xuất hiện đủ tọa độ.
        2. Fill tọa độ bị khuyết (copy từ frame trước/sau).
        """
        data = np.array(all_frames, dtype=np.float32)
        if len(data) == 0:
            return data

        num_frames, num_points, channels = data.shape

        # 1. Tìm các frame có dữ liệu điểm tay hợp lệ (tay bắt đầu từ index 12 đến 53)
        hand_valid = []
        for f in range(num_frames):
            # Nếu có bất kỳ điểm tay nào khác 0.0 -> Hợp lệ
            if np.any(data[f, 12:] != 0.0):
                hand_valid.append(True)
            else:
                hand_valid.append(False)

        valid_indices = np.where(hand_valid)[0]
        if len(valid_indices) > 0:
            # Cắt bỏ các frame tĩnh ở đầu và cuối
            start_idx = valid_indices[0]
            end_idx = valid_indices[-1]
            data = data[start_idx:end_idx + 1]
        else:
            # Không có tay trong toàn bộ video
            return np.array([])

        # Cập nhật lại số frame sau khi cắt
        num_frames = data.shape[0]

        # 2. Fill các điểm mất tracking (0.0 hoặc NaN)
        for f in range(num_frames):
            for p in range(num_points):
                if (data[f, p, 0] == 0.0 and data[f, p, 1] == 0.0) or np.isnan(data[f, p, 0]):
                    if f > 0:
                        # Lấy từ frame trước (Forward-fill)
                        data[f, p] = data[f - 1, p]
                    else:
                        # Nếu là frame đầu tiên, lấy từ frame sau (Backward-fill)
                        for next_f in range(f + 1, num_frames):
                            if not (data[next_f, p, 0] == 0.0 and data[next_f, p, 1] == 0.0):
                                data[f, p] = data[next_f, p]
                                break

        return data

    def predict_isolated_word(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        all_frames = []

        # Đọc toàn bộ frame và trích xuất điểm
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.holistic.process(frame_rgb)

            features = self._extract_frame_landmarks(results)
            all_frames.append(features)

        cap.release()

        # Tiền xử lý (Cắt frame thừa và Fill)
        processed_data = self._trim_and_fill_skeleton(all_frames)

        if len(processed_data) == 0:
            return {"status": "error", "message": "Không tìm thấy chuyển động tay rõ ràng trong video."}

        total_frames = processed_data.shape[0]
        best_gloss = "Unknown"
        best_conf = -1.0

        # Chuẩn bị cửa sổ trượt
        windows = []
        if total_frames <= self.WINDOW_SIZE:
            # Nếu tổng frame nhỏ hơn 70, nội suy (scale) để đủ 70
            indices = np.linspace(0, total_frames - 1, self.WINDOW_SIZE).astype(int)
            windows.append(processed_data[indices])
        else:
            # Sliding window: Size 70, Step 10
            for start in range(0, total_frames - self.WINDOW_SIZE + 1, self.STEP):
                windows.append(processed_data[start : start + self.WINDOW_SIZE])

            # Thêm window cuối cùng nếu bước nhảy bỏ sót
            if (total_frames - self.WINDOW_SIZE) % self.STEP != 0:
                windows.append(processed_data[-self.WINDOW_SIZE:])

        # Dự đoán và lấy Conf lớn nhất
        for w_data in windows:
            input_data = np.expand_dims(w_data, axis=0).astype(np.float32)
            logits = self.session.run([self.output_name], {self.input_name: input_data})[0]

            # Tính Softmax -> Confidence
            probs = np.exp(logits[0] - np.max(logits[0]))
            probs /= np.sum(probs)
            confidence = float(np.max(probs) * 100)

            if confidence > best_conf:
                best_conf = confidence
                predicted_id = np.argmax(logits, axis=1)[0]
                best_gloss = self.id2label.get(predicted_id, "Unknown")

        return {
            "status": "success",
            "predicted_gloss": best_gloss,
            "confidence": best_conf,
            "total_processed_frames": total_frames,
            "total_windows_checked": len(windows)
        }

    def close(self):
        self.holistic.close()
