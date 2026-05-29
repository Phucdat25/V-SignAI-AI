# 1. Khởi tạo service khi start server (Chỉ làm 1 lần duy nhất để nạp model)
# service = VSignExtractorAndPredictor(
#     onnx_model_path="đường_dẫn_đến/spoter_model.onnx", 
#     gloss_to_id_path="đường_dẫn_đến/gloss_to_id.json"
# )

# 2. Khi có request từ client gửi video lên, gọi hàm này để nhận diện:
# result = service.predict_sentence("video_input.mp4", stride=15, confidence_threshold=70.0)
# print(result["final_sentence"]) # Kết quả dạng chuỗi: "XIN CHÀO -> CẢM ƠN"

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
        """
        Khởi tạo hệ thống: Nạp Model ONNX, Từ điển và Cấu hình MediaPipe DUY NHẤT 1 LẦN.
        """
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

    def _extract_frame_landmarks(self, results):
        """Trích xuất 54 điểm và nhân cứng với không gian 1080x1080"""
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
        return extracted_points * 1080

    def extract_features_from_video(self, video_path: str) -> np.ndarray:
        """Đọc video và chuyển thành mảng landmarks trong bộ nhớ RAM (Không ghi file disk)"""
        cap = cv2.VideoCapture(str(video_path))
        video_sequences = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.holistic.process(frame_rgb)
            
            frame_features = self._extract_frame_landmarks(results)
            video_sequences.append(frame_features)
            
        cap.release()
        return np.array(video_sequences, dtype=np.float32)

    def predict_sentence(self, video_path: str, stride: int = 15, confidence_threshold: float = 70.0) -> dict:
        """
        Hàm chính: Nhận đường dẫn video -> Trích xuất đặc trưng -> Chạy cửa sổ trượt -> Trả kết quả JSON
        """
        # Bước 1: Trích xuất đặc trưng trực tiếp vào RAM
        data = self.extract_features_from_video(video_path)
        if len(data) == 0:
            return {"status": "error", "message": "Video không có dữ liệu hoặc không đọc được."}
            
        total_frames = data.shape[0]
        window_size = self.TARGET_FRAMES
        raw_results = []

        # Bước 2: Thuật toán Cửa sổ trượt (Sliding Window)
        if total_frames < window_size:
            indices = np.linspace(0, total_frames - 1, self.TARGET_FRAMES).astype(int)
            window_data = data[indices]
            window_data = window_data.reshape(self.TARGET_FRAMES, -1)
            input_data = np.expand_dims(window_data, axis=0).astype(np.float32)

            logits = self.session.run([self.output_name], {self.input_name: input_data})[0]
            predicted_id = np.argmax(logits, axis=1)[0]
            probs = np.exp(logits[0] - np.max(logits[0]))
            probs /= np.sum(probs)
            
            raw_results.append({
                "start": 0, "end": total_frames,
                "gloss": self.id2label.get(predicted_id, "Unknown"),
                "confidence": float(np.max(probs) * 100)
            })
        else:
            for start_idx in range(0, total_frames - window_size + 1, stride):
                end_idx = start_idx + window_size
                window_data = data[start_idx:end_idx]

                window_data = window_data.reshape(self.TARGET_FRAMES, -1)
                input_data = np.expand_dims(window_data, axis=0).astype(np.float32)

                logits = self.session.run([self.output_name], {self.input_name: input_data})[0]
                predicted_id = np.argmax(logits, axis=1)[0]

                probs = np.exp(logits[0] - np.max(logits[0]))
                probs /= np.sum(probs)
                confidence = float(np.max(probs) * 100)
                predicted_gloss = self.id2label.get(predicted_id, "Unknown")

                if confidence >= confidence_threshold:
                    raw_results.append({
                        "start": int(start_idx), "end": int(end_idx),
                        "gloss": predicted_gloss, "confidence": confidence
                    })

        # Bước 3: Lọc nhiễu & Gộp từ liên tiếp
        final_sentence = []
        last_gloss = None
        for res in raw_results:
            if res['gloss'] != last_gloss:
                final_sentence.append(res['gloss'].upper())
                last_gloss = res['gloss']

        return {
            "status": "success",
            "total_frames": total_frames,
            "windows": raw_results,
            "final_sentence": " -> ".join(final_sentence) if final_sentence else ""
        }

    def close(self):
        self.holistic.close()