# 1. Khởi tạo service khi start server (Chỉ làm 1 lần duy nhất để nạp model)
# service = VSignExtractorAndPredictor(
#     onnx_model_path="đường_dẫn_đến/ctrgcn_model.onnx", 
#     gloss_to_id_path="đường_dẫn_đến/gloss_to_id.json"
# )

# 2. Khi có request từ client gửi video lên, gọi hàm này để nhận diện:
# result = service.predict_sentence("video_input.mp4", confidence_threshold=50.0)
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
        self.NUM_POINTS = 54 # Thêm thuộc tính này để sử dụng khi reshape
        self.IN_CHANNELS = 2 # Thêm thuộc tính này để sử dụng khi reshape

    def _extract_frame_landmarks(self, results):
        """Trích xuất 54 điểm, KHÔNG nhân với 1080 để giữ nguyên chuẩn hóa 0-1 khớp với lúc train"""
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

    def _is_active_frame(self, results):
        """Kiểm tra xem tay có đang được đưa lên để thực hiện ngôn ngữ ký hiệu không"""
        if not results.pose_landmarks:
            return False

        # Nếu phát hiện được bàn tay, coi như đang hoạt động
        if results.left_hand_landmarks or results.right_hand_landmarks:
            return True

        # Kiểm tra vị trí cổ tay so với hông (trục Y hướng xuống dưới)
        left_wrist = results.pose_landmarks.landmark[15]
        right_wrist = results.pose_landmarks.landmark[16]
        left_hip = results.pose_landmarks.landmark[23]
        right_hip = results.pose_landmarks.landmark[24]

        active = False
        # Cổ tay cao hơn hông (y nhỏ hơn) và có độ tin cậy nhất định
        if left_wrist.visibility > 0.5 and left_wrist.y < left_hip.y:
            active = True
        if right_wrist.visibility > 0.5 and right_wrist.y < right_hip.y:
            active = True

        return active

    def extract_features_and_segments(self, video_path: str):
        """Đọc video, trích xuất features và gom thành các đoạn (segments) dựa trên cử động tay"""
        cap = cv2.VideoCapture(str(video_path))
        segments = []
        current_segment = []

        is_active = False
        empty_frames = 0
        MAX_EMPTY = 10  # Số frame rỗng tối đa cho phép trước khi cắt segment
        MIN_SEGMENT_LEN = 15 # Chiều dài tối thiểu của 1 đoạn

        start_frame = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.holistic.process(frame_rgb)

            frame_features = self._extract_frame_landmarks(results)
            active_now = self._is_active_frame(results)

            if active_now:
                if not is_active:
                    is_active = True
                    start_frame = frame_count
                    current_segment = []
                empty_frames = 0
                current_segment.append(frame_features)
            else:
                if is_active:
                    empty_frames += 1
                    if empty_frames < MAX_EMPTY:
                        current_segment.append(frame_features)
                    else:
                        # Kết thúc đoạn
                        if len(current_segment) >= MIN_SEGMENT_LEN:
                            segments.append({
                                'start': start_frame,
                                'end': frame_count,
                                'data': np.array(current_segment, dtype=np.float32)
                            })
                        is_active = False
                        current_segment = []

            frame_count += 1

        # Xử lý đoạn cuối nếu video kết thúc khi tay vẫn đang đưa lên
        if is_active and len(current_segment) >= MIN_SEGMENT_LEN:
            segments.append({
                'start': start_frame,
                'end': frame_count,
                'data': np.array(current_segment, dtype=np.float32)
            })

        cap.release()
        return segments, frame_count

    def predict_sentence(self, video_path: str, confidence_threshold: float = 70.0) -> dict:
        """
        Hàm chính: Trích xuất các đoạn (segments) khi tay đưa lên -> Dự đoán từng đoạn -> Trả kết quả JSON
        """
        segments, total_frames = self.extract_features_and_segments(video_path)

        if not segments:
            return {"status": "error", "message": "Không phát hiện được cử chỉ tay rõ ràng trong video."}

        raw_results = []
        final_sentence = []
        last_gloss = None

        for seg in segments:
            data = seg['data']
            seg_frames = data.shape[0]

            # Đưa các đoạn về 70 frames bằng nội suy
            indices = np.linspace(0, seg_frames - 1, self.TARGET_FRAMES).astype(int)
            window_data = data[indices]

            # CTRGCN input: (batch_size, num_frames, num_points, in_channels)
            input_data = np.expand_dims(window_data, axis=0).astype(np.float32)

            logits = self.session.run([self.output_name], {self.input_name: input_data})[0]
            predicted_id = np.argmax(logits, axis=1)[0]

            probs = np.exp(logits[0] - np.max(logits[0]))
            probs /= np.sum(probs)
            confidence = float(np.max(probs) * 100)
            predicted_gloss = self.id2label.get(predicted_id, "Unknown")

            if confidence >= confidence_threshold:
                raw_results.append({
                    "start": seg['start'], "end": seg['end'],
                    "gloss": predicted_gloss, "confidence": confidence
                })

                if predicted_gloss != last_gloss:
                    final_sentence.append(predicted_gloss.upper())
                    last_gloss = predicted_gloss

        return {
            "status": "success",
            "total_frames": total_frames,
            "segments": raw_results,
            "final_sentence": " -> ".join(final_sentence) if final_sentence else ""
        }

    def close(self):
        self.holistic.close()
