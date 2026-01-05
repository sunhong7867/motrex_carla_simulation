# yolo_detector.py

from ultralytics import YOLO
from pathlib import Path

class YOLODetector:
    def __init__(self, weight_name="best.pt", device="cpu", conf=0.2):
        # ... (기존 초기화 코드 동일) ...
        base_dir = Path(__file__).resolve().parent
        weight_path = base_dir / weight_name
        
        if not weight_path.exists():
            raise FileNotFoundError(f"YOLO weight not found: {weight_path}")

        self.model = YOLO(str(weight_path))
        self.device = device
        self.conf = conf
        print("[YOLO] class names:", self.model.names)

    def infer(self, img_bgr, conf=None):
        # 인자로 conf가 들어오면 그것을 쓰고, 없으면 초기 설정값(self.conf) 사용
        target_conf = conf if conf is not None else self.conf
        
        results = self.model.predict(
            source=img_bgr,
            conf=target_conf,
            device=self.device,
            verbose=False,
            stream=False,
            
            iou=0.4,           # 겹침 허용치 (낮을수록 중복 박스를 더 잘 지움. 보통 0.4~0.5 추천)
            agnostic_nms=True   # 클래스가 달라도 겹치면 하나로 합침 (차량끼리 겹칠 때 유용)
        )

        r = results[0]
        dets = []

        if r.boxes is None:
            return []

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append({
                "bbox": (x1, y1, x2, y2),
                "cls": int(box.cls[0]),
                "conf": float(box.conf[0]),
            })

        return dets