# -----------------------------------------------------------
# YOLOv5 Wrapper with Ultralytics Library (Using cfg)
# written by Jeongmin Kim (jm.kim@dankook.ac.kr)
# -----------------------------------------------------------
from ultralytics import YOLO


class YOLOv5:
    def __init__(self, cfg):
        """
        YOLOv5 모델 클래스 (Ultralytics 활용, cfg 기반 설정)

        Args:
            cfg (dict): 설정 파일 정보를 포함한 딕셔너리
        """
        model_name = cfg.MODEL.NAME.lower()  # 모델 이름
        phase = cfg.PHASE
        pretrained = cfg.MODEL.PRETRAINED if cfg.MODEL.PRETRAINED else None # Load Pretrained YOLO


        if phase == 'inference':
            self.model = None
            self.load_pretrained(cfg.MODEL.EXTRA.PT)
            return

        if pretrained:
            # 사전 학습된 모델 로드
            self.model = YOLO(f"{model_name}s.pt")
        else:
            # YAML 파일로 새로운 모델 정의
            self.model = YOLO(f"{model_name}.yaml")

    def train(self, cfg):
        """
        모델 학습

        Args:
            cfg (dict): 설정 파일 정보를 포함한 딕셔너리
        """
        device = cfg.DEVICE.lower()  # 소문자로 변환
        if device == "gpu":  # 'gpu'를 '0'으로 매핑
            device = "0"
        elif device.startswith("gpu:"):  # 'gpu:1' 형태 처리
            device = device.split(":")[1]
        elif device == "cpu":
            device = "cpu"
        else:
            raise ValueError(f"Invalid device: {cfg.DEVICE}. Use 'gpu', 'gpu:0', or 'cpu'.")

        self.model.train(
            data=cfg.DATASET.JSON,  # data.json 경로
            epochs=cfg.TRAIN.END_EPOCH,
            imgsz=cfg.MODEL.EXTRA.INPUT_SIZE[0],
            batch=cfg.TRAIN.BATCH_SIZE_PER_GPU,
            device=device
        )

    def val(self, cfg):
        """
        모델 검증

        Args:
            cfg (dict): 설정 파일 정보를 포함한 딕셔너리

        Returns:
            dict: 검증 결과 (mAP, Precision, Recall 등)
        """
        device = cfg.DEVICE.lower()  # 소문자로 변환
        if device == "gpu":  # 'gpu'를 '0'으로 매핑
            device = "0"
        elif device.startswith("gpu:"):  # 'gpu:1' 형태 처리
            device = device.split(":")[1]
        elif device == "cpu":
            device = "cpu"
        else:
            raise ValueError(f"Invalid device: {cfg.DEVICE}. Use 'gpu', 'gpu:0', or 'cpu'.")


        return self.model.val(
            data=cfg.DATASET.JSON,
            batch=cfg.TEST.BATCH_SIZE_PER_GPU,
            device=device
        )

    def predict(self, source, save=False, device=0):
        """
        이미지 또는 비디오에서 객체를 탐지합니다.

        Args:
            source (str): 입력 데이터 경로 (이미지, 비디오, 폴더 등)
            save (bool): 탐지 결과를 저장할지 여부
            device (int or str): 추론에 사용할 디바이스 (GPU ID 또는 'cpu')

        Returns:
            list: 탐지된 결과
        """
        results = self.model.predict(source=source, save=save, device=device)
        return results

    def save_checkpoint(self, checkpoint_path):
        """
        학습된 가중치를 저장합니다.

        Args:
            checkpoint_path (str): 저장할 경로
        """
        self.model.save(checkpoint_path)

    def load_pretrained(self, checkpoint_path):
        """
        저장된 체크포인트에서 가중치를 로드합니다.

        Args:
            checkpoint_path (str): 체크포인트 경로
        """
        self.model = YOLO(checkpoint_path)