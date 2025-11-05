from ultralytics import YOLO
import torch

def train_thermal_model():
    # 사전학습된 YOLOv8n 모델 로드
    model = YOLO('yolov8n.pt')
    
    # 학습 설정 및 실행
    results = model.train(
        data='dataset/thermal_dataset.yaml',  # 데이터셋 설정 파일
        epochs=100,                           # 학습 에포크 수
        imgsz=640,                           # 입력 이미지 크기
        batch=16,                            # 배치 크기
        patience=50,                         # Early stopping patience
        save=True,                           # 최상의 모델 저장
        device='0' if torch.cuda.is_available() else 'cpu',  # GPU 사용 (가능한 경우)
        project='thermal_detection',          # 프로젝트 이름
        name='exp1',                         # 실험 이름
        exist_ok=True                        # 기존 실험 덮어쓰기
    )
    
    return results

def validate_model():
    # 학습된 모델 로드
    model = YOLO('thermal_detection/exp1/weights/best.pt')
    
    # 검증 수행
    metrics = model.val()
    print("Validation metrics:", metrics)

def test_on_image(image_path):
    # 학습된 모델 로드
    model = YOLO('thermal_detection/exp1/weights/best.pt')
    
    # 이미지에서 테스트
    results = model(image_path)
    
    # 결과 저장
    results.save(save_dir='thermal_detection/exp1/results')

if __name__ == "__main__":
    print("Starting thermal detection model training...")
    
    # 학습 실행
    results = train_thermal_model()
    
    # 검증 실행
    validate_model()
    
    # 테스트 이미지로 테스트 (예시 이미지 경로)
    test_image = "dataset/val/images/test_image.jpg"  # 이미지 경로는 실제 테스트 이미지로 변경 필요
    test_on_image(test_image)