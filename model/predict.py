import pickle
import torch
import os
import glob
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm 
from mtc import MTC  # mtc 모델만을 사용
from dataset import CustomListDataset
from preprocess import preprocess_data

def predict_from_pkl(model: MTC, batch_size: int = 128, device: str = 'cuda:0'):
    """
    Task1과 Task2를 예측하고 결과를 출력하는 함수
    Args:
        model: MTC 모델
        batch_size: 배치 사이즈
        device: 사용할 디바이스 ('cuda' 또는 'cpu')
    """
    # Check if GPU is available, otherwise switch to CPU
    if not torch.cuda.is_available():
        print('Fail to use GPU, switching to CPU')
        device = 'cpu'

    # 동일 디렉토리에 있는 pkl 파일을 모두 불러오기
    pkl_files = glob.glob('./*.pkl')

    # pkl 파일이 없을 경우 에러 처리
    if not pkl_files:
        print("pkl 파일이 없습니다.")
        return

    # 모델을 디바이스로 이동
    model = model.to(device)
    model.eval()

    for pkl_file in pkl_files:
        print(f'Predicting for file: {pkl_file}')

        # pkl 파일 로드
        with open(pkl_file, 'rb') as f:
            test_data_rows = pickle.load(f)

        # Dataset 및 DataLoader 생성
        test_dataset = CustomListDataset(test_data_rows)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Task1, Task2 결과 저장용 리스트
        task1_outputs = []
        task2_outputs = []

        # 예측 수행
        with torch.no_grad():
            pbar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc=f"Predicting")
            for batch_idx, (inputs, labels_task1, labels_task2, _) in pbar:
                inputs = inputs.to(device)

                # 모델에 입력하여 Task1과 Task2 예측
                outputs1, outputs2, _ = model(inputs)

                # Task1, Task2 예측 결과를 리스트에 저장
                task1_outputs.extend(torch.argmax(outputs1, dim=1).cpu().numpy())
                task2_outputs.extend(torch.argmax(outputs2, dim=1).cpu().numpy())

        # Task1, Task2에서 가장 많이 나온 라벨 계산
        most_frequent_task1 = np.argmax(np.bincount(task1_outputs))
        most_frequent_task2 = np.argmax(np.bincount(task2_outputs))

        # 예측 결과를 출력
        print(f"Most frequent label for Task1 in {pkl_file}: {most_frequent_task1}\n")
        print(f"Most frequent label for Task2 in {pkl_file}: {most_frequent_task2}")

if __name__ == '__main__':
    # MTC 모델을 로드하고 예측 수행
    model = MTC()
    model.load_state_dict(torch.load('MTC_model.pt', map_location=torch.device('cpu')))

    preprocess_data()
    predict_from_pkl(model)
