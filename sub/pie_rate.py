import os
import glob
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from collections import Counter
from datetime import datetime
from scapy.all import rdpcap
from data_processing import ask_predict  # 데이터 전처리 모듈
from dotenv import load_dotenv  # 환경 변수 로드

# .env 파일 로드
load_dotenv()

app = Flask(__name__)

@app.route('/batch/ask_predic', methods=['POST'])
def ask_predic():
    data = request.get_json()
    selected_date = data.get('date')
    start_time = data.get('start_time')
    end_time = data.get('end_time')

    traffic_count, app_count, prediction_messages = ask_predict(selected_date, start_time, end_time)
    
    return jsonify({
        'traffic_count': traffic_count,
        'app_count': app_count,
        'messages': prediction_messages
    })


# 비율 계산 함수 (트래픽 및 애플리케이션 유형)
@app.route('/batch/pie_rate', methods=['GET'])
def send_pie():
    selected_date = request.args.get('date')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')

    if not selected_date:
        return jsonify({'error': '날짜를 선택해주세요.'}), 400

    traffic_count, app_count, prediction_messages = ask_predict(selected_date, start_time, end_time)
    
    total_traffic = sum(traffic_count.values())
    total_app = sum(app_count.values())

    traffic_ratios = {
        'CHAT': traffic_count[0] / total_traffic if total_traffic > 0 else 0,
        'VOIP': traffic_count[1] / total_traffic if total_traffic > 0 else 0,
        'STREAMING': traffic_count[2] / total_traffic if total_traffic > 0 else 0,
    }

    app_ratios = {
        'facebook': app_count[0] / total_app if total_app > 0 else 0,
        'discord': app_count[1] / total_app if total_app > 0 else 0,
        'skype': app_count[2] / total_app if total_app > 0 else 0,
        'line': app_count[3] / total_app if total_app > 0 else 0,
        'youtube': app_count[4] / total_app if total_app > 0 else 0,
    }

    return jsonify({
        'traffic_ratios': traffic_ratios,
        'app_ratios': app_ratios,
        'prediction_messages': prediction_messages
    })

# 예측 결과 반환 API
@app.route('/batch/receive_prediction', methods=['GET'])
def receive_prediction():
    selected_date = request.args.get('date')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')

    if not selected_date:
        return jsonify({'error': '날짜를 선택해주세요.'}), 400

    traffic_count, app_count, prediction_messages = ask_predict(selected_date, start_time, end_time)
    
    return jsonify({
        'prediction_messages': prediction_messages
    })

# 전체 패킷 흐름도 반환 API
@app.route('/batch/packet_flow', methods=['GET'])
def packet_flow():
    selected_date = request.args.get('date')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')

    if not selected_date:
        return jsonify({'error': '날짜를 선택해주세요.'}), 400

    # 전체 패킷 흐름 데이터를 반환하는 부분 추가 (데이터 가공 필요)
    # 임시 반환 값으로 예시 데이터
    packet_flow_data = {
        'flow_start_time': start_time,
        'flow_end_time': end_time,
        'packets': [
            {'time': '12:00:01', 'src': '192.168.1.1', 'dst': '192.168.1.2', 'protocol': 'TCP'},
            {'time': '12:00:05', 'src': '192.168.1.2', 'dst': '192.168.1.1', 'protocol': 'UDP'}
        ]
    }

    return jsonify(packet_flow_data)

if __name__ == '__main__':
    app.run(debug=True)
