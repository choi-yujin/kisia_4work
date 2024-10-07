#server.py

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import subprocess
import json

server = Flask(__name__)  # Flask 웹 서버 생성
server.config['SECRET_KEY'] = 'secret!'  # 웹소켓 기능 활성화를 위한 비밀키 설정
socketio = SocketIO(server)  # Flask 서버에 SocketIO 추가 (실시간 양방향 통신)


packets = []  # 수신된 패킷을 저장할 리스트

# 기본 웹페이지 라우트
@server.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # index1.html 반환 (패킷 실시간 표시용)

# 패킷 수신 엔드포인트
@server.route('/receive_packet', methods=['POST'])
def receive_packet():
    packet_data = request.json  # client에서 보낸 JSON 형식의 패킷 데이터 받아오기
    packets.append(packet_data)  # 패킷 데이터 리스트에 추가
    print('received packet data:', json.dumps(packet_data, indent=4))  # 수신된 패킷 출력

    # 실시간으로 수신된 패킷을 웹페이지로 전송
    socketio.emit('new_packet', packet_data)  # 웹소켓으로 패킷 전송
    return jsonify({'status': 'success', 'received_packets': packet_data})  # 성공 응답 반환

# @socketio.on('model_prediction')
# def handle_model_prediction(data):
#     print('model prediction received:',json.dump(data, indent=4))
#     socketio.emit('new prediction',data)

predictions=[]

@server.route('/receive_prediction',methods=['POST'])
def receive_prediction():
    pred_data=request.json
    predictions.append(pred_data)
    print(f'received prediction data: ',json.dumps(pred_data,indent=4))

    socketio.emit('new_prediction',pred_data)
    return jsonify({'status': 'success', 'received_prediction': pred_data}) 

#메인페이지 실시간 파이차트 예측 후 index.html에 결과 반환 개발 중
import sub.ml_server2 as sub
import os

@server.route('/predict_pie', methods=['GET'])
def send_pie():
    pcap_folder = 'C:\Users\김재민\Documents\GitHub\kisia_4work\pcaps'  # PCAP 파일이 저장된 폴더 경로
    pcap_files = [f for f in os.listdir(pcap_folder) if f.endswith('.pcap')]

    if not pcap_files:
        return jsonify({'error': 'PCAP 파일이 존재하지 않습니다.'}), 400
    
    traffic_count = {0: 0, 1: 0, 2: 0}
    app_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for pcap_file in pcap_files:
        pcap_path = os.path.join(pcap_folder, pcap_file)
        file_traffic_count, file_app_count = sub.predict(pcap_path)
        
        for key in traffic_count:
            traffic_count[key] += file_traffic_count[key]
        
        for key in app_count:
            app_count[key] += file_app_count[key]

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
        'app_ratios': app_ratios
    })





# 웹소켓 연결 처리
@socketio.on('connect')
def handle_connect():
    print('Client connected')  # 웹소켓 연결 시 출력

# # 트래픽 분석 결과 반환
# @server.route('/analyze-traffic')
# def analyze_traffic():
#     start_time = request.args.get('start')
#     end_time = request.args.get('end')

    # # 예시 분석 결과 (여기서는 임의의 값, 실제 분석 로직 추가 가능)
    # result = {
    #     "f1_score": 0.92,
    #     "accuracy": 0.89,
    #     "traffic_distribution": [30, 15, 10, 20, 5, 20]  # 각 유형별 트래픽 비율 (예시)
    # }
    # return jsonify(result)  # 분석 결과 JSON 형태로 반환


if __name__ == '__main__':
    socketio.run(server, host='0.0.0.0', port=5000)  # Flask 서버 실행 (WebSocket 포함)