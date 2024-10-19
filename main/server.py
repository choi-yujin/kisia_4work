#server.py

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import subprocess
import json
import os
import logging
import time

logging.basicConfig(level=logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR) #패킷 캡쳐가 너무 많이 돼서 터미널이 온통 요청 로그로 가득차는 걸 방지하기 위함...

print('http://192.168.219.109:5000/')

server = Flask(__name__)  # Flask 웹 서버 생성
server.config['SECRET_KEY'] = 'secret!'  # 웹소켓 기능 활성화를 위한 비밀키 설정
socketio = SocketIO(server)  # Flask 서버에 SocketIO 추가 (실시간 양방향 통신)


packets = []  # 수신된 패킷을 저장할 리스트
ratios = {  # 비율 데이터를 저장할 전역 변수
    'traffic_ratios': {
        'CHAT': 0,
        'VOIP': 0,
        'STREAMING': 0,
    },
    'app_ratios': {
        'facebook': 0,
        'discord': 0,
        'skype': 0,
        'line': 0,
        'youtube': 0,
    }
}

# 기본 웹페이지 라우트
@server.route('/', methods=['GET'])
def index():
    return render_template('dashboard.html')  # index1.html 반환 (패킷 실시간 표시용)

# 패킷 수신 엔드포인트
@server.route('/stream/receive_packet', methods=['POST'])
def receive_packet():
    packet_data = request.json  # client에서 보낸 JSON 형식의 패킷 데이터 받아오기
    packets.append(packet_data)  # 패킷 데이터 리스트에 추가
    # print('received packet data:', json.dumps(packet_data, indent=4))  # 수신된 패킷 출력

    # 실시간으로 수신된 패킷을 웹페이지로 전송
    socketio.emit('new_packet', packet_data)  # 웹소켓으로 패킷 전송
    return jsonify({'status': 'success', 'received_packets': packet_data})  # 성공 응답 반환

# 모델 예측 결과 수신 엔드포인트
predictions=[]
@server.route('/stream/receive_prediction',methods=['POST'])
def receive_prediction():
    pred_data=request.json #예측 데이터 수신
    predictions.append(pred_data) #데이터 저장
    # print(f'received prediction data: ',json.dumps(pred_data,indent=4))

    socketio.emit('new_prediction',pred_data) #웹서버로 예측 데이터 전송
    return jsonify({'status': 'success', 'received_prediction': pred_data})  #상태 응답, 예측 데이터

# 파이차트 ratio 수신 엔드포인트
@server.route('/stream/pie_rate', methods=['POST'])  
def receive_pie():
    global ratios  # ratios를 전역 변수로 사용
    ratio_data = request.json  # 클라이언트에서 보낸 JSON 형식의 비율 데이터 받아오기
    print(f'received ratio data: {ratio_data}')

    ratios['traffic_ratios'] = ratio_data['traffic_ratios']  # traffic_ratios 업데이트
    ratios['app_ratios'] = ratio_data['app_ratios']  # app_ratios 업데이트
    print(f'update ratios: {ratios}')
    return jsonify({'status': 'success', 'updated_ratios': ratios})  # 업데이트 상태 반환

@server.route('/stream/pie_rate', methods=['GET'])
def send_pie():
    # ratios 데이터를 반환
    return jsonify(ratios)

# 그래프 생성, 웹소켓으로 업데이트
graph_image_path='./static/flow_graph.png'
def create_graph():
    try:
        subprocess.run(['python', 'graph.py'], check=True)

        # 그래프 이미지 파일이 존재하는지 확인
        if os.path.exists(graph_image_path):
            # 웹서버로 실시간 그래프 업데이트 알림을 웹소켓으로 전송
            socketio.emit('new_graph', {'status': 'graph_updated'})
            return send_file(graph_image_path, mimetype='image/png')
        else:
            print('Graph file not found')
            return None

    except Exception as e:
        print(f'Error in generating graph: {str(e)}')
        return None
    
# 그래프 생성 코드 실행 엔드포인트
@server.route('/stream/packet_flow', methods=['POST'])
def generate_graph():
    return create_graph()  

# 그래프를 주기적으로 업데이트하는 함수
def update_graph_periodically(interval):
    while True:
        time.sleep(interval)  # 주기 설정
        with server.app_context():  # 애플리케이션 컨텍스트 설정
            create_graph()  

       
# packet.py와 MTC_model.py 자동 실행
def start_packet_capture():
    try:
        subprocess.Popen(['python', 'packet.py'])
        print("Packet capture started.")
    except Exception as e:
        print(f'Error starting packet capture: {e}')

def start_model_prediction():
    try:
        subprocess.Popen(['python', 'mtc_model.py'])
        print("Model prediction started.")
    except Exception as e:
        print(f'Error starting model prediction: {e}')
        
# 웹소켓 연결 처리
@socketio.on('connect')
def handle_connect():
    print('Client connected')  # 웹소켓 연결 시 출력

if __name__ == '__main__':
    # 서버가 가동된 후 패킷 캡처 및 모델 예측 시작
    start_packet_capture()
    start_model_prediction()

    socketio.start_background_task(update_graph_periodically, 30)  # 30초마다 그래프 업데이트

    socketio.run(server, host='0.0.0.0', port=5000)  # Flask 서버 실행 (WebSocket 포함)
