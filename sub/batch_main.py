import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from data_processing import ask_predict  # 데이터 전처리 및 예측 모듈
from pie_rate import send_pie  # 트래픽/애플리케이션 유형 비율 모듈
from history import inquiry_history, save_history, delete_history  # 히스토리 관련 모듈

# .env 파일 로드
load_dotenv()

app = Flask(__name__)

# 1. 사용자 요청을 받아 비실시간 패킷 예측 수행
@app.route('/batch/ask_predic', methods=['POST'])
def ask_predic():
    """사용자 요청을 받아 패킷 예측을 실행하는 엔드포인트"""
    data = request.get_json()
    selected_date = data.get('date')
    start_time = data.get('start_time')
    end_time = data.get('end_time')

    # 비실시간 패킷 예측 결과 가져오기
    response = ask_predict(selected_date, start_time, end_time)
    
    return jsonify(response)

# 2. 비실시간 트래픽 및 애플리케이션 유형 비율 계산
@app.route('/batch/pie_rate', methods=['GET'])
def pie_rate():
    """트래픽 및 애플리케이션 유형 비율을 반환하는 엔드포인트"""
    selected_date = request.args.get('date')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')

    # 비율 계산 결과 반환
    response = send_pie(selected_date, start_time, end_time)
    
    return jsonify(response)

# 3. 예측 결과 히스토리 조회
@app.route('/history/inquiry', methods=['GET'])
def inquiry():
    """저장된 예측 결과 히스토리를 조회하는 엔드포인트"""
    return inquiry_history()

# 4. 예측 결과 히스토리 저장
@app.route('/history/save', methods=['POST'])
def save():
    """예측 결과를 히스토리에 저장하는 엔드포인트"""
    data = request.json
    return save_history(data)

# 5. 예측 결과 히스토리 삭제
@app.route('/history/delete', methods=['DELETE'])
def delete():
    """특정 히스토리 레코드를 삭제하는 엔드포인트"""
    data = request.json
    return delete_history(data)

if __name__ == '__main__':
    app.run(debug=True)
