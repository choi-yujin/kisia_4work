import os
import sqlite3
from flask import Flask, jsonify, request, g
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = Flask(__name__)

# 데이터베이스 연결 함수
def get_db():
    db_path = os.getenv("DATABASE_PATH")
    
    if not db_path:
        raise ValueError("DATABASE_PATH 환경 변수가 설정되지 않았습니다.")

    # 디렉토리 경로가 없다면 생성
    db_dir = os.path.dirname(db_path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    # 데이터베이스 연결
    if 'db' not in g:
        g.db = sqlite3.connect(db_path)
    return g.db

# 데이터베이스 연결 종료 함수
@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# 테이블 생성 함수 (앱 시작 시 한번만 실행)
def create_table():
    db = get_db()
    
    # 테이블 생성
    db.execute('''CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    traffic_count TEXT,
                    app_count TEXT,
                    prediction_messages TEXT)''')
    db.commit()

# 1. 히스토리 조회 기능
@app.route('/history/inquiry', methods=['GET'])
def inquiry_history():
    db = get_db()
    cursor = db.execute('SELECT * FROM prediction_history ORDER BY date DESC')
    rows = cursor.fetchall()
    
    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "date": row[1],
            "start_time": row[2],
            "end_time": row[3],
            "traffic_count": row[4],
            "app_count": row[5],
            "prediction_messages": row[6]
        })
    
    return jsonify({"history": history})

# 2. 히스토리 저장 기능
@app.route('/history/save', methods=['POST'])
def save_history():
    data = request.json
    selected_date = data.get('date')
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    traffic_count = data.get('traffic_count')
    app_count = data.get('app_count')
    prediction_messages = data.get('messages')

    if not selected_date:
        return jsonify({'error': '날짜를 입력해주세요.'}), 400

    db = get_db()
    db.execute('INSERT INTO prediction_history (date, start_time, end_time, traffic_count, app_count, prediction_messages) VALUES (?, ?, ?, ?, ?, ?)',
               (selected_date, start_time, end_time, str(traffic_count), str(app_count), str(prediction_messages)))
    db.commit()

    return jsonify({'message': '히스토리가 저장되었습니다.'}), 200

# 3. 히스토리 삭제 기능
@app.route('/history/delete', methods=['DELETE'])
def delete_history():
    data = request.json
    record_id = data.get('id')

    if not record_id:
        return jsonify({'error': '삭제할 레코드의 ID를 입력해주세요.'}), 400

    db = get_db()
    result = db.execute('DELETE FROM prediction_history WHERE id = ?', (record_id,))
    db.commit()

    if result.rowcount == 0:
        return jsonify({'error': '해당 ID의 레코드가 없습니다.'}), 404

    return jsonify({'message': '히스토리가 삭제되었습니다.'}), 200

# 초기 테이블 생성
with app.app_context():
    create_table()

if __name__ == '__main__':
    app.run(debug=True)
