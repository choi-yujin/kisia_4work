# app.py

from flask import Flask, request, jsonify, render_template
import sqlite3
from datetime import datetime

app = Flask(__name__)

# 데이터베이스 초기화
def init_db():
    with sqlite3.connect('packet_history.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS packet_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                summary TEXT,
                src_ip TEXT,
                dst_ip TEXT
            )
        ''')
    conn.close()

# 패킷 데이터 저장 함수
def save_packet(packet_data):
    with sqlite3.connect('packet_history.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO packet_history (timestamp, summary, src_ip, dst_ip)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
              packet_data['summary'], 
              packet_data['src_ip'], 
              packet_data['dst_ip']))
        conn.commit()

# 특정 기간의 패킷 히스토리 조회
@app.route('/history', methods=['GET'])
def get_history():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    with sqlite3.connect('packet_history.db') as conn:
        cursor = conn.cursor()
        query = '''
            SELECT timestamp, summary, src_ip, dst_ip 
            FROM packet_history 
            WHERE timestamp BETWEEN ? AND ?
        '''
        cursor.execute(query, (start_date, end_date))
        rows = cursor.fetchall()
        
        # 결과를 JSON 형식으로 반환
        history = [{'timestamp': row[0], 'summary': row[1], 'src_ip': row[2], 'dst_ip': row[3]} for row in rows]
        return jsonify(history)

# 패킷 데이터 수신
@app.route('/receive_packet', methods=['POST'])
def receive_packet():
    packet_data = request.get_json()
    if packet_data:
        save_packet(packet_data)
        return jsonify({'status': 'success'}), 200
    return jsonify({'status': 'error'}), 400

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)
