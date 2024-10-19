#packet.py

import os
from scapy.all import sniff, wrpcap
from datetime import datetime
import time
import requests
import json
from dotenv import load_dotenv

# 환경 변수
load_dotenv()

server_url=os.getenv('SERVER_URL')
iface_name=os.getenv('IFACE_NAME')
iface_name = iface_name.replace('\\\\', '\\') 

# 파일 생성 간격 설정 (초)
file_interval = int(os.getenv('FILE_INTERVAL') ) 
last_file_time = time.time()

#파이차트에 쓸 pcap을 저장할 폴더
pcaps_folder = os.getenv('PCAP_FOLDER')
    
if not os.path.exists(pcaps_folder):
    os.makedirs(pcaps_folder)  # 폴더가 없으면 생성

current_file = os.path.join(pcaps_folder,f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcap")

# 패킷을 처리하고 저장하며 서버로 전송하는 함수
def packet_handler(packet):
    global last_file_time, current_file

    # 현재 시간 확인
    current_time = time.time()

    # 일정 시간이 지나면 새로운 파일 생성
    if current_time - last_file_time >= file_interval: # 현재-마지막 >= 60초 면 새로운 파일 생성
        last_file_time = current_time
        current_file = os.path.join(pcaps_folder, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcap")
        print(f"New file: {current_file}")

    # 패킷을 현재 파일에 추가
    wrpcap(current_file, packet, append=True)

    # 패킷 요약 정보 및 데이터 처리
    packet_summary = packet.summary()
    packet_data = {}

    if packet.haslayer('IP'):  # IP 패킷만 처리
        src_ip = packet['IP'].src  # 출발지 IP
        dst_ip = packet['IP'].dst  # 목적지 IP

        packet_data = {
            'summary': packet_summary,  # 패킷 요약 정보
            'src_ip': src_ip,  # 출발지 IP
            'dst_ip': dst_ip  # 목적지 IP
        }

        # # 서버에 packet_data 전송
        requests.post(f'{server_url}/stream/receive_packet', json=packet_data)
        
        
# sniff(인터페이스, 패킷 처리 함수, store=0 >> 메모리에 저장X )
sniff(iface=iface_name, prn=packet_handler, count=0, store=0)
