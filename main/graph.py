import os
from scapy.all import rdpcap, TCP, UDP
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import time
from dotenv import load_dotenv

load_dotenv()
file_interval = os.getenv('FILE_INTERVAL')

# 플로우를 식별하기 위한 함수 (5-튜플: src IP, src Port, dst IP, dst Port, 프로토콜)
def get_flow(packet):
    if TCP in packet:
        protocol = 'TCP'
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
    elif UDP in packet:
        protocol = 'UDP'
        src_port = packet[UDP].sport
        dst_port = packet[UDP].dport
    else:
        return None  # TCP/UDP 패킷만 처리

    return (packet[0][1].src, src_port, packet[0][1].dst, dst_port, protocol)

# 특정 애플리케이션 트래픽 식별 (여러 애플리케이션 기반으로 필터링 가능)
def get_application_type(flow):
    app_dict = {
        80: 'HTTP',
        443: 'HTTPS',
        53: 'DNS',
    }
    
    # Check if source or destination port matches known application ports
    if flow[1] in app_dict:
        return app_dict[flow[1]]
    elif flow[3] in app_dict:
        return app_dict[flow[3]]
    else:
        return 'Other'

# 패킷을 플로우별로 그룹화하고 시간대별로 패킷 수를 계산
def analyze_pcap(file_path):
    packets = rdpcap(file_path)
    flows = defaultdict(list)

    for packet in packets:
        flow = get_flow(packet)
        if flow:
            try:
                # packet.time을 float으로 변환하여 datetime으로 변환
                timestamp = float(packet.time)
                timestamp = datetime.fromtimestamp(timestamp)
                app_type = get_application_type(flow)
                protocol = flow[4]  # TCP 또는 UDP
                flows[(flow, app_type, protocol)].append(timestamp)
            except Exception as e:
                print(f"Error processing packet: {e}")

    return flows

# 여러 pcap 파일의 플로우를 시각화
def plot_flows_over_time(all_flows, file_names):
    plt.figure(figsize=(14, 8))

    flow_data = defaultdict(lambda: defaultdict(int))

    for file_name, flows in all_flows.items():
        for (flow, app_type, protocol), timestamps in flows.items():
            # 패킷 수를 시간대별로 집계 (30초 단위)
            df = pd.DataFrame(timestamps, columns=['timestamp'])
            df['half_minute'] = df['timestamp'].dt.floor('30s')  # 30초 단위로 내림
            packet_counts = df.groupby('half_minute').size()

            # 플로우 데이터 저장
            for time, count in packet_counts.items():
                flow_data[time][(app_type, protocol)] += count

    # 각 시간대별로 플로우를 누적하는 영역 그래프
    times = sorted(flow_data.keys())
    traffic_types = set()  # 애플리케이션과 프로토콜 유형을 구분하기 위해 set 사용

    for time_data in flow_data.values():
        traffic_types.update([app_proto for app_proto in time_data.keys()])  # set에 애플리케이션 및 프로토콜 유형 추가

    # 애플리케이션 및 프로토콜 유형별로 다중 인덱스를 생성
    columns = pd.MultiIndex.from_tuples(traffic_types, names=['Application', 'Protocol'])
    flow_matrix = pd.DataFrame(index=times, columns=columns).fillna(0)

    for time, flow_counts in flow_data.items():
        for (app_type, protocol), count in flow_counts.items():
            flow_matrix.loc[time, (app_type, protocol)] += count

    # 영역 그래프 그리기
    flow_matrix.plot(kind='area', stacked=True, figsize=(14, 8), alpha=0.7)

    # 곡선 그래프 추가 (drawstyle='steps-post' 옵션 사용)
    for app_proto in flow_matrix.columns:
        app_type, protocol = app_proto
        plt.plot(flow_matrix.index, flow_matrix[app_proto], label=f'{app_type} ({protocol})', drawstyle='steps-post')

    plt.xlabel('Time (30-second intervals)')
    plt.ylabel('Packet Count')
    plt.title('Application and Protocol Packet Flow Over Time')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # 디렉토리 존재 여부 확인
    output_dir = './static'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 디렉토리 생성

    plt.savefig(os.path.join(output_dir, 'line_graph.png'))  # 이미지 저장
    plt.close()

# 디렉토리에 있는 모든 pcap 파일을 분석
def analyze_multiple_pcaps(pcap_dir,check_interval=70):
    all_flows = {}

    for file_name in os.listdir(pcap_dir):
        if file_name.endswith('.pcap'):
            file_path = os.path.join(pcap_dir, file_name)
            time.sleep(file_interval)
            print(f"Analyzing {file_name}...")

            # pcap 파일 분석
            flows = analyze_pcap(file_path)

            # time.sleep(check_interval) #굳이 안 해도 될 듯

            # 플로우 저장
            all_flows[file_name] = flows

    # 모든 pcap 파일의 플로우를 시각화
    plot_flows_over_time(all_flows, list(all_flows.keys()))

# pcap 파일이 저장된 디렉토리 경로
pcap_dir = './pcaps'  # 실제 pcap 파일이 있는 디렉토리 경로로 변경, 왜 pcap_dir이 아닌 pcap_directory를 쓴 것? -> 이러면 분석 함수에서 경로를 못 받아올텐데

# 분석 실행
if __name__=="__main__":
    analyze_multiple_pcaps(pcap_dir)
