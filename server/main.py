from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import subprocess
import json
import os
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
import requests
import pickle
import random
import glob
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Any, Dict, Union
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from scapy.all import sniff, wrpcap, Ether, IP, TCP, UDP
from scapy.compat import raw
from scapy.layers.dns import DNS
from scapy.packet import Padding, Packet
from scipy import sparse
from sklearn.model_selection import train_test_split

from torcheval.metrics import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
from dotenv import load_dotenv


# .env 파일 로드
load_dotenv('.env')

app = Flask(__name__,static_folder='../static', template_folder='../templates')
socketio = SocketIO(app)  # Flask 서버에 SocketIO 추가 (실시간 양방향 통신)
#CORS(app)


PREFIX_TO_TRAFFIC_ID = {
    'chat': 0,
    'voip':1,
    'streaming': 2,
}

# Application labels
PREFIX_TO_APP_ID = {
    'facebook': 0,
    'discord':1,
    'skype':2,
    'line':3,
    'youtube': 4
}

# Auxiliary task labels
AUX_ID = {
    'all_chat': 0,
    'all_voip': 1,
    'all_streaming': 2,
}
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'  # 모든 도메인 허용
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

def load_data() -> Tuple[Any, Any, Any]:
    """Load data from pickle files"""
    with open('data/train_data_rows.pkl', 'rb') as f:
        train_data_rows = pickle.load(f)

    with open('data/val_data_rows.pkl', 'rb') as f:
        val_data_rows = pickle.load(f)

    with open('data/test_data_rows.pkl', 'rb') as f:
        test_data_rows = pickle.load(f)

    print(f'Amount of train data: {len(train_data_rows)}')
    print(f'Amount of val data: {len(val_data_rows)}')
    print(f'Amount of test data: {len(test_data_rows)}')

    return train_data_rows, val_data_rows, test_data_rows


def id_to_one_hot_tensor(
        id_value: Union[int, torch.Tensor],
        num_classes: int
):
    # Convert int to tensor if single value
    if isinstance(id_value, int):
        id_value = torch.tensor(id_value)

    one_hot_tensor = torch.nn.functional.one_hot(id_value, num_classes=num_classes)
    return one_hot_tensor.to(torch.float32)


class CustomListDataset(Dataset):
    """Subclass of Dataset class"""

    def __init__(self, rows: List[Dict[str, Any]]):
        """ Initialize dataset.

        Args:
            rows: Data samples in a list of dict of features and labels.
        """
        self.data = rows
        self.n_traffic = len(PREFIX_TO_TRAFFIC_ID)
        self.n_app = len(PREFIX_TO_APP_ID)
        self.n_aux = len(AUX_ID)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        # Convert class index to one-hot encoding
        y_traffic = id_to_one_hot_tensor(d['traffic_label'], self.n_traffic)
        y_app = id_to_one_hot_tensor(d['app_label'], self.n_app)
        y_aux = id_to_one_hot_tensor(d['aux_label'], self.n_aux)

        # Concat a data sample including a sparse matrix converted
        sample = (torch.from_numpy(d['feature'].toarray()), y_traffic, y_app, y_aux)

        return sample


def get_dataset(data_rows: List[Dict[str, Any]]) -> Dataset:
    """Create a dataset with data samples"""
    ds = CustomListDataset(data_rows)
    return ds

print('dataset code done')

class CustomEmbedding(nn.Module):
    """Embedding layer"""

    def __init__(self, n_channels: int, n_dims: int):
        """
        Args:
            n_channels: Channels of embedding.
            n_dims: The dimensions of embedding.
        """
        super(CustomEmbedding, self).__init__()
        self.n_channels = n_channels
        self.n_dims = n_dims
        self.embedding = nn.Linear(self.n_dims, self.n_dims)

    def forward(self, input_data):
        """

        Args:
            input_data: Intput data with shape(B, C, L)

        Returns:
            Data with shape(B, Channel, Dimension)
        """
        input_data = input_data.reshape(-1, self.n_channels, self.n_dims)
        embedded = self.embedding(input_data)

        return embedded


class Bottleneck(nn.Module):
    """Bottleneck block in 1D-CNN blocks"""

    def __init__(self, in_channels, mid_channels, out_channels, residual_channels=0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = nn.Conv1d(mid_channels + residual_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.07)

    def forward(self, x, residual_1=None):
        residual_input = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        if residual_1 is not None:
            x = torch.concat((x, residual_1), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        residual_2 = self.dropout(x)

        x = self.conv3(residual_2)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Residual connection
        x += residual_input
        return x, residual_2


class FusionDownBlock(nn.Module):
    """Fusion block for down-sampling which connect 1D-CNN to transformer block."""

    def __init__(self,
                 down_cnn_in_channels, down_cnn_out_channels):
        super(FusionDownBlock, self).__init__()

        # 1x1 convolution to match dimensions
        self.conv_down = nn.Conv1d(down_cnn_in_channels, down_cnn_out_channels, kernel_size=1)
        self.layer_norm = nn.LayerNorm([down_cnn_out_channels, 50])

    def forward(self, cnn_features):
        # Down-sample
        # Match dimensions using 1x1 convolution for CNN features
        down_sampled_features = self.conv_down(cnn_features)
        # Down-sample CNN features using average pooling
        down_sampled_features = F.avg_pool1d(down_sampled_features, kernel_size=down_sampled_features.size(1))
        down_sample_out = self.layer_norm(down_sampled_features)

        return down_sample_out


class FusionUpBlock(nn.Module):
    """Fusion block for up-sampling which connect transformer block to 1D-CNN."""

    def __init__(self, up_cnn_in_channels, up_cnn_out_channels, interpolate_size):
        super(FusionUpBlock, self).__init__()
        self.interpolate_size = interpolate_size
        self.conv_up = nn.Conv1d(up_cnn_in_channels, up_cnn_out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(up_cnn_out_channels)

    def forward(self, transformer_features):
        # Up-sample
        up_sampled_features = self.conv_up(transformer_features)
        up_sampled_features = self.batch_norm(up_sampled_features)
        # Transformer features using interpolation
        up_sampled_out = F.interpolate(
            up_sampled_features, size=self.interpolate_size, mode='linear', align_corners=True
        )

        return up_sampled_out


class MTC(nn.Module):
    """Main network to construct transformer, 1D-CNN and fusion blocks"""

    def __init__(
            self,
            seq_len: int = 1500,
            embed_n: int = 30,
            embed_d: int = 50,
            trans_h: int = 5,
            trans_d1: int = 1024
    ):
        super(MTC, self).__init__()
        self.seq_len = seq_len
        self.embed_n = embed_n
        self.embed_d = embed_d

        # Create embedding layer
        self.embedding = CustomEmbedding(embed_n, embed_d)

        # Define the transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_d, nhead=trans_h, dim_feedforward=trans_d1,
                                                   batch_first=True)

        # Create transformer blocks
        self.transformer_blk1 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.transformer_blk2 = nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.transformer_blk3 = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Create each bottleneck as 1D-CNN blocks
        self.cnn_blk1_b1 = Bottleneck(1, 50, 100)
        self.cnn_blk2_b1 = Bottleneck(100, 50, 100)
        self.cnn_blk2_b2 = Bottleneck(100, 50, 100, residual_channels=50)
        self.cnn_blk3_b1 = Bottleneck(100, 100, 100)  # paper said third dim should be 200
        self.cnn_blk3_b2 = Bottleneck(100, 100, 100, residual_channels=100)  # paper said third dim should be 200

        # Create fusion blocks
        self.fusion_blk1_up = FusionUpBlock(30, 50, interpolate_size=seq_len)
        self.fusion_blk1_down = FusionDownBlock(50, 30)

        self.fusion_blk2_up = FusionUpBlock(30, 100, interpolate_size=seq_len)
        self.fusion_blk2_down = FusionDownBlock(100, 30)

        # Create layer normalization layers
        self.layer_norm_1 = nn.LayerNorm([30, 50])
        self.layer_norm_2 = nn.LayerNorm([30, 50])

        # Down sample for cnn
        self.fc = nn.Linear(seq_len, 50)
        self.norm = nn.BatchNorm1d(100)

        # Task-specific layers
        self.task1_output = nn.Linear(50 * 100, len(PREFIX_TO_TRAFFIC_ID))
        self.task2_output = nn.Linear(seq_len, len(PREFIX_TO_APP_ID))
        self.task3_output = nn.Linear(50 * 100, len(AUX_ID))

    def forward(self, x):
        t_x = self.embedding(x)  # Shape(batch, embed_n, embed_d)  # (128, 30, 50)
        t_x = self.transformer_blk1(t_x)  # (128, 30, 50)
        c_x, _ = self.cnn_blk1_b1(x)  # (128, 100, 1500)

        c_x, residual_c_x = self.cnn_blk2_b1(c_x)  # (128, 100, 1500), (128, 50, 1500)
        residual_c_x = self.fusion_blk1_down(residual_c_x)  # (128, 30, 50)
        t_x = self.layer_norm_1(t_x + residual_c_x)  # (128, 30, 50)
        t_x = self.transformer_blk2(t_x)  # (128, 30, 50)
        residual_t_x = self.fusion_blk1_up(t_x)  # (128, 50, 1500)
        c_x, _ = self.cnn_blk2_b2(c_x, residual_t_x)  # (128, 100, 1500), (128, 100, 1500)

        c_x, residual_c_x = self.cnn_blk3_b1(c_x)  # (128, 100, 1500), (128, 100, 1500)
        residual_c_x = self.fusion_blk2_down(residual_c_x)  # (128, 30, 50)
        t_x = self.layer_norm_2(t_x + residual_c_x)  # (128, 30, 50)
        t_x = self.transformer_blk3(t_x)  # (128, 30, 50)

        residual_t_x = self.fusion_blk2_up(t_x)  # (128, 100, 1500)
        c_x, _ = self.cnn_blk3_b2(c_x, residual_t_x)  # (128, 100, 1500)

        c_x = F.relu(self.norm(self.fc(c_x)))

        t_x = torch.flatten(t_x, start_dim=1)
        c_x = torch.flatten(c_x, start_dim=1)

        output1 = self.task1_output(c_x)
        output2 = self.task2_output(t_x)
        output3 = self.task3_output(c_x)

        return output1, output2, output3

print('train code done')

def reduce_tcp(
        packet: Packet,
        n_bytes: int = 20
) -> Packet:
    if TCP in packet:
        # Calculate the TCP header length
        tcp_header_length = packet[TCP].dataofs * 32 / 8

        # Check if the TCP header length is greater than 20 bytes
        if tcp_header_length > n_bytes:
            # Reduce the TCP header length to 20 bytes
            packet[TCP].dataofs = 5  # 5 * 4 = 20 bytes
            del packet[TCP].options  # Remove any TCP options beyond the 20 bytes

            # Recalculate the TCP checksum
            del packet[TCP].chksum
            del packet[IP].chksum
            packet = packet.__class__(bytes(packet))  # Recreate the packet to recalculate checksums

            # Display the modified packet
            # print("Modified Packet:")
            # print(packet.show())
    return packet


def pad_udp(packet: Packet):
    """ Pad the UDP header to 20 bytes with zero.

    Args:
        packet: Scapy packet.

    Returns:
        IP packet.
    """
    if UDP in packet:
        # Get layers after udp
        layer_after = packet[UDP].payload.copy()

        # Build a padding layer
        pad = Padding()
        pad.load = "\x00" * 12

        # Concat the origin payload with padding layer
        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

    return packet


def packet_to_sparse_array(
        packet: Packet,
        max_length: int = 1500
) -> sparse.csr_matrix:
    """ Normalize the byte string and convert to sparse matrix

    Args:
        packet: Scapy packet.
        max_length: Max packet length

    Returns:
        Sparse matrix.
    """
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0:max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)

    arr = sparse.csr_matrix(arr, dtype=np.float32)
    return arr


def filter_packet(pkt: Packet):
    """ Filter packet approach following MTC author.

    Args:
        pkt: Scapy packet.

    Returns:
        Scapy packet if pass all filtering rules. Or None.
    """
    # eliminate Ethernet header with the physical layer information
    if Ether in pkt:
        # print('Ethernet header in packet')
        pkt = pkt[Ether].payload
    else:
        # print('Ethernet header not in packet')
        pass

    # IP header was changed to 0.0.0.0
    if IP in pkt:
        # print('IP header in packet')
        pkt[IP].src = "0.0.0.0"
        pkt[IP].dst = "0.0.0.0"
        # print(pkt[IP].src, pkt[IP].dst, 'after modification')
    else:
        # print('IP header not in packet')
        return None

    if TCP in pkt:
        # print('TCP header in packet')
        # print(f'Len of TCP packet: {len(pkt[TCP])}, payload: {len(pkt[TCP].payload)}')
        pkt = reduce_tcp(pkt)
        # print(f'Len of TCP packet: {len(pkt[TCP])}, payload: {len(pkt[TCP].payload)} after reducing')
    elif UDP in pkt:
        # print('UDP header in packet')
        # print(f'Len of UDP packet: {len(pkt[UDP])}, payload: {len(pkt[UDP].payload)}')
        pkt = pad_udp(pkt)
        # print(f'Len of UDP packet: {len(pkt[UDP])}, payload: {len(pkt[UDP].payload)} after padding')
    else:
        return None

    # Pre-define TCP flags
    FIN = 0x01
    SYN = 0x02
    RST = 0x04
    PSH = 0x08
    ACK = 0x10
    URG = 0x20
    ECE = 0x40
    CWR = 0x80

    # Parsing transport layer protocols using Scapy
    # Checking if it is an IP packet
    if IP in pkt:
        # Obtaining data from the IP layer
        ip_packet = pkt[IP]

        # If it is a TCP protocol
        if TCP in ip_packet:
            # Obtaining data from the TCP layer
            tcp_packet = ip_packet[TCP]
            # Checking for ACK, SYN, FIN flags
            if tcp_packet.flags & 0x16 in [ACK, SYN, FIN]:
                # print('TCP has ACK, SYN, and FIN packets')
                # print(pkt)
                # Returning None (or an empty packet b'')
                return None
        # If it is a UDP protocol
        elif UDP in ip_packet:
            # Obtaining data from the UDP layer
            udp_packet = ip_packet[UDP]
            # Checking for DNS protocol (assuming the value is 53)
            if udp_packet.dport == 53 or udp_packet.sport == 53 or DNS in pkt:
                # print('UDP has DNS packets')
                # print(pkt)
                # Returning None (or an empty packet b'')
                return None
        else:
            # Not a TCP or UDP packet
            return None

        # Valid packet
        return pkt

    else:
        # Not an IP packet
        return None


def preprocess_data(
        pcap_files: List[str],
        limited_count: int = None
):
    """ Perform data preprocessing

    Args:
        pcap_files: List of PCAP file paths.
        limited_count: Limited amount of record to fetch from *.pcap* files.

    Returns:

    """
    data_rows = []

    for pcap_f_name in pcap_files:
        print(f'Load file: {pcap_f_name}')

        pkt_arrays = []

        # Callback function for sniffing
        def method_filter(pkt):
            # Eliminate Ethernet header with the physical layer information
            pkt = filter_packet(pkt)
            # A valid packet would be returned
            if pkt is not None:
                # Convert to sparse matrix
                ary = packet_to_sparse_array(pkt)
                pkt_arrays.append(ary)

        # Limit the number of data for testing
        if limited_count:
            sniff(offline=pcap_f_name, prn=method_filter, store=0, count=limited_count)
        else:
            sniff(offline=pcap_f_name, prn=method_filter, store=0)

        # Concat feature and labels
        for array in pkt_arrays:
            #무작위 값
            row = {
                "app_label": 0,
                "traffic_label": 0,
                "aux_label": 0,
                "feature": array
            }
            data_rows.append(row)

        # Release memory
        del pkt_arrays

    print(f'Save data with {len(data_rows)} rows')

    # Save a preprocessed data to pickle file
    with open('data/test_data_rows.pkl', 'wb') as f:
        pickle.dump(data_rows, f)

import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, MulticlassF1Score, \
    MulticlassConfusionMatrix
from tqdm import tqdm
from typing import List

def test_op(
        model: nn.Module,
        batch_size: int = 128,
        device: str = 'cuda:0',
):
    """ Perform testing.

    Args:
        model: The model for testing.
        batch_size: Batch size.
        device: Device number to serve model.

    Returns:
        Metrics for tasks including the average and the per-class results.
    """
    if not torch.cuda.is_available():
        print('Fail to use GPU')
        device = 'cpu'

    _, _, test_data_rows = load_data()
    test_dataset = get_dataset(test_data_rows)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    task1_outputs = []
    task2_outputs = []
    task3_outputs = []
    predicted_labels = {1: [], 2: [], 3: []}  # 각 task에 대한 예측 레이블 저장

    with torch.no_grad():
        pbar = tqdm(enumerate(test_data_loader), total=len(test_data_loader), desc=f"Testing")
        for batch_idx, (inputs, labels_task1, labels_task2, labels_task3) in pbar:
            inputs = inputs.to(device)
            outputs1, outputs2, outputs3 = model(inputs)

            outputs1 = outputs1.cpu()
            outputs2 = outputs2.cpu()
            outputs3 = outputs3.cpu()

            task1_outputs.append((outputs1, labels_task1))
            task2_outputs.append((outputs2, labels_task2))
            task3_outputs.append((outputs3, labels_task3))

            # 가장 높은 확률의 레이블 인덱스 추출
            predicted_labels[1].extend(torch.argmax(outputs1, dim=1).numpy())
            predicted_labels[2].extend(torch.argmax(outputs2, dim=1).numpy())
            predicted_labels[3].extend(torch.argmax(outputs3, dim=1).numpy())

    task_metrics = []
    for task_outputs, n_classes in zip([task1_outputs, task2_outputs, task3_outputs],
                                       [len(PREFIX_TO_TRAFFIC_ID), len(PREFIX_TO_APP_ID), len(AUX_ID)]):
        total_loss = 0.0
        total_batches = 0

        for outputs, labels in task_outputs:
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches

        task_metrics.append(avg_loss)

    # 각 task별 예측된 레이블 중 가장 많이 등장한 레이블 추출
    final_predicted_labels = {task: max(set(pred_labels), key=pred_labels.count)
                              for task, pred_labels in predicted_labels.items()}

    return task_metrics, final_predicted_labels  # 최종 예측 레이블 반환

def predict():
    pcap_folder = os.getenv("PCAP_FOLDER")  # pcap 파일이 있는 폴더 경로
    pcap_files = glob.glob(f'{pcap_folder}/*.pcap')  # pcap 파일들

    try:
        model = MTC()
        model.load_state_dict(torch.load(os.getenv('MODEL_PATH'), map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Model loading error: {e}")
        return

    for pcap_file in pcap_files:
        try:
            preprocess_data([pcap_file])  # pcap 파일 전처리
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            continue

        start_time = datetime.fromtimestamp(os.path.getctime(pcap_file)).strftime('%Y-%m-%d %H:%M:%S')
        end_time = datetime.fromtimestamp(os.path.getmtime(pcap_file)).strftime('%Y-%m-%d %H:%M:%S')

        try:
            # 모델 예측 수행
            _, final_predicted_labels = test_op(model)

            TRAFFIC_ID_TO_PREFIX = {v: k for k, v in PREFIX_TO_TRAFFIC_ID.items()}
            APP_ID_TO_PREFIX = {v: k for k, v in PREFIX_TO_APP_ID.items()}
            predicted_traffic = TRAFFIC_ID_TO_PREFIX[final_predicted_labels[1]]
            predicted_app = APP_ID_TO_PREFIX[final_predicted_labels[2]]


            # 단일 파일의 예측 결과 전송
            prediction_data = {
                'traffic_count': predicted_traffic,  # 이 파일의 트래픽 카운트
                'app_count': predicted_app,       # 이 파일의 앱 카운트
                'start_time': start_time,
                'end_time' : end_time        
            }

            pie_data = {
                'traffic_count': {final_predicted_labels[1]:1},  # 이 파일의 트래픽 카운트
                'app_count': {final_predicted_labels[2]:1}
            }

            # # numpy.int64를 Python 기본 데이터 유형으로 변환
            pie_data['traffic_count'] = {int(k): int(v) for k, v in pie_data['traffic_count'].items()}
            pie_data['app_count'] = {int(k): int(v) for k, v in pie_data['app_count'].items()}

            # API에 데이터 전송
            response = requests.post('http://localhost:5000/stream/receive_prediction', json=prediction_data)
            response = requests.post('http://localhost:5000/stream/pie_rate', json=pie_data)
            print(f"Prediction data sent to API for file {pcap_file} - Status code: {response.status_code}")


            # live_packet_flow 데이터 생성 및 전송
            file_flows = analyze_pcap(pcap_file)  # PCAP 파일 분석
            processed_data = process_flows(file_flows)  # 흐름 데이터 처리
            try:
                flow_response = requests.post('http://localhost:5000/stream/live_packet_flow', json=processed_data)
                if flow_response.status_code == 200:
                    print(f"Packet flow data sent for file {pcap_file}.")
                else:
                    print(f"Failed to send packet flow data for file {pcap_file}. Status: {flow_response.status_code}")
            except Exception as e:
                print(f"Error sending packet flow data: {e}")


        except Exception as e:
            print(f"Prediction error for file {pcap_file}: {e}")
            continue

        try:
            # 처리된 임시 파일 삭제
            file_path = os.getenv('TEST_PKL')
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"File removal error: {e}")


def model_pcap(pcap_dir):
    processed_files = set()

    while True:  # 무한 루프로 PCAP 계속 처리하기 위해
        
        pcap_files = glob.glob(pcap_dir)
        pcap_files.sort(key=os.path.getctime)  # 생성순으로 정리
        print("model_pcap")

        # 새로 발견된 PCAP 파일에 대해서만 처리
        new_files_found = False
        for pcap_file in pcap_files:
            if pcap_file not in processed_files:  # PCAP 파일이 processed_files에 없을 때
                print(f'Processing PCAP file: {pcap_file}')

                #time.sleep()  # 대기

                preprocess_data([pcap_file])  # PCAP 파일 전처리
                processed_files.add(pcap_file)  # 처리 완료된 파일 추가
                new_files_found = True  # 새 파일이 발견되었음을 표시

                # PCAP 파일 처리 후 테스트 수행
                predict()
                 # receive_prediction 함수를 직접 호출하여 예측 결과 전달
                #receive_prediction(traffic_count, app_count, prediction_messages)
               
        if not new_files_found:
            print("No new PCAP files found...")

        time.sleep(5) #잠깐 대기 >> pcap 생성이 완벽하게 180초가 아닐 때가 있음

#-------------------
load_dotenv('.env')

# 패킷 캡처 관련 환경 변수 설정
server_url = os.getenv('SERVER_URL')
iface_name = os.getenv('IFACE_NAME')
#iface_name = iface_name.replace('\\\\', '\\') 
file_interval = int(os.getenv('FILE_INTERVAL', 15))
pcaps_folder = os.getenv('PCAP_FOLDER')

if not os.path.exists(pcaps_folder):
    os.makedirs(pcaps_folder)  # 패킷 저장 폴더가 없으면 생성

current_file = os.path.join(pcaps_folder, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcap")
last_file_time = time.time()

# Flask 애플리케이션 설정
logging.basicConfig(level=logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

# CORS 헤더 추가
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# 패킷 데이터를 저장하는 리스트
packets = []
ratios = {
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


# 패킷 캡처 및 처리 함수
def packet_handler(packet):
    global last_file_time, current_file

    # 시간 확인
    current_time = time.time()
    if current_time - last_file_time >= file_interval:
        last_file_time = current_time
        current_file = os.path.join(pcaps_folder, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pcap")
        print(f"New file: {current_file}")

    # 패킷 저장
    wrpcap(current_file, packet, append=True)

    packet_summary = packet.summary()
    packet_data = {}

    if packet.haslayer('IP'):
        src_ip = packet['IP'].src
        dst_ip = packet['IP'].dst

        packet_data = {
            'summary': packet_summary,
            'src_ip': src_ip,
            'dst_ip': dst_ip
        }

        # 패킷 데이터를 서버로 전송
        print("captured success")
        requests.post(f'{server_url}/stream/receive_packet', json=packet_data)

        # 서버로 전송된 패킷 데이터를 WebSocket을 통해 클라이언트로 전송
        socketio.emit('new_packet', packet_data)
        packets.append(packet_data)

# 패킷 데이터를 저장할 리스트
packets = []

# POST 요청으로 패킷 데이터를 받아 서버에 저장
@app.route('/stream/receive_packet', methods=['POST'])
def receive_packet():
    packet_data = request.json
    if packet_data:
        print(f'Received packet data: {packet_data}')
        packets.append(packet_data)  # 패킷 데이터를 저장
        socketio.emit('new_packet', packet_data)  # WebSocket으로 클라이언트에 전송
        return jsonify({'status': 'success', 'received_packets': packet_data})
    else:
        return jsonify({'status': 'error', 'message': 'No packet data'}), 400

# GET 요청으로 저장된 패킷 데이터를 반환
@app.route('/stream/receive_packet', methods=['GET'])
def get_packets():
    return jsonify({'packets': packets})  # 저장된 패킷 리스트를 반환


predictions = []
# POST 요청으로 예측 데이터를 받아 predictions 리스트에 추가
@app.route('/stream/receive_prediction', methods=['POST'])
def receive_prediction():
    print("call receive_prediction")
    pred_data = request.json
    if not pred_data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400

    # 필요한 데이터 추출
    traffic_count = pred_data.get("traffic_count", "")
    app_count = pred_data.get("app_count", "")
    start_time = pred_data.get("start_time", "")
    end_time = pred_data.get("end_time", "")

    # 유효성 검사
    if not traffic_count or not app_count or not start_time or not end_time:
        return jsonify({'status': 'error', 'message': 'Invalid data provided'}), 400

    # 예측 데이터를 리스트에 추가
    global predictions
    predictions.append({
        'traffic_count': traffic_count,
        'app_count': app_count,
        'start_time': start_time,
        'end_time': end_time
    })

    # React로 WebSocket 메시지 전송
    socketio.emit('new_prediction_message', {
        'traffic_count': traffic_count,
        'app_count': app_count,
        'start_time': start_time,
        'end_time': end_time
    })

    print("new_prediction_message", predictions)
    # HTTP 응답으로 반환
    return jsonify({'status': 'success', 'prediction': pred_data})

# GET 요청으로 전체 예측 데이터를 반환
@app.route('/stream/receive_prediction', methods=['GET'])
def get_predictions():
    global predictions
    return jsonify({'predictions': predictions})  # 저장된 예측 리스트 반환


@app.route('/stream/pie_rate', methods=['POST'])
def receive_pie():
    """
    Receive traffic and application data, update global ratios.
    """
    global ratios

    # Traffic ID to name mapping
    TRAFFIC_ID_TO_NAME = {0: 'CHAT', 1: 'VOIP', 2: 'STREAMING'}
    APP_ID_TO_NAME = {0: 'facebook', 1: 'discord', 2: 'skype', 3: 'line', 4: 'youtube'}

    try:
        # Parse incoming JSON data
        ratio_data = request.get_json(force=True)
        if isinstance(ratio_data, str):
            ratio_data = json.loads(ratio_data)  # Parse string to dict if necessary

        # Update traffic ratios
        for key, value in ratio_data.get('traffic_count', {}).items():
            # Map numeric traffic ID to traffic name
            traffic_name = TRAFFIC_ID_TO_NAME.get(int(key), key)
            ratios['traffic_ratios'][traffic_name] = ratios['traffic_ratios'].get(traffic_name, 0) + value

        # Update application ratios
        for key, value in ratio_data.get('app_count', {}).items():
            # Map numeric app ID to app name
            app_name = APP_ID_TO_NAME.get(int(key), key)
            ratios['app_ratios'][app_name] = ratios['app_ratios'].get(app_name, 0) + value

        return jsonify({'status': 'success', 'updated_ratios': ratios})
    except Exception as e:
        print(f"Error in receive_pie: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/stream/pie_rate', methods=['GET'])
def send_pie_rate():
    """
    Send traffic and application ratios with names instead of numeric IDs.
    """
    total_traffic = sum(ratios['traffic_ratios'].values())
    total_app = sum(ratios['app_ratios'].values())

    traffic_ratios = {
        name: value / total_traffic * 100 if total_traffic > 0 else 0
        for name, value in ratios['traffic_ratios'].items()
    }

    app_ratios = {
        name: value / total_app * 100 if total_app > 0 else 0
        for name, value in ratios['app_ratios'].items()
    }

    print("Mapped Traffic Ratios:", traffic_ratios)
    print("Mapped Application Ratios:", app_ratios)

    return jsonify({
        'traffic_ratios': traffic_ratios,
        'app_ratios': app_ratios
    })



# 패킷 캡처 시작 함수
def start_packet_capture():
    try:
        print("Starting packet capture...")
        sniff(iface=iface_name, prn=packet_handler, count=0, store=0)
    except Exception as e:
        print(f'Error starting packet capture: {e}')

# 웹소켓 연결 처리
@socketio.on('connect')
def handle_connect():
    print('Client connected')
import os
from scapy.all import rdpcap, TCP, UDP
from datetime import datetime
import pandas as pd
from collections import defaultdict
from flask import Flask, jsonify

# Function to identify flows based on 5-tuple
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
        return None  # Only process TCP/UDP packets

    return (packet[0][1].src, src_port, packet[0][1].dst, dst_port, protocol)

# Identify application traffic based on port number
def get_application_type(flow):
    app_dict = {
        80: 'HTTP',
        443: 'HTTPS',
        53: 'DNS',
    }
    if flow[1] in app_dict:
        return app_dict[flow[1]]
    elif flow[3] in app_dict:
        return app_dict[flow[3]]
    else:
        return 'Other'

# Analyze pcap file and group flows by time
def analyze_pcap(file_path):
    packets = rdpcap(file_path)
    flows = defaultdict(list)

    for packet in packets:
        flow = get_flow(packet)
        if flow:
            try:
                timestamp = float(packet.time)
                timestamp = datetime.fromtimestamp(timestamp)  # Ensure datetime format
                app_type = get_application_type(flow)
                protocol = flow[4]  # TCP or UDP
                flows[(flow, app_type, protocol)].append(timestamp)
            except Exception as e:
                print(f"Error processing packet: {e}")

    return flows


# Group and count protocols by 30-second intervals
from datetime import datetime

def process_flows(flows):
    """
    Process flow data and group by 30-second intervals.
    """
    processed_data = {}
    
    for (flow, app_type, protocol), timestamps in flows.items():
        for timestamp in timestamps:
            # Ensure timestamp is a datetime object
            if not isinstance(timestamp, datetime):
                timestamp = datetime.fromtimestamp(timestamp)  # Convert if it's not already a datetime
            
            # Round to the nearest 30-second interval
            rounded_time = timestamp.replace(second=(timestamp.second // 30) * 30, microsecond=0)
            rounded_time_str = rounded_time.strftime('%Y-%m-%d %H:%M:%S')

            if rounded_time_str not in processed_data:
                processed_data[rounded_time_str] = {'TCP': 0, 'UDP': 0, 'Other': 0}
            
            # Increment protocol count
            if protocol == 'TCP':
                processed_data[rounded_time_str]['TCP'] += 1
            elif protocol == 'UDP':
                processed_data[rounded_time_str]['UDP'] += 1
            else:
                processed_data[rounded_time_str]['Other'] += 1

    return processed_data

@app.route('/stream/live_packet_flow', methods=['GET'])
def live_packet_flow():
    pcap_dir = os.getenv('PCAP_FOLDER')
    all_flows = defaultdict(list)

    # Analyze all PCAP files in the directory
    for file_name in os.listdir(pcap_dir):
        if file_name.endswith('.pcap'):
            file_path = os.path.join(pcap_dir, file_name)
            file_flows = analyze_pcap(file_path)  # Analyze the PCAP file
            
            # Ensure the timestamps are added correctly
            for (flow, app_type, protocol), timestamps in file_flows.items():
                all_flows[(flow, app_type, protocol)].extend(timestamps)  # Accumulate timestamps

    # Process the flows and count protocols
    try:
        processed_data = process_flows(all_flows)
        return jsonify(processed_data)
    except Exception as e:
        print(f"Error processing flows: {e}")
        return jsonify({"error": "Failed to process flows"}), 500


# Process multiple pcap files and return JSON data
@app.route('/stream/pie_rate', methods=['GET'])
def get_packet_data():
    pcap_dir = os.getenv('PCAP_FOLDER') # Adjust to your actual pcap file path
    all_flows = {}

    for file_name in os.listdir(pcap_dir):
        if file_name.endswith('.pcap'):
            file_path = os.path.join(pcap_dir, file_name)
            flows = analyze_pcap(file_path)
            all_flows[file_name] = flows

    # Convert flows into a format suitable for JSON response
    flow_data = defaultdict(lambda: defaultdict(int))

    for file_name, flows in all_flows.items():
        for (flow, app_type, protocol), timestamps in flows.items():
            df = pd.DataFrame(timestamps, columns=['timestamp'])
            df['half_minute'] = df['timestamp'].dt.floor('30s')  # Round to 30s intervals
            packet_counts = df.groupby('half_minute').size()

            for time, count in packet_counts.items():
                flow_data[time][(app_type, protocol)] += count

    # Convert defaultdict to normal dict for JSON response
    flow_data_json = {str(time): {f"{app_type}_{protocol}": count for (app_type, protocol), count in counts.items()} for time, counts in flow_data.items()}

    return jsonify(flow_data_json)

# 애플리케이션 시작
if __name__ == '__main__':
    pcap_dir = 'pcaps/*.pcap'
    socketio.start_background_task(model_pcap, pcap_dir)

        # 패킷 캡처를 위한 별도 스레드 시작
    packet_capture_thread = threading.Thread(target=start_packet_capture, daemon=True)
    packet_capture_thread.start()

    socketio.run(app, host='0.0.0.0', port=5000)
