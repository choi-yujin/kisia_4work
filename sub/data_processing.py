import pickle
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Any, Dict, Union
import random
from typing import Any, List, Union
# import dpkt
import datetime
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scapy.all import sniff
from scapy.compat import raw
from scapy.layers.dns import DNS
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding, Packet
from scipy import sparse
from sklearn.model_selection import train_test_split
import glob
from scapy.all import rdpcap
from flask import Flask, jsonify, render_template, request
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv('.env')

app = Flask(__name__)

# 트래픽과 애플리케이션 타입을 매핑하는 딕셔너리
PREFIX_TO_TRAFFIC_ID = {
    'chat': 0,
    'voip': 1,
    'streaming': 2,
}

PREFIX_TO_APP_ID = {
    'facebook': 0,
    'discord': 1,
    'skype': 2,
    'line': 3,
    'youtube': 4
}

AUX_ID = {
    'all_chat': 0,
    'all_voip': 1,
    'all_streaming': 2,
}

def id_to_one_hot_tensor(id_value: Union[int, torch.Tensor], num_classes: int):
    if isinstance(id_value, int):
        id_value = torch.tensor(id_value)

    one_hot_tensor = torch.nn.functional.one_hot(id_value, num_classes=num_classes)
    return one_hot_tensor.to(torch.float32)

class CustomListDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.data = rows
        self.n_traffic = len(PREFIX_TO_TRAFFIC_ID)
        self.n_app = len(PREFIX_TO_APP_ID)
        self.n_aux = len(AUX_ID)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        y_traffic = id_to_one_hot_tensor(d['traffic_label'], self.n_traffic)
        y_app = id_to_one_hot_tensor(d['app_label'], self.n_app)
        y_aux = id_to_one_hot_tensor(d['aux_label'], self.n_aux)
        sample = (torch.from_numpy(d['feature'].toarray()), y_traffic, y_app, y_aux)
        return sample

def get_dataset(data_rows: List[Dict[str, Any]]) -> Dataset:
    ds = CustomListDataset(data_rows)
    return ds

def load_data() -> Tuple[Any, Any, Any]:
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
        """ Initialize MTC model with expected inputs with shape(B, C, L). `B` is batch size, `C` is channel and
         `L` is sequence length.

        Args:
            seq_len: Sequence length.
            embed_n: Embedding channels.
            embed_d: Embedding dimensions.
            trans_h: Number of heads in transformer block.
            trans_d1: Dimensions of the first feedforward net in transformer block.
        """
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




def reduce_tcp(packet: Packet, n_bytes: int = 20) -> Packet:
    if TCP in packet:
        tcp_header_length = packet[TCP].dataofs * 32 / 8
        if tcp_header_length > n_bytes:
            packet[TCP].dataofs = 5
            del packet[TCP].options
            del packet[TCP].chksum
            del packet[IP].chksum
            packet = packet.__class__(bytes(packet))
    return packet

def pad_udp(packet: Packet):
    if UDP in packet:
        layer_after = packet[UDP].payload.copy()
        pad = Padding()
        pad.load = "\x00" * 12
        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after
    return packet

def packet_to_sparse_array(packet: Packet, max_length: int = 1500) -> sparse.csr_matrix:
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0:max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    arr = sparse.csr_matrix(arr, dtype=np.float32)
    return arr

def filter_packet(pkt: Packet):
    if Ether in pkt:
        pkt = pkt[Ether].payload
    if IP in pkt:
        pkt[IP].src = "0.0.0.0"
        pkt[IP].dst = "0.0.0.0"
    else:
        return None

    if TCP in pkt:
        pkt = reduce_tcp(pkt)
    elif UDP in pkt:
        pkt = pad_udp(pkt)
    else:
        return None

    if IP in pkt:
        if TCP in pkt and (pkt[TCP].flags & 0x16 in [0x10, 0x02, 0x01]):
            return None
        elif UDP in pkt and (pkt[UDP].dport == 53 or pkt[UDP].sport == 53 or DNS in pkt):
            return None
        return pkt
    return None


def preprocess_data(pcap_files: str, limited_count: int = None) -> List[Dict[str, Any]]:
    """패킷을 필터링하고 전처리하여 데이터 리스트를 반환하는 함수"""
    data_rows = []
    print(f'Load file: {pcap_files}')
    pkt_arrays = []

    def method_filter(pkt):
        pkt = filter_packet(pkt)
        if pkt is not None:
            ary = packet_to_sparse_array(pkt)
            pkt_arrays.append(ary)

    if limited_count:
        sniff(offline=pcap_files, prn=method_filter, store=0, count=limited_count)
    else:
        sniff(offline=pcap_files, prn=method_filter, store=0)

    for array in pkt_arrays:
        row = {
            "app_label": 0,
            "traffic_label": 0,
            "aux_label": 0,
            "feature": array
        }
        data_rows.append(row)

    del pkt_arrays
    print(f'Save data with {len(data_rows)} rows')

    return data_rows  # 데이터 리스트 반환

# @app.route('/batch/ask_predict', methods=['POST'])
def ask_predict():
    """사용자 요청에 따라 패킷을 필터링하고 예측 결과를 반환하는 API"""
    data = request.json
    selected_date = data.get('selected_date')
    start_time = data.get('start_time')
    end_time = data.get('end_time')

    if not selected_date:
        return jsonify({"error": "selected_date is required"}), 400

    # PCAP 파일 로드
    pcap_folder = os.getenv("PCAP_FOLDER")
    pcap_files = glob.glob(f'{pcap_folder}/{selected_date}*.pcap')

    if start_time and end_time:
        start_time = start_time.replace(':', '')
        end_time = end_time.replace(':', '')
        pcap_files = [
            f for f in pcap_files
            if start_time <= f[-9:-5] <= end_time
        ]

    # 예측 데이터 준비
    all_data_rows = []
    for pcap_file in pcap_files:
        data_rows = preprocess_data(pcap_file)
        all_data_rows.extend(data_rows)

    # 모델 로드 및 예측
    model = MTC()  # 모델 인스턴스 생성
    model.load_state_dict(torch.load(os.getenv('MODEL_PATH')))
    model.eval()

    traffic_labels = []
    app_labels = []
    prediction_messages = []

    with torch.no_grad():
        for row in all_data_rows:
            features = torch.from_numpy(row['feature'].toarray()).unsqueeze(0)  # 배치 차원 추가
            outputs = model(features)
            traffic_pred = torch.argmax(outputs[0], dim=1).item()  # 트래픽 예측
            app_pred = torch.argmax(outputs[1], dim=1).item()  # 애플리케이션 예측

            traffic_labels.append(traffic_pred)
            app_labels.append(app_pred)

            # 예측 메시지 작성
            predicted_traffic = next((k for k, v in PREFIX_TO_TRAFFIC_ID.items() if v == traffic_pred), "Unknown")
            predicted_app = next((k for k, v in PREFIX_TO_APP_ID.items() if v == app_pred), "Unknown")
            message = f"{pcap_file}에서 {predicted_app}를 사용하여 {predicted_traffic} traffic이 발생된 것으로 예상됩니다."
            prediction_messages.append(message)

    # 예측 결과 비율 계산
    traffic_count = Counter(traffic_labels)
    app_count = Counter(app_labels)

    return jsonify({
        "traffic_count": dict(traffic_count),
        "app_count": dict(app_count),
        "messages": prediction_messages
    })

if __name__ == '__main__':
    app.run(debug=True)