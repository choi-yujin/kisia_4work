#파이차트 관련 예측하는 모델 파일
import pickle
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Any, Dict, Union
import random
from typing import Any, List, Union
import dpkt
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
import os

from dotenv import load_dotenv


# .env 파일 로드
load_dotenv('mtc.env')

app = Flask(__name__,static_folder='../static', template_folder='../templates')


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
    """
    Convert an ID to a one-hot encoded tensor using PyTorch.

    Parameters:
    - id_value (int or Tensor): The ID value(s) to be converted to a one-hot tensor.
    - num_classes (int): Total number of classes/categories.

    Returns:
    - one_hot_tensor (Tensor): The one-hot encoded tensor.
    """
    # Convert int to tensor if single value
    if isinstance(id_value, int):
        id_value = torch.tensor(id_value)

    one_hot_tensor = torch.nn.functional.one_hot(id_value, num_classes=num_classes  )
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




def reduce_tcp(
        packet: Packet,
        n_bytes: int = 20
) -> Packet:
    """ Reduce the size of TCP header to 20 bytes.

    Args:
        packet: Scapy packet.
        n_bytes: Number of bytes to reserve.

    Returns:
        IP packet.
    """
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
        Scapy packet if pass all filtering rules. Or `None`.
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
        pcap_files:str,
        limited_count: int = None
):
    """ Perform data preprocessing

    Args:
        pcap_files: List of PCAP file paths.
        limited_count: Limited amount of record to fetch from *.pcap* files.

    Returns:

    """
    data_rows = []

   
    print(f'Load file: {pcap_files}')

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
        sniff(offline=pcap_files, prn=method_filter, store=0, count=limited_count)
    else:
        sniff(offline=pcap_files, prn=method_filter, store=0)

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


def get_packet_times(pcap_file):
    with open(pcap_file, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)

        start_timestamp = None
        last_timestamp = None

        # 패킷의 타임스탬프를 반복하여 첫 번째와 마지막 타임스탬프를 기록
        for i, (timestamp, _) in enumerate(pcap):
            if i == 0:  # 첫 번째 패킷의 타임스탬프
                start_timestamp = timestamp
            last_timestamp = timestamp  # 마지막 패킷의 타임스탬프는 계속 업데이트

        if start_timestamp and last_timestamp:
            # Unix timestamp를 사람이 읽을 수 있는 시간 형식으로 변환
            start_time = datetime.datetime.utcfromtimestamp(start_timestamp).strftime('%H:%M:%S')
            last_time = datetime.datetime.utcfromtimestamp(last_timestamp).strftime('%H:%M:%S')
            return start_time, last_time
        else:
            return None, None


def predict(selected_date, start_time=None, end_time=None):
    pcap_folder = os.getenv("PCAP_FOLDER")  # pcap 파일이 있는 폴더 경로
    pcap_files = glob.glob(f'{pcap_folder}/{selected_date}*.pcap')  # 날짜에 해당하는 모든 pcap 파일

    if start_time and end_time:
        # ':' 문자를 제거하여 시분을 정리합니다.
        start_time = start_time.replace(':','')  # 1130 형식으로 변환
        end_time = end_time.replace(':','')      # 1230 형식으로 변환
        for f in pcap_files:
            file_time = f[-9:-5]  # HHMM 부분 추출
            print(f"File: {f}, Extracted Time: {file_time}")
        # 선택된 날짜의 파일 중 시간 범위 내에 있는 파일을 필터링합니다.
        pcap_files = [
            f for f in pcap_files
            if f.endswith('.pcap') and len(f) >= len(f"{pcap_folder}/{selected_date}") + 4 and  # 파일 이름이 충분히 긴지 확인
            start_time <= f[-9:-5] <= end_time  # 마지막 4자리 (시간) 비교
        ]
    # 이하 기존 코드 동일

    traffic_labels = []
    app_labels = []
    prediction_messages = []  # 예측 문장 리스트

    model = MTC()
    model.load_state_dict(torch.load(os.getenv('MODEL_PATH')))
    for pcap_file in pcap_files:  # 여러 pcap 파일을 처리
        preprocess_data(pcap_file)
        print(pcap_file)
        _, final_predicted_labels = test_op(model)
        start_time, last_time = get_packet_times(pcap_file)

        # 예측 결과 저장
        traffic_labels.append(final_predicted_labels[1])
        app_labels.append(final_predicted_labels[2])

        TRAFFIC_ID_TO_PREFIX = {v: k for k, v in PREFIX_TO_TRAFFIC_ID.items()}
        APP_ID_TO_PREFIX = {v: k for k, v in PREFIX_TO_APP_ID.items()}

        predicted_traffic = TRAFFIC_ID_TO_PREFIX[final_predicted_labels[1]]
        predicted_app = APP_ID_TO_PREFIX[final_predicted_labels[2]]

        if start_time and last_time:
            message = f"{start_time} ~ {last_time} ip 192.168.132.252에서 {predicted_app}를 사용하여 {predicted_traffic} traffic이 발생된걸로 예상됩니다"
        else:
            message = "패킷 정보를 불러올 수 없습니다."

        prediction_messages.append(message)

        # 중간 파일 제거
        file_path = os.getenv('TEST_PKL')
        if os.path.exists(file_path):
            os.remove(file_path)

    # 예측 결과 비율 계산
    traffic_count = Counter(traffic_labels)
    app_count = Counter(app_labels)

    return traffic_count, app_count, prediction_messages

import os
import glob
from flask import jsonify, request

def count_protocols(pcap_file):
    # 프로토콜 카운트 초기화
    protocol_counts = {
        'TCP': 0,
        'UDP': 0,
        'ICMP': 0,
        'HTTP': 0,
        'HTTPS': 0,
        'FTP': 0,
        'SMTP': 0,
        'DNS': 0,
        'SSH': 0,
        'ARP': 0,
        'Other': 0,
    }

    # pcap 파일 읽기
    packets = rdpcap(pcap_file)

    # 각 패킷을 분석
    for packet in packets:
        if packet.haslayer('IP'):
            protocol = packet['IP'].proto  # IP 레이어의 프로토콜 번호 가져오기
            
            # 프로토콜 번호에 따라 카운트 업데이트
            if protocol == 6:  # TCP
                protocol_counts['TCP'] += 1
                if packet.haslayer('HTTP'):
                    protocol_counts['HTTP'] += 1
                elif packet.haslayer('HTTPS'):
                    protocol_counts['HTTPS'] += 1
                elif packet.haslayer('FTP'):
                    protocol_counts['FTP'] += 1
            elif protocol == 17:  # UDP
                protocol_counts['UDP'] += 1
                if packet.haslayer('DNS'):
                    protocol_counts['DNS'] += 1
            elif protocol == 1:  # ICMP
                protocol_counts['ICMP'] += 1
            elif protocol == 204:  # ARP
                protocol_counts['ARP'] += 1
            else:
                protocol_counts['Other'] += 1  # 기타 프로토콜

    return protocol_counts

def get_protocol_counts(date, start_time=None, end_time=None):
    pcap_folder = os.getenv("PCAP_FOLDER")  # pcap 파일이 있는 폴더 경로
    pcap_files = glob.glob(f'{pcap_folder}/{date}*.pcap')  # 날짜에 해당하는 모든 pcap 파일

    if start_time and end_time:
        # ':' 문자를 제거하여 시분을 정리합니다.
        start_time = start_time.replace(':','')  # 1130 형식으로 변환
        end_time = end_time.replace(':','')      # 1230 형식으로 변환
        for f in pcap_files:
            file_time = f[-9:-5]  # HHMM 부분 추출
            print(f"File: {f}, Extracted Time: {file_time}")
        # 선택된 날짜의 파일 중 시간 범위 내에 있는 파일을 필터링합니다.
        pcap_files = [
            f for f in pcap_files
            if f.endswith('.pcap') and len(f) >= len(f"{pcap_folder}/{date}") + 4 and  # 파일 이름이 충분히 긴지 확인
            start_time <= f[-9:-5] <= end_time  # 마지막 4자리 (시간) 비교
        ]

    print(f"Initial PCAP Files: {pcap_files}")  # 디버깅을 위한 출력
    protocol_counts = {}

    for pcap_file in pcap_files:
        counts = count_protocols(pcap_file)
        
        # 프로토콜 카운트를 총합
        for protocol, count in counts.items():
            if protocol in protocol_counts:
                protocol_counts[protocol] += count
            else:
                protocol_counts[protocol] = count
    
    # 0이 아닌 프로토콜만 필터링
    protocol_counts = {k: v for k, v in protocol_counts.items() if v > 0}
    return protocol_counts


@app.route('/protocol_counts', methods=['GET'])
def protocol_counts():
    date = request.args.get('date')  # 클라이언트에서 전송한 날짜를 받음
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    
    if not date:
        return jsonify({'error': '날짜를 선택해주세요.'}), 400
    
    protocol_counts = get_protocol_counts(date, start_time, end_time)
    return jsonify(protocol_counts)

@app.route('/predict', methods=['GET'])
def send_pie():
    selected_date = request.args.get('date')  # 클라이언트에서 전송한 날짜를 받음
    start_time = request.args.get('start_time')  # 선택된 시작 시간
    end_time = request.args.get('end_time')      # 선택된 종료 시간

    if not selected_date:
        return jsonify({'error': '날짜를 선택해주세요.'}), 400

    # 선택된 날짜와 시간대를 기반으로 예측 함수 호출
    traffic_count, app_count, prediction_messages = predict(selected_date, start_time, end_time)

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


@app.route('/')
def index():
    return render_template('sub.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)