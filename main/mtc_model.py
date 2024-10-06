import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, List, Any, Dict, Union

import pickle
import random
from typing import Any, List, Union
import matplotlib.pyplot as plt
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

from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, MulticlassF1Score, \
    MulticlassConfusionMatrix
from tqdm import tqdm
from typing import List

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
        """ Initialize MTC model with expected inputs with shape(B, C, L). B is batch size, C is channel and
         L is sequence length.

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

print('train code done')

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
                print(f'packet processed: {pkt}') #유효 패킷
            # else:
            #      print(f'packet filtered out: {pkt}') #필터링 처리된 패킷

        print(f'sniff from: {pcap_f_name}')

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

# def model_pcap(pcap_dir):
#     processed_files=set()

#     while True: #무한 루프로 pcap 계속 처리하기 위해
        
#         pcap_files = glob.glob(pcap_dir)  
#         pcap_files.sort(key=os.path.getctime) #생성순으로 정렬해서 중복 생기지 않게 하려고

#         for pcap_file in pcap_files:
#             if pcap_file not in processed_files: #pcap_file processed_files에 있지 않을 때
#                 print(f'pcap file: {pcap_file}')
#                 preprocess_data(pcap_file)
#                 processed_files.add(pcap_file) #전처리 완료된 패킷은 processed_files에 추가

#         print("before time_sleep")
#         time.sleep(10)
    
# if __name__ == "__main__":

#     pcap_dir='/*.pcap'
#     model_pcap(pcap_dir)
    
    
    

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


def testing():
    testing_models = [MTC]  # MTC 모델만 테스트하도록 설정
    model_metrics = dict()
    prediction_results={}

    for module in testing_models:
        m = module()
        print(f'Test {m.__class__.__name__} model...')
        m.load_state_dict(torch.load('MTC_model_1.pt',weights_only=True)) #weights_only: 보안 문제 경고 해결
        metrics, final_predicted_labels = test_op(m)
        model_metrics[m.__class__.__name__] = metrics

        # 최종 예측된 레이블 출력
        print(f'Final predicted label for {m.__class__.__name__}:')
        print(f'Traffic ID: {final_predicted_labels[1]}')
        print(f'APP ID: {final_predicted_labels[2]}')
        print(f'AUX ID  : {final_predicted_labels[3]}')

        # from flask_socketio import SocketIO
        import requests
        import json

        # 반대로 매핑 >> key 받아오고, flask 출력했을 때 0,1,2 대신 chat,voip 등으로 표시
        TRAFFIC_ID_TO_PREFIX = {v: k for k, v in PREFIX_TO_TRAFFIC_ID.items()}
        APP_ID_TO_PREFIX = {v: k for k, v in PREFIX_TO_APP_ID.items()}
        AUX_ID_TO_PREFIX = {v: k for k, v in AUX_ID.items()}

        prediction_results['Traffic ID'] = TRAFFIC_ID_TO_PREFIX.get(final_predicted_labels[1])
        prediction_results['APP ID'] = APP_ID_TO_PREFIX.get(final_predicted_labels[2])
        prediction_results['AUX ID'] = AUX_ID_TO_PREFIX.get(final_predicted_labels[3])
        # print(f'final_labels: {final_predicted_labels}')
        print(f'prediction results: {prediction_results}')

         # 유효성 검사
        if None in prediction_results.values():
            raise ValueError("ID:None")
        
        response=requests.post('http://192.168.219.109:5000/receive_prediction',json=prediction_results)
        print(f'-Status code: {response.status_code}')

    
def model_pcap(pcap_dir):
    processed_files = set()

    while True:  # 무한 루프로 PCAP 계속 처리하기 위해
        
        pcap_files = glob.glob(pcap_dir)
        pcap_files.sort(key=os.path.getctime)  # 생성순으로 정렬

        # 새로 발견된 PCAP 파일에 대해서만 처리
        new_files_found = False
        for pcap_file in pcap_files:
            if pcap_file not in processed_files:  # PCAP 파일이 processed_files에 없을 때
                print(f'Processing PCAP file: {pcap_file}')

                time.sleep(180)  # 대기

                preprocess_data([pcap_file])  # PCAP 파일 전처리
                processed_files.add(pcap_file)  # 처리 완료된 파일 추가
                new_files_found = True  # 새 파일이 발견되었음을 표시

                # PCAP 파일 처리 후 테스트 수행
                testing()

        if not new_files_found:
            print("No new PCAP files found...")

        time.sleep(10) #잠깐 대기 >> pcap 생성이 완벽하게 180초가 아닐 때가 있음

if __name__ == '__main__':
    pcap_dir = '*.pcap'  # PCAP 파일 경로 설정
    model_pcap(pcap_dir)
