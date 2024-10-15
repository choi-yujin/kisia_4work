import pickle
import random
from typing import Any, List, Union

import numpy as np
import pandas as pd
from scapy.all import sniff
from scapy.compat import raw
from scapy.layers.dns import DNS
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding, Packet
from scipy import sparse
import glob

# Traffic type labels
PREFIX_TO_TRAFFIC_ID = {
    'chat': 0,
    'voip': 1,
    'streaming': 2
}

# Application labels
PREFIX_TO_APP_ID = {
    'facebook': 0,
    'discord': 1,
    'skype': 2,
    'line': 3,
    'youtube': 4
}

# Auxiliary task labels
AUX_ID = {
    'all_chat': 0,
    'all_voip': 1,
    'all_streaming': 2
}


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



def preprocess_data(limited_count: int = None):
    """pcap 파일을 전처리하여 pkl 형식으로 저장하는 함수."""
    pcap_files = glob.glob(f'./*.pcap')
    
    if not pcap_files:
        print(f"pcap 파일이 디렉토리에 존재하지 않습니다.")
        return

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
        output_file = f'./{pcap_f_name.split("./")[-1].replace(".pcap", ".pkl")}'
        with open(output_file, 'wb') as f:
            pickle.dump(data_rows, f)
