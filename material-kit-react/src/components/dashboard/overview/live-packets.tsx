'use client';
import * as React from 'react'; 
import { useEffect, useState } from 'react';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import Divider from '@mui/material/Divider';
import { ArrowClockwise as ArrowClockwiseIcon } from '@phosphor-icons/react/dist/ssr/ArrowClockwise';

// 패킷 데이터 타입 정의
interface Packet {
  timestamp: string;
  summary: string;
  src_ip: string;
  dst_ip: string;
}

export function Livepacket(): React.JSX.Element {
  const [packets, setPackets] = useState<Packet[]>([]); // 패킷 배열에 대한 타입 명시

  // 서버로부터 실시간 패킷 데이터를 받아오는 함수
  const fetchLivePackets = () => {
    fetch('/stream/live_packets')  // Flask에서 패킷 데이터를 제공하는 API
      .then(res => res.json())
      .then(data => setPackets(data.packets))  // data.packets로 리스트에 접근
      .catch(err => console.error("Failed to fetch live packets:", err));
  };

  useEffect(() => {
    // 컴포넌트가 처음 렌더링될 때 패킷을 받아오고, 5초마다 갱신
    fetchLivePackets();
    const interval = setInterval(fetchLivePackets, 5000);

    return () => clearInterval(interval);  // 컴포넌트가 사라질 때 인터벌을 정리
  }, []);

  return (
    <Card>
      <CardHeader
        action={
          <Button color="inherit" size="small" onClick={fetchLivePackets} startIcon={<ArrowClockwiseIcon fontSize="var(--icon-fontSize-md)" />}>
            Sync
          </Button>
        }
        title="Live Packets"
      />
      <CardContent>
        <Divider />
        <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
          {packets.length === 0 ? (
            <p>No packets received yet.</p>
          ) : (
            packets.map((packet, index) => (
              <div key={index} style={{ marginBottom: '10px' }}>
                <strong>Timestamp:</strong> {packet.timestamp} <br />
                <strong>Summary:</strong> {packet.summary} <br />
                <strong>Source IP:</strong> {packet.src_ip} <br />
                <strong>Destination IP:</strong> {packet.dst_ip} <br />
                <Divider />
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  );
}
