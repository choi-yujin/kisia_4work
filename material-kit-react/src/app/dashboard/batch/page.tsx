// import * as React from 'react';
// import type { Metadata } from 'next';
// import Grid from '@mui/material/Unstable_Grid2';
// import dayjs from 'dayjs';

// import { config } from '@/config';
// import { Predict_traffic } from '@/components/dashboard/batch/tls-prediction';
// import { Calender } from '@/components/dashboard/batch/calender';
// import { Traffic } from '@/components/dashboard/batch/traffic';
// import { Application } from '@/components/dashboard/batch/application';
// import { Protocol } from '@/components/dashboard/batch/protocol';
// import { History } from '@/components/dashboard/batch/history';
// import { height } from '@mui/system';

// export const metadata = { title: `batch | Dashboard | ${config.site.name}` } satisfies Metadata;

// export default function Page(): React.JSX.Element {
//   return (
//     <Grid container spacing={3}justifyContent="center">
//       <Grid lg={3} xs={12} sx={{ height: '100%' }}>
//         <Calender/>
//       </Grid>
//       <Grid lg={3} md={6} xs={12}>
//         <Traffic chartSeries={[0, 0, 0]} labels={['Chat', 'Voip', 'Streaming']} sx={{ height: '100%'}}/>
//       </Grid>
//       <Grid lg={3} md={6} xs={12}>
//         <Application chartSeries={[0, 0, 0, 0,0]} labels={['Facebook', 'Discord', 'Skype','Line','Youtube']} sx={{ height: '100%' }} />
//       </Grid>
//       <Grid lg={3} md={6} xs={12}>
//         <Protocol chartSeries={[0, 0, 0]} labels={['TCP', 'UDP', 'DNS']} sx={{ height: '100%' }} />
//       </Grid>
//       <Grid lg={8} md={12} xs={12}>
//         <Predict_traffic
//           predicts={[
//           ]}
//           sx={{ height: '100%' }}
//         />
//       </Grid>
//       <Grid lg={4} md={6} xs={12}>
//         <History
//           products={[
//           ]}
//           sx={{ height: '100%' }}
//         />
//       </Grid>
//     </Grid>
//   );
// }
// Page.tsx
'use client';
import * as React from 'react';
import Grid from '@mui/material/Unstable_Grid2';
import { config } from '@/config';
import { Predict_traffic } from '@/components/dashboard/batch/tls-prediction';
import { Calender } from '@/components/dashboard/batch/calender';
import { Traffic } from '@/components/dashboard/batch/traffic';
import { Application } from '@/components/dashboard/batch/application';
import { Protocol } from '@/components/dashboard/batch/protocol';
import { History } from '@/components/dashboard/batch/history';

export default function Page(): React.JSX.Element {
  // 상태를 정의합니다.
  const [predictionData, setPredictionData] = React.useState([]); // 예측 데이터 상태
  const [trafficData, setTrafficData] = React.useState<Record<string, number> | null>(null); // 트래픽 데이터 상태
  const [appData, setAppData] = React.useState<Record<string, number> | null>(null); // 애플리케이션 데이터 상태
  const [protocolData, setProtocolData] = React.useState<Record<string, number> | null>(null); // 프로토콜 데이터 상태

  // Calender에서 호출되는 데이터 처리 함수
  const handleFetchPredictions = (data: any) => {
    // 데이터를 상태에 저장합니다.
    setPredictionData(data.predictions);
    setTrafficData(data.traffic);
    setAppData(data.applications);
    setProtocolData(data.protocols);

    console.log('Prediction Data:', data.predictions);
    console.log('Traffic Data:', data.traffic);
    console.log('Application Data:', data.applications);
    console.log('Protocol Data:', data.protocols);
  };

  return (
    <Grid container spacing={3} justifyContent="center">
      <Grid lg={3} xs={12} sx={{ height: '100%' }}>
        {/* Calender 컴포넌트에 onFetchPredictions 전달 */}
        <Calender onFetchPredictions={handleFetchPredictions} />
      </Grid>
      <Grid lg={3} md={6} xs={12}>
        <Traffic 
          chartSeries={trafficData ? Object.values(trafficData).map(value => value * 100) : [0, 0, 0]} 
          labels={trafficData ? Object.keys(trafficData) : ['Chat', 'VoIP', 'Streaming']} 
          sx={{ height: '100%' }} 
        />
      </Grid>
      <Grid lg={3} md={6} xs={12}>
        <Application 
          chartSeries={appData ? Object.values(appData).map(value => value * 100) : [0, 0, 0, 0, 0]} 
          labels={appData ? Object.keys(appData) : ['Facebook', 'Discord', 'Skype', 'Line', 'Youtube']} 
          sx={{ height: '100%' }} 
        />
      </Grid>
      <Grid lg={3} md={6} xs={12}>
        <Protocol 
          chartSeries={protocolData ? Object.values(protocolData) : [0, 0, 0]} 
          labels={['TCP', 'UDP', 'DNS']} 
          sx={{ height: '100%' }} 
        />
      </Grid>
      <Grid lg={8} md={12} xs={12}>
        <Predict_traffic predicts={predictionData} sx={{ height: '100%' }} />
      </Grid>
      <Grid lg={4} md={6} xs={12}>
        <History products={[]} sx={{ height: '100%' }} />
      </Grid>
    </Grid>
  );
}
