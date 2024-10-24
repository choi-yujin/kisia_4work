// 'use client';

// import { fetchPredictions } from './fetchPredictions'; // Flask API 호출 함수
// import { useState } from 'react';
// import dayjs, { Dayjs } from 'dayjs';
// import * as React from 'react';
// import Button from '@mui/material/Button';
// import Card from '@mui/material/Card';
// import CardActions from '@mui/material/CardActions';
// import CardContent from '@mui/material/CardContent';
// import CardHeader from '@mui/material/CardHeader';
// import Divider from '@mui/material/Divider';
// import { DatePicker, TimePicker } from '@mui/x-date-pickers'; // Date and Time Pickers
// import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
// import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs'; // Adapter for dayjs
// import { ArrowRight as ArrowRightIcon } from '@phosphor-icons/react/dist/ssr/ArrowRight';

// export function Calender(): React.JSX.Element {
//   const [selectedDate, setSelectedDate] = useState<Dayjs | null>(dayjs());  // 날짜 선택 상태
//   const [startTime, setStartTime] = useState<Dayjs | null>(dayjs());  // 시작 시간 선택 상태
//   const [endTime, setEndTime] = useState<Dayjs | null>(dayjs());  // 종료 시간 선택 상태
//   const [loading, setLoading] = useState(false);  // 로딩 상태
//   const [error, setError] = useState<string | null>(null);  // 에러 상태

//   const [predictionData, setPredictionData] = useState<string[] | null>(null);  // 예측 데이터 상태
//   const [trafficData, setTrafficData] = useState<Record<string, number> | null>(null);  // 트래픽 데이터 상태
//   const [appData, setAppData] = useState<Record<string, number> | null>(null);  // 애플리케이션 데이터 상태
//   const [protocolData, setProtocolData] = useState<Record<string, number> | null>(null);  // 프로토콜 데이터 상태

//   // handlePredict 함수에서 fetchPredictions 사용
//   const handlePredict = async () => {
//     if (!selectedDate || !startTime || !endTime) {
//       setError('Please select both date and time');
//       return;
//     }
  
//     const formattedDate = selectedDate.format('YYYYMMDD');
//     const formattedStartTime = startTime.format('HH:mm');
//     const formattedEndTime = endTime.format('HH:mm');
  
//     try {
//       setLoading(true);
      
//       // fetchPredictions 호출 후 데이터 받기
//       const result = await fetchPredictions(formattedDate, formattedStartTime, formattedEndTime);
  
//       if (result) {
//         const { predictionData, pieData, protocolData } = result;
  
//         // 각 데이터를 해당 컴포넌트 상태에 저장
//         setPredictionData(predictionData.prediction_messages); // TLS Prediction Component
//         setTrafficData(pieData.traffic_ratios);                 // Traffic Component
//         setAppData(pieData.app_ratios);                         // Application Component
//         setProtocolData(protocolData);                          // Protocol Component
//       } else {
//         setError('Failed to fetch prediction data');
//       }
  
//       setLoading(false);
//     } catch (err) {
//       console.error('Error predicting traffic:', err);
//       setError('Failed to fetch prediction data');
//       setLoading(false);
//     }
//   };

//   return (
//     <Card>
//       <CardHeader title="Predict Traffic" />
//       <CardContent>
//         <LocalizationProvider dateAdapter={AdapterDayjs}>
//           {/* Date Picker */}
//           <DatePicker
//             label="Select Date"
//             value={selectedDate}
//             onChange={(newValue) => setSelectedDate(newValue)} // 날짜 선택 상태 업데이트
//           />

//           {/* Start Time Picker */}
//           <TimePicker
//             label="Select Start Time"
//             value={startTime}
//             onChange={(newValue) => setStartTime(newValue)} // 시간 선택 상태 업데이트
//           />

//           {/* End Time Picker */}
//           <TimePicker
//             label="Select End Time"
//             value={endTime}
//             onChange={(newValue) => setEndTime(newValue)} // 시간 선택 상태 업데이트
//           />
//         </LocalizationProvider>

//         {/* 로딩 상태 표시 */}
//         {loading && <p>Loading predictions...</p>}
//         {/* 에러 메시지 표시 */}
//         {error && <p style={{ color: 'red' }}>{error}</p>}
//       </CardContent>

//       <Divider />

//       {/* API 호출 버튼 */}
//       <CardActions sx={{ justifyContent: 'flex-end' }}>
//         <Button
//           color="inherit"
//           endIcon={<ArrowRightIcon fontSize="var(--icon-fontSize-md)" />}
//           size="small"
//           onClick={handlePredict}  // 예측 요청 함수 실행
//           disabled={loading}  // 로딩 중에는 버튼 비활성화
//         >
//           Predict Traffic
//         </Button>
//       </CardActions>
//     </Card>
//   );
// }
'use client';

import { fetchPredictions } from './fetchPredictions'; // Flask API 호출 함수
import { useState } from 'react';
import dayjs, { Dayjs } from 'dayjs';
import * as React from 'react';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import Divider from '@mui/material/Divider';
import { DatePicker, TimePicker } from '@mui/x-date-pickers'; // Date and Time Pickers
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs'; // Adapter for dayjs
import { ArrowRight as ArrowRightIcon } from '@phosphor-icons/react/dist/ssr/ArrowRight';

export function Calender({ onFetchPredictions }: { onFetchPredictions: (data: any) => void }): React.JSX.Element {
  const [selectedDate, setSelectedDate] = useState<Dayjs | null>(dayjs());  // 날짜 선택 상태
  const [startTime, setStartTime] = useState<Dayjs | null>(dayjs());  // 시작 시간 선택 상태
  const [endTime, setEndTime] = useState<Dayjs | null>(dayjs());  // 종료 시간 선택 상태
  const [loading, setLoading] = useState(false);  // 로딩 상태
  const [error, setError] = useState<string | null>(null);  // 에러 상태

  // handlePredict 함수에서 fetchPredictions 사용
  const handlePredict = async () => {
    if (!selectedDate || !startTime || !endTime) {
      setError('Please select both date and time');
      return;
    }

    const formattedDate = selectedDate.format('YYYYMMDD');
    const formattedStartTime = startTime.format('HH:mm');
    const formattedEndTime = endTime.format('HH:mm');

    try {
      setLoading(true);

      // fetchPredictions 호출 후 데이터 받기
      const result = await fetchPredictions(formattedDate, formattedStartTime, formattedEndTime);

      console.log('API Response:', result); // API 응답 로그

      if (result) {
      // onFetchPredictions 함수 호출하여 결과 전달
        onFetchPredictions({
        predictions: result.predictionData.prediction_messages,
        traffic: result.pieData.traffic_ratios,
        applications: result.pieData.app_ratios,
        protocols: result.protocolData,
      });
      } else {
        setError('Failed to fetch prediction data');
      }

      setLoading(false);
    } catch (err) {
      console.error('Error predicting traffic:', err);
      setError('Failed to fetch prediction data');
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardHeader title="Predict Traffic" />
      <CardContent>
        <LocalizationProvider dateAdapter={AdapterDayjs}>
          {/* Date Picker */}
          <DatePicker
            label="Select Date"
            value={selectedDate}
            onChange={(newValue) => setSelectedDate(newValue)} // 날짜 선택 상태 업데이트
          />

          {/* Start Time Picker */}
          <TimePicker
            label="Select Start Time"
            value={startTime}
            onChange={(newValue) => setStartTime(newValue)} // 시간 선택 상태 업데이트
          />

          {/* End Time Picker */}
          <TimePicker
            label="Select End Time"
            value={endTime}
            onChange={(newValue) => setEndTime(newValue)} // 시간 선택 상태 업데이트
          />
        </LocalizationProvider>

        {/* 로딩 상태 표시 */}
        {loading && <p>Loading predictions...</p>}
        {/* 에러 메시지 표시 */}
        {error && <p style={{ color: 'red' }}>{error}</p>}
      </CardContent>

      <Divider />

      {/* API 호출 버튼 */}
      <CardActions sx={{ justifyContent: 'flex-end' }}>
        <Button
          color="inherit"
          endIcon={<ArrowRightIcon fontSize="var(--icon-fontSize-md)" />}
          size="small"
          onClick={handlePredict}  // 예측 요청 함수 실행
          disabled={loading}  // 로딩 중에는 버튼 비활성화
        >
          Predict Traffic
        </Button>
      </CardActions>
    </Card>
  );
}
