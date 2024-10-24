// // Predict_traffic 컴포넌트
// 'use client';
// import * as React from 'react';
// import Box from '@mui/material/Box';
// import Card from '@mui/material/Card';
// import CardHeader from '@mui/material/CardHeader';
// import Divider from '@mui/material/Divider';
// import Table from '@mui/material/Table';
// import TableBody from '@mui/material/TableBody';
// import TableCell from '@mui/material/TableCell';
// import TableHead from '@mui/material/TableHead';
// import TableRow from '@mui/material/TableRow';

// export interface Predict_traffic {
//   prediction_messages: string;
//   traffic_count: string;
//   app_count: string;
// }

// export interface Predict_trafficProps {
//   predicts?: Predict_traffic[];
// }

// export function Predict_traffic({ predicts = [] }: Predict_trafficProps): React.JSX.Element {
//   console.log('Predict_traffic Component Props:', { predicts }); // 로그 추가

//   return (
//     <Card>
//       <CardHeader title="SSL/TLS Traffic Prediction" />
//       <Divider />
//       <Box sx={{ overflowX: 'auto' }}>
//         <Table sx={{ minWidth: 800 }}>
//           <TableHead>
//             <TableRow>
//               <TableCell>Traffic ID</TableCell>
//               <TableCell>Application ID</TableCell>
//               <TableCell>Prediction Messages</TableCell>
//             </TableRow>
//           </TableHead>
//           <TableBody>
//             {predicts.length > 0 ? (
//               predicts.map((predict, index) => (
//                 <TableRow hover key={index}>
//                   <TableCell>{predict.traffic_count}</TableCell>
//                   <TableCell>{predict.app_count}</TableCell>
//                   <TableCell>{predict.prediction_messages}</TableCell>
//                 </TableRow>
//               ))
//             ) : (
//               <TableRow>
//                 <TableCell colSpan={3} align="center">No Predictions available</TableCell>
//               </TableRow>
//             )}
//           </TableBody>
//         </Table>
//       </Box>
//       <Divider />
//     </Card>
//   );
// }
// Predict_traffic 컴포넌트
'use client';
import * as React from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardHeader from '@mui/material/CardHeader';
import Divider from '@mui/material/Divider';
import Typography from '@mui/material/Typography';

export interface Predict_trafficProps {
  predicts?: string[]; // 예측 메시지를 문자열 배열로 받음
}

export function Predict_traffic({ predicts = [] }: Predict_trafficProps): React.JSX.Element {
  console.log('Predict_traffic Component Props:', { predicts }); // 로그 추가

  return (
    <Card>
      <CardHeader title="SSL/TLS Traffic Prediction" />
      <Divider />
      <Box sx={{ padding: 2 }}>
        {predicts.length > 0 ? (
          predicts.map((message, index) => (
            <Typography key={index} variant="body1" sx={{ marginBottom: 1 }}>
              {message} {/* 예측 메시지를 출력 */}
            </Typography>
          ))
        ) : (
          <Typography variant="body1" align="center">No Predictions available</Typography>
        )}
      </Box>
      <Divider />
    </Card>
  );
}
