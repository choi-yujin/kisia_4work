import React, { useState, useEffect } from 'react';
import TlsPrediction from './TlsPrediction'; // tls-prediction.tsx import
import Traffic from './Traffic'; // Traffic 컴포넌트 import
import Application from './Application'; // Application 컴포넌트 import
import Protocol from './Protocol'; // Protocol 컴포넌트 import
import { fetchPredictions } from './fetchPredictions'; // fetchPredictions 함수 import

function App() {
  // State to store data from batch prediction
  const [batchData, setBatchData] = useState({
    predictionData: null,
    pieData: null,
    protocolData: null
  });

  // Fetch data from Batch prediction APIs
  useEffect(() => {
    const selectedDate = '2024-10-21';
    const startTime = '10:00';
    const endTime = '11:00';

    fetchPredictions(selectedDate, startTime, endTime)
      .then((result) => {
        if (result) {
          setBatchData(result);
        }
      })
      .catch((error) => {
        console.error('Error fetching batch data:', error);
      });
  }, []);

  return (
    <div>
      <h1>Batch Prediction Data</h1>
      <h2>Predictions</h2>
      <pre>{JSON.stringify(batchData.predictionData, null, 2)}</pre>

      {/* Traffic Component */}
      {batchData.pieData && (
        <Traffic
          chartSeries={Object.values(batchData.pieData.traffic_ratios).map((value) => value * 100)}
          labels={Object.keys(batchData.pieData.traffic_ratios)}
        />
      )}

      {/* Application Component */}
      {batchData.pieData && (
        <Application
          chartSeries={Object.values(batchData.pieData.app_ratios).map((value) => value * 100)}
          labels={Object.keys(batchData.pieData.app_ratios)}
        />
      )}

      {/* Protocol Component */}
      {batchData.protocolData && (
        <Protocol protocolCounts={batchData.protocolData} />
      )}

      <h1>Main Server Data</h1>
      {/* 예전 Main Server 데이터를 유지하고 싶으면 아래 코드를 추가 */}
      <h2>Packets</h2>
      <pre>{JSON.stringify(mainData.packets, null, 2)}</pre>
      <h2>Predictions</h2>
      <pre>{JSON.stringify(mainData.predictions, null, 2)}</pre>
      <h2>Traffic Ratios</h2>
      <pre>{JSON.stringify(mainData.ratios.traffic_ratios, null, 2)}</pre>
      <h2>App Ratios</h2>
      <pre>{JSON.stringify(mainData.ratios.app_ratios, null, 2)}</pre>

      <h1>Sub Server Data</h1>
      <h2>Batch Prediction</h2>
      <pre>{JSON.stringify(subData.predictionResult, null, 2)}</pre>
      <h2>Pie Rate</h2>
      <pre>{JSON.stringify(subData.pieRate, null, 2)}</pre>
      <h2>History</h2>
      <pre>{JSON.stringify(subData.history, null, 2)}</pre>

      {/* Prediction Component */}
      <TlsPrediction />
    </div>
  );
}

export default App;
