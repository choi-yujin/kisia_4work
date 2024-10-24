// export const fetchPredictions = async (selectedDate, startTime, endTime) => {
//   try {
//     // 첫 번째 요청: receive_prediction API
//     const response = await fetch("http://localhost:5000/batch/receive_prediction", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({
//         selected_date: selectedDate,
//         start_time: startTime,
//         end_time: endTime,
//       }),
//     });

//     if (!response.ok) {
//       throw new Error("Failed to fetch data from receive_prediction");
//     }

//     const predictionData = await response.json(); // 예측 데이터 파싱

//     // 두 번째 요청: send_pie API
//     const pieResponse = await fetch("http://localhost:5000/batch/send_pie", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({
//         selected_date: selectedDate,
//         start_time: startTime,
//         end_time: endTime,
//       }),
//     });

//     if (!pieResponse.ok) {
//       throw new Error("Failed to fetch data from send_pie");
//     }

//     const pieData = await pieResponse.json(); // 파이 데이터 파싱

//     // 세 번째 요청: protocol_rate API
//     const protocolResponse = await fetch("http://localhost:5000/batch/protocol_rate", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({
//         selected_date: selectedDate,
//         start_time: startTime,
//         end_time: endTime,
//       }),
//     });

//     if (!protocolResponse.ok) {
//       throw new Error("Failed to fetch data from protocol_rate");
//     }

//     const protocolData = await protocolResponse.json(); // 프로토콜 데이터 파싱

//     // 모든 데이터를 객체로 반환
//     return {
//       predictionData, // receive_prediction의 예측 메시지
//       pieData,        // send_pie의 트래픽/앱 비율
//       protocolData    // protocol_rate의 프로토콜 데이터
//     };
//   } catch (error) {
//     console.error("Error in fetchPredictions:", error);
//     return null;
//   }
// };
export const fetchPredictions = async (selectedDate, startTime, endTime) => {
  try {
    // 1. receive_prediction API 호출
    const receivePredictionResponse = await fetch("http://localhost:5000/batch/receive_prediction", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date: selectedDate, start_time: startTime, end_time: endTime }),
    });

    if (!receivePredictionResponse.ok) {
      throw new Error('Failed to fetch prediction data');
    }

    const predictionData = await receivePredictionResponse.json();
    console.log('Prediction Data:', predictionData);  // 예측 데이터를 로그로 출력

    // 2. send_pie API 호출
    const sendPieResponse = await fetch(`http://localhost:5000/batch/send_pie?date=${selectedDate}&start_time=${startTime}&end_time=${endTime}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    if (!sendPieResponse.ok) {
      throw new Error('Failed to fetch pie chart data');
    }

    const pieData = await sendPieResponse.json();
    console.log('Pie Data:', pieData);  // 파이 데이터를 로그로 출력

    // 3. protocol_rate API 호출
    const protocolRateResponse = await fetch(`http://localhost:5000/batch/protocol_rate?date=${selectedDate}&start_time=${startTime}&end_time=${endTime}`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    if (!protocolRateResponse.ok) {
      throw new Error('Failed to fetch protocol rate data');
    }

    const protocolData = await protocolRateResponse.json();
    console.log('Protocol Rate Data:', protocolData);  // 프로토콜 데이터를 로그로 출력

    // 모든 데이터를 한 객체로 반환
    return {
      predictionData,
      pieData,
      protocolData,
    };
  } catch (error) {
    console.error('Error fetching predictions:', error);
    return null;
  }
};
