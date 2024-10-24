# kisia_4work

2024 KISIA AI보안 네트워크반 4조 '넷트워크' 팀 레포지토리
<br>

## 서비스소개 
![TLScope](https://github.com/choi-yujin/kisia_4work/blob/main/TLScope_logo.png)

본 프로젝트는 'SSL/TLS 암호화 패킷 가시화'를 주제로 진행되었습니다.
본 서비스는 복호화 없이 인공지능 모델을 활용하여 SSL/TLS 암호화 패킷의 트래픽(traffic)과 어플리케이션(application)을 가시화해주는 서비스로,
프록시 서버를 이용해 사용자의 패킷을 캡쳐 후 인공지능 모델에 패킷을 전달하여 분석하는 구조입니다.
<br>
<br>

## 기능소개
본 서비스는 실시간 페이지와 비 실시간 페이지로 나눠집니다.

실시간 페이지의 경우 사용자의 현재 트래픽을 캡쳐하여 분석 및 시각화하는 기능을 제공하고,<br>
비실시간 페이지의 경우 디렉토리에 저장된 패킷을 원하는 시간대로 필터링하여 분석 및 관련 통계치를 제공합니다. 

본 서비스에서는 다음과 같은 피쳐들을 시각화 합니다.

| Application | Traffic | 
|----------|----------|
| Facebook    | chat | 
| Discord    | chat, voip | 
| Skype    | chat, voip | 
| Line    | chat | 
| Youtube    | streaming | 

<br>
<br>

## 참조
본 프로젝트의 인공지능 파트는 논문 'MTC: A Multi-Task Model for Encrypted Network Traffic Classification Based on Transformer and 1D-CNN'을 참조하였습니다<br>
https://github.com/yuchengml/MTC


