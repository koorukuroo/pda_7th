# 주식 시장 개장 직후 트래픽 집중 대응을 위한 EC2 인스턴스 최적화 전략

## Introduction
### 1. 프로젝트 배경
주식 시장 개장 직후는 거래량이 폭증하여 초당 수천 건의 트랜잭션이 발생하는 특수한 환경임. 이 시점의 지연 시간(Latency)은 매매 체결 가격에 직접적인 영향을 미치므로 최적의 인프라 구성이 목적임.

### 2. 실험 목적
  - Peak-Time 가용성 검증: 장 개시 직후 폭발적인 트래픽 상황에서 단일 노드의 생존력 및 임계 처리량 확인.
  - 세대별 성능 우위 규명: 최신 Graviton3(m7g)와 이전 세대(c6g) 프로세서의 실질적 데이터 처리 성능 비교.

## 실험 설계 및 환경 구성 (Experimental Design)
### 1. 통제 변수 설정
실험의 변별력을 위해 아래 항목을 모든 인스턴스에 동일하게 고정함.
    - 인스턴스 크기: .medium (2 vCPU / 4 GiB RAM 기준)
    - 스토리지: gp3 / 20 GB (IOPS 변동성 차단)
    - 운영체제: Amazon Linux 2023 (AWS 최적화 커널)
    - 리전: 서울 (ap-northeast-2, 네트워크 거리 통제)

### 2. 비교 대상 선정
```최신 세대 범용(m7g)```과 ```이전 세대 컴퓨팅 최적화(c6g)```를 비교하여 하드웨어 아키텍처의 세대 교체가 실질적인 성능 향상으로 이어지는지 검증함.

## System Architecture
### 1. 데이터 수집 엔진
Python yfinance 라이브러리를 활용하여 Yahoo Finance로부터 6개 주요 종목의 실시간 시세 및 10일간의 데이터를 병렬(Multi-threading)로 수집함.

### 2. 프론트엔드 대시보드
HTML5 Canvas 및 Chart.js를 사용하여 시계열 데이터를 시각화함. 사용자에게 데이터 최신성을 보장하기 위해 서버 수집 시각을 표기하는 'Sync' 타임스탬프 기능을 구현함.

### 3. 스케줄드 스케일링
장 시작 전 인스턴스를 선제적으로 3개까지 확장하도록 Auto Scaling Group(ASG)을 설정하여 초기 트래픽 스파이크에 대응함.

<img width="800" height="800" alt="Image" src="https://github.com/user-attachments/assets/cc30ce07-3549-4cf2-be7b-ec079d17e76f" />

<img width="1440" height="747" alt="Image" src="https://github.com/user-attachments/assets/166ca5b9-af50-46bb-8da7-2755d9da50ac" />

## 부하 테스트 시나리오 (Load Test Scenarios)
### 1. 테스트 도구
HTTP 부하 발생 도구인 hey를 사용하여 실제 접속 상황을 모사함.

### 2. 시나리오 설계
- ```hey -n 2000 -c 100 (저부하 테스트)```: 동시 접속자 100명 상황에서 서버가 지연 없이 즉각적인 응답을 내놓는지 **순수 속도(Baseline)**를 측정

- ```hey -n 50000 -c 100 (고부하 테스트)```: 장 개시 직후 지속되는 트래픽 폭주 상황을 가정하여, 서버가 지치지 않고 일관된 성능을 보여주는지 확인

- ```hey -n 100000 -c 500 (임계점 가혹 테스트)```: 의도적으로 서버를 마비시켜 임계 동시 접속자 수를 파악하고, 로드 밸런서 기반 확장성이 왜 필수적인지 논리적 근거를 확보

### 3. 로드 밸런서 배제 사유
분산 처리 장치인 ALB를 거치지 않고 각 인스턴스에 부하를 직접 가함으로써, 개별 컴퓨팅 노드가 가진 순수 연산 한계점과 네트워크 스택의 임계 성능을 정밀 측정하고자 함.

## 성능 결과 분석 (Performance Analysis)
### 1. m7g.medium vs c6g.medium 성능 비교

m계열이 c 계열보다 더 높은 처리량(rps)과 더 짧은 총 처리 시간을 기록함. 평균, p95, p99, 최대 지연 시간으로 구성된 latency profile에서도 m 계열은 지연 분포가 좁고 안정적인 반면, c 계열은 지연이 전반적으로 더 크고 tail latency가 크게 증가하는 모습을 확인함.

<img width="900" height="666" alt="Image" src="https://github.com/user-attachments/assets/3acb7843-6684-4fba-b067-7da715c3454a" />

<img width="900" height="666" alt="Image" src="https://github.com/user-attachments/assets/4492b14f-41a8-4130-af5f-2c6542dd2da5" />

<img width="634" height="470" alt="Image" src="https://github.com/user-attachments/assets/39de691f-2104-479d-86a7-35b498bc519c" />

<img width="634" height="470" alt="Image" src="https://github.com/user-attachments/assets/ae5ee9c5-49c1-4574-803b-d0392d54041a" />


### 2. 결론
- 처리량(Throughput): m7g가 우세
    - 퍼블릭 호출(응답 12B, n=2000)과 localhost(응답 5886B, n=2000/50000) 모두에서 m7g.medium이 더 높은 RPS를 기록. 특히 localhost + n=50,000 + 응답 5,886B 조건에서 m7g는 c6g 대비 약 39% 높은 RPS를 보여, 고부하에서의 처리능력 차이가 명확히 관측됨.
    - 이로부터, 해당 거래 데이터 API 워크로드는 단일 vCPU 환경에서 CPU/메모리 처리 성능에 민감하며, 플랫폼 성능 차이가 곧바로 처리량 차이로 연결될 수 있음을 확인 함.

- 평균 지연시간: m7g가 낮음
    - 평균 응답시간은 모든 시나리오에서 m7g가 더 낮았음.
    - 이는 “요청을 얼마나 많이 처리하느냐”뿐 아니라, **평균 체감 품질(평균 latency)**도 m7g가 유리함을 확인함.

## 추가적인 분석

### 가혹 테스트 결과 비교 (100,000 Requests / 500 Concurrency)

1. 한계 성능을 파악하기 위해 실시한 초고부하 테스트 결과, 최신 세대인 m7g 인스턴스가 c6g 대비 모든 지표에서 우월한 성능을 기록하였음.

<img width="697" height="354" alt="Image" src="https://github.com/user-attachments/assets/8fd0069c-a7bb-49c0-86d4-b1b351799831" />

2. 결과 요약 및 시사점
    - 처리 효율의 격차: 동일한 vCPU와 RAM 환경임에도 m7g(Graviton3)가 초당 처리량에서 약 44%의 우위를 보인 것은 최신 아키텍처의 연산 효율성이 주식 데이터 처리와 같은 실시간 워크로드에 더 적합함을 파악함.

    - 응답 안정성: 특히 99% Latency 지표에서 m7g가 c6g 대비 약 5.2초 더 빠른 응답을 기록한 점은, 트래픽 폭주 시에도 대다수의 사용자에게 일관된 서비스를 제공할 수 있음을 의미함.

    - 인프라 결론: 두 인스턴스 모두 10만 건 이상의 요청에서 타임아웃 에러가 발생한 점을 미루어 볼 때, 단일 노드의 수직적 확장(Scale-up)보다는 로드 밸런서를 활용한 수평적 확장(Scale-out) 전략이 필수적임을 정량적으로 확인함.

### 2. 추가 최적화 실험: Nginx 역방향 프록시 도입

1. 목표: 단일 Flask 서버(Direct) 방식은 클라이언트의 요청을 파이썬 프로세스가 직접 수용하므로, 대규모 트래픽 발생 시 컨텍스트 스위칭 오버헤드와 연결 관리 부담이 큼. 이에 따라 고성능 웹 서버인 Nginx를 앞단에 배치하여 다음과 같은 효과를 검증하고자 함.

2. 결과

<img width="1280" height="615" alt="Image" src="https://github.com/user-attachments/assets/75ca7009-13fd-4b0c-a588-21a55ddbbe86" />

