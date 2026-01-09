# 증권 거래 시스템을 위한 Burstable과 Memory-Optimized 인스턴스 심층 분석

## 1. 목적

본 성능 테스트의 목적은 다양한 EC2 인스턴스의 성능을 분석하고, 특정 워크로드에 대한 비용 대비 효율성을 평가하는 것입니다. 실험에서는 `sysbench`를 사용하여 CPU, 메모리, 파일 I/O 등 기본적인 시스템 성능과 함께, 데이터베이스(MariaDB) 부하 테스트를 통해 실제 애플리케이션 환경에서의 성능을 종합적으로 측정했습니다.

성능 분석은 다음 세 가지 주요 기준을 바탕으로 진행했습니다:

1.  **인스턴스 크기별 성능 차이** (예: large vs xlarge)
2.  **인스턴스 패밀리 간 성능 차이** (t, m, r 계열)
3.  **CPU 아키텍처별 성능 차이** (ARM 기반 Graviton vs x86 기반 Intel)

---

## 2. 테스트 환경 및 방법

### 2.1. 사용 인스턴스 목록

| 아키텍처 | 시리즈 | 인스턴스 타입 | vCPU | 메모리(GiB) |
|:--- |:--- |:---|:---:|:---:|
| **ARM (Graviton)** | T-시리즈 | `t4g.medium` | 2 | 4 |
| | M-시리즈 | `m7g.large` | 2 | 8 |
| | | `m7g.xlarge` | 4 | 16 |
| | R-시리즈 | `r7g.large` | 2 | 16 |
| **x86 (Intel)** | M-시리즈 | `m6i.large` | 2 | 8 |

### 2.2. 테스트 도구 및 환경

- **테스트 도구**: `sysbench 1.0.20`
- **데이터베이스**: `MariaDB 10.11.13`
- **스토리지**: 모든 인스턴스에서 `gp3` 타입 EBS 볼륨으로 통일

### 2.3. 테스트 명령어

- **CPU 성능 측정**:
  ```bash
  sysbench cpu --cpu-max-prime=50000 --threads=2 --time=60 run
  ```
- **메모리 성능 측정**:
  ```bash
  sysbench memory --memory-block-size=1K --memory-total-size=100G --threads=2 run
  ```
- **파일 I/O 성능 측정**:
  ```bash
  sysbench fileio --file-total-size=10G --file-test-mode=rndrw --threads=4 --time=60 run
  ```
- **데이터베이스(OLTP) 성능 측정**:
  ```bash
  sysbench oltp_read_write --mysql-host=localhost --mysql-user=root --mysql-password=1234 --mysql-db=sbtest --threads=100 --time=120 run
  ```

---

## 섹션 1: 전체 인스턴스 성능 비교 분석

이 섹션에서는 다양한 인스턴스 타입과 시리즈에 걸쳐 진행된 벤치마크 결과를 종합적으로 비교 분석합니다.

### 1.1. CPU 성능 비교

![CPU Performance Comparison](https://raw.githubusercontent.com/koorukuroo/pda_7th/DBDBDeep/DBDBDeep/DBphoto/cpu_performance.png)

| 지표 (Metric) | t4g.medium | m7g.large | m7g.xlarge | r7g.large | m6i.large (x86) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **CPU Speed (events/sec)** | 593.05 | 628.95 | 628.78 | **628.89** | 334.85 |
| **Total Time (s)** | 60.0021 | 60.0024 | 60.0007 | 60.0030 | 60.0019 |
| **Total Events** | 35,585 | 37,739 | 37,728 | 37,736 | 20,092 |
| **Latency (avg, ms)** | 3.37 | 3.18 | 3.18 | 3.18 | 5.97 |
| **Latency (95th percentile, ms)** | 3.36 | 3.19 | 3.19 | 3.19 | 5.99 |

*   **분석**: Graviton(g) 기반 인스턴스들이 x86(i) 기반 인스턴스보다 CPU 처리량에서 약 88% 높은 성능을 보입니다. 이는 워크로드가 ARM 아키텍처에 더 최적화되어 있음을 시사합니다.

### 1.2. 메모리 성능 비교

![Memory Performance Comparison](https://raw.githubusercontent.com/koorukuroo/pda_7th/DBDBDeep/DBDBDeep/DBphoto/memory_performance.png)

| 지표 (Metric) | t4g.medium | m7g.large | m7g.xlarge | r7g.large | m6i.large (x86) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **전송 속도 (MiB/sec)** | 24996.27 | 40787.57 | **40845.75** | 40607.31 | 19108.85 |
| **초당 연산 (ops/sec)** | 24996.27 | 40787.57 | **40845.75** | 40607.31 | 19108.85 |
| **Total Time (s)** | 4.0952 | 2.5094 | 2.5058 | 2.5206 | 5.3571 |
| **Latency (95th percentile, ms)** | 0.08 | **0.05** | **0.05** | **0.05** | 0.12 |

*   **분석**: 메모리 성능 역시 Graviton 기반 인스턴스들이 x86보다 약 2.1배 뛰어납니다. `t4g.medium`은 다른 g 시리즈에 비해 약 61% 수준의 성능을 보여, T-시리즈의 특성이 메모리 대역폭에도 영향을 줄 수 있음을 암시합니다.

### 1.3. 파일 I/O 성능 비교

![File I/O Performance Comparison](https://raw.githubusercontent.com/koorukuroo/pda_7th/DBDBDeep/DBDBDeep/DBphoto/file_io_performance.png)

| 지표 (Metric)                       | t4g.medium | m7g.large | m7g.xlarge |  r7g.large  | m6i.large (x86) |
| :-------------------------------- | :--------: | :-------: | :--------: | :---------: | :-------------: |
| **읽기 속도 (reads/s)**               |  2171.39   |  2985.44  |  4512.79   | **4513.21** |     3003.57     |
| **쓰기 속도 (writes/s)**              |  1447.57   |  1990.27  |  3008.55   | **3008.81** |     2002.36     |
| **읽기 처리량 (read, MiB/s)**          |   33.93    |   46.65   |   70.51    |  **70.52**  |      46.93      |
| **쓰기 처리량 (written, MiB/s)**       |   22.62    |   31.10   |   47.01    |  **47.01**  |      31.29      |
| **Latency (95th percentile, ms)** |    1.86    |   1.82    |  **1.76**  |  **1.76**   |      1.82       |

*   **분석**: 파일 I/O 성능은 인스턴스 크기에 비례하여 증가하는 경향을 보입니다. vCPU가 4개인 `m7g.xlarge`와 `r7g.large`가 가장 뛰어난 성능을 보였습니다.

### 1.4. 데이터베이스(OLTP) 성능 비교

<img width="1200" height="700" alt="image" src="https://github.com/user-attachments/assets/82547302-8b99-429e-aab1-80861905aa90" />


| 지표 (Metric) | t4g.medium | m7g.large | m7g.xlarge | r7g.large | m6i.large (x86) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **처리량 (TPS)** | **229.43** | 226.57 | 225.50 | 226.40 | 224.88 |
| **쿼리 처리량 (QPS)** | **4588.63** | 4531.49 | 4509.96 | 4527.93 | 4497.55 |
| **Latency (95th percentile, ms)**| 831.46 | 773.68 | **759.88** | **759.88** | 773.68 |

*   **분석**: 모든 인스턴스가 비슷한 TPS를 기록, 이는 테스트 부하가 디스크 I/O 병목에 먼저 도달했음을 시사합니다. 하지만 Latency는 인스턴스 등급이 높을수록 낮아져 안정성이 향상됨을 알 수 있습니다.

---

## 섹션 2: T-시리즈 심층 분석: 크레딧과 '성능 절벽'

### 2.1. '크레딧 절벽' 현상 재현 시나리오

T-시리즈의 CPU 크레딧 기반 성능 변화를 확인하기 위해 `t4g.medium` 인스턴스를 대상으로 아래와 같이 3가지 상태를 비교 측정했습니다.

1.  **최고 성능 상태**: CPU 크레딧이 충분할 때의 DB 성능
2.  **성능 저하 상태**: `sysbench cpu run &` 명령으로 CPU 크레딧을 모두 소진시킨 후의 DB 성능
3.  **비교군**: 크레딧 개념이 없는 고정 성능 `r7g.large` 인스턴스의 DB 성능

### 2.2. 원본 데이터 비교

#### 1. `t4g.medium` 최고 성능 (크레딧 소진 전)
```
SQL statistics:
    transactions:                        34361  (571.99 per sec.)
    queries:                             788271 (13121.92 per sec.)
Latency (ms):
         95th percentile:                      450.77
```

#### 2. `t4g.medium` 성능 저하 (크레딧 소진 후)
```
SQL statistics:
    transactions:                        7169   (118.80 per sec.)
    queries:                             166257 (2755.01 per sec.)
Latency (ms):
         95th percentile:                     2320.55
```

#### 3. `r7g.large` 고정 성능
```
SQL statistics:
    transactions:                        45577  (758.88 per sec.)
    queries:                             1049455 (17473.88 per sec.)
Latency (ms):
         95th percentile:                      337.94
```
![A](https://raw.githubusercontent.com/koorukuroo/pda_7th/DBDBDeep/DBDBDeep/DBphoto/A.png)
![B](https://raw.githubusercontent.com/koorukuroo/pda_7th/DBDBDeep/DBDBDeep/DBphoto/B.png)
![C](https://raw.githubusercontent.com/koorukuroo/pda_7th/DBDBDeep/DBDBDeep/DBphoto/C.png)
### 2.3. 분석: T-시리즈의 '크레딧 절벽'과 성능 붕괴

| 지표 (Metric) | `t4g.medium` (최고 성능) | `t4g.medium` (크레딧 소진 후) | `r7g.large` (고정 성능) |
|:---|:---:|:---:|:---:|
| **처리량 (TPS)** | 571.99 건/초 | 118.80 건/초 | **758.88 건/초** |
| **95% 응답시간 (Latency)** | 450.77 ms | 2320.55 ms | **337.94 ms** |

![DB Performance (TPS): T-Series vs R-Series](https://raw.githubusercontent.com/koorukuroo/pda_7th/DBDBDeep/DBDBDeep/DBphoto/t_vs_r_tps_comparison.png)
*<그래프 1: 처리량(TPS) 비교. 크레딧 소진 후 t4g.medium의 성능이 급격히 저하됨을 보여줍니다.>*

![DB Performance (Latency): T-Series vs R-Series](https://raw.githubusercontent.com/koorukuroo/pda_7th/DBDBDeep/DBDBDeep/DBphoto/t_vs_r_latency_comparison.png)
*<그래프 2: 95% 응답시간(Latency) 비교. 크레딧 소진 후 t4g.medium의 응답시간이 치명적인 수준으로 급증했음을 보여줍니다.>*

- **처리량(TPS) 79% 감소**: 크레딧이 소진되자 `t4g.medium`의 DB 처리 능력은 초당 572건에서 119건으로 **약 5분의 1 토막**이 났습니다.
- **응답시간(Latency) 5.1배 급증**: 사용자가 체감하는 반응 속도는 0.45초에서 **2.3초로 5배 이상 느려졌습니다.** 증권 거래 시스템에서 이는 치명적인 장애로 이어질 수 있습니다.

반면, **`r7g.large`** 는 `t4g.medium`의 최고 성능 상태보다도 **약 32% 더 높은 처리량**과 **25% 더 빠른 응답시간**을 변동 없이 꾸준히 제공했습니다.

---

## 3. 최종 결론

**T-시리즈 (`t4g.medium`)**:
- **장점**: 비용이 저렴하고, 트래픽이 적을 때는 준수한 성능을 보여줍니다.
- **단점**: CPU 크레딧 소진 시 성능이 예측 불가능하게 급락하는 '크레딧 절벽' 리스크가 존재합니다.
- **추천 용도**: 개발/테스트 서버, 내부 관리 도구 등 안정성이 중요하지 않은 비핵심 업무에 적합합니다.

**R-시리즈 (`r7g.large`)**:
- **장점**: CPU 성능이 고정되어 있어 어떠한 부하 상황에서도 일관되고 예측 가능한 고성능을 제공합니다.
- **단점**: T-시리즈에 비해 비용이 높습니다.
- **추천 용도**: **증권 거래 시스템**, 실시간 금융 데이터 분석, 대규모 이커머스 등 찰나의 지연도 허용되지 않는 **미션 크리티컬(Mission-Critical)한 서비스의 운영 DB 서버**로 절대적으로 추천됩니다.

**최종 제언**: 증권 거래 시스템의 데이터베이스 서버를 선택할 때, 단순히 비용만으로 T-시리즈를 고려하는 것은 매우 위험합니다. 순간적인 트래픽 증가로 CPU 크레딧이 소진될 경우, 시스템 전체가 마비될 수 있습니다. 따라서 안정성과 신뢰성이 최우선인 금융 서비스에는 **R-시리즈와 같은 고정 성능 인스턴스를 사용하는 것이 필수적**입니다.
