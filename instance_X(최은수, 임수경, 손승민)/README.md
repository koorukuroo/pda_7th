
<aside>
🚀

# **AWS 인스턴스별 금융 실시간 시세 처리 성능 비교**

</aside>

# 주제 선택 배경

금융 실시간 시세 처리는 데이터가 끊임없이 들어오는 스트리밍 워크로드이고, 장 시작/마감이나 변동성 확대 구간에서는 짧은 시간에 데이터 유입량이 크게 늘어난다. 이때 시스템이 감당하지 못하면 처리 지연이 발생하고, 실시간 모니터링·알림·분석 파이프라인 전반의 품질이 떨어진다.

하지만 AWS 인스턴스는 가격이 비싸다고 항상 빠른 것이 아니고 또한 같은 비용대에서도 CPU 성능, 메모리 대역폭, 네트워크 처리, I/O 특성이 달라 실제 성능이 크게 달라질 수 있다. 따라서 “실시간 시세 처리 시스템에 가장 적합한 인스턴스는 무엇인가?”를  지연 시간, 처리량, 자원 사용, S3 업로드 성능, 비용 효율성을 기준으로 실측 비교해 선택할 필요가 있어 본 주제를 선택했다.

# 실험 배경 및 목적

구현한 “실시간 시세 수집 → 지표 계산/분석 → S3 저장” 파이프라인은 금융권에서 실제로 사용되는 핵심 로직과 유사하다.

1. **실시간 시세 수집 및 전처리**
    - 증권사 HTS/MTS, 시세 중계기, 마켓데이터 수집기가 수행하는 역할과 같다.
    - 틱 데이터가 지속적으로 유입되므로, 비동기 처리와 안정적 버퍼링이 필수이며, 이 구간 성능이 전체 지연과 처리량을 좌우한다.
2. **실시간 지표 계산/연산**
    - 이동평균, 거래량 변화, 변동성, 이상징후 탐지 등 연산을 동시에 수행하며, CPU·메모리 사용률이 급격히 올라간다.
    - 현업의 고난도 연산 상황을 가정한 스트레스 테스트로 의미 있는 성능 평가가 가능하다.
3. **클라우드 데이터 레이크/아카이빙**
    - 분석 결과와 로그를 S3에 저장해 장기 보관, 백테스팅, 장애 분석에 활용한다.
    - 금융 데이터 특성상 빠르고 안전한 저장 설계가 중요하며, 업로드 성능이 병목이 되기 쉽다.

결론적으로, 동일한 코드라도 AWS 인스턴스 유형에 따라 처리량·지연·업로드 속도·비용 효율성이 달라지므로, 실측 비교가 필요하다.

# 실험 개요

- 문제 유형: Hybrid I/O-Compute Bound
    
    외부 API 응답 대기(Network I/O)와 실시간 지표 계산(CPU 연산)이 결합된 비동기 처리 문제
    
- 입력 조건:
    - 대상 종목: 국내 주요 주식 3종(삼성전자, SK하이닉스, 네이버)
    - 데이터 소스: KIS 모의투자 서버 REST API
    - 병렬 실행: asyncio 기반 비동기 Task (수집+분석+S3 업로드 동시 수행)
- 측정 항목:
    - 성능 지표: 평균 지연 시간(Average Latency), 꼬리 지연 시간 (Tail latency), 초당 처리량(Throughput).
    - 자원 지표: 평균 CPU 사용률(%), 최대 메모리 점유율(MB)
    - 네트워크: S3 업로드 전송 속도(KB/s)
    - 경제성: 비용 효율성 (1달러 당 처리량)

# 실험 방법

실험은 정확한 비교를 위해 다음과 같이 4단계로 구성되었으며, 모든 인스턴스에 동일한 환경을 적용한다.

1. 환경 설정 및 인스턴스 배치
- 대상 AWS 인스턴스(5종):

| **인스턴스 유형** | **vCPU** | **RAM (GiB)** | **온디맨드 요금 (시간당)** | **특징** |
| --- | --- | --- | --- | --- |
| **t3.medium** | 2 | 4 | **$0.052** | x86 범용, 버스터블 성능 |
| **t4g.medium** | 2 | 4 | **$0.0416** | Graviton2(ARM), 최저가 |
| **m6i.large** | 2 | 8 | **$0.107** | 인텔 기반 범용 표준 |
| **c6i.large** | 2 | 4 | **$0.107** | 컴퓨팅 최적화 |
| **r6i.large** | 2 | 16 | **$0.158** | 메모리 최적화, 최고가 |
2. 실시간 데이터 수집 및 연산 로직
- 데이터 요청: KIS 서버에 실시간 현재가를 비동기로 요청한다.
- 지표 계산: 수집된 Raw Data를 바탕으로 다음을 실시간 계산한다.
- 데이터 저장: 수집/계산된 데이터를 s3에 업로드한다.
3. 시스템 모니터링 및 성능 측정
- 자원 추적: psutil 라이브러리를 통해 1초 단위로 CPU 점유율과 메모리 변화를 기록한다.
- 네트워크 벤치마크: 분석이 완료된 데이터를 1분 단위로 CSV 파일로 생성하고, 이를 AWS S3 저장소로 업로드하며 전송 속도와 지연을 측정한다.
- 지연 시간 측정: API 요청 시작점부터 분석 결과가 S3 업로드 대기열에 들어가는 시점까지의 전체 소요 시간(End-to-End Latency)을 기록한다.
4. 데이터 로깅 및 분석
- 모든 결과는 CSV로 로그 파일에 기록되며, 테스트 종료 후 다음 산식을 통해 비용 효율성을 도출한다.
    - 비용 효율성 = 초당 처리량(Throughput) / 인스턴스 시간당 요금

# 사용 코드

### 증권 시세 데이터 수집

```java
async def fetch_current_price(stock_code: str):
    await ensure_token()
    url = f"{BASE}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "authorization": f"Bearer {access_token}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": "FHKST01010100",
        "custtype": "P",
        "content-type": "application/json",
    }
    params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": stock_code}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params, timeout=5)
        return response.json()
```

### S3 업로드 속도 측정

```java
async def upload_to_s3_with_bench():
    if not S3_BUCKET:
        print("[경고] S3_BUCKET_NAME 설정 없음")
        return
    s3_client = boto3.client("s3")
    while True:
        await asyncio.sleep(60)  # 1분마다 전송 테스트
        if os.path.exists(LOG_FILE):
            try:
                file_size = os.path.getsize(LOG_FILE) / 1024
                start_time = time.time()
                time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                s3_path = f"logs/{INSTANCE_NAME}/{time_str}_{LOG_FILE}"
                s3_client.upload_file(LOG_FILE, S3_BUCKET, s3_path)
                duration = time.time() - start_time
                speed = file_size / duration if duration > 0 else 0
                with open(S3_LOG_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.time(), file_size, duration, speed])
                print(
                    f" >>> [S3전송] {file_size:.1f}KB | {duration:.2f}s | {speed:.1f}KB/s"
                )
            except Exception as e:
                print(f"[S3 에러] {e}")
```

### 거래량 변동분 및 1분 이동평균 및 거래량 분석

```java
vol_delta = (
    vol - LAST_VOLUMES[symbol]
    if LAST_VOLUMES[symbol] is not None
    else 0
)
LAST_VOLUMES[symbol] = vol
DATA_QUEUE[symbol].append((now, price, vol_delta))

recent = [d for d in DATA_QUEUE[symbol] if d[0] > now - 60]
ma_1m = sum(d[1] for d in recent) / len(recent) if recent else 0
vol_1m = sum(d[2] for d in recent) if recent else 0
```

### 시스템 리소스 및 성능 측정

```java
cpu = psutil.cpu_percent()
mem = psutil.virtual_memory().percent
latency = time.time() - start_req

processed_count += 1
throughput = processed_count / max(
    time.time() - start_time_global, 0.001
)
cost_per_tx = INSTANCE_HOURLY_COST / (throughput * 3600)
```

# 실험 결과 예측

| 분석 항목 | 예상 결과 |
| --- | --- |
| **속도 (Latency)** | 연산 및 메모리 최적화 모델인 r6i, c6i가 압도적으로 빠를 것이다. |
| **비용 효율 (Efficiency)** | 저가형 모델인 t3, t4g가 단위 비용 대비 처리 효율이 가장 좋을 것이다. |
| **처리량 (Throughput)** | 고사양 인스턴스(Large급)가 초당 더 많은 데이터를 처리할 것이다. |
| **자원 사용 (CPU)** | 전용 코어가 아닌 공유 자원 기반의 버스터블 아키텍처인 **t 시리즈**가 동일 작업 시 CPU 사용률이 높을 것이다. |
| **S3 업로드 속도** | 네트워크 밸런스가 좋은 **m 시리즈**가 전송 안정성 면에서 우세할 것이다. |

# 실험 결과

| 인스턴스 | 꼬리 지연 시간 (Tail latency) | 평균 지연 시간 (s) | 초당 처리량 (TX/s) | CPU 사용률 (%) | S3 전송 속도 (KB/s) | 비용 효율성 **(1달러당 처리량)** |
| --- | --- | --- | --- | --- | --- | --- |
| **t4g.medium** | 0.3200 | 0.203 | 0.280 | 27.5% | 997.4 | **30,034건** |
| **t3.medium** | 0.2744 | 0.169 | **0.317** | 18.7% | 183.0 | **27,429건** |
| **m6i.large** | 0.2310 | 0.137 | 0.272 | 15.3% | **1,166.7** | 10,205건 |
| **c6i.large** | 0.2282 | 0.132 | 0.204 | **10.3%** | 364.7 | 8,633건 |
| **r6i.large** | **0.2256** | **0.131** | 0.175 | 12.3% | 206.7 | 4,992건 |

### 꼬리 지연 시간 (Tail latency) & 평균 지연 시간

<img width="45%" height="600" alt="Code_Generated_Image-8" src="https://github.com/user-attachments/assets/e512f894-d499-4f2c-8f4f-235b95fa8f02" /> <img width="45%" height="600" alt="Code_Generated_Image-7" src="https://github.com/user-attachments/assets/d2849e12-453c-4543-b6c4-9d1057149be3" />



- **결과**: c6i.large와 r6i.large가 가장 낮고(빠름), t4g.medium과 t3.medium이 가장 높게 나타났다.
- **이유**:
    - **인스턴스 등급 차이**: c6i, m6i, r6i는 전용 자원을 사용하는 **매니지드 인스턴스이**다. 반면 t 시리즈는 **버스터블(Burstable)** 인스턴스로, 평소에는 낮은 성능을 유지하다 필요할 때만 CPU 크레딧을 소모해 성능을 높인다. 이 과정에서 지연 시간이 발생하거나 성능 변동성(꼬리 지연 시간 (Tail latency))이 커질 수 있다.
    - **CPU 세대**: c6i와 r6i는 최신 Intel Xeon Scalable(Ice Lake) 프로세서를 사용하여 연산 처리 속도가 근본적으로 빠르다.

### 평균 처리량 (Average Throughput)

<img width="1000" height="600" alt="Code_Generated_Image-6" src="https://github.com/user-attachments/assets/944452ab-a6bf-48a6-9424-94e2acc98e24" />

- **결과**: t3.medium과 t4g.medium이 상대적으로 높은 수치를 기록했다.
- **이유**:
    - **테스트 시나리오의 특성**: 만약 실행된 작업이 매우 무거운 연산이 아니라 가벼운 트랜잭션 위주였다면, 저사양 인스턴스에서도 충분히 많은 수를 처리할 수 있다.
    - **데이터의 특이점**: 로그상에서 t 시리즈가 더 활발하게 처리량을 기록한 것은, 해당 인스턴스들이 테스트 당시 '버스트 모드'를 활발히 사용하여 일시적으로 높은 성능을 냈기 때문일 가능성이 크다.

### 평균 CPU 사용률 (Average CPU Usage)

<img width="1000" height="600" alt="Code_Generated_Image-5" src="https://github.com/user-attachments/assets/78009926-5ac2-4e7b-b862-f56d76b56038" />


- **결과**: c6i.large가 가장 낮고, t4g.medium이 가장 높다.
- **이유**:
    - **연산 효율성**: c6i는 'Compute Optimized' 인스턴스로 CPU 성능이 가장 강력하다. 똑같은 작업을 수행해도 CPU 자원을 훨씬 적게(약 10%) 사용하여 처리할 수 있다.
    - **아키텍처 차이**: t4g는 ARM 기반의 Graviton 프로세서이다. 특정 x86 최적화 코드나 연산에서 Intel 프로세서보다 더 많은 연산 주기를 사용해야 했을 수 있으며, 인스턴스 자체의 절대적 코어 성능이 c6i보다 낮아 사용률이 높게 찍히게 된다.

### S3 전송 속도 (Average S3 Upload Speed)

<img width="1000" height="600" alt="Code_Generated_Image-4" src="https://github.com/user-attachments/assets/17f10b81-d4cf-4341-83fb-4dea8473d171" />


- **결과**: m6i.large가 압도적으로 빠르고, r6i나 t3는 상대적으로 낮다.
- **이유**:
    - **네트워크 대역폭 할당**: AWS는 인스턴스 크기와 타입에 따라 네트워크 대역폭을 차등 할당한다. m6i(General Purpose)는 네트워크와 연산, 메모리 밸런스가 가장 잘 잡혀 있어 S3와의 통신에서 안정적인 고속 전송을 보여준다.
    - **EBS/네트워크 최적화**: r6i는 메모리에 최적화되어 있어 네트워크 처리량에서는 m 시리즈보다 우선순위가 낮게 설정될 수 있으며, t 시리즈는 기본 네트워크 대역폭 자체가 매우 작게 제한되어 있다.

### 비용 효율성(1달러 당 처리량)

<img width="1000" height="600" alt="Code_Generated_Image-3" src="https://github.com/user-attachments/assets/735e3e09-ea7f-4c32-a548-eeca2b0f255c" />

- **결과**: t4g.medium과 t3.medium이 대형 인스턴스들보다 수 배 이상 높았습니다.
- **이유**:
    - **압도적인 가격 차이**: t4g.medium은 r6i.large에 비해 약 3배 저렴하다.
    - **효율성**: 비록 개별 처리 속도(Latency)는 느릴지라도, 워낙 가격이 저렴하기 때문에 "1달러를 투입했을 때 나오는 총 결과물" 측면에서는 저사양 인스턴스가 훨씬 유리하다. 대규모 단순 반복 작업이라면 t 시리즈가 훨씬 경제적임을 보여준다.

### CPU와 Memory 사용률의 차이

<img width="1200" height="700" alt="Code_Generated_Image-9" src="https://github.com/user-attachments/assets/5c274b42-2ca8-4d3b-879f-4fb48339571b" />


- **결과**: r6i는 메모리 사용률이 매우 낮고, t4g는 CPU 사용률이 매우 높다.
- **이유**:
    - **r6i (Memory Optimized)**: 이 인스턴스는 동일 등급 대비 RAM 용량이 훨씬 크다(Large 기준 16GB). 따라서 다른 인스턴스와 똑같은 양의 데이터를 메모리에 올려도 사용률(%)은 훨씬 낮게 표시된다. 메모리 부족 현상 없이 안정적인 서비스가 가능함을 시사한다.
    - **t4g (ARM/Burstable)**: 작은 메모리(4GB)와 상대적으로 낮은 CPU 성능을 가지고 있어, 테스트 부하 유입 시 CPU 및 메모리 자원이 설계 임계치(Threshold)에 근접하여 동작하고 있다.
    
    .

# 실험 결과 요약

### **1. 비용 대비 처리량 우수(비용 효율성): t4g.medium & t3.medium**

- **결과**: **t4g.medium**이 1달러당 약 3만 건의 데이터를 처리하며 가장 우수하다.
- **이유**: 인스턴스 대여 비용이 매우 저렴하고 평균 처리량 높아 투입 비용 대비 압도적으로 우수하다.

### **2. 지연 및 연산 안정성: r6i.large & c6i.large**

- **결과**: **r6i.large**가 가장 낮은 지연 시간(0.131s)을, **c6i.large**가 가장 낮은 CPU 부하(10.3%)를 기록하였다.
- **이유**: 최신 인텔 6세대 프로세서의 연산 능력이 구형이나 저가형 모델보다 훨씬 뛰어나다. 또한, c6i의 경우는 'Compute Optimized' 인스턴스로 CPU 성능이 가장 우수하다.

### **3. S3 업로드 속도 우수: m6i.large**

- **결과**: S3 업로드 속도가 **1,166.7 KB/s**로 다른 인스턴스들보다 압도적으로 빠르다.
- **이유**: 범용(M) 인스턴스는 네트워크 대역폭이 균형 있게 설계되어 대용량 로그 전송에 유리하다.

# 추가 테스트 결과

### 가설 vs 실제 데이터 대조표

| 분석 항목 | 초기 가설 (Assumption) | 실제 결과 (Actual) | 가설 일치 여부 |
| --- | --- | --- | --- |
| **속도 (Latency)** | R6i,C6i가 압도적으로 빠를 것이다. | R6i(0.131s), C6i(0.132s)로 **가장 빠름.** | **O (일치)** |
| **가성비 (Efficiency)** | T3,T4g가 비용 대비 효율이 좋을 것이다. | T4g가 1달러당 처리량 **1위**. | **O (일치)** |
| **처리량 (Throughput)** | 고사양 인스턴스가 초당 더 많이 처리할 것이다. | T3(0.31)가 R6i(0.17)보다 **높게 나옴.** | **X (불일치)** |
| **자원 사용 (CPU)** | t시리즈가 CPU사용률이 높을 것이다. | T4g(27%)가 C6i(10%)보다 훨씬 높음. | **O (일치)** |

### **예상 결과와의 괴리 분석**

**처리량이** T3(0.31)가 R6i(0.17)보다 **높게 나왔다.**

- 외부 API의 병목 지점 존재
- 오버스펙의 확인
    - R6i의 CPU 사용률은 10%대로 서버의 가진 힘을 제대로 활용하지 못했을 것이다.
    - T3의 CPU 사용률은 30%대로 서버를 잘 활용하였을 것이다.
    - 데이터 수집 / 분석 / 파일 업로드 로직이 R6i에게 너무 가벼운 작업이었을 가능성이 있다.
- t계열의 버스터블 특성
    - t계열은 평소 낮은 성능을 유지하다 필요할 때 CPU 크래딧을 소모해(버스트 모드) 일시적으로 높은 성능을 낸다.

### t3.medium과 r6i.large의 처리량 추가 테스트 실시

새로 뽑은 t3.medium과 r6i.large의 로그를 바탕으로 상세 비교 분석을 진행하였다. 어제의 데이터와 오늘의 데이터에서 보이는 가장 큰 변화는 **두 인스턴스의 처리량(Throughput)이 매우 비슷해졌다는 점**이다.

| 지표 | **t3.medium (오늘)** | **r6i.large (오늘)** | 비고 |
| --- | --- | --- | --- |
| **평균 지연 시간 (Latency)** | 0.149*s* | **0.115*s*** | **r6i**가 약 23% 더 빠름 |
| **초당 처리량 (Throughput)** | 0.23 *tx*/*s* | 0.32 *tx*/*s* | **유사함** (차이 미미) |
| **평균 CPU 사용률** | 27.9% | 28.7% | 연산 부하가 비슷하게 작용 |
| **평균 메모리 사용률** | 13.6% | **3.7%** | **r6i**의 자원 여유 압도적 |
| **S3 업로드 속도** | 742.7 *KB*/*s* | **1010.1 *KB*/*s*** | **r6i**의 네트워크 우세 |
| **비용 효율성 (1달러당 처리량)** | **15,762건** | 9,091건 | **t3**가 약 1.7배 우위 |

### 추가 테스트 결론

인스턴스 내부 연산 속도가 아무리 빨라도, 전체 루프의 90% 이상을 외부 API(한국투자증권) 응답 대기가 차지하고 있다. 따라서 어제 R6i의 처리량이 낮았던 것은 인스턴스의 문제가 아니라, 해당 시점의 네트워크 혼잡도나 API 서버의 응답 지연이 R6i에게 더 불리하게 작용했기 때문으로 추측된다.

만약 인스턴스 간 처리량의 진정한 변별력을 확인하고자 한다면, 감시 종목 수를 100개 이상으로 늘려 동시성(Concurrency) 테스트를 진행하거나, 연산 부하 로직의 복잡도를 10배 이상 상향하여 CPU 집중형(Compute Intensive) 환경을 구축해야 할 것으로 보인다.

# 결론

본 테스트 결과, **실시간 주식 가격 처리**와 같이 0.1초의 응답 속도가 중요한 서비스에는 최저 지연 시간(0.2256s)을 기록한 인스턴스인 **r6i.large**가 가장 적합하다. 반면, 데이터 전송이 빈번한 작업에는 S3 속도가 가장 빠른 인스턴스인 **m6i.large**가 유리하며, 단순 연산이나 배치 작업처럼 **비용 효율**이 최우선인 경우에는 달러당 처리량이 가장 높은 인스턴스인 **t4g.medium**을 선택하는 것이 전략적이다. 
결과적으로 선택하고자하는 서비스의 비즈니스 목적이 안정성이냐 가성비이냐 등 무엇에 해당하는지를 파악하고 그에 맞춰 해당 인스턴스 유형을 결정해야 한다.
