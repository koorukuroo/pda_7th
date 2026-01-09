# 📊 AWS EC2 인스턴스 성능 비교: 금융 데이터 분석 벤치마크

## 1. 실험 목적

이 프로젝트는 **금융 데이터 분석 워크로드**를 기준으로 다양한 AWS EC2 인스턴스의 성능을 정량적으로 비교하여, 용도별 최적 인스턴스를 선정하는 것을 목표로 하였다.

특히 KOSPI와 NASDAQ 지수 간의 **상관관계 분석 및 회귀 분석**과 같은 통계 연산을 반복 수행하여, 순수 CPU 연산 성능을 측정하였다.

---

## 2. 실험 환경 및 설정

### 2.1 테스트 대상 인스턴스

| 인스턴스 | vCPU | RAM | 아키텍처 | 특징 | 시간당 요금 |
|---------|------|-----|---------|------|-----------|
| t3.micro | 2 | 1GB | x86 (Intel) | 버스트형, 최저가 | $0.0104 |
| t3.small | 2 | 2GB | x86 (Intel) | 버스트형 | $0.0208 |
| t3.medium | 2 | 4GB | x86 (Intel) | 버스트형 | $0.0416 |
| t3.large | 2 | 8GB | x86 (Intel) | 버스트형 | $0.0832 |
| m4.large | 2 | 8GB | x86 (Intel) | 구세대 범용형 | $0.1000 |
| m5.large | 2 | 8GB | x86 (Intel) | 최신 범용형 | $0.0960 |
| c7i.large | 2 | 4GB | x86 (Intel) | CPU 최적화 | $0.0725 |

### 2.2 실험 조건

- **데이터 소스**: Stooq CSV API (KOSPI: ^KS11, NASDAQ: ^IXIC)
- **분석 기간**: 2016-01-01 ~ 현재 (약 2,500일)
- **총 데이터**: 약 4,976 rows
- **벤치마크 설정**:
  - `USE_CACHE=True`: 네트워크 영향 제거 (로컬 CSV 재사용)
  - `BENCH_REPEAT=80`: CPU 부하 확대를 위한 80회 반복 실행
  - `WARMUP_REPEAT=3`: 캐시 워밍업 3회
  - `SHOW_PLOTS=False`: 그래프 출력 비활성화 (순수 연산만 측정)

### 2.3 측정 지표

- **실행 시간**: 다운로드+정렬 시간, 분석 반복 시간
- **처리량**: rows/sec (총 데이터 행 수 × 반복 횟수 ÷ 분석 시간)
- **CPU 사용률**: 평균 및 최대 사용률 (%)
- **메모리 사용량**: RSS 기준 평균 및 최대 (MB)
- **비용 대비 효율**: 처리량 ÷ 시간당 요금

---

## 3. 테스트 코드 설명

### 3.1 전체 코드 구조

```python
# ============================================================
# KOSPI vs NASDAQ : Stooq CSV HTTP + Analysis + Benchmark
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os, time, platform
import numpy as np
import pandas as pd
import statsmodels.api as sm
import requests
from io import StringIO

# 성능 측정 준비
import psutil
_proc = psutil.Process()
_cpu_samples = []
_mem_samples = []
```

### 3.2 주요 함수

#### (1) 데이터 다운로드 및 캐싱

```python
def download_stooq_close(stooq_symbol: str, start: str, end=None, timeout=20) -> pd.Series:
    """
    Stooq CSV를 HTTP로 가져와 Close 가격을 Series로 반환.
    USE_CACHE=True면 로컬 파일에 저장 후 재사용하여 네트워크 영향 제거.
    """
    cache_file = _cache_path(stooq_symbol)

    # 캐시 로드
    if USE_CACHE and os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
    else:
        # 네트워크 다운로드
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        df = pd.read_csv(StringIO(r.text))

        # 캐시 저장
        if USE_CACHE:
            df.to_csv(cache_file, index=False)

    # 날짜 필터링 및 전처리
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] >= pd.to_datetime(start)]

    return df.set_index("Date")["Close"]
```

#### (2) 통계 분석 함수

```python
def step_analysis_only(px: pd.DataFrame):
    """
    금융 데이터 분석 워크로드:
    1. 로그 수익률 계산
    2. OLS 회귀 분석 (베타 계산)
    3. 롤링 상관계수 (60일 윈도우)
    """
    # 수익률 계산
    kospi_r = returns_from_price(px[KOSPI], USE_LOG_RETURN)
    nasdaq_r = returns_from_price(px[NASDAQ], USE_LOG_RETURN)

    df = pd.concat([kospi_r, nasdaq_r], axis=1).dropna()
    df.columns = ["KOSPI", "NASDAQ"]
    df["NASDAQ_lag1"] = df["NASDAQ"].shift(1)

    # 상관계수
    corr_same = df["KOSPI"].corr(df["NASDAQ"])
    corr_lag1 = df["KOSPI"].corr(df["NASDAQ_lag1"])

    # OLS 회귀 분석
    alpha, beta, r2, pval, n = ols_beta(df["KOSPI"], df["NASDAQ"])

    # 롤링 상관계수
    roll_corr = df["KOSPI"].rolling(ROLL_WINDOW).corr(df["NASDAQ"])

    return df, corr_same, corr_lag1, alpha, beta, r2, pval, roll_corr
```

#### (3) 벤치마크 실행

```python
# 워밍업 (캐시/BLAS 최적화 완료)
for _ in range(WARMUP_REPEAT):
    _ = step_analysis_only(px)

# 본 벤치마크 (분석만 80회 반복)
t0 = time.perf_counter()
for _ in range(BENCH_REPEAT):
    last = step_analysis_only(px)
t_analysis = time.perf_counter() - t0

# 처리량 계산
rows = len(df)
analysis_throughput = (rows * BENCH_REPEAT / t_analysis)
```

### 3.3 성능 측정 방식

**네트워크와 CPU 연산을 분리하여 측정**하였다:

1. **다운로드+정렬 시간**: 첫 실행 시 네트워크 다운로드 포함 (1회만)
2. **분석 반복 시간**: 캐시된 데이터로 순수 CPU 연산만 80회 반복
3. **처리량**: `(데이터 행 수 × 80) ÷ 분석 시간`으로 계산

이를 통해 **네트워크 지연의 영향을 제거**하고, 순수한 인스턴스 연산 성능만을 비교할 수 있었다.

---

## 4. 실험 결과

### 4.1 전체 성능 비교

| 인스턴스 | 분석시간(s) | 처리량(rows/s) | CPU평균(%) | 메모리(MB) | $/시간 | 가성비* |
|---------|------------|---------------|-----------|-----------|--------|---------|
| **c7i.large** | **0.463** | **410,443** | 65.4 | 183 | $0.0725 | 5,661k |
| **m5.large** | 0.778 | 244,262 | 86.8 | 183 | $0.0960 | 2,544k |
| m4.large | 0.815 | 233,164 | 67.9 | 184 | $0.1000 | 2,332k |
| t3.small | 0.855 | 222,132 | 62.3 | 185 | $0.0208 | **10,679k** |
| t3.medium | 0.908 | 209,259 | 67.8 | 184 | $0.0416 | 5,029k |
| t3.micro | 0.929 | 204,554 | 64.1 | 183 | $0.0104 | **19,669k** |
| t3.large | 1.079 | 175,965 | 72.3 | 183 | $0.0832 | 2,115k |

*가성비 = 처리량 ÷ 시간당 요금*

### 4.2 성능 순위

#### 처리 속도 (빠른 순)
1. **c7i.large** - 410,443 rows/s (기준)
2. m5.large - 244,262 rows/s (1.68배 느림)
3. m4.large - 233,164 rows/s (1.76배 느림)
4. t3.small - 222,132 rows/s (1.85배 느림)
5. t3.medium - 209,259 rows/s (1.96배 느림)
6. t3.micro - 204,554 rows/s (2.01배 느림)
7. t3.large - 175,965 rows/s (2.33배 느림) ⚠️

#### 가성비 (효율적인 순)
1. **t3.micro** - 19,669k (압도적)
2. t3.small - 10,679k
3. c7i.large - 5,661k
4. t3.medium - 5,029k
5. m5.large - 2,544k
6. m4.large - 2,332k
7. t3.large - 2,115k (최악)

---

## 5. 분석 및 인사이트

### 5.1 CPU 최적화 인스턴스의 압도적 우위

**c7i.large**가 2위인 m5.large 대비 **1.68배 빠른 성능**을 보였다. 이는 금융 데이터 분석이 다음과 같은 특성을 가지기 때문이다:

- **CPU 집약적**: 수익률 계산, 상관계수, OLS 회귀는 순수 연산 작업
- **메모리 요구량 낮음**: 모든 인스턴스에서 184MB만 사용
- **I/O 최소화**: 캐시 사용으로 디스크/네트워크 영향 제거

따라서 **CPU 클럭 속도와 캐시 구조**가 성능을 좌우하며, C 계열 인스턴스가 이러한 작업에 최적화되어 있음을 확인하였다.

### 5.2 범용형 인스턴스 세대 차이

**m5.large vs m4.large 비교**:

| 항목 | m5.large (최신) | m4.large (구세대) | 차이 |
|------|----------------|------------------|------|
| 처리량 | 244,262 rows/s | 233,164 rows/s | **+4.8%** |
| CPU 사용률 | 86.8% | 67.9% | 높은 활용도 |
| 가격 | $0.096/h | $0.100/h | **4% 저렴** |

동일한 스펙(2 vCPU, 8GB RAM)임에도 **m5가 m4보다 빠르고 저렴**하였다. 이는 Intel Xeon 프로세서 세대 차이(Broadwell/Haswell → Skylake/Cascade Lake)에 기인한 것으로 추정된다.

**결론**: 구세대 인스턴스는 피하고, **최신 세대를 선택하는 것이 중요**하다.

### 5.3 버스트 인스턴스의 예상치 못한 결과

#### ⚠️ t3.large가 가장 느린 이유

**예상**: t3 계열에서 스펙이 높을수록 성능이 좋을 것
**실제**: t3.large가 **t3.micro보다 14% 느림**

```
t3.micro  (1GB, $0.0104)  →  204,554 rows/s
t3.small  (2GB, $0.0208)  →  222,132 rows/s  (+8.6%)
t3.medium (4GB, $0.0416)  →  209,259 rows/s  (-5.8%)
t3.large  (8GB, $0.0832)  →  175,965 rows/s  (-16.0%) ❌
```

#### 왜 이런 결과가 나왔을까?

**1. CPU 크레딧 소진 (버스트 특성)**

t3 인스턴스는 **버스트 성능 모델**을 사용한다:
- **베이스라인 CPU**: 지속 가능한 기본 성능 (20~40%)
- **CPU 크레딧**: 일시적으로 100% 성능을 낼 수 있는 토큰

우리 실험은:
- 80회 반복 실행 (총 약 1초)
- 짧은 시간이지만 **지속적인 고부하**

t3.large는 **크레딧 소진 후 베이스라인으로 제한**되었을 가능성이 높다. 반면 t3.micro/small은 짧은 작업이라 크레딧 내에서 처리되었을 것으로 추정된다.

**2. 메모리 과잉 (낭비)**

모든 인스턴스에서 **메모리 사용량 183~185MB로 동일**하였다:
- t3.micro (1GB): 18% 사용
- t3.large (8GB): **2.3% 사용** → 7GB 이상 낭비

이 워크로드는 메모리 집약적이지 않아, RAM 증가가 **전혀 도움되지 않았다**.

**3. 단일 스레드 워크로드**

pandas, numpy, statsmodels는 기본적으로 **단일 스레드 연산**을 수행한다:
- vCPU 2개로 충분
- t3.large도 vCPU 2개 → **CPU 코어 수 증가 효과 없음**

**4. AWS 공유 하드웨어 특성**

EC2는 물리 서버를 여러 인스턴스가 공유한다:
- 순간적인 **노이즈/간섭** 발생 가능
- t3.large 실행 시점에 다른 인스턴스의 부하가 있었을 수 있음

#### 다른 팀의 유사 사례

- **Mew팀**: "t4g.large까지는 성능 차이 미미, xlarge(4 vCPU)에서만 급증"
- **투팍팀**: "t2.xlarge는 요금 대비 비효율, nano/micro가 가성비 최고"

→ **T 계열은 large 이상 가면 가성비 급락**하는 공통 패턴을 확인하였다.

### 5.4 인스턴스 패밀리별 특성 정리

| 패밀리 | 특징 | 이 워크로드 적합도 | 추천 용도 |
|--------|------|------------------|----------|
| **C 계열** | CPU 최적화, 고클럭 | ⭐⭐⭐ 최고 | 금융 분석, ML 학습 |
| **M 계열** | 범용형, 균형잡힌 스펙 | ⭐⭐ 양호 | 웹/API 서버, DB |
| **T 계열 (micro/small)** | 버스트형, 저가 | ⭐⭐ 가성비 최고 | 개발/테스트, 간헐적 작업 |
| **T 계열 (large 이상)** | 버스트형, 과한 스펙 | ❌ 비효율 | 이 작업엔 부적합 |

---

## 6. 다른 팀 실험과의 비교

| 팀 | 주요 발견 | 우리 실험과의 공통점 |
|---|---------|------------------|
| **3.2.6팀** | c5 CPU 최고, i3 Storage 최고 | ✅ C 계열 연산 성능 압도적 |
| **투팍팀** | c5.large 최고 가성비, nano/micro 효율적 | ✅ C 계열 + 소형 T 계열 추천 동일 |
| **Mew팀** | t4g.large→xlarge에서만 성능 급증 | ✅ T 계열 large 이상 비효율 일치 |
| **SuperI팀** | c7g(ARM) 최고 효율 | ✅ C 계열 압도적 성능 동일 |
| **말하는감자팀** | m7g/c7g 장시간 안정적 | ✅ 최신 세대 우수 동일 |

**종합**: 모든 팀 실험에서 **"C 계열 = 연산 최고, T 계열 소형 = 가성비 최고"**라는 일관된 결론을 얻었다.

---

## 7. 결론 및 권장 사항

### 7.1 시나리오별 최적 인스턴스

| 시나리오 | 추천 인스턴스 | 이유 |
|---------|-------------|------|
| **프로덕션: 성능 최우선** | **c7i.large** | 410k rows/s, CPU 최적화 |
| **프로덕션: 균형잡힌 선택** | **m5.large** | 안정적 성능, 범용성 |
| **개발/테스트 환경** | **t3.micro** | 압도적 가성비 (19M rows/$h) |
| **간헐적 배치 작업** | **t3.small** | 적당한 성능 + 좋은 가성비 |
| ❌ **비추천** | t3.large, m4.large | 낮은 효율, 구세대 |

### 7.2 의사결정 트리

```
📊 금융 데이터 분석 인스턴스 선택

1. 예산이 충분한가?
   ├─ YES → c7i.large (최고 성능, 1.68배 빠름)
   └─ NO  → 2단계로

2. 프로덕션 환경인가?
   ├─ YES → m5.large (안정성 + 성능)
   └─ NO  → 3단계로

3. 개발/테스트인가?
   ├─ YES → t3.micro (가성비 최고, 19배 효율)
   └─ NO  → t3.small (절충안)

⚠️ 피해야 할 선택:
- t3.large: 8배 비싼데 14% 느림
- m4.large: 구세대, m5보다 느리고 비쌈
```

### 7.3 핵심 메시지

> **"워크로드 특성에 따라 10배 이상 가성비 차이 발생"**

- **순수 연산 성능**: c7i.large (410k rows/s)
- **가성비**: t3.micro (19M rows/$h)
- **균형**: m5.large (안정성 + 범용성)

### 7.4 실용적 조언

1. **테스트부터 시작**: t3.micro로 개발 후 성능 필요 시 업그레이드
2. **최신 세대 선택**: m4→m5, c5→c7처럼 신세대가 항상 우수
3. **스펙 맹신 금지**: t3.large처럼 큰 인스턴스가 항상 빠른 건 아님
4. **실제 워크로드 측정**: 벤치마크로 직접 확인 필수

---

## 8. 한계점 및 개선 방향

### 8.1 현재 실험의 한계

1. **단일 워크로드**: 금융 분석만 테스트 (I/O, 네트워크 부하 미포함)
2. **짧은 실행 시간**: 1초 미만 작업 (장시간 안정성 미확인)
3. **단일 실행**: 1회만 측정 (통계적 신뢰도 낮음)
4. **x86 아키텍처만 테스트**: ARM(Graviton) 미포함

### 8.2 추가 실험 제안

- **I/O 중심 워크로드**: 대용량 CSV 읽기/쓰기, 데이터베이스 연동
- **메모리 집약적**: 대형 행렬 연산(SVD, PCA), 인메모리 처리
- **병렬 처리**: 멀티 스레드/프로세스 활용 테스트
- **장시간 실행**: 10분 이상 지속 부하 (GC, 메모리 누수 확인)
- **ARM 비교**: c7g(Graviton3) vs c7i(Intel) 성능 비교

---

## 9. 실행 방법

### 9.1 환경 설정

```bash
# 필요 라이브러리 설치
pip install pandas numpy matplotlib statsmodels requests psutil

# 코드 다운로드
git clone <repository_url>
cd <directory>
```

### 9.2 실행

```bash
# 기본 실행 (캐시 사용, 그래프 미출력)
python benchmark.py

# 그래프 출력 포함
# 코드 내 SHOW_PLOTS = True로 변경 후
python benchmark.py

# 반복 횟수 조정 (더 정밀한 측정)
# 코드 내 BENCH_REPEAT = 200으로 변경 후
python benchmark.py
```

### 9.3 결과 확인

실행 후 다음과 같은 출력을 확인할 수 있다:

```
======================================================================
:압정: 성능 측정 결과(구간 분리 + CPU 부하 반복)
0) 전체 실행 시간(TOTAL)      : 0.863 sec
1) 다운로드+정렬 시간          : 0.053 sec  (USE_CACHE=True)
2) 분석만 반복 실행 시간        : 0.778 sec  (repeat=80, warmup=3)
3) 분석 처리량(대략)           : 244,262.1 rows/sec
4) CPU 사용률(평균/최대)        : 86.84% / 172.10%
5) 메모리 사용량(RSS)          : 183.17 MB(avg) / 184.98 MB(max)
6) 디스크 I/O(변화량)          : read 0.00 MB, write 0.00 MB
7) 다운로드 데이터 행(누적)      : 4,975 rows
======================================================================
```

---

## 10. 참고 자료

- [AWS EC2 인스턴스 유형](https://aws.amazon.com/ko/ec2/instance-types/)
- [Stooq 금융 데이터](https://stooq.com/)
- [pandas 문서](https://pandas.pydata.org/)
- [statsmodels 문서](https://www.statsmodels.org/)

---

**작성일**: 2026-01-09
**실험 환경**: Python 3.x, pandas 2.x, statsmodels 0.14.x
**데이터**: Stooq (KOSPI, NASDAQ, 2016-01-01 ~ 현재)
**측정 도구**: psutil, time.perf_counter()
