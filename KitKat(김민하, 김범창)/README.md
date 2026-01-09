# 🚀 AWS EC2 Instance Performance 분석 리포트

**주제:** EC2 인스턴스 패밀리 & 아키텍처(x86 vs ARM) 성능 비교  
**지표:** Throughput (rows/s) — **높을수록 더 빠름**  
**데이터:** 총 **24,000 lines**, 동일 입력으로 반복 측정(**REPEATS=5**, **median 기준**)

---

## 1) 한 줄 결론

- ✅ **전체 1등:** `i8g.xlarge` *(ARM, Graviton4)*
- ✅ **x86에서 가장 안정적인 상위권:** `c6id.xlarge`
-⚠️ **`d2.xlarge`는 대용량 저장용** *(연산/DB/랜덤 I/O에는 부적합)*
- 📌 **ARM이 무조건 빠른 건 아님:** `i8g`만 “전 영역 강함”, `m6g/c6g`는 **CPU만 우세**하고 **I/O/SQLite는 약세**

---

## 2) 실험 구성 요약

### 🎯 목적
- 워크로드 특성에 맞는 **최적 인스턴스 선택**
- **x86 vs ARM(Graviton)** 체감 성능 비교
- 스토리지 성향(HDD/Nitro SSD/NVMe 등)이 성능에 미치는 영향 확인

### 🛠️ 테스트 항목

| 테스트 | 의미 | 무엇이 강하면 유리? |
|---|---|---|
| **CPU_HASH** | SHA256 해싱(순수 CPU) | 연산/암호화/배치 |
| **IO_BUFFERED_WRITE** | 버퍼드 파일 쓰기(캐시 포함, fsync X) | 로그/파일 생성/스트리밍 |
| **SQLITE_PRAGMAS** | SQLite insert(WAL/NORMAL/mmap/cache) | DB 적재/ETL/삽입 |

<details>
<summary><b>상세 설명</b></summary>
<div markdown="1">
1. 🧠 CPU_HASH (두뇌의 계산 속도)
"우리 사무실 직원이 얼마나 복잡한 암산을 빠르게 처리하는가?"

설명: CPU_HASH는 CPU(중앙처리장치)라는 컴퓨터의 두뇌가 얼마나 똑똑하고 빠른지 측정하는 테스트입니다.

방법: 모든 데이터 행에 대해 SHA256이라는 매우 복잡한 암호화 계산을 수행합니다. 이때 여러 명의 직원(멀티코어)을 동시에 투입하여 계산을 분담시킵니다.

의미: 이 수치가 높을수록 복잡한 알고리즘 연산, 암호화 처리, 인공지능 계산 등을 더 빨리 처리할 수 있다는 뜻입니다. 이번 테스트에서 ARM(Graviton) 아키텍처가 이 부분에서 특히 뛰어난 효율을 보여주었습니다.

2. ✍️ IO_BUFFERED_WRITE (받아쓰기 및 정리 속도)
"사무실 바구니에 서류를 모았다가 한꺼번에 서류함에 넣는 속도가 얼마나 빠른가?"

설명: 컴퓨터의 저장 공간(SSD/HDD)에 데이터를 기록하는 '쓰기 대역폭'을 측정하는 테스트입니다.

방법: 데이터를 한 줄씩 디스크에 직접 쓰는 대신, 메모리(RAM)라는 '바구니'에 약 5,000줄 정도를 모았다가 한 번에 디스크로 보냅니다. 이는 운영체제(OS)의 캐시 기능을 최대한 활용하는 방식입니다.

의미: 대용량 로그 파일을 기록하거나 대규모 데이터를 백업할 때의 성능을 대변합니다. NVMe SSD를 탑재한 인스턴스들이 일반 EBS(네트워크 저장소) 인스턴스보다 압도적으로 유리한 항목입니다.

3. 🗄️ SQLITE_PRAGMAS (전문 서기 전담팀의 능력)
"노련한 비서가 특별한 기술(치트키)을 써서 장부를 얼마나 효율적으로 정리하는가?"

설명: 데이터베이스(DB)라는 '장부'에 정보를 체계적으로 기록하는 성능을 측정합니다.

방법: PRAGMA라는 특수한 명령(치트키)을 사용하여 SQLite라는 DB 엔진의 설정을 변경합니다. 예를 들어, "기록할 때마다 일일이 확인하지 말고 나중에 한꺼번에 확정해(WAL 모드)"와 같은 최적화 기법을 적용합니다.

의미: 실제 웹 서비스나 애플리케이션에서 사용자 정보를 저장하거나 게시글을 올리는 등의 '데이터 관리' 성능을 가장 잘 나타내는 지표입니다. 하드웨어 성능뿐만 아니라 소프트웨어 설정 최적화가 성능에 얼마나 큰 영향을 미치는지 확인할 수 있습니다.

</div>
</details>

---

## 3) 분석 대상 인스턴스

- 모든 인스턴스는 vCPU 4개로 통일
- 패밀리 별 메모리 크기 통일
  
| 아키텍처 | 타입 | 메모리 | 포지션 |
|---|---|---:|---|
| x86 | `m6i.xlarge` | 16 GiB | 범용 |
| x86 | `i4i.xlarge` | 32 GiB | 스토리지 최적화 |
| x86 | `d2.xlarge` | 30.5 GiB | 대용량 저장(저지연 X) |
| x86 | `c6id.xlarge` | 8 GiB | 컴퓨팅 + 로컬 NVMe |
| ARM | `m6g.xlarge` | 16 GiB | 범용(Graviton2) |
| ARM | `i8g.xlarge` | 32 GiB | 최신(Graviton4) |
| ARM | `c6g.xlarge` | 8 GiB | 컴퓨팅(Graviton2) |

---

## 4) 📈 핵심 결과

### 4-1) x86 내부 비교: `c6id`가 전 테스트 1등

| rows/s | m6i | i4i | d2 | **c6id** |
|---|---:|---:|---:|---:|
| **CPU_HASH** | 39,171 | 39,258 | 28,789 | **39,642** |
| **IO_BUFFERED_WRITE** | 2,960,253 | 3,012,751 | 2,419,675 | **3,071,144** |
| **SQLITE_PRAGMAS** | 458,570 | 454,075 | 385,815 | **466,799** |

**해석**
- `c6id`: **x86에서 가장 안전한 상위권** (특히 쓰기/SQLite 강함)
- `d2`: 모든 항목 하위권 → **설계 목적(저장)**과 **테스트 목적(처리량)**이 다름

---

### 4-2) x86 vs ARM 전체 비교: `i8g`가 전 영역 1등

| rows/s | x86(c6id) | ARM(m6g) | **ARM(i8g)** | ARM(c6g) |
|---|---:|---:|---:|---:|
| **CPU_HASH** | 39,642 | 52,299 | **99,299** | 52,373 |
| **IO_BUFFERED_WRITE** | 3,071,144 | 2,428,265 | **3,183,217** | 2,205,210 |
| **SQLITE_PRAGMAS** | 466,799 | 290,800 | **470,632** | 292,266 |

**핵심 관찰**
- ARM은 **세대/제품군별 편차**가 큼
- `i8g(Graviton4)`: **CPU + I/O + SQLite 모두 강함**
- `m6g/c6g(Graviton2)`: **CPU는 강한데 I/O/SQLite는 약세**

<img width="3000" height="1800" alt="benchmark_cpu_hash" src="https://github.com/user-attachments/assets/7884fa9c-3a14-4def-8584-ccfbba5e12f8" />
<img width="3000" height="1800" alt="benchmark_io_buffered_write" src="https://github.com/user-attachments/assets/6703e4e2-edba-4ef3-b069-ff15557807c4" />
<img width="3000" height="1800" alt="benchmark_sqlite_pragmas" src="https://github.com/user-attachments/assets/d7dead97-3396-4698-8e68-6593052c5291" />


---

## 5) 패밀리별 정리

### 🆚 M 패밀리 (m6i vs m6g)
- CPU: ARM(m6g) **+33.5%**
- I/O: ARM(m6g) **-18.0%**
- SQLite: ARM(m6g) **-36.6%**

✅ 결론: **범용 서비스 + DB/쓰기면 `m6i(x86)`**, CPU 비중 높으면 `m6g(ARM)` 고려

---

### 🆚 I 패밀리 (i4i vs i8g)
- CPU: i8g **+152.9% (2.53배)**
- I/O: i8g **+5.7%**
- SQLite: i8g **+3.6%**

✅ 결론: **대량 처리/ETL/적재는 `i8g(ARM)`가 최강**

---

### 🆚 C 패밀리 (c6id vs c6g)
- CPU: c6g **+32.1%**
- I/O: c6g **-28.2%**
- SQLite: c6g **-37.4%**

✅ 결론: **DB 적재/쓰기 중심이면 `c6id(x86)`**, 순수 연산 중심이면 `c6g(ARM)`

➡️ CPU처리 ARM 우세, 쓰기/SQLite는 x86 우세한 경향 확인

---

## 6) 왜 이런 결과가 나왔나?

### 💾 스토리지는 워크로드와 “궁합”이 중요
- SQLite/파일 쓰기는 **랜덤 접근 + 메타데이터 작업**이 많아 **저지연 SSD/NVMe**가 유리
- `d2`는 **대용량 저장 목적**이라 이 실험(처리량 중심)에서 불리할 수 있음

### 🧠 ARM은 “Graviton 세대” 영향이 큼
- `i8g(Graviton4)`는 아키텍처/플랫폼 개선 폭이 커서 **우세**
- `m6g/c6g(Graviton2)`는 CPU는 잘 나오지만 **I/O/DB 경로**에서 기대만큼 안 나올 수 있음

📌 결론: **‘ARM vs x86’가 아니라 ‘인스턴스/세대/스토리지 조합’ 싸움**

---

## 7) 🏆 최종 추천

| 워크로드 | 추천 | 이유 |
|---|---|---|
| 최고 성능 | **`i8g.xlarge`** | CPU/I/O/SQLite 모두 1등 |
| x86 유지 + 상위권 | **`c6id.xlarge`** | x86 중 전 테스트 1등 |
| 범용 서비스(안정) | `m6i.xlarge` | 밸런스 + I/O/SQLite 안정적 |
| 연산 위주(가성비) | `c6g.xlarge` | CPU_HASH 우세 |
| 대용량 저장 | `d2.xlarge` | 저장 목적에 최적(처리량 목적엔 비추천) |


<details>
<summary>코드</summary>
<div markdown="1">

```python
"""
EC2 Instance Family Micro-Benchmark (rows/s)

목적
- 인스턴스 패밀리/아키텍처(x86 vs ARM)에 따른 성능 차이를 비교
- CPU 연산(해시), 파일 쓰기(버퍼드/선택적 fsync), SQLite insert(PRAGMA 튜닝) 측정

출력
- 각 테스트별 median/p95/min/max 시간 + throughput(rows/s)
"""

import csv
import hashlib
import os
import platform
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional

# ==========================================================
# 0) SETTINGS
# ==========================================================
CSV_PATH = "sap500_data.csv"
OUTDIR   = "."

LIMIT_ROWS = 0
REPEATS = 5                   # 반복 횟수
WORKERS = "auto"

# --- CPU 테스트 강도 (해시 반복 횟수)
CPU_ROUNDS = 30

# --- 파일 쓰기 테스트 설정
BLOCK_LINES = 5000            # buffered write block 크기
FSYNC_LINES = 0               

# --- SQLite insert 테스트 설정
SQLITE_COMMIT_EVERY = 1000    # 커밋 단위
# ==========================================================

# ==========================================================
# 1) 유틸리티 (시간/통계/시스템 정보)
# ==========================================================
def now_perf():
    return time.perf_counter()

def fmt_sec(x: float) -> str:
    return f"{x:.4f}s"

def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def sys_info():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
    }

def summarize_times(times: List[float]) -> str:
    med = statistics.median(times)
    p95 = percentile(times, 0.95)
    mn = min(times)
    mx = max(times)
    return f"median={fmt_sec(med)}, p95={fmt_sec(p95)}, min={fmt_sec(mn)}, max={fmt_sec(mx)}"
# ==========================================================

# ==========================================================
# 2) 데이터 로딩 (모든 테스트가 동일 데이터 사용)
# ==========================================================
@dataclass
class CsvData:
    header: List[str]
    rows: List[List[str]]

def load_csv_data(file_path: str, limit: Optional[int] = None) -> CsvData:
    rows: List[List[str]] = []
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, r in enumerate(reader):
            rows.append(r)
            if limit is not None and (i + 1) >= limit:
                break
    return CsvData(header=header, rows=rows)

def build_csv_lines_bytes(data: CsvData) -> List[bytes]:
    lines: List[bytes] = []
    lines.append((",".join(data.header) + "\n").encode("utf-8"))
    for r in data.rows:
        lines.append((",".join(r) + "\n").encode("utf-8"))
    return lines
# ==========================================================

# ==========================================================
# 3) TEST 1 : CPU-bound
#    - multiprocessing으로 코어를 최대한 활용
# ==========================================================
def _hash_worker(args):
    chunk_rows, rounds = args
    out = 0
    for _ in range(rounds):
        for r in chunk_rows:
            h = hashlib.sha256()
            for v in r:
                h.update(v.encode("utf-8", errors="ignore"))
            out ^= int.from_bytes(h.digest()[:8], "little")
    return out

def bench_cpu_hash(rows: List[List[str]], rounds: int, workers: int) -> float:
    n = len(rows)
    w = max(1, workers)
    chunk_size = (n + w - 1) // w
    chunks = [rows[i:i + chunk_size] for i in range(0, n, chunk_size)]

    start = now_perf()
    if w == 1:
        _hash_worker((rows, rounds))
    else:
        with Pool(processes=w) as pool:
            pool.map(_hash_worker, [(c, rounds) for c in chunks])
    return now_perf() - start
# ==========================================================

# ==========================================================
# 4) TEST 2: I/O write
# ==========================================================
def bench_io_write(lines: List[bytes], out_path: str, fsync_every: int, block_lines: int) -> float:
    start = now_perf()
    with open(out_path, "wb", buffering=1024 * 1024) as f:
        buf: List[bytes] = []
        for i, line in enumerate(lines, start=1):
            buf.append(line)
            if len(buf) >= block_lines:
                f.write(b"".join(buf))
                buf.clear()

            if fsync_every > 0 and (i % fsync_every == 0):
                f.flush()
                os.fsync(f.fileno())

        if buf:
            f.write(b"".join(buf))

        f.flush()
        if fsync_every > 0:
            os.fsync(f.fileno())

    return now_perf() - start
# ==========================================================

# ==========================================================
# 5) TEST 3: SQLite insert
# ==========================================================
def bench_sqlite_insert_pragmas(
    rows: List[List[str]],
    header: List[str],
    db_path: str,
    commit_every: int,
) -> float:
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")   # ~200MB
    cur.execute("PRAGMA mmap_size=268435456;")  # 256MB

    # 테이블 생성 (CSV 컬럼을 전부 TEXT로)
    col_def = ", ".join([f'"{c}" TEXT' for c in header])
    cur.execute(f"CREATE TABLE test_table ({col_def})")

    # INSERT 준비
    placeholders = ",".join(["?"] * len(header))
    q = f"INSERT INTO test_table VALUES ({placeholders})"

    # INSERT + 주기적 commit
    start = now_perf()
    pending = 0
    cur.execute("BEGIN;")
    for r in rows:
        cur.execute(q, r)
        pending += 1
        if commit_every > 0 and pending >= commit_every:
            conn.commit()
            cur.execute("BEGIN;")
            pending = 0

    conn.commit()
    conn.close()
    return now_perf() - start
# ==========================================================

# ==========================================================
# 6) Main
# ==========================================================
def main():
    # (A) 입력/출력 경로 확인
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH} (경로를 하드코딩 값으로 수정하세요)")

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    # (B) 실행 파라미터 결정
    workers = cpu_count() if WORKERS == "auto" else max(1, int(WORKERS))
    limit = None if LIMIT_ROWS == 0 else LIMIT_ROWS

    # (C) 헤더 출력(재현성 기록)
    print("=" * 72)
    print("System:", sys_info())
    print("CSV_PATH:", CSV_PATH)
    print("OUTDIR:", str(outdir.resolve()))
    print(f"workers={workers}, repeats={REPEATS}, cpu_rounds={CPU_ROUNDS}")
    print(f"block_lines={BLOCK_LINES}, fsync_lines={FSYNC_LINES}, sqlite_commit_every={SQLITE_COMMIT_EVERY}")
    print("=" * 72)

    # (D) 데이터 로딩 (모든 테스트가 동일 데이터 사용)
    data = load_csv_data(CSV_PATH, limit=limit)
    nrows = len(data.rows)
    print(f"[*] Loaded rows: {nrows:,}, cols: {len(data.header)}")

    # 파일쓰기용 bytes 라인 생성(1회)
    lines = build_csv_lines_bytes(data)

    # ------------------------------------------------------
    # (1) CPU_HASH
    # ------------------------------------------------------
    cpu_times = []
    for _ in range(REPEATS):
        cpu_times.append(bench_cpu_hash(data.rows, rounds=CPU_ROUNDS, workers=workers))
    cpu_thr = nrows / statistics.median(cpu_times)

    # ------------------------------------------------------
    # (2) IO_BUFFERED_WRITE
    # ------------------------------------------------------
    io_buf_times = []
    csv_buf_path = str(outdir / "bench_buffered.csv")
    for _ in range(REPEATS):
        io_buf_times.append(bench_io_write(lines, csv_buf_path, fsync_every=0, block_lines=BLOCK_LINES))
    io_buf_thr = nrows / statistics.median(io_buf_times)

    # ------------------------------------------------------
    # (3) IO_FSYNC_WRITE
    # ------------------------------------------------------
    io_fsync_times = []
    io_fsync_thr = None
    if FSYNC_LINES > 0:
        csv_fsync_path = str(outdir / "bench_fsync.csv")
        for _ in range(REPEATS):
            io_fsync_times.append(bench_io_write(lines, csv_fsync_path, fsync_every=FSYNC_LINES, block_lines=1))
        io_fsync_thr = nrows / statistics.median(io_fsync_times)

    # ------------------------------------------------------
    # (4) SQLITE_PRAGMAS
    # ------------------------------------------------------
    sqlite_pragmas_times = []
    db_pragmas_path = str(outdir / "bench_pragmas.db")
    for _ in range(REPEATS):
        sqlite_pragmas_times.append(
            bench_sqlite_insert_pragmas(data.rows, data.header, db_pragmas_path, commit_every=SQLITE_COMMIT_EVERY)
        )
    sqlite_pragmas_thr = nrows / statistics.median(sqlite_pragmas_times)

    # (E) 결과 출력
    print("\n" + "=" * 72)
    print(f"RESULTS (rows={nrows:,})")
    print("=" * 72)

    print(f"\n[CPU_HASH (rounds={CPU_ROUNDS}, workers={workers})]")
    print("  " + summarize_times(cpu_times))
    print(f"  throughput: {cpu_thr:,.2f} rows/s")

    print(f"\n[IO_BUFFERED_WRITE (block_lines={BLOCK_LINES})]")
    print("  " + summarize_times(io_buf_times))
    print(f"  throughput: {io_buf_thr:,.2f} rows/s")

    if FSYNC_LINES > 0:
        print(f"\n[IO_FSYNC_WRITE (fsync_every={FSYNC_LINES} lines)]")
        print("  " + summarize_times(io_fsync_times))
        print(f"  throughput: {io_fsync_thr:,.2f} rows/s")

    print(f"\n[SQLITE_PRAGMAS (commit_every={SQLITE_COMMIT_EVERY})]")
    print("  " + summarize_times(sqlite_pragmas_times))
    print(f"  throughput: {sqlite_pragmas_thr:,.2f} rows/s")

if __name__ == "__main__":
    main()

```

</div>
</details>
