# AI가 먼저 알아야 할 맥락

## 현재 목표와 상태

사용자는 제어기 자체의 성능 경쟁보다 RS02 하드웨어 조건 아래의 보행 범위·한계 원인 확인을 원한다.
20cm 성공을 위해 조건이나 정책을 계속 바꾸는 것은 한계 측정과 다르다.
최신 사용자 목표는 보존 보고서 `RS02_동적보행_계획.md` 6절에 있다.

채택 시연 정책은 rhythm이다. 15cm: model_500.pt, 20cm: model_1000.pt.
체크포인트 위치/해시는 [policies.json](../../config/policies.json)에 있다.
15cm rhythm500은 6/6, 20cm rhythm1000은 6/6; rhythm500의 기존 20cm 결과는 4/6이다.
모든 결과는 명목 시뮬레이션의 제한된 반복 표본이며 하드웨어 안전 검증은 아니다.

## 필요한 코드만 읽기

| 작업 | 파일 |
|---|---|
| 실행 옵션/체크포인트 선택 | `src/run_rs02_stairs.py` |
| 모델·관측·PD·공통 평가 | `src/rs02_rl.py` |
| 지형/계단 커리큘럼 | `src/rs02_rl_stairs.py` |
| 위상/지지/방향/채택 보상 | `rs02_rl_gait.py` → `rs02_rl_support.py` → `rs02_rl_route.py` → `rs02_rl_rhythm.py` (모두 src/) |
| 완주/출구 판정 | `src/rs02_course_validation.py` |
| 독립 프로세스 반복 | `src/evaluate_rs02_stairs_matrix.py` |
| 이전 MPC/crawl 비교 | `src/rs02_mpc.py`, `src/rs02_walk.py` |
| 경로 변경 | `src/robost_paths.py` |

flat 모듈 이름은 동적 import 및 정책 어댑터 호환성을 위해 유지했다.
평가 source_snapshot에는 `robost_paths.py`를 포함한다. 이전 스냅샷은 정리 전 경로를 담은 증거이며 자동 재실행 패키지가 아니다.

## 고정 조건

- 총질량 16.940606kg, 12관절, hip/thigh/calf 토크 한계 17/17/25.2Nm.
- 정책 50Hz, 물리 500Hz. 관절 hard range와 실제 접촉 유지.
- 기본 코스 5단 상승 → 1m 평지 → 5단 하강, 깊이 0.30m, 폭 1.6m.
- 명령 속도 0.25m/s, 평가 최대 40초, seed 42/43/44를 2회씩.
- 발별 출구 착지·모든 발 x>4.30m 이후 2초 생존. 첫 실패를 영구 기록.
- 종아리 접촉과 정격 대비 RMS 문제가 남아 있으며 실물 최대 높이 미확정.

## 문서 유지 원칙

현재 상태는 [STATUS](../user/STATUS.md) 한 곳에 요약한다.
상세 실패·수정 근거는 [보고서 목록](../reports/README.md)에서 필요한 문서만 읽는다.
새로운 수치에는 정책/조건/원본 결과 경로를 붙이고, 시연과 고정 정책의 한계 시험을 구별한다.
