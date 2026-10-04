# 진행 기록

## 현재 상태 · 2026-10-05

최신 계단 형상과 TPU 발 접촉 설정에서 같은 RS06 정책으로 **12cm 6/6, 18cm 6/6 완주**를 확인했다. 평지는 **60초 생존과 수치 품질 기준 3/3**이다. 기본 파일은 `assets/policies/rs06.pt`, SHA-256은 `6f8ffe21ab7a148f149233866a05185e8ada56081805a9c389c17bea0e19851d`다. 가중치가 저장소에 포함되므로 후임 개발자는 학습 없이 `robost-stairs --stairs-cm 12` 또는 `--stairs-cm 18`로 실행할 수 있다.

| 조건 | 독립 평가 결과 | 자세 RMS / 피크 최댓값 | 접지 발 미끄럼 최댓값 | 종아리 접촉력 피크 | 모터 RMS/정격 최대비 |
|---|---|---:|---:|---:|---:|
| 평지 | 60초 생존·수치 품질 3/3 | 1.32° / 2.25° | 0.0251m/s | 0.0N | 0.702 |
| 12cm | 완주 6/6, 35.00–35.68초 | 12.21° / 26.26° | 0.0428m/s | 294.7N | 1.005 |
| 18cm | 완주 6/6, 35.68–35.70초 | 18.27° / 35.56° | 0.0484m/s | 480.2N | 0.956 |

계단은 seed 42/43/44 × 각각 독립 프로세스 2회, 평지는 같은 3개 seed × 1회다. 속도 0.25m/s, 시험 상한 60초, 환경 1개, 자동 리셋 없음이다. 계단 시험은 출구 완주 시 종료하므로 계단의 생존 표시는 완주 시점까지를 뜻한다. 완주는 각 발이 출구 바닥에 접지하고 모든 발이 x=7.612m 밖에서 2초 유지한 결과이며, 기존 실패·우회·관절 제한 판정을 유지했다.

계단의 `numerical_gate_pass=false`는 완주 판정과 별개다. 그 수치 기준은 평지용이고, 계단에는 자세 변화와 종아리 접촉이 남아 있다. `hardware_pass=null`이며 실물 성공·모터 열·인지 성능의 증거는 아니다. 위 피크는 50Hz 저장 표본이고 500Hz 물리 하위 스텝의 피크를 보증하지 않는다. 현재 정책의 6/10/14/16/20cm 및 외란 성능은 새로 검증하지 않았다.

## 물리 설정과 최소 변경

- 끝단 geom 미끄럼 계수를 1.35에서 **1.25**로 변경했다. 승인된 32/6/0.5/0.3/3/18cm 단면과 10단 상승·하강 형상은 유지한다.
- 네 TPU 발 geom만 `condim=4`, `friction=(0.8, 0.003, 0.0001)`로 설정했다. 명시적 발–지형 pair의 일반 표면 미끄럼 계수는 0.8, 끝단은 1.25이고 비틀림 계수는 둘 다 0.003m다. 구름 계수는 저장하지만 condim4에서 작동하지 않는다. 다른 로봇 충돌과 URDF 관성·관절·토크·PD는 유지했다. 해당 물리 변경 커밋은 `2ec824d`이다.
- 기존 지형 스캔의 아래 방향 광선이 높은 계단 내부에서 시작하면 윗면 대신 바닥·아랫면을 읽는 문제가 있었다. `RaisedGridPattern`으로 광선 시작점을 베이스 위 **1m**로 높이고 실제 베이스 높이 기준과 관측 크기·순서를 유지했다. 이상적 지형 관측을 고친 것이며 실물 센서 검증을 뜻하지 않는다.
- `target` 학습 단계는 **12/14/16/18cm 네 행에 처음부터 분산**한다. 최소 커리큘럼 높이는 12cm, `gait_contact` 가중치는 1.5다. 액션 스케일·하드웨어 한계·평가 종료 조건을 완화하지 않았다.

## 선택한 정책의 학습 계보

초기 정책 SHA-256: `4ee77201d9895c682317ea5cb34851bfa8c2707d6a2a97a41b19f9e56e0b91d3`. 기존 정책의 명목 물리·관측 계보를 이어서 actor·critic·normalizer를 불러왔으며 각 실행의 optimizer는 새로 시작했다. std는 고정했고 엔트로피 가중치는 0이다.

| 실행 | 초기 checkpoint | seed | 요청 업데이트 | 선택한 checkpoint / 사용 업데이트 | 환경×스텝 | std | normalizer pseudocount |
|---|---|---:|---:|---|---|---:|---:|
| target_v1 | 기존 rs06.pt | 42 | 400 | model_399.pt / 400 | 512×24 | 0.15 | 1,000,000 |
| target_v2 | v1/model_399.pt | 43 | 800 | model_799.pt / 800 | 512×24 | 0.15 | 1,000,000 |
| target_v3 | v2/model_799.pt | 44 | 800 | **model_799.pt / 800** | 512×24 | 0.12 | 10,000,000 |

선택 정책에 반영된 추가 업데이트는 **2,000회**, 전이는 **24,576,000개**다. 독립 평가를 통과한 checkpoint를 선택했다. 요청한 전체 학습량과 선택 정책에 반영된 학습량은 구분한다. 기존 3,100회 적응 이력은 아래 역사 자료에 보존했다.

정확한 입력 가중치는 `git show 2ec824d:assets/policies/rs06.pt`로 복원할 수 있다. 현재 코드와 `target` 설정에서 README의 세 단계 명령을 사용하고 `target_v3/model_799.pt`를 평가한다. 출력 경로는 새로 지정해야 한다. GPU 연산은 같은 seed에서도 비트 단위 재현을 보장하지 않으므로 재학습 결과는 별도 독립 평가가 필요하다.

선택 파일 해시, 단계별 입력·선택 가중치 해시, 실제 설정, 소스·물리 입력 해시, 각 시험의 간략 결과는 `config/rs06_policy.json`에 포함한다. 원시 로그·영상·중간 가중치는 Git에 넣지 않으며 RS06 기본 가중치는 하나만 포함한다.

## 남은 과제

완주한 계단에도 큰 자세 변화·종아리 접촉이 있다. 발 지지 전환과 내려가는 구간의 자세를 개선하며 현재 12/18cm와 평지 성능을 함께 재검증해야 한다. TPU 계수는 설정값이며 실제 출력 경도·접촉면·바닥 재질별로 눌림, 미끄럼, 비틀림을 측정해야 한다. 실측 토크-속도·열·지연·탄성·인지 오차, CPU/GPU 접촉 동등성 및 실물 동작은 미검증이다.

## 이전 정책과 결과 · 역사 자료

이하 내용은 **이전 형상·마찰·관측 설정과 이전 정책**의 개발 당시 기록이다. 이 부분의 '현재'는 당시 시점을 가리키며 최신 정책의 성능으로 읽으면 안 된다. 기존 6/10/12cm 성공이나 14/18cm 실패는 이번 정책의 결과와 구분한다.

<details>
<summary>2026-10-04 이전 정책의 상세 개발 기록</summary>

### 이전 정책 상태 · 2026-10-04

RS06 v5 URDF와 실제 계단 치수를 사용하는 MuJoCo 모델을 구성하고 정책을 조정했다. 현재 기본 정책은 `assets/policies/rs06.pt`, SHA-256은 `4ee77201d9895c682317ea5cb34851bfa8c2707d6a2a97a41b19f9e56e0b91d3`이다. 모든 높이에 이 가중치 하나를 사용한다. 실제 18cm 계단 완주는 아직 확인하지 못했다.

| 조건 | 기존 독립 평가 결과 | 관측 |
|---|---|---|
| 평지 | 60초 생존 3/3, 수치 품질 기준 3/3 | 평균 0.258m/s, 자세 RMS 약 0.86°, 종아리 접촉 없음 |
| 6cm | 완주 3/3, 33.38~34.08초 | 종아리 접촉력 최대 127N |
| 10cm | 완주 3/3, 35.02~35.74초 | 자세 RMS 약 10.2°, 종아리 접촉력 최대 363N |
| 12cm | 완주 3/3, 36.30~36.72초 | 자세 RMS 약 11.9°, 종아리 접촉력 최대 299N, 모터 RMS/정격 최대비 약 1.012 |
| 14cm | 완주 0/3 | 6.14~19.94초에 몸체 접촉·넘어짐 |
| 18cm | 완주 0/3 | 5.92~9.62초에 정체·넘어짐·몸체 접촉, 모터 RMS/정격 최대비 약 1.43 |

조건은 seed 42/43/44, 속도 0.25m/s, 상한 60초, 환경 1개, 자동 리셋 없음이다. 표는 정책 조정 당시 확인한 결과이며 구조 변경 후 새로 재현한 평가로 간주하지 않는다. 시험한 높이 외의 중간 높이나 외란에 대한 안정성을 뜻하지 않는다. 원시 중간 실험 자료는 현재 실행 패키지에 포함하지 않는다. 정책·훈련 입력 해시와 높이별 집계는 `config/rs06_policy.json`에 있다.

### 완료한 개발

- RS06 v5 원본 질량·COM·관성·관절 범위·35개 충돌 형상을 유지한 모델을 구성했다. 기본 전장 포함 18.082756kg, 12관절, hip/thigh/calf 토크 한계 17/23/30Nm다.
- 계단은 높이 18cm·깊이 32cm·상승 10단/하강 10단·중간 평지 1m·폭 1.6m·5mm×6cm 돌출부다. CPU 장면과 RL 환경이 같은 지형 생성 코드를 사용한다.
- 평지 300회 → 낮은 계단 800회 → 높은 계단 1,200회 → 혼합 800회, 총 3,100회 PPO 학습을 실행했다. 256환경×24스텝 기준 19,046,400개 전이를 수집했다.
- 높은 계단 전용 후보는 평지 0/3·6cm 2/3으로 퇴화했다. 혼합 학습에서 평지 25%와 낮은 계단 25%를 유지해 퇴화를 줄였다. 높은 계단 후보의 14cm 1/3에서 현재 혼합 후보의 0/3으로 감소한 결과도 함께 고려했다.
- 같은 가중치·seed에서도 미세한 초기 수치 차이가 누적되어 완주와 실패가 달라졌다. GPU 비트 단위 재현을 보장하지 않는다.

### 당시 실행 확인

통합 `robost` 환경은 Python 3.11.16, MuJoCo/mujoco-warp 3.11.0, PyTorch 2.14.0을 사용한다. CPU 검사 33개·RL 검사 44개, 컴파일·코드 형식·패키지 의존성 검사를 통과했다. 별도의 깨끗한 CPU 환경에서도 Torch·mjlab 없이 모델 컴파일과 2초 서기를 확인했다. 저장 XML/MJB에는 `stand` 초기 자세가 포함된다.

현재 체크포인트로 seed 42를 추가 확인한 결과는 평지 60초 생존, 12cm 36.38초 완주, 18cm 8.82초 넘어짐이다. 각 조건 1회 확인이며 위의 기존 18회 평가를 새로 재현한 결과가 아니다. 16개 환경에서 PPO 1 iteration과 새 체크포인트 생성도 확인했으며, 이 검사용 가중치는 기본 정책에 적용하지 않았다.

### 남은 문제

18cm에서는 둘째 단 진입 중 앞뒤 발의 지지 높이 차이와 발 배치 문제가 관찰됐다. 지지가 부족한 상태에서 몸체가 크게 기울거나 thigh가 돌출부에 닿았다. 현재 정책의 실패이며 하드웨어의 근본적 불가능성을 증명하지 않는다.

완주한 10·12cm에도 종아리 접촉과 모터 부하가 남아 있다. 기존 전체 50Hz 저장 표본에서 17/23/30Nm 상한이나 하드 관절 범위를 넘지 않았지만, 500Hz 물리 하위 스텝의 피크를 보증하지 않는다. 완료한 코스·보행 품질·하드웨어 안전은 별개이며 `hardware_pass=null`이다.

물리는 명목 고정 armature와 제한된 이상적 PD다. 무릎 모터 환산은 CAD 링크비 표를 사용하며, 관측에는 이상적인 시뮬레이터 지형·위치 정보가 포함된다. 실측 토크-속도·열·지연·탄성·인지 오차와 CPU/GPU 접촉 동등성, 실물 동작은 미검증이다.

다음 정책 조정은 발 배치·지지 전환·자세를 개선하면서 평지와 낮은 계단 성능을 함께 확인하는 방향이다. 모델·토크·충돌·출구 판정을 유리하게 바꾸지 않는다. 20cm 성공 자체는 필수 목표가 아니다.

하드웨어와 실제 계단의 제공 출처: [하드웨어 수정사항](https://app.notion.com/p/3ea243d742fa80a0820de79cf3eb9cc8), [실제 계단 정보](https://app.notion.com/p/3ea243d742fa8036af79fb879860a846).

</details>

<details>
<summary>이전 정책의 manifest 전체 — 해시·설정·입력·소스 계보 보존</summary>

```json
{
  "adapter": "rs06",
  "path": "assets/policies/rs06.pt",
  "sha256": "4ee77201d9895c682317ea5cb34851bfa8c2707d6a2a97a41b19f9e56e0b91d3",
  "status": "experimental",
  "course_18cm_completed": false,
  "speed_m_s": 0.25,
  "duration_s": 60.0,
  "seeds": [
    42,
    43,
    44
  ],
  "by_height": {
    "0": {
      "total": 3,
      "completed": null,
      "survived": 3,
      "failures": {}
    },
    "6": {
      "total": 3,
      "completed": 3,
      "survived": 3,
      "failures": {}
    },
    "10": {
      "total": 3,
      "completed": 3,
      "survived": 3,
      "failures": {}
    },
    "12": {
      "total": 3,
      "completed": 3,
      "survived": 3,
      "failures": {}
    },
    "14": {
      "total": 3,
      "completed": 0,
      "survived": 0,
      "failures": {
        "illegal_contact": 1,
        "fell_over": 2
      }
    },
    "18": {
      "total": 3,
      "completed": 0,
      "survived": 0,
      "failures": {
        "stalled": 1,
        "fell_over": 1,
        "illegal_contact": 1
      }
    }
  },
  "hardware_pass": null,
  "selection_note": "Mixed replay restores flat and lower-course performance; fixed policy across heights. Completion does not certify gait quality or hardware safety. Heights 14/18 cm did not complete.",
  "validation_date": "2026-10-04",
  "validation_record": "PROGRESS.md",
  "training": {
    "stage": "mixed",
    "iterations": 800,
    "total_adaptation_iterations": 3100,
    "num_envs": 256,
    "num_steps_per_env": 24,
    "initial_checkpoint_sha256": "0eed125720ee9f2de9108e6fad0519be3296bde14736d8bf18dd1ff7157f401b",
    "settings": {
      "description": "RS06 v5 nominal fixed-physics policy adaptation; training resets are not evaluation passes.",
      "speed_m_s": 0.25,
      "learning_rate": 0.0003,
      "learning_rate_schedule": "adaptive",
      "normalizer_pseudocount": 1000000,
      "exploration_std": 0.15,
      "save_interval": 50,
      "num_steps_per_env": 24,
      "stages": {
        "flat": {
          "episode_seconds": 15.0,
          "min_rise_m": 0.0,
          "max_rise_m": 0.0,
          "terrain_rows": 0,
          "gait_contact_weight": 0.8,
          "swing_obstacle_weight": 0.0,
          "pose_weight": 1.0
        },
        "low": {
          "episode_seconds": 55.0,
          "min_rise_m": 0.02,
          "max_rise_m": 0.1,
          "terrain_rows": 5,
          "gait_contact_weight": 1.5,
          "swing_obstacle_weight": -1.0,
          "pose_weight": 0.5
        },
        "stairs": {
          "episode_seconds": 70.0,
          "min_rise_m": 0.08,
          "max_rise_m": 0.18,
          "terrain_rows": 6,
          "gait_contact_weight": 3.0,
          "swing_obstacle_weight": -1.0,
          "pose_weight": 0.5
        },
        "mixed": {
          "episode_seconds": 70.0,
          "min_rise_m": 0.0,
          "max_rise_m": 0.18,
          "terrain_rows": 10,
          "max_initial_level": 6,
          "replay_fractions": {
            "flat": 0.25,
            "low_2_4_6_cm": 0.25,
            "adaptive": 0.5
          },
          "gait_contact_weight": 1.5,
          "swing_obstacle_weight": -1.0,
          "pose_weight": 0.5
        }
      }
    },
    "mujoco_version": "3.11.0",
    "torch_version": "2.14.0",
    "source_snapshot_sha256": {
      "rl/__init__.py": "6e83d855b10511db192670fff885edc641abb80a0fa76ea35f2a68130873466e",
      "paths.py": "12e9566086e630b6c9f31ac59b1780daae73a38623227a82a505667af547af44",
      "simulation/__init__.py": "822859153c52e8064ffb84897ffee7e6d9130a9bbe17f4c846cb888b8b30267a",
      "simulation/hardware.py": "89100d6a9c418050d9b9a97627969796d1695492a7ef5455f8fb9d33001ff337",
      "simulation/terrain.py": "4a5667c77c1f1a92d6b1fca4a3d24eaa9c1277ceedbaa7c8d092a10ee831db4f",
      "rl/base.py": "b13922fd8e67764fb14b7d5bca65e7eae468756689689c18cf48de78ba7f7f89",
      "rl/stairs.py": "c988ef48d4721a7830e74e016f747c2a30e927a2b4882f1ce607c128b5b9c2ba",
      "rl/gait.py": "a5c95dac5be274ecbbe7dcd5cb528102b65734cb0dad74a230ff67d16b91ede4",
      "rl/support.py": "98fecf1c59ed8742e8f49e5e03f21a2f565cadd7ff809bf86d36b25ed6eec5d4",
      "rl/route.py": "2a3c2e1ed5a4778aa56c265af89a9cbd7f04c381054a0a25aa6b4fb9450adff1",
      "rl/rhythm.py": "aec843f469cf842188597ad27015b687082afdecdd4160cf940c32346e9ae804",
      "simulation/model.py": "a3903820f948000989c551b7223140dd0b2aa305849e6a173193e905bed4ed93",
      "rl/rs06.py": "c814c6682e9c7a5b86d256601303476ed124716ecaa29750ad22d0d7d8a980ac"
    },
    "model_input_sha256": {
      "rs06_quadruped.urdf": "1d9f7164a38fadf1103a311fdbd0dc0374e0c8c3af23e4364a681ac466ce0552",
      "actuator_params.yaml": "e561e18dbb4f5ca14690c093256d9246eb32ac671d5cb779c5912d7baedd1557",
      "knee_linkage.csv": "8d7510ae3258f0fbd717986ae143adae5e1148d9a10c1e476aa03d20af92f777",
      "urdf_summary.json": "a7d55d2bf397afb69c50014378d96e4a9a14343e8da2d834e6e86466d23efee1",
      "courses.json": "38ddca0f485901b4df62457613e66bcfa43d7fa861ba1104b74f098e680c4ff2",
      "rs06_training.json": "88bb9cd9a966573f2126348098a990c46eb81c5886c458b220424a855e6b9dd8"
    },
    "hardware": {
      "mass_kg": 18.082756,
      "joint_order": [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint"
      ],
      "gravity_m_s2": [
        0.0,
        0.0,
        -9.81
      ],
      "physics_dt_s": 0.002,
      "control_dt_s": 0.02,
      "urdf_sha256": "1d9f7164a38fadf1103a311fdbd0dc0374e0c8c3af23e4364a681ac466ce0552",
      "controller": "unchanged bounded PD60/60/80, damping2"
    },
    "normalizer_counts_before": {
      "actor": 8372800.0,
      "critic": 8372800.0
    },
    "normalizer_counts_start": {
      "actor": 1000000.0,
      "critic": 1000000.0
    }
  }
}
```

</details>
