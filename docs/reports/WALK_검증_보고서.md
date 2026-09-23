> 보존 보고서: 정리 이전의 파일명·실행 명령·진행상황이 포함됩니다. 현재 기준은 [진행상황](../user/STATUS.md), 실행은 [실행 안내](../user/RUN.md)를 먼저 보세요.

# RS02 보행 제어기 교체 및 하드웨어 검토

> **사용자 목표 재정의:** 이 문서의 계단 crawl 완주는 저속 동작 가능성 시험이다.
> 연속적이고 안정적인 ‘잘 걷는 보행’의 통과 근거가 아니다. 현재 작업 기준은
> [동적 보행 계획](RS02_동적보행_계획.md)이며, 공개 학습 환경의 RS02 이식·검증을 진행한다.

## 목적과 판정 범위

목적은 제어기를 연구하는 것 자체가 아니라, RS02의 질량·관절 범위·모터 제약 아래에서
평지와 15 / 20 cm 계단을 이동할 수 있는지 확인하는 것이다. 제어기의 실패만으로
하드웨어 불가능을 판정하지 않는다. 반대로 시뮬레이션 완주만으로 실물 하드웨어를 승인하지 않는다.
JSON의 `success`는 시뮬레이션 이동 시험 결과이며 `hardware_pass`는 미판정(`null`)이다.

## 확인된 결과 (2026-09-21)

| 시험 | 결과 | 주의점 |
|---|---|---|
| 평지 0.25 m/s 명령, 30초 | 넘어짐 없이 3.816m, 평균 0.127m/s | 뒷무릎 모터 RMS 최대 6.67N·m |
| 평지 0.50 m/s 명령, 30초 | 넘어짐 없이 10.184m, 평균 0.339m/s | 뒷무릎 모터 RMS 최대 7.05N·m |
| 15cm, 상승→회전→후진 하강 | **전체 완주**, 698.826초, 193회 발 이동 | 종아리 접촉·작은 관절 한계 초과 있음 |
| 20cm, 상승→회전→후진 하강 | 537.692초에서 착지 확인 실패 | 내리막 도중 FL 발, 전체 완주 아님 |

평지 평균은 초기 정지·가속 구간을 포함한다. 계단 명령은 0.04m/s지만 15cm의 전체 평균은
약 0.00661m/s다. 회전과 지지 이동을 포함한 매우 느린 가능성 시험이며 실용 주행 속도 검증이 아니다.

15cm 결과: `output/walk/validation_stairs15_reverse/stairs_15cm_course_report.json`.
진단 재시작 없이 처음부터 진행했으며 `success: true`, `diagnostic_resume: null`이다.
최대 모터 환산 RMS 5.390N·m, 종아리 접촉 최대 침투 약 2.27mm,
최소 관절 여유 -0.00383rad(약 -0.22도)다. MuJoCo의 부드러운 접촉·관절 한계 때문에 발생한
작은 침투/초과도 제거하지 않고 보고한다. **완주가 충돌·관절 범위 검증 합격을 의미하지 않는다.**
약 10Hz의 6,989개 자세를 별도 검사했을 때 1mm 초과 자체 충돌은 검출되지 않았다.

20cm 결과: `output/walk/validation_stairs20_reverse/stairs_20cm_course_report.json`.
5단 상승과 상단 회전 후 내리막에서 멈췄다. 최대 모터 환산 RMS 5.566N·m,
몸통/hip 접촉 및 종아리 약 5.29mm 침투, 관절 최소 여유 -0.0130rad가 기록됐다.
QP 오류는 없지만 실제 착지가 확인되지 않았다. ‘QP가 풀림’과 ‘물리 보행 성공’은 다른 조건이다.

15cm 검증 뒤의 20cm 보완은 20cm 지형에만 적용했다. 후미가 높은 단에 남아 있는데 선두가
두 단 아래로 먼저 내려가는 것을 막고, 앞뒤 발 간격과 후미 착지 위치를 조절했다.
추가 착지 위치 실험은 `output/walk/validation_stairs20_landing/`에 분리하여 보존한다.
후미 착지 여유를 10cm에서 18cm로 늘린 추가 실험은 504.176초, x=3.124m에서 FR 착지 실패로
더 일찍 멈췄다. 따라서 이 변경은 채택하지 않고 현재 코드는 10cm 착지 설정으로 복원했다.
각 결과의 `source_sha256`은 해당 시험 시작 시점의 코드이며 개발 시험 사이에 코드가 달라질 수 있다.

## 재사용한 코드와 변경 범위

- 평지: 제공받은 패키지의 `07_스크립트/URDF_시뮬레이션/sim_gaits.py`에 있던
  Raibert 발 위치 조절 + 역기구학 + 관절 PD 방식을 `RaibertTrot`에 이식했다.
  기존 MPC와 독립적인 비교 기준이다. 원본 패키지는 수정하지 않았다.
- 계단: MIT Cheetah의 `FootSwingTrajectory.cpp`를 수정 없이 C++ 공유 라이브러리로
  빌드해 수평 발 궤적에 실제 사용한다. 계단 모서리를 피하는 상승·이동·하강 분할,
  수직 궤적, 지지삼각형·착지 계획 및 MuJoCo 전신 제어는 RS02용 어댑터다.
- **MIT Cheetah 전체 제어기/WBIC를 그대로 실행한 것이 아니다.** 원본의 동역학 모델은
  RS02와 다르다. 전신 QP와 기구학 피드백은 공개 WBC 구조를 참고해 MuJoCo 모델에 맞게 구현했다.
- CHAMP, OCS2 계열도 조사했지만, ROS 의존성과 로봇 모델 이식이 필요하다.
  다른 로봇의 학습된 정책을 RS02에서 검증 없이 그대로 쓰지 않았다.

출처: [MIT Cheetah Software](https://github.com/mit-biomimetics/Cheetah-Software),
[CHAMP](https://github.com/chvmp/champ),
[OCS2/MuJoCo ROS2 예제](https://github.com/HexiangZhou/Quadruped-Control-OCS2-ROS2).

고정된 소스 버전:

- Cheetah: `c71c5a138d3e418cc833e94e25357ceea8955daa`, MIT 라이선스
- Eigen 3.3.7: `cf794d3b741a6278df169e58461f8529f43bce5d`, 포함된 라이선스 파일 유지
- 원본·라이선스는 `third_party/`에 보존했다. ROS 설치 없이 실행한다.

## 물리 시험 조건

- 총질량 16.940606 kg, 기존 5 kg payload 포함
- 최대 관절 토크: hip / thigh / calf = 17 / 17 / 25.2 N·m
- 무릎 링크비 1.48로 모터 토크와 관절 토크를 구분
- 기존 관절 범위, 중력, 로봇-계단 접촉 유지
- 몸통 고정, `qpos` 강제 이동, 외부 지지력, 중력 제거 없음
- 계단: 깊이 30 cm, 폭 160 cm, 5회 상승 → 상단 평지 1 m → 5회 하강
- 지형 높이를 정확히 알고 있는 시험이다. 센서 기반 계단 인식 성공을 의미하지 않는다.
- 보수적인 한 발씩 옮기는 crawl로 계단을 시험한다. 속도 추종·고속 트롯 성능 시험과 다르다.

기존 문제를 줄이기 위해 몸통 pitch/roll을 발 높이에 맞추고, 기계적 무릎 한계에 따른
도달 범위를 반영했다. 실제 지지삼각형 안으로 무게중심을 이동한 후 발을 들고,
실제 발 접촉과 목표점 오차를 확인한 후 다음 발로 넘어간다. 앞발만 계속 먼저 올라가지
않도록 앞·뒷발 간격을 제한한다. QP에는 동역학, 지지 접촉, 마찰, 관절 토크 제약을 넣었다.
전신 역기구학 관절 목표에 PD 피드백을 더하며 최종 토크도 같은 한계로 제한한다.

## 실행

```bash
cd /home/bang/robost
conda activate rs02-mujoco

# 공유 라이브러리가 없을 때 자동 빌드되지만, 명시적으로도 가능(g++ 필요)
python build_cheetah_swing.py

# 평지: 원본 패키지 기반 Raibert 트롯
python rs02_walk.py --terrain flat --speed 0.25 --duration 30

# 계단: 전신 제어 + 저속 crawl, 상단 회전 후 후진 하강
python rs02_walk.py --terrain stairs --step-height-cm 15 --speed 0.04 --reverse-descent --duration 900
# 20cm는 아직 전체 완주가 검증되지 않은 실험 모드
python rs02_walk.py --terrain stairs --step-height-cm 20 --speed 0.04 --reverse-descent --duration 900
```

`--speed` 단위는 m/s다. 계단은 지지 이동·착지 확인·안전한 발 이동 시간이 우선되어
실제 속도가 명령보다 느릴 수 있다. 큰 명령을 넣어도 안전 시간 제한을 해제하지 않는다.
`--reverse-descent`가 없으면 전진 내리막으로 진행하며 현재 완주가 검증되지 않았다.
완주/실패 시 프로그램이 종료되고 결과를 저장한다. GUI는 시뮬레이션 실제 시간으로 실행한다.
15cm 전체 실행은 약 12분을 잡아야 한다. 빨리 검산하려면 `--headless`를 붙인다.
`--headless`는 창 없이 실행, `--video`는 영상 저장, `--output 경로`는 결과 분리다.
`rs02_mpc.py`는 이전 MPC 비교용으로 남겼으며 새 보행기는 `rs02_walk.py`다.

기본 결과는 `output/walk/`의 `flat_*`, `stairs_15cm_course_*`, `stairs_20cm_course_*`다.
개발 중 매개변수 비교 결과는 그 아래 개별 실험 폴더에 보존한다.

## 기록과 재검증

- `torque_peak_Nm`, `torque_rms_Nm`: 관절 토크
- `motor_equivalent_peak_Nm`, `motor_equivalent_rms_Nm`: 무릎 링크비 환산 후 모터 토크
- `motor_above_6Nm_seconds`: 모터별 정격 6 N·m 초과 누적 시간
- `joint_limit_min_margin_rad`: 기계적 관절 범위까지 최소 여유(음수면 초과)
- `nonfoot_contact_pairs`: 종아리·몸통 등 발 이외의 지형 접촉
- `stop_reason`: 착지 지연, 지지 이동 지연, 자세 기준 초과, QP 실패 등을 구분
- `log`: 자세·발 위치·관절 상태 로그. 결과를 재생하고 별도 충돌 검토 가능

```bash
python -m unittest test_rs02_walk test_rs02_terrain
python audit_rs02_run.py output/walk/stairs_15cm_course_report.json
```

`audit_rs02_run.py`는 저장한 자세에서 로봇 자체 충돌을 별도로 검사한다. 이는
자체 충돌을 켠 상태의 동역학 재시험을 대체하지 않으며, 약 10 Hz 로그 샘플 사이의 충돌은 놓칠 수 있다.

## 실물 판단에 남는 제한

1. 현재 무릎은 실제 4절 링크가 아니라 직결 회전축 + 고정 링크비로 근사했다.
2. 회전자 관성·모터 토크-속도 곡선·CAN 지연·감속기 마찰·백래시가 실측으로 보정되지 않았다.
3. 연속 6 N·m 사양은 특정 방열 조건의 정격이다. RMS가 6 이하라도 현재 로봇의 방열을 보증하지 않는다.
4. 충돌 형상은 원본의 박스·구 근사다. 얕은 종아리 접촉도 실제 CAD 형상으로 다시 확인해야 한다.
5. 기본 동역학에서는 자체 충돌을 끈 상태다. 별도 로그 검사는 추가 선별일 뿐이다.
6. payload 질량·무게중심은 배치 확정 후 다시 맞춰야 한다. 실제 마찰과 배터리 전압도 별도 변수다.

따라서 결과는 **모델 조건에서의 보행 가능성/위험 신호**로 해석하고, 제작 승인이나
실물 무손상 운전 보증으로 해석하지 않는다.

## 발표 자료

- `output/presentation/RS02_보행모델_설계_발표.pptx`: 10장, 발표자 노트와 출처 포함
- `보행모델_설계_발표대본.md`: 8~10분용 설명, 코드 연결, 예상 질문
- `output/walk/validation_stairs15_reverse/stairs_15cm_course_replay.mp4`: 실제 물리 상태 로그 8배속 재생
- `output/walk/validation_flat_050/flat_replay.mp4`: 평지 물리 로그 재생
