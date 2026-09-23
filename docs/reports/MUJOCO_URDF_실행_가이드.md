> 보존 보고서: 정리 이전의 파일명·실행 명령·진행상황이 포함됩니다. 현재 기준은 [진행상황](../user/STATUS.md), 실행은 [실행 안내](../user/RUN.md)를 먼저 보세요.

# RS02 URDF를 MuJoCo에서 여는 방법

> 2026-09-21 보행 시험 추가: 모델 표시만 보려면 이 문서의 실행기를 사용한다.
> 실제 평지/계단 보행은 `rs02_walk.py`와 [새 검증 보고서](WALK_검증_보고서.md)를 참고한다.
> 15cm는 회전 후 후진 하강 전체 완주, 20cm는 미완료이며 하드웨어 승인 결과가 아니다.

## 이번에 구성한 환경

- conda 환경 이름: `rs02-mujoco`
- Python: 3.11
- MuJoCo: 3.13.0
- 실행기: `run_rs02_mujoco.py`
- 읽는 모델: `RS02_4족로봇_최종정리_2026-09-15/04_URDF_IsaacSim/rs02_quadruped/rs02_quadruped.urdf`

원본 `07_스크립트`에는 Windows 절대경로가 들어 있으므로, 현재 Linux 폴더에서는
새 실행기를 사용한다. 원본 패키지 파일은 수정하지 않는다.

## 처음부터 따라 하기

터미널에서 다음 순서로 실행한다.

```bash
cd /home/bang/robost
conda env create -f environment.yml
conda activate rs02-mujoco
python run_rs02_mujoco.py
```

환경이 이미 만들어져 있다면 마지막 두 줄만 실행하면 된다.

```bash
cd /home/bang/robost
conda activate rs02-mujoco
python run_rs02_mujoco.py
```

기본값은 `rs02_mujoco_standing_pose.png`와 같은 서기 자세다. 창은 자세를 고정해서
보여주는 미리보기이며, 충돌 박스는 숨긴다. CAD 저장 자세를 보려면 다음과 같이 실행한다.

```bash
python run_rs02_mujoco.py --pose cad
```

창을 닫으면 실행기가 종료된다. 마우스 왼쪽 드래그는 회전, 오른쪽 드래그는 이동,
휠은 확대/축소에 쓴다.

## GUI 없이 확인하기

URDF, STL, 관절 수, 전체 질량이 정상 로드되는지만 확인한다.

```bash
conda run -n rs02-mujoco python run_rs02_mujoco.py --check
```

정상이면 월드 바디를 포함해 `body=19`, `joint=12`, `mesh=13`,
`mass=16.9406 kg`가 표시된다.

PNG 한 장으로 렌더링하려면 다음 명령을 쓴다.

```bash
conda run -n rs02-mujoco python run_rs02_mujoco.py \
  --pose stand --snapshot /home/bang/robost/rs02_mujoco_snapshot.png
```

## 문제 해결

- `EnvironmentNameNotFound`: `conda env create -f environment.yml`을 먼저 실행한다.
- `DISPLAY ... 없습니다`: 데스크톱 세션의 터미널에서 실행한다. SSH라면 X11 전달이 필요하다.
- 모델 파일을 못 찾음: `run_rs02_mujoco.py`와 RS02 자료 폴더가 지금처럼 같은
  `/home/bang/robost` 아래에 있어야 한다.
- 검은 화면 또는 OpenGL 오류: 그래픽 드라이버가 사용 가능한 로컬 데스크톱에서 다시
  실행하고, 먼저 `--check`로 모델 로드 자체가 정상인지 구분한다.

## 이 실행기가 보정하는 점

MuJoCo가 URDF를 기본값으로 바로 읽으면 visual STL과 fixed link를 합치거나 버릴 수 있다.
실행기는 임시 compiler 설정으로 `discardvisual=false`, `fusestatic=false`를 적용한다.
그래서 13개 STL 외형, 5 kg payload, 발 링크까지 보존된 전체 16.9406 kg 모델을 표시한다.
기본 표시 자세는 서기 자세(허벅지 0.68022 rad, 무릎 -1.3664 rad)다.
CAD 저장 자세는 `--pose cad`로 선택한다.

2026-09-17 GUI 수정: 기존 GUI의 `viewer.launch()`는 기본 표시 옵션으로 충돌 박스까지
보여줬지만 PNG 렌더러는 박스를 숨기고 있었다. 이제 `launch_passive()`의 카메라와
geom group을 PNG와 동일하게 설정한다. 모델 파일이 다른 문제가 아니었다.

## 실제 물리 시뮬레이션: MPC 트롯

이번에 이미 의존성을 설치했다. 다른 PC에 기존 환경만 있다면 다음 명령으로 갱신한다.

```bash
cd /home/bang/robost
conda env update -f environment.yml
conda activate rs02-mujoco
```

평지 트롯은 명령 속도 **0.50 m/s**, 최대 12초로 설정되어 있다.

```bash
python rs02_mpc.py --terrain flat
```

계단 트롯은 명령 속도 **0.24 m/s**, 최대 30초로 설정되어 있다.
계단은 **단높이 15 cm / 20 cm 두 버전**이며, 기본값은 15 cm다.
두 버전 모두 **5단 오르막 → 상단 평지 1 m → 5단 내리막**이다.
**디딤판 깊이 30 cm, 폭 160 cm**이며 최상단 높이는 15 cm 버전 75 cm,
20 cm 버전 100 cm다. 단수는 수직 단차 기준이다. 오르막의 다섯 번째 단은 상단
평지로 연결되고 내리막의 다섯 번째 단은 바닥으로 연결된다.
일반 디딤판 면적은 0.96→0.48 m²로 절반이며, 상단 평탄부 깊이도 2→1 m로 줄였다.

```bash
python rs02_mpc.py --terrain stairs --step-height-cm 15
python rs02_mpc.py --terrain stairs --step-height-cm 20
```

### 매 실행마다 속도 변경하기

`--speed` 뒤 숫자가 전진 속도 명령이며 단위는 **m/s**다. 코드 수정 없이 변경한다.
생략하면 평지 0.50 m/s, 계단 0.24 m/s다. 0 이상의 유한한 숫자만 허용한다.
실제 달성 속도는 균형과 접촉 상태에 따라 다르며 명령값과 같다고 보장하지 않는다.
낮은 속도로 전체 코스를 시험하려면 최대 실행 시간도 `--duration`으로 늘린다.

```bash
# 15 cm 계단: 0.10 m/s, 최대 90초
python rs02_mpc.py --terrain stairs --step-height-cm 15 --speed 0.10 --duration 90

# 20 cm 계단: 0.20 m/s, 최대 60초
python rs02_mpc.py --terrain stairs --step-height-cm 20 --speed 0.20 --duration 60

# 평지도 동일한 속도 옵션 사용
python rs02_mpc.py --terrain flat --speed 0.30
```

위 명령은 설정 예시이지 보행 성공이 검증된 속도 조합은 아니다.
새 코스의 기본 속도 0.24 m/s 재시험은 15 cm에서 7.726초, 20 cm에서 6.252초에
자세 각도 기준으로 중단됐다. 오르막·상단 평지·내리막 전체 완주는 아직 미통과다.

앞선 시험에서 평지는 약 4.23초, 이전 22.5 cm 계단은 약 21.35초에
몸통 높이 정지 기준에 도달했다. 15 / 20 cm 버전의 시험 결과는 각각 별도 JSON에 저장한다.
`stop_reason`은 중단 원인, `success`는 시험 통과 여부다. 계단의 높이 기준은 발 아래
지형 높이의 평균을 사용하므로 중단을 곧바로 실제 전복으로 해석하면 안 된다.
이전 성공 영상과 보고서는 `output/history/2026-09-17/`에 보관했다.

종료 후 창을 최종 자세로 유지하려면 `--hold`를 붙인다. 이때 계산은 완료되어 화면만
유지된다. 다시 보려면 창을 닫고 명령을 재실행한다.

```bash
python rs02_mpc.py --terrain stairs --hold
```

창 없이 검증하고 MP4까지 저장하려면:

```bash
python rs02_mpc.py --terrain flat --headless --video
python rs02_mpc.py --terrain stairs --step-height-cm 15 --headless --video
python rs02_mpc.py --terrain stairs --step-height-cm 20 --headless --video
```

결과는 `output/mpc/`에 저장된다. 같은 지형·높이를 다시 실행하면 해당 버전의 JSON 결과와
장면 XML을 갱신하며, `--video`를 붙이면 MP4와 PNG도 갱신한다.

- 파일 접두사는 `flat`, `stairs_15cm_course`, `stairs_20cm_course`로 구분된다.
- 각 접두사 뒤 `_mpc.mp4`: 보행 영상
- `_report.json`: 성공 여부, 자세·발 위치 로그, 토크, QP 결과
- `_scene.xml`: 지형과 모터를 포함한 MuJoCo 장면
- `_start.png`, `_end.png`: 시작/종료 장면
- 기존 `stairs_*` 파일은 이전 22.5 cm 시험 결과이며 새 두 버전이 덮어쓰지 않는다.

기존 `stairs_15cm_*`, `stairs_20cm_*` 중 `_course`가 없는 파일은 이전 3단 결과다.
같은 높이의 코스를 다른 속도로 재실행하면 해당 코스 결과가 갱신된다.

장면 XML만 단독 실행하면 MPC 제어기가 연결되지 않는다. 보행은 반드시
`rs02_mpc.py`로 실행한다. 상세 설정과 검증 수치는 `MPC_트롯_검증_보고서.md`를 참고한다.
