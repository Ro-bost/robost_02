> 보존 보고서: 정리 이전의 파일명·실행 명령·진행상황이 포함됩니다. 현재 기준은 [진행상황](../user/STATUS.md), 실행은 [실행 안내](../user/RUN.md)를 먼저 보세요.

# ROBOST / RS02 4족 로봇 파일 설명 보고서

작성일: 2026-09-17
검토 위치: `/home/bang/robost`

## 1. 한 줄 결론

이 폴더는 RS02 모터 12개를 쓰는 4족 로봇의 **최종 CAD, 설계 기록, 구조 해석,
Isaac Sim용 URDF, MuJoCo 검증 영상, 모터 자료, 생성 스크립트**를 모은 전달용
사본이다. 바로 시뮬레이션에 쓸 핵심은
`04_URDF_IsaacSim/rs02_quadruped/rs02_quadruped.urdf`와 `meshes/`이고,
설계 원본은 `01_CAD_최종모델/ROBOT_LR_r11.f3d`이다.

현재 Linux PC에는 `rs02-mujoco` conda 환경을 만들었고, URDF를 MuJoCo 3.13.0에
실제로 로드해 visual STL 13개, 회전 관절 12개, 총질량 16.9406 kg을 확인했다.
기본 표시 자세는 사용자가 요청한 PNG와 동일한 서기 자세다. CAD 저장 자세는
`--pose cad`로 별도 선택한다.

## 2. 검토 범위와 방법

- `00_먼저_읽어주세요.html`을 가장 먼저 끝까지 읽었다.
- 폴더 안의 실제 파일 212개, 총 376 MB를 이름·형식·크기·해시 기준으로 조사했다.
- Markdown, HTML, JSON, URDF, XML, Python, MATLAB 소스는 구조와 핵심 내용을 읽었다.
- 두 F3D의 내부 목록과 미리보기를 확인하고 STEP 헤더를 검사했다.
- URDF ZIP은 압축 무결성 검사를 통과했고, 두 풀린 패키지가 서로 완전히 동일함을
  디렉터리 비교로 확인했다.
- STL은 MuJoCo가 실제로 13개를 로드하는 것으로 확인했다.
- 주요 설계·응력·서기 자세 이미지를 직접 확인했다.
- RS02 공식 PDF는 80페이지이며, 전체 텍스트와 목차를 확인하고 사양·토크 곡선·과부하
  시간·드라이버 페이지를 렌더링해 읽었다.
- MP4는 파일 구성, 크기, 중복 여부와 문서에 기록된 장면 설명을 대조했다. 영상 전체를
  프레임 단위로 재평가한 것은 아니다.

## 3. `00_먼저_읽어주세요`의 핵심 지시

이 폴더는 `C:\fusion 360\` 원본의 **사본**이다. 기존 파일을 수정할 때는 Windows
원본에서 작업하고 필요한 결과만 다시 복사하라는 지시가 있다. 따라서 이번 작업에서는
패키지 내부 원본을 바꾸지 않고, 새 보고서·환경 파일·MuJoCo 실행기만
`/home/bang/robost` 최상위에 추가했다.

처음 볼 때 권장 순서는 다음과 같다.

1. `02_문서/QUADRUPED_DESIGN_REPORT.html` - 전체 설계와 해석을 시각적으로 이해
2. `05_영상/MuJoCo_시뮬레이션/mujoco_gaits_all.mp4` - 보행 결과 확인
3. `01_CAD_최종모델/ROBOT_LR_r11.f3d` - 최종 Fusion 설계
4. `04_URDF_IsaacSim/rs02_quadruped_urdf_2026-09-15.zip` - RL 담당자 전달본

## 4. 폴더 지도

| 위치 | 파일 수 | 실제 크기 | 역할 |
|---|---:|---:|---|
| `00_먼저_읽어주세요.html` | 1 | 16 KB | 전체 안내와 최종 사양 |
| `01_CAD_최종모델` | 99 | 263 MB | 최종 로봇·단일 다리 F3D/STEP와 F3D 내부 전개본 |
| `02_문서` | 3 | 6.6 MB | 종합 보고서, 상세 작업 일지, 공차 사양 |
| `03_해석결과` | 18 | 1.0 MB | 응력·토크·베어링·몸통 해석 그림과 JSON |
| `04_URDF_IsaacSim` | 53 | 64 MB | URDF 패키지 2벌과 전달용 ZIP |
| `05_영상` | 20 | 40 MB | Fusion 동작 영상과 MuJoCo 보행 영상 |
| `06_모터자료` | 2 | 2.1 MB | RS02 요약과 공식 80페이지 매뉴얼 |
| `07_스크립트` | 16 | 188 KB | URDF·문서·해석·영상 재생성 스크립트 사본 |

안내 HTML에 적힌 개수·크기보다 실제 폴더가 큰 이유는 F3D 내부 전개 폴더와 동일한
URDF 패키지의 복제본이 함께 있기 때문이다.

## 5. 설계 사양 요약

| 항목 | 현재 자료의 값 |
|---|---|
| 구조 | 4족, 다리당 3축, 총 12개 구동 관절 |
| 링크 길이 | 허벅지 220 mm, 종아리 220 mm |
| 몸통 | 311 × 230 × 130 mm |
| 고관절 위치 | x ±260 mm, y ±57.8~60 mm 부근 |
| 모터 | RobStride RS02 × 12 |
| 모터 정격/최대 | 연속 6 N·m, 최대 17 N·m |
| 모터 전압/속도 | 정격 48 V, 동작 24~60 V, 무부하 410 rpm |
| 재질 | 몸통 6061-T6, 허벅지·커버 7075-T6, 일부 Ti/PC/TPU |
| CAD 기구 질량 | 11.9402 kg |
| URDF 총질량 | 16.9406 kg = 기구 약 11.94 kg + 가정 payload 5 kg |
| 배터리 계획 | 12S2P 21700, 43.2 V, 9 Ah, 약 389 Wh |
| 권장 서기 높이 | 고관절-발 중심 340~360 mm |
| URDF 기본 base 높이 | 0.36661 m, 발 반지름 포함 |

안내 문서의 “전체 약 16.7 kg”은 개략치이고, 현재 URDF 검증값은 16.9406 kg이다.
전자장비 배치와 실제 배터리 질량이 확정되면 payload를 반드시 갱신해야 한다.

## 6. 폴더별 파일 설명

### 6.1 `01_CAD_최종모델`

- `ROBOT_LR_r11.f3d`: 최신 전체 로봇 설계. 조인트와 재질 정보가 있는 주 편집본이다.
- `ROBOT_LR_r11.step`: 다른 CAD 또는 가공 업체 전달용. 형상은 전달되지만 Fusion 조인트
  이력은 없다.
- `SINGLE_LEG_LR_r19.f3d`: 최종 단일 다리. r16 형상에 재질을 정리하고 허벅지·커버를
  7075-T6로 바꾼 버전이다.
- `SINGLE_LEG_LR_r19.step`: 단일 다리 중립 CAD 교환본이다.
- `SAVEPOINTS.md`: r01~r19, 로봇 r05~r11의 변경 이력과 B1~B4 분기 결정을 기록한다.
- 같은 이름의 디렉터리 두 개는 F3D 아카이브의 내부 데이터와 미리보기 전개본이다.
  `BREP.*`, `Fusion*Stream.dat`, `ProteinAsset.*`는 Fusion 내부 형식이므로 직접 편집할
  대상이 아니다.

최종 선택은 전체 로봇 r11, 단일 다리 r19다. 형상을 다시 크게 바꿀 때는 조인트가 들어간
r11에서 무리하게 편집하기보다 작업 기록이 권한 r08 계열로 돌아가 형상을 바꾸고 조인트를
다시 넣는 흐름이 안전하다.

### 6.2 `02_문서`

- `QUADRUPED_DESIGN_REPORT.html`: 20개 PNG와 1개 MP4가 내부에 포함된 자체 완결형
  시각 보고서다. 로봇 개요, 무릎 링크, 경량화, 가동범위, 응력, 공차, 보행을 한 번에
  설명한다.
- `LEG_REDESIGN_PLAN_2026-09-14.md`: 실제 설계 의사결정의 원장에 가깝다. 8-1~8-27에
  실패한 시도, 롤백, 해석 조건, 교훈까지 남아 있어 “왜 이렇게 설계했는가”를 확인할 때
  가장 중요하다.
- `FIT_TOLERANCE_SPEC_2026-09-15.md`: 베어링·핀의 R6/M7/h6/H7 등 끼워맞춤,
  표면조도, 경도, 조립법을 정리한 제작 도면용 문서다.

### 6.3 `03_해석결과`

`그림/`은 응력 분포와 설계 비교를 사람이 빠르게 보는 자료이고, `수치/`의 JSON은
보고서 그래프와 결론의 원자료다.

주요 결론은 다음과 같다.

- 허벅지 r16 형상은 6061 기준 좌우 복합 하중에서 국부 안전율이 약 1.33이었다.
  형상 필렛 r17/r18은 최대 응력 위치만 옮겨 미채택했고, 허벅지·커버를 7075-T6로
  바꿔 최종 안전율을 약 2.4로 높였다.
- 몸통 비틀림 FEA는 27.267 N·m에서 비틀림 0.0734°, 대표 최대 응력 21.38 MPa,
  안전율 약 12.9다. 뚜껑 없는 보수적 조건이다.
- RNAF6138N 로드 베어링은 대표 복합 하중에서 정적 안전율 2.08,
  모터 한계 전체에서는 1.53이다. 주문 전 제조사 표로 재확인하라는 주의가 있다.
- HK0808 두 개는 대표 복합 하중에서 정적 안전율 4.08이다.
- 무릎은 낮게 앉을수록 연속 토크·발열 여유가 나빠진다. 340~360 mm가 권장 범위다.

### 6.4 `04_URDF_IsaacSim`

실질적인 시뮬레이션 전달물이다.

- `rs02_quadruped/`: 바로 사용할 풀린 패키지
- `rs02_quadruped_urdf_2026-09-15/rs02_quadruped/`: 위 폴더와 완전히 동일한 복제본
- `rs02_quadruped_urdf_2026-09-15.zip`: RL 담당자에게 전달할 무결한 ZIP

패키지 구성:

- `rs02_quadruped.urdf`: 링크 18개, 조인트 17개 중 revolute 12개와 fixed 5개
- `meshes/*.stl`: base 1개와 다리별 hip/thigh/calf 12개, 총 visual mesh 13개
- collision: 학습 속도를 위해 box와 sphere 기본 형상 사용
- `urdf_summary.json`: 질량, 관성, 조인트, 기본 자세를 만드는 수치 원본
- `validation_report.json`: 오류 0, 경고 0, 발 지면 오차 약 0.002 mm,
  전체 COM이 지지 사각형 안임을 기록
- `docs/joint_table.md`: 축, 부모/자식, 한계, effort, velocity, 부호 정의
- `docs/standing_pose.*`: 12개 관절의 기본 서기 각도
- `docs/mujoco_check_model.xml`: 기존 MuJoCo 검증용 MJCF. 단, meshdir가
  `C:/fusion 360/...`로 고정돼 현재 Linux에서는 그대로 로드할 수 없다.

관절별 핵심값:

- hip roll: x축, 최대 토크 17 N·m, 속도 42.94 rad/s
- thigh pitch: y축, 최대 토크 17 N·m, 속도 42.94 rad/s
- calf/knee: y축, 4절 링크비 1.48을 반영해 25.2 N·m, 29.01 rad/s
- 기본 서기: hip 0, thigh 0.68022 rad, calf -1.3664 rad

중요한 모델링 단순화:

- 실제 무릎은 4절 링크지만 URDF의 닫힌 링크 제약 때문에 직결 회전 관절로 등가화했다.
- 크랭크와 로드 질량은 CAD 기준 자세에서 thigh 링크에 합쳤다.
- payload 5 kg은 200×150×80 mm 균일 박스로 가정했다.
- 회전자 관성은 공식 자료에 없어 armature를 추정해야 한다.
- 케이블은 질량 모델에서 제외했다.

### 6.5 `05_영상`

- `mujoco_gaits_all.mp4`: walk, 제자리 trot, 0.3/0.6/1.0 m/s trot,
  1.4/2.0 m/s flying trot의 7장면 연결본이다.
- `보행_장면별/`: 위 7장면의 개별 파일이다.
- `mujoco_stand_squat.mp4`: 낙하 후 서기와 스쿼트 검증이다.
- `mujoco_joint_sweep.mp4`: hip/thigh/calf 방향과 추종 검증이다.
- `Fusion_녹화/`: 전체 로봇 트롯 3구도와 앞 오른쪽 다리 상세 6구도다.

기록상 1.4 m/s 명령까지는 안정적이고 실제 속도는 약 1.23 m/s였다. 2.0 m/s는
넘어지지는 않았지만 속도 추종에 실패하고 옆으로 약 1.6 m 틀어졌다. 이는 RL 결과가
아니라 단순 스크립트 제어기의 결과다.

### 6.6 `06_모터자료`

- `RS02_motor_datasheet.md`: URDF에 필요한 사양만 뽑은 요약본이다.
- `RS02_User_Manual_260713.pdf`: 80페이지 공식 매뉴얼이다. 기계 치수, 전기 사양,
  드라이버 연결, PC 설정, CAN/CANopen/MIT 프로토콜, 샘플 코드와 버전 이력이 있다.

매뉴얼에서 특히 주의할 점:

- 17 N·m는 연속 토크가 아니라 피크다.
- 6 N·m 연속 정격도 260×280 mm 방열판과 100 rpm 시험 조건이 붙는다.
- 표에 따르면 회전 중 17 N·m 운전 허용 시간은 약 10초이고, 정지 상태에서는 약 6초다.
- 제어 모드는 구동 중 바로 바꾸지 말고 정지 명령 후 바꾸라고 명시한다.
- 드라이버 CAN 속도는 1 Mbps, 엔코더는 14-bit 절대형이다.

### 6.7 `07_스크립트`

URDF 생성 흐름은 `build_urdf.py → validate_urdf.py → make_docs.py`, 동작 검증은
`sim_mujoco.py`, `sim_joint_sweep.py`, `sim_gaits.py`다. 해석 쪽에는 하중·무릎·베어링
계산, STL 용접, MATLAB FEA, HTML 보고서 생성기가 있다.

이 디렉터리는 **완전한 재현 환경이 아니라 재생성용 사본**이다.

- 대부분 `C:\fusion 360\outputs\...` 절대경로를 사용한다.
- `build_urdf.py`가 요구하는 `r11_urdf_source.json`, `_body_stl/manifest.json`과 원시
  body STL은 이 전달 폴더에 없다.
- 일부 해석 스크립트가 요구하는 `voxfea` 모듈, 중간 STL/JSON, MATLAB PDE Toolbox도
  별도로 필요하다.
- `make_gait_videos.py`는 Windows용 ffmpeg 경로와 Malgun Gothic 글꼴 경로가 고정돼 있다.

따라서 현재 사본만으로 “CAD부터 URDF까지 완전 재생성”할 수는 없지만, 이미 만들어진
URDF의 로드·검증·시뮬레이션은 가능하다.

## 7. 이번에 만든 MuJoCo 환경과 검증 결과

추가된 파일:

- `/home/bang/robost/environment.yml`
- `/home/bang/robost/run_rs02_mujoco.py`
- `/home/bang/robost/MUJOCO_URDF_실행_가이드.md`
- `/home/bang/robost/rs02_mujoco_snapshot.png`

생성한 환경:

```text
conda env: rs02-mujoco
Python:    3.11.16
MuJoCo:   3.13.0
NumPy:    2.4.6
Pillow:   12.3.0
```

실제 확인값:

```text
body=19       # world 포함
joint=12      # 회전 관절
geom=30       # visual + collision
mesh=13       # STL visual
mass=16.9406 kg
```

기존 MJCF의 Windows 경로를 직접 고치는 대신, 실행기가 원본 URDF에 임시 MuJoCo compiler
설정을 넣는다. `discardvisual=false`, `fusestatic=false`를 적용해 visual mesh, payload,
foot fixed link가 사라지지 않게 한다. 임시 파일은 모델을 읽은 직후 삭제한다.

CAD와 URDF 외형 대조 결과, STEP 204개 솔리드의 체적은 4,892.517 cm³이고 URDF
13개 visual STL의 체적 합은 4,891.670 cm³로 약 0.017% 차이다. CAD 저장 자세로
조립한 전체 바운딩박스도 STEP 813.223×348.140×305.000 mm, URDF
813.223×348.144×305.000 mm로 최대 차이 0.004 mm다. 외형이 다르게 보이는 주된
이유에는 자세·재질 차이가 있다. 이후 사용자의 GUI 캡처에서 충돌 박스가 외형 위에
겹쳐 표시되는 별도 오류를 발견했고 수정했다. 체적·바운딩박스의 일치는 모델 출처를
뒷받침하지만 모든 부품의 국부 형상이나 동역학 일치까지 증명하는 것은 아니다.

실행 명령:

```bash
cd /home/bang/robost
conda activate rs02-mujoco
python run_rs02_mujoco.py
```

기본값은 서기 자세이며, CAD 저장 자세는 `python run_rs02_mujoco.py --pose cad`로 연다.

2026-09-17 추가: `rs02_mpc.py`로 MPC 기반 평지 트롯과 단높이 0.15 m 계단 3개
오르기를 구현했다. `MPC_트롯_검증_보고서.md`에 결과와 한계를 정리했다.
`output/mpc/`의 XML·JSON·MP4는 이번에 새로 실행해 만든 결과다.

2026-09-21 갱신: 계단 깊이 30 cm/높이 22.5 cm, 평지·계단 명령 속도 0.50/0.24 m/s로
변경했다. 새 조건에서 두 시험 모두 몸통 높이 기준으로 중단되어 통과하지 못했다.
현재 `output/mpc/`는 새 결과이며, 이전 성공 자료는 `output/history/2026-09-17/`에 있다.

검증 전용 명령:

```bash
conda run -n rs02-mujoco python run_rs02_mujoco.py --check
```

## 8. 앞으로의 우선순위

1. 전자장비·배터리 실제 배치와 질량을 확정해 URDF payload의 질량, COM, 관성을 갱신한다.
2. 실제 RS02 토크-속도 곡선, 열 제한, 출력축 환산 회전자 관성을 시뮬레이터에 반영한다.
3. 4절 링크 등가화가 실제 기구 토크·속도에 주는 오차를 별도 transmission 또는
   제어 매핑으로 보정한다.
4. 부품별 STEP 분리, 2D 제작 도면, 공차 사양, BOM을 완성한다.
5. Isaac Sim/Isaac Lab에서 접촉·PD gain·armature를 보정한 뒤 RL 학습을 진행한다.
6. 제작 전 RNAF6138N 끼워맞춤과 정격을 최신 제조사 표로 재확인한다.

## 9. 지금 기억하면 되는 파일 6개

| 목적 | 파일 |
|---|---|
| 전체 설계를 빠르게 이해 | `02_문서/QUADRUPED_DESIGN_REPORT.html` |
| 설계 이유와 시행착오 확인 | `02_문서/LEG_REDESIGN_PLAN_2026-09-14.md` |
| 최종 CAD 편집 | `01_CAD_최종모델/ROBOT_LR_r11.f3d` |
| 제작 공차 | `02_문서/FIT_TOLERANCE_SPEC_2026-09-15.md` |
| 시뮬레이션 모델 | `04_URDF_IsaacSim/rs02_quadruped/rs02_quadruped.urdf` |
| 다른 사람에게 전달 | `04_URDF_IsaacSim/rs02_quadruped_urdf_2026-09-15.zip` |
