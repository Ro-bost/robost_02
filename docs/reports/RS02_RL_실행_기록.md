> 보존 보고서: 정리 이전의 파일명·실행 명령·진행상황이 포함됩니다. 현재 기준은 [진행상황](../user/STATUS.md), 실행은 [실행 안내](../user/RUN.md)를 먼저 보세요.

# RS02 공개 동적 보행 학습 환경 이식 기록

> 최신: 후속 rhythm 정책의 선택 후보로15cm/20cm 각각 고정 반복6/6 완주했다.
> 학습 종료 후 사용자용 launcher 추가 재현도15cm24.66초/20cm25.34초 통과했다.
> 이전 후보3/4 등의 실패 기록은 아래 이력으로 보존하며 최신 성공률과 섞지 않는다.
> 실행 방법은 `RS02_계단_실행방법.md`, 개선 과정은 `RS02_계단_개선_실험.md`를 참고한다.
> 아래 계단 실패 기록은 이전 실험 이력이다. 코스 완주와 하드웨어 안전 통과는 구분한다.

## 환경

- conda `rs02-rl` / Python 3.11.16
- mjlab 1.6.0, 커밋 `27577db821fe321c819072a851bdda234b89f32d`
- MuJoCo 3.11.0 + MuJoCo Warp 3.11.0 / Warp 1.17.0
- PyTorch 2.14.0 / RSL-RL 5.5.1 / GPU RTX5000 Ada Laptop16GB
- 기존 `rs02-mujoco`의 MuJoCo3.13.0 및 기존 실행 파일은 보존했다.
- 설치된 패키지 버전 스냅샷: `requirements-rs02-rl.lock`.
  conda의 로컬 빌드 경로인 pip 항목은 제외하고 mjlab 경로는 프로젝트 상대 경로로 바꾸었다.
  다른 OS/GPU에서도 설치·실행됨을 보증하는 범용 잠금 파일은 아니다.

새 Linux 환경에서 재현할 때는 저장소 루트에서 아래처럼 실행한다(현재 환경을 덮어쓸 필요 없음).

```bash
conda create -n rs02-rl-repro python=3.11.16 pip -y
conda activate rs02-rl-repro
pip install uv==0.12.17
uv pip install --python "$CONDA_PREFIX/bin/python" -r requirements-rs02-rl.lock \
  --extra-index-url https://pypi.nvidia.com/ --index-strategy unsafe-best-match
```

재설치 명령은 안내이며 새 환경 전체 재설치까지 반복 시험하지는 않았다.

버전이 다르므로 CPU MuJoCo3.13.0에서의 재검증 전에는 학습 시뮬레이터 밖에서도 같은 성능이라고
말할 수 없다. GPU 물리도 실물 검증을 대체하지 않는다.

## 공개 코드 재사용 범위

`third_party/mjlab` 원본을 수정하지 않고 `rs02_rl.py`에서 Go1 평지 속도 추종 환경 팩토리와
PPO 네트워크/학습 설정을 호출한다. 관측(몸통 속도·각속도·중력방향·관절 상태·이전 입력·속도 명령),
보상, 접촉 센서, 종료 관리, RSL-RL 학습 루프를 재사용한다.
RS02 관절 이름·질량·관성·가동범위·토크 상한·PD·입력 크기로 교체한다.
이전 `Crawl`이나 `Swing`은 import하지 않는다.
정책은50Hz로12개 관절 목표를 동시에 출력하고,500Hz 물리의 토크 제한 PD가 이를 실행한다.
기존처럼 ‘한 다리 착지를 기다려 다음 다리로’ 넘어가는 프로그램이 아니다.

## 계획 → 검증 → 수정 → 실행

1. 통과 기준을 `RS02_동적보행_계획.md`에 먼저 정의했다.
2. `test_rs02_rl.py`: 원본 대비 몸체 질량/관성·관절 범위·토크 제한·충돌 마스크·외부지지 없음 확인.
3. GPU16환경 정지시험: 질량16.940606kg, 관절12, actor관측48/critic72,2초 후 높이평균0.35185m,
   관측 유한값, 종료0개. 이것은 환경 정상 여부 검사이지 보행 시험이 아니다.
4. `flat_v1`: 1,024환경 PPO1차 학습. 중간100/200iteration 정책은10회 평가 전부 약0.22초에
   다리를 접고 지면에 주저앉아 실패했다. 보상 상승을 보행 성공으로 해석하지 않는다.
5. `flat_v2`: air-time보상을 원본 Go1 기본값0으로 복원, 종료 페널티-100(시간스케일0.02적용),
   평지 몸통 높이0.35m 보상 추가, 초기 탐색 표준편차1.0→0.4로 감소 후 재학습.
   기존 질량·토크·접촉 형상을 바꿔 실패를 숨기지 않았다.

## 실행 명령

```bash
cd /home/bang/robost
conda activate rs02-rl

# 설치를 다시 할 경우(기존 환경은 재설치할 필요 없음)
pip install uv
uv pip install --python "$CONDA_PREFIX/bin/python" -e ./third_party/mjlab \
  --extra-index-url https://pypi.nvidia.com/ --index-strategy unsafe-best-match

# 검증
uv run --no-project --python "$CONDA_PREFIX/bin/python" -m unittest test_rs02_rl
uv run --no-project --python "$CONDA_PREFIX/bin/python" rs02_rl.py smoke \
  --num-envs 16 --output output/rl/smoke

# 새 학습: output은 기존 학습 폴더와 다른 이름을 사용
uv run --no-project --python "$CONDA_PREFIX/bin/python" rs02_rl.py train \
  --num-envs 1024 --iterations 500 --output output/rl/my_run

# 체크포인트 평가: --speed만 변경하여 같은 조건에서 비교
uv run --no-project --python "$CONDA_PREFIX/bin/python" rs02_rl.py evaluate \
  --checkpoint output/rl/my_run/model_499.pt --num-envs 10 \
  --speed 0.25 --duration 30 --video --output output/rl/my_eval025
```

실제 체크포인트 파일명은 학습 폴더에서 확인한다. 평가는 50Hz수치와25fps1배속 영상을 저장한다.
첫 실패를 영구 기록한다. 이후 자동 초기화로 계속 표시되는 동작은 성공으로 세지 않는다.
실물 열/토크속도 검증은 별도 단계다.
외부로그 업로드 없이 로컬 TensorBoard에만 기록한다.

## 2026-09-22 확인된 평가 결과

검증한 정책은 `output/rl/flat_v2/model_200.pt`이다. 마지막 체크포인트가 자동으로 최선인 것은 아니다.
각 시험은 seed42에서 초기 위치·각도가 약간 다른 10개 환경, 30초이다.
독립된 학습 시드 10개나 실물 10대 시험을 뜻하지 않는다.

| 시험 | 수치 기준 통과 | 측정 결과 | 결과 폴더 |
|---|---:|---|---|
| 평지 0.25m/s | 10/10 | 평균 속도 0.25193m/s | `output/rl/eval_v2_200_025` |
| 평지 0.50m/s | 10/10 | 평균 속도 0.50942m/s | `output/rl/eval_v2_200_050` |
| 정지→0.5m/s→정지→재출발 | 10/10 | 정지 안정 구간 최대 속도 0.001142m/s | `output/rl/eval_v2_200_stopgo` |
| 평지 정책→15cm 계단 | 실패 | 첫 단에서 막힘, 최대 몸통 x≈0.548m | `output/rl/transfer_stairs15` |
| 평지 정책→20cm 계단 | 실패 | 첫 단에서 막힘, 최대 몸통 x≈0.548m | `output/rl/transfer_stairs20` |

계단 시작은 x=0.75m이며 앞발은 몸통보다 앞에 있다. 넘어지지 않았지만 전진하지 못하므로 실패다.
계단 시험 종아리 접촉 시간 비율은 약98.5%였다. 접촉 형상을 삭제하여 통과시키지 않는다.
각 폴더의 `evaluation_1x.mp4`는 실제 물리 궤적을 1배속으로 재생한 영상이다.
수치 통과는 영상의 보행 품질 판단이나 하드웨어 인증을 대신하지 않는다.

### 검증한 평지 정책 직접 실행

```bash
cd /home/bang/robost
conda activate rs02-rl
SPEED=0.50
uv run --no-project --python "$CONDA_PREFIX/bin/python" rs02_rl.py play \
  --checkpoint output/rl/flat_v2/model_200.pt --speed "$SPEED"
```

위 GUI 실행은0.5m/s·5초 제한으로 실행/정상 종료까지 확인했다. 결과표의 정량 수치는 별도 `evaluate` 시험이다.
GUI는 실패 시 초기화하므로 계속 움직인다는 이유로 통과로 판단하면 안 된다.

### 계단용 신규 학습

`rs02_rl_stairs.py`는 별도 정책이다. 평지 관측 끝에 187개 지형 높이를 추가하고,
첫 신경망 층의 추가 입력 가중치를 0으로 확장하여 기존 보행을 초기값으로 보존한다.
학습 전용 2→20cm 난이도 지형에서도 발판0.3m, 폭1.6m, 5단 상승→1m 평지→5단 하강을 유지한다.
몸통 높이 보상은 절대 높이가 아닌 몸통 아래 지형과의 상대 높이로 수정했다.
기존 평지 정책/원본 mjlab은 수정하지 않는다. 이상적인 지형 센서를 사용하므로 인지 오차는 미검증이다.

```bash
uv run --no-project --python "$CONDA_PREFIX/bin/python" -m unittest test_rs02_rl test_rs02_rl_stairs
uv run --no-project --python "$CONDA_PREFIX/bin/python" rs02_rl_stairs.py train \
  --checkpoint output/rl/flat_v2/model_200.pt --num-envs 512 --iterations 500 \
  --output output/rl/my_stairs_run
uv run --no-project --python "$CONDA_PREFIX/bin/python" rs02_rl_stairs.py evaluate \
  --checkpoint output/rl/my_stairs_run/model_499.pt --num-envs 1 --stairs-cm 15 \
  --speed 0.25 --duration 40 --video --output output/rl/my_stairs15_eval
```

20cm은 `--stairs-cm 20`, 속도는 `--speed`로 변경한다. 출력 폴더는 새 이름을 사용한다.
500회 학습 자체가 계단 통과를 뜻하지 않는다. 고정 코스 완주·낙상·접촉·토크 평가를 별도로 확인한다.

### 계단 실험의 수정 이력

- `stairs_smoke_v1`: 32환경×2 iteration 실행 완료. 학습 루프가 동작한다는 검사다.
- `stairs_v1`: 512환경×500 iteration 완료. 중간200정책은15cm에서 멈췄고,
  최종499정책도 별도4cm 코스에서 멈췄다. 낮은 계단의 기초 보행도 아직 검증되지 않았다.
- `stairs_v2`: 상승/하강의 수직 속도를 수평 속도 오차에 포함하지 않도록 수정했다.
  옆으로 코스를 벗어나는 경우는 실패로, 전방 끝에 도달한 경우는 정상 종료로 분리했다.
  수평 속도 추종 가중치2→4, 코스 방향/중심 이탈 벌점 추가, 초기 탐색 표준편차0.30으로 재설정했다.
  종아리 충돌을 없애거나 토크를 늘리지 않았다.
- 각 학습 폴더의 `adapter_source.py`와 `training.log`가 해당 실행의 설정/기록이다.
  현재 파일이 발전하므로 이전 실험 재현에는 당시 소스 스냅샷도 확인한다.
- 새 평가에는 `evaluator_source.py`, `policy_adapter_source.py`와 해시도 저장한다.
  평가 도중 파일을 수정해도 시작 당시 소스가 기록되도록 했다.
- `stairs_v2/model_200.pt`의4cm 시험은 몸통x=2.878m까지 전진했으나 완주하지 못했다.
  관절 범위 최소 여유가-0.02425rad로 음수여서 하드웨어 가능성 근거로 채택할 수 없다.
- 이에 현재 계단 어댑터는 관절 목표를 원본 범위보다0.03rad 안쪽으로 제한하고,
  관절 한계 벌점도 강화했다. 이미 진행 중이던 `stairs_v2` 학습에는 소급 적용되지 않는다.
  이후 현재 파일로 하는 평가/새 학습에는 적용된다. 목표 제한은 관성에 의한 실제 각도 초과를
  원천 보장하지 않으므로 실제 각도도 계속 검사한다.
- `stairs_v2/model_500.pt` + 목표 제한의15cm 시험은 최대x=0.346m에서 멈췄다.
  관절 최소 여유0.0815rad, 종아리 접촉0이지만 **계단 보행 실패**다.

### GUI 종료 오류 진단

MuJoCo3.11 passive viewer를 닫고 즉시 프로세스가 종료되면 렌더링 스레드와
GLFW 종료가 겹쳐 `GLXBadDrawable`/segmentation fault가 발생했다.
로봇이 없는 구 모형으로도 재현했다. `play`에서는 GLFW를 사용하고, 생성한 렌더링 스레드의
종료를 기다리도록 수정했다. 이는 물리/정책 변경이 아닌 뷰어 생명주기 수정이다.
`--play-seconds 5`로 제한된 GUI 실행을 확인할 수 있다. 기본0은 창을 닫을 때까지 실행한다.
수정 후 실제 RS02 정책의0.5m/s·5초 실행은 종료코드0으로 완료했다.

### 목표 각도 제한을 포함한 추가 학습

```bash
uv run --no-project --python "$CONDA_PREFIX/bin/python" rs02_rl_stairs.py train \
  --checkpoint output/rl/stairs_v2/model_800.pt --num-envs 512 --iterations 500 \
  --initial-level 4 --exploration-std 0.20 --output output/rl/my_stairs_safe
```

이번 실행 폴더는 `output/rl/stairs_v3_safe`이다. 초기 난이도는2~10cm에서 무작위 배치하고,
같은5단 상승/평지/5단 하강 코스의 진행에 따라 최대20cm까지 올리는 설정이다.
목표 각도 제한은 학습 때부터 적용한다. 사용자 최종 코스는 계속15cm와20cm 두 가지다.
`stairs_v1`500회, `stairs_v2`1,000회, `stairs_v3_safe`500회 모두 실행 완료했다.
v3의 부모는 v2의800번째 체크포인트이며, 1,000회 최종 정책을 부모로 썼다는 뜻은 아니다.
학습 환경의 최고 난이도 도달 여부와 별개로 아래 독립 평가에서 통과 여부를 결정한다.

## 이번 실행의 최종 계단 평가 — 실패, 미완료

정책 `output/rl/stairs_v3_safe/model_499.pt`, 결정론적 추론, seed42, 명령0.25m/s,
30초, 코스별1회이다. 성공률을 추정할 수 있는 다회 평가가 아니며, 이1회부터 실패했다.

| 코스 | 완주 | 최대 몸통x | 최대 발 중심 높이 | 관절 범위 최소 여유 | 종아리 접촉 시간 비율 |
|---|---|---:|---:|---:|---:|
| 15cm | 실패 | 0.413m | 0.1615m | +0.00439rad | 0% |
| 20cm | 실패 | 0.373m | 0.1631m | +0.03793rad | 0% |

두 경우 모두 첫 단 앞에서 정체되었고,3초 이후 평균 전진 속도는 사실상0이었다.
넘어지지 않음·종아리 접촉0·관절 범위 준수만으로 보행 성공으로 판정하지 않는다.
발 반지름0.02661m를 고려하면 첫 단 위 착지에 필요한 발 중심 높이는 각각0.17661/0.22661m다.
측정 최대치가 이에 못 미쳤다. **발의 계단 진입 높이 부족이 확인된 실패 요인**이며,
그 높이를 확보하면 반드시 전체 코스를 통과한다는 뜻은 아니다.

- 15cm 영상/수치: `output/rl/stairs_v3_499_eval15/evaluation_1x.mp4`, `evaluation.json`
- 20cm 영상/수치: `output/rl/stairs_v3_499_eval20/evaluation_1x.mp4`, `evaluation.json`
- 회귀 테스트8개 통과. GUI 평지0.5m/s·5초 실행 정상 종료. 학습/평가 프로세스 완료.
- 평지용으로 계속 권장하는 검증 체크포인트는 `flat_v2/model_200.pt`다.
  계단 최신 정책을 평지 검증 정책의 대체품으로 사용하지 않는다.

### 남은 작업

지형 높이에 맞춘 전방 발 들기/착지 보상과 정체 처리를 검토하고, 학습 중 무작위 탐색을 제거한
정책이 낮은 계단부터 재현 가능하게 완주하는지를 우선 통과시켜야 한다.
그 뒤15/20cm 상승·평지·하강 전체를 여러 초기 조건과 속도에서 다시 평가한다.
현재 결과로 하드웨어 계단 보행에 문제가 없다고 결론 내릴 수 없다.
