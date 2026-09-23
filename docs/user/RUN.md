# 설치와 실행

모든 명령은 저장소 루트에서 실행한다. 기본 지원 방식은 Linux의 source checkout이다.
`python`은 선택한 conda 환경의 Python 3.11을 뜻한다. CPU MuJoCo와 GPU RL 환경은 버전이 달라 분리한다.

## 기존 환경에서 실행

```bash
conda activate rs02-rl
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m run_rs02_stairs --stairs-cm 15 --speed 0.25
python -m run_rs02_stairs --stairs-cm 20 --speed 0.25
```

15cm는 rhythm500, 20cm는 rhythm1000을 선택한다. GPU와 표시 가능한 데스크톱이 필요하다.
`--play-seconds 3`은 GUI를 3초 뒤 종료한다. GUI 리셋은 평가 통과가 아니다.

## 새 CPU 환경: 자세 확인·기존 MPC/crawl

```bash
conda env create -f config/environment.yml
conda activate rs02-mujoco
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/bootstrap_third_party.py --only Cheetah-Software eigen3
python -m run_rs02_mujoco --check
python -m run_rs02_mujoco
python -m build_cheetah_swing
python -m rs02_walk --terrain flat --speed 0.25 --duration 30 --headless
make test-core
```

C++ 빌드에는 `g++`이 필요하다. 생성물은 `build/libcheetah_swing.so`에 둔다.
conda 대신 별도 Python 환경에 `pip install -r config/requirements-mujoco.txt`로 같은 CPU 의존성을 설치할 수 있다.

## 새 GPU RL 환경

```bash
python scripts/bootstrap_third_party.py --only mjlab
conda create -n rs02-rl-repro python=3.11.16 pip -y
conda activate rs02-rl-repro
pip install uv==0.12.17
uv pip install --python "$CONDA_PREFIX/bin/python" -r config/requirements-rs02-rl.lock   --extra-index-url https://pypi.nvidia.com/ --index-strategy unsafe-best-match
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
make test-rl
```

기존 lock은 Linux/CUDA 13 환경 스냅샷이다. 다른 GPU/OS의 범용 lock이 아니며 새 환경 전체 재설치는 이번 정리에서 검증하지 않았다.
소스 묶음에는 checkpoint가 없으므로 별도 `robost-policies.tar.gz`를 받아 아래처럼 복원한다.
압축 안에 `robost/` 접두사가 있으므로 **저장소 루트**에서 한 단계 제거한다.

```bash
tar -xzf /path/to/robost-policies.tar.gz --strip-components=1
```

복원 위치는 `output/rl/stairs_rhythm_v1/model_500.pt`, `model_1000.pt`다.
선택과 SHA-256은 [정책 목록](../../config/policies.json)에 있다.
로컬 기존 프로젝트에는 이미 이 파일들이 있으므로 복원할 필요 없다.

## 점수와 1배속 영상 저장

```bash
python -m run_rs02_stairs --stairs-cm 20 --speed 0.25 --headless --seed 42 --duration 40
python -m evaluate_rs02_stairs_matrix   --adapter rhythm --checkpoint output/rl/stairs_rhythm_v1/model_1000.pt   --heights 20 --seeds 42 43 44 --repeats 2 --speed 0.25   --output output/rl/my_repeat20
```

반복 출력에는 새 이름을 사용한다. `evaluation.json`은 판정·측정,
`evaluation_1x.mp4`는 영상, `qpos_env0.npy`와 `scene.mjb`는 원본 상태·모델,
`source_snapshot/`은 평가 당시 코드다. 과거 결과는 보존한다.
커스텀 정책은 `--checkpoint 경로 --adapter gait|support|route|rhythm`을 함께 지정한다.
속도 변경은 재평가 대상이며 느리다고 항상 안정적인 것은 아니다.

```bash
python -m render_rs02_trial output/rl/새평가폴더
python -m plot_rs02_trial output/rl/새평가폴더
```

첫 명령은 해당 폴더에 영상이 없을 때 사용한다. 학습/평가 상세 옵션은
`python -m rs02_rl_rhythm --help`, `python -m rs02_rl --help`로 확인한다.

## 짧은 명령을 쓰고 싶다면

```bash
python -m pip install --no-deps --no-build-isolation -e .
rs02-stairs --stairs-cm 15 --speed 0.25
rs02-preview --check
```

editable 설치는 코드 경로와 명령만 연결한다. MuJoCo/GPU 의존성과 checkpoint는 위 절차로 준비한다.
일반 wheel에는 외부 모델/결과가 포함되지 않으므로 source checkout + editable 방식만 사용한다.
