# ROBOST · RS02 4족 로봇

RS02의 원본 질량·관절·구동 조건을 유지하면서 MuJoCo에서 보행 가능 범위와 실패 원인을 확인하는 연구 프로젝트입니다.
현재 계단 시연은 PPO 기반 rhythm 정책을 사용합니다. 15/20cm에서 서로 다른 체크포인트로 각각 6/6 완주한 기록이 있으며, 실물 안전성과 최대 계단 높이는 미확정입니다.

| 목적 | 먼저 볼 문서 |
|---|---|
| 내가 실행하기 | [설치·실행방법](docs/user/RUN.md) |
| 현재 어디까지 했는지 | [진행상황과 다음 과제](docs/user/STATUS.md) |
| 폴더와 코드 이해 | [프로젝트 구조](docs/user/STRUCTURE.md) |
| 선행연구·재사용 범위 | [선행연구 안내](docs/research/README.md) |
| 과거 실험·발표 자료 | [보고서 목록](docs/reports/README.md) |
| GitHub 배포 준비 | [배포 안내](docs/user/DEPLOY.md) |
| AI에게 작업 맡기기 | [AGENTS.md](AGENTS.md) → [AI 맥락](docs/ai/CONTEXT.md) |
| 이번 정리 내역 | [정리·검증 기록](docs/user/ORGANIZATION.md) |

이미 구성된 환경에서는 저장소 루트에서 실행합니다.

```bash
conda activate rs02-rl
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m run_rs02_stairs --stairs-cm 15 --speed 0.25
```

새로 받은 저장소에는 학습 가중치가 없습니다. [실행 안내](docs/user/RUN.md)의 가중치 복원 절차를 따르세요.
영상·학습 결과 약 9GB와 CAD 원본은 로컬에 보존하고 GitHub 소스에서는 제외합니다.

![RS02 원본 서기 자세](docs/assets/rs02_mujoco_standing_pose.png)
