# 선행연구와 실제 재사용 범위

기존 프로젝트 조사·구현 기록을 읽기 쉽게 모은 색인이다. 이번 정리에서 새로운 문헌 조사를 수행한 것은 아니다.

| 출처 | 프로젝트에서의 용도 | 재사용 범위와 제한 |
|---|---|---|
| [mjlab](https://github.com/mujocolab/mjlab) | 현재 GPU 학습·평가 기반 | Go1 속도 추종 MDP/PPO 구성을 RS02 모델에 이식. Go1 가중치를 그대로 쓰지 않음 |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | PPO 학습 구현 | mjlab runner를 통해 사용, 설치 버전은 RL lock에 기록 |
| [MIT Cheetah Software](https://github.com/mit-biomimetics/Cheetah-Software) | 과거 crawl의 발 스윙 | FootSwingTrajectory.cpp와 얇은 C ABI 연결만 사용. 전체 WBIC 이식 아님 |
| [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways) | 위상·접촉 보상 설계 참고 | 설계 참고이며 사전학습 정책을 직접 사용한 것이 아님 |
| [CHAMP](https://github.com/chvmp/champ) | 기존 후보 조사 | 현재 실행 의존성 아님 |
| [OCS2 ROS2 예제](https://github.com/HexiangZhou/Quadruped-Control-OCS2-ROS2) | 기존 후보 조사 | 현재 실행 의존성 아님 |

근거는 [동적 보행 계획](../reports/RS02_동적보행_계획.md),
[RL 실행 기록](../reports/RS02_RL_실행_기록.md),
[계단 개선 실험](../reports/RS02_계단_개선_실험.md),
[WALK 보고서](../reports/WALK_검증_보고서.md)다.

이전 제어기와 현재 PPO를 비교할 때 속도·코스·판정·정책 선택 조건을 함께 제시한다.
699초 crawl 완주를 동적 보행 성능으로 발표하지 않는다.
외부 코드의 고정 커밋은 [버전 목록](../../config/third-party.lock.json),
배포 시 출처와 라이선스 처리는 [외부 자료 안내](../../THIRD_PARTY_NOTICES.md)에 있다.
