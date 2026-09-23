# 외부 코드

원본 checkout은 로컬에 보존하며 Git 및 소스 배포에서 제외한다.
저장소 URL/커밋/라이선스 정보는 [고정 목록](../config/third-party.lock.json)에 있다.
`python scripts/bootstrap_third_party.py`로 재현한다. 기존 checkout이 수정되었거나 버전이 다르면 중단한다.
CPU 보행만 필요하면 `--only Cheetah-Software eigen3`, RL만 필요하면 `--only mjlab`을 사용한다.
서드파티 라이선스는 해당 checkout에 보존하며 재배포 시 함께 제공해야 한다.
