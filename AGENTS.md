# AI 작업 시작점

1. `docs/ai/CONTEXT.md`를 먼저 읽고, 상태 판단에는 `docs/user/STATUS.md`를 사용한다.
2. 실행 명령은 `docs/user/RUN.md`, 파일 위치는 `docs/user/STRUCTURE.md`를 따른다.
3. 현재 목표는 같은 정책·조건에서 보행 가능한 범위와 실패 원인을 밝히는 것이다. 20cm 성공 자체는 필수 목표가 아니다.
4. 코드 변경은 `src/`, 테스트는 `tests/`, 설정은 `config/`에 둔다. 새 보고서를 루트에 만들지 않는다.
5. 과거 `output/` 결과, 체크포인트, 코드 스냅샷과 `archive/` 원본은 덮어쓰지 않는다. 실패도 증거로 보존한다.
6. 질량·관절 범위·토크 한계·충돌·중력 또는 통과 판정을 유리하게 바꾸지 않는다. 변경이 연구상 필요하면 조건과 영향을 명시한다.
7. 평가는 자동 리셋으로 실패를 숨기지 않는다. 코스 완주, 보행 품질, 하드웨어 안전을 별개로 보고한다.
8. 일반 탐색은 `src tests config docs`로 범위를 제한한다. `output`, `archive`, `third_party` 전체를 먼저 읽지 않는다.
9. 경로는 `src/robost_paths.py`를 기준으로 한다. 저장소 밖 설치를 위한 독립 wheel은 지원하지 않으며 source checkout에서 실행한다.
10. CPU 변경은 rs02-mujoco 환경의 `make test-core`, RL 변경은 rs02-rl 환경의 `make test-rl`, 문서·배포 변경은 `make check`로 확인한다. GPU 동작은 별도 smoke test가 필요하다.
11. 작업 완료 후 `docs/user/STATUS.md`와 관련 문서를 실제 확인 결과로 갱신한다. 기존 연구 결과를 새로 재현했다고 표현하지 않는다.
