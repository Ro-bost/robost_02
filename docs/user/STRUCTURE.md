# 프로젝트 파일 구조

```text
robost/
├── README.md                 전체 안내
├── AGENTS.md                 AI 작업 규칙
├── src/                      실행·제어·평가 Python 모듈
├── tests/                    회귀 테스트
├── native/                   MIT swing C++ 연결 코드
├── models/                   실행에 필요한 원본 모델·메시
├── config/                   환경·고정 버전·정책 해시
├── scripts/                  외부 코드 복원·검사·배포 묶음 생성
├── docs/
│   ├── user/                 실행·구조·상태·배포·정리 결과
│   ├── ai/                   AI가 읽을 핵심 맥락
│   ├── research/             선행연구와 재사용 범위
│   ├── reports/              기존 보고서와 선별한 반복 결과 JSON
│   └── assets/               문서용 이미지
├── .github/workflows/        CPU 검사와 소스 묶음 생성
├── third_party/              고정 버전 외부 코드 (로컬)
├── output/                   실험·영상·체크포인트 (로컬)
├── archive/                  CAD·원본 자료·정리 전 파일 (로컬)
├── build/                    다시 만들 수 있는 C++ 라이브러리 (로컬)
└── dist/                     GitHub 전달용 압축 묶음 (로컬)
```

루트의 `RS02_4족로봇_최종정리_2026-09-15`는 원본 폴더를 가리키는 로컬 호환 링크다.
과거 결과 XML의 절대경로를 유지하기 위한 것으로 GitHub 배포에는 들어가지 않는다.
새 코드는 `models/`를 사용한다. 연구 스크립트, CAD 원본, 모터 자료, 과거 영상은
`archive/RS02_4족로봇_최종정리_2026-09-15/`에 기존 하위 구조 그대로 있다.

`src/run_rs02_stairs.py`가 현재 계단 진입점이며 `run_rs02_mujoco.py`는 자세 확인용이다.
`rs02_rl_damped.py`, `rs02_rl_guard.py`는 보존된 비교 실험으로 기본 정책이 아니다.
기타 audit/render/plot 모듈은 결과 분석용이다. 기능별 상세 설명은 [AI 코드 지도](../ai/CONTEXT.md)를 참고한다.

새 결과는 `output/rl/새로운_실험명/`, 새 사용자 안내는 `docs/user/`, 상세 보고서는
`docs/reports/`에 저장한다. 원본 결과를 덮어쓰거나 성공 사례만 남기지 않는다.
