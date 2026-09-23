# GitHub 배포용 구성

현재 디렉터리는 정리 시작 시 Git 저장소가 아니었으며 연결된 원격 저장소도 없었다.
배포 대상은 [Ro-bost/robost_02](https://github.com/Ro-bost/robost_02)이다.
현재 GitHub 앱의 파일 생성 요청이 403으로 거부되어 최초 업로드는 대기 중이다.
로컬 소스 커밋과 원격 연결을 준비했으며, 쓰기 인증이 연결되면 아래 명령으로 업로드한다.

```bash
git push -u origin main
```
이 프로젝트는 로봇 시뮬레이션 코드이므로 웹사이트 배포가 아닌 소스/실행 자료 배포다.

## 로컬에서 배포 파일 만들기

```bash
python scripts/check_project.py
python scripts/export_release.py --with-policies
```

- `dist/robost-source.tar.gz`: 코드·테스트·설정·문서·필수 모델·GitHub Actions.
- `dist/robost-policies.tar.gz`: 기본 rhythm500/1000과 정책 SHA-256 목록.
- 소스는 `SOURCE_MANIFEST.json`, 가중치는 `POLICY_MANIFEST.json`에 파일별 SHA-256/크기를 기록한다.

이미 파일이 있으면 덮어쓰지 않는다. 다음 버전은
`--output dist/새버전/robost-source.tar.gz --with-policies`로 생성한다.
9GB 실험 결과, CAD 원본, 전체 외부 저장소, 캐시, 이전 빌드 파일은 포함되지 않는다.
모델 원본은 메시를 단순화하지 않고 필요한 파일만 선별했으며 `models/manifest.json`으로 무결성을 확인한다.

## GitHub에 올릴 소스 폴더

빈 작업 폴더에서 source 묶음을 풀면 최상위 `robost/`가 생긴다.
그 폴더를 GitHub 저장소로 사용하면 로컬 연구 데이터가 섞이지 않는다.
소스 폴더에서 `git init`, 저장소 생성·remote 연결·push를 진행한다.
정책 묶음은 소스 Git에 넣지 않고 별도 Release 첨부 파일로 전달한다.
업로드 대상 저장소와 자료 공개 범위는 소유자가 정한다. 라이선스 현황은 [외부 자료 안내](../../THIRD_PARTY_NOTICES.md)에 있다.

## 자동 검사

`.github/workflows/check.yml`은 push/PR/수동 실행 시 Python 3.11에서 CPU 의존성과 고정 외부 소스를 준비하고
`make check test-core` 및 소스 압축을 실행한다. GPU RL 시험은 일반 GitHub runner에 포함하지 않는다.
소스 묶음은 Actions artifact로 생성하며 원격 Release를 자동 공개하지 않는다.
워크플로는 로컬에서 대응 명령을 검증했으며 GitHub 원격 실행은 아직 하지 않았다.

## 배포 후 확인

새 위치에서 [실행 안내](RUN.md)에 따라 환경/외부 코드를 준비한다.
CPU는 `python -m run_rs02_mujoco --check`, RL은 가중치 복원 후
`python -m run_rs02_stairs --stairs-cm 15 --headless --duration 40`으로 확인한다.
새 환경에서 원래 6/6 성능을 주장하려면 동일한 정책·조건의 반복 평가를 다시 수행해야 한다.
