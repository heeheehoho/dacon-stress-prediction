# 🧠 스트레스 지수 예측 AI 모델

데이콘 Basic 스트레스 지수 예측 대회에 참여하며 진행한 머신러닝 프로젝트입니다.  
건강 데이터를 기반으로 개인의 스트레스 점수를 예측하는 AI 모델을 개발하였습니다.

## 📌 대회 정보

- **대회명**: DACON Basic 스트레스 지수 예측
- **주최**: DACON
- **목표**: 신체 정보, 수면, 활동 데이터를 바탕으로 개인의 `stress_score` 예측
- **평가 기준**: MAE (Mean Absolute Error)

## 🗂️ 데이터 설명

- `train.csv`: 학습 데이터 (target 포함)
- `test.csv`: 테스트 데이터 (target 없음)
- `sample_submission.csv`: 제출 양식

| Column | Description |
|--------|-------------|
| gender, age, height, weight | 기본 신체 정보 |
| cholesterol, blood_pressure, glucose | 건강 수치 |
| bone_density, activity, smoke_status | 생활 습관 |
| sleep_pattern, edu_level | 수면/학력 |
| mean_working | 주당 평균 근로시간 |
| stress_score | 🎯 예측 대상 (train에만 존재) |

## ⚙️ 사용 기술

- Python 3.10
- Pandas, NumPy, Scikit-learn
- LightGBM
- Matplotlib, Seaborn (EDA)

## 📈 모델링 과정

1. 데이터 전처리
   - 결측치 처리
   - Label Encoding
2. 학습/검증 분할
3. LightGBM 기본 모델 학습
4. MAE 기반 성능 평가
5. 제출 파일 생성

## 🧪 주요 성능

- 모델: LightGBM (baseline)
- 검증 성능 (MAE): 약 **△△.△△△△**
- 향후 개선 방향:
  - 하이퍼파라미터 튜닝
  - 교차 검증 도입
  - XGBoost, CatBoost 등 다양한 모델 실험
  - 앙상블 전략 적용

## 📂 폴더 구조

```
stress-predict/
├── data/                   # 데이터셋 (train, test, sample_submission)
├── notebook/               # EDA 및 실험용 ipynb
├── models/                 # 모델 결과 파일 저장
├── submissions/            # 제출 파일 저장
├── .venv/                    # Conda 가상환경 (옵션)
├── code.py                 # 학습/추론 스크립트
└── README.md               # 프로젝트 설명 문서
```

## 📤 제출 방법

```bash
python code.py
# 또는
jupyter notebook lgbm_baseline.ipynb
```

생성된 `baseline_lgbm_submission.csv` 파일을 제출

---

## 🙋‍♀️ Contact

- 작성자: [jungheeho](https://github.com/heeheehoho)
- 문의: heraokok@naver.com

> 본 프로젝트는 DACON 스트레스 지수 예측 대회를 기반으로 진행되었습니다.
