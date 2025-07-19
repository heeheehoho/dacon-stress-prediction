# 1. 라이브러리 불러오기
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 2. 데이터 불러오기
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')

# 3. 결측치 처리
fillna_cat_cols = ['medical_history', 'family_medical_history', 'edu_level']
for col in fillna_cat_cols:
    train[col] = train[col].fillna('Missing')
    test[col] = test[col].fillna('Missing')

train['mean_working'] = train['mean_working'].fillna(train['mean_working'].median())
test['mean_working'] = test['mean_working'].fillna(train['mean_working'].median())

# 4. 인코딩
from sklearn.preprocessing import LabelEncoder

cat_cols = train.select_dtypes(include='object').drop(columns='ID').columns
for col in cat_cols:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])

# 5. 피처 및 타겟 정의
X = train.drop(columns=['ID', 'stress_score'])
y = train['stress_score']
X_test = test.drop(columns=['ID'])

# 6. 데이터 분할
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. LightGBM 모델 학습
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)

# 8. 검증 점수 출력
val_pred = lgb_model.predict(X_valid)
mae = mean_absolute_error(y_valid, val_pred)
print(f"Validation MAE: {mae:.4f}")

# 9. 테스트 데이터 예측
test_pred = lgb_model.predict(X_test)

# 10. 제출 파일 생성
submission = sample_submission.copy()
submission['stress_score'] = test_pred
submission.to_csv('baseline_lgbm_submission.csv', index=False)
print("✅ 제출 파일 저장 완료: baseline_lgbm_submission.csv")
