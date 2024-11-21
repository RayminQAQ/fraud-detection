import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# 讀取資料
train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("data/test_data.csv")

# 合併資料以便統一處理
combine_normal = pd.concat([train_data, test_data], axis=0, ignore_index=True)

# 資料預處理
combine_normal['att1'] = pd.to_datetime(combine_normal['att1']).apply(lambda x: x.timestamp())
combine_normal = combine_normal.drop(columns=[ 'att9', 'att10', 'att11'])
combine_normal = pd.get_dummies(combine_normal)

# 分割訓練和測試集
train = combine_normal[combine_normal['fraud'].notna()].copy()
test = combine_normal[combine_normal['fraud'].isna()].copy()

# 提取特徵和標籤
train_X = train.drop(columns=['Id', 'fraud'])
train_Y = train['fraud']
test_X = test.drop(columns=['Id', 'fraud'])
test_ids = test['Id']

# 定義超參數搜索空間
random_grid = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'min_child_weight': [1, 2, 3, 4, 5],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0, 0.01, 0.1, 1],
}

# 初始化 XGBoost 分類模型
xg4 = xgb.XGBClassifier(use_label_encoder=False, eval_metric="auc")

# 設置 RandomizedSearchCV
model = RandomizedSearchCV(
    estimator=xg4, 
    param_distributions=random_grid, 
    n_iter=100, 
    cv=3, 
    verbose=2, 
    random_state=42, 
    n_jobs=-1
)

# 訓練模型
model.fit(train_X, train_Y)

# 預測測試集
predictions = model.predict(test_X)

# 生成提交結果
result_df = pd.DataFrame({
    'Id': test_ids.astype(int),
    'fraud': predictions.astype(int)
})

# 輸出結果
result_df.to_csv("submission.csv", index=False)
print("預測結果已儲存為 submission.csv")
