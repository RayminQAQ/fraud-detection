import pandas as pd
import xgboost as xgb

# Load data
train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("data/test_data.csv")

# Prepare training and test sets
X_train = train_data.drop(['fraud'], axis=1)
y_train = train_data['fraud']
combine_normal =  pd.concat([train_data, test_data], axis=0)

# process on data
combine_normal['att1'] = pd.to_datetime(combine_normal['att1']).apply(lambda x: x.timestamp())
combine_normal = combine_normal.drop(columns=['att8', 'att9', 'att10', 'att11'])
combine_normal = pd.get_dummies(combine_normal)

# test & train
train = combine_normal[combine_normal['Id'].isna()]
train = train.drop(columns=['Id'])
test = combine_normal[combine_normal['fraud'].isna()]
test = test.drop(columns=['fraud'])

# Train_x, Train_y
train_X = train.drop(columns=['fraud'])
train_Y = train['fraud']

# Looping: Test -> predict
result = [
    ["Id", "fraud"],
]

for index in range(len(test)):
    data = test.iloc[index]

# 初始化 XGBoost 分類模型
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# 訓練模型
model.fit(train_X, train_Y)

# 預測測試集
test_ids = test['Id']  # 假設 'Id' 欄位存在於 test DataFrame 中
test_X = test.drop(columns=['Id'])  # 測試集的輸入特徵

for index in range(len(test_X)):
    data = test_X.iloc[index]
    prediction = model.predict([data])[0]  # 取得單筆預測結果
    result.append([int(test_ids.iloc[index]), int(prediction)])  # 將 Id 和預測結果添加到 result

# 將結果轉換成 DataFrame 並輸出成 CSV
result_df = pd.DataFrame(result[1:], columns=result[0])  # 排除 header 列
result_df.to_csv("submission.csv", index=False)

print("預測結果已儲存為 submission.csv")
