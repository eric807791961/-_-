# 台北房價預測

利用Ensemble Learning組成兩層式的架構，第一層由CatBoost、XGBoost、KNN Regressor、Linear Regression、及Random Forest組成。
再將將各個Model的預測結果 Feed in 至第二層的Linear Regression做出最終的預測

模型示意圖
![Model](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/ensemble.png)

資料來源為內政部不動產成交案件資料供應系統之台北市104~109年第2季交易資料
資料共12萬筆 ; 每筆資料具有17種屬性，包含該不動產所處之區、土地面積、建物面積、停車格面積、有無隔間、有無管理室、建物類型、樓層數、用途。。。等

## 資料清理

## Cat Boost



