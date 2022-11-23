# 台北房價預測

利用Ensemble Learning組成兩層式的架構，第一層由CatBoost、XGBoost、KNN Regressor、Linear Regression、及Random Forest組成。
再將將各個Model的預測結果 Feed in 至第二層的Linear Regression做出最終的預測

模型示意圖
![Model](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/ensemble.png)

資料來源為內政部不動產成交案件資料供應系統之台北市104~109年第2季交易資料
資料共12萬筆 ; 每筆資料具有17種屬性，包含該不動產所處之區、土地面積、建物面積、停車格面積、有無隔間、有無管理室、建物類型、樓層數、用途。。。等

## 資料清理

![Cleaning](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/cleaning.png)
在房價的預測上，屋齡也是一大影響價格的重要因素，由於原始資料裡只有建造年月日，因此我們將年份提取出來轉化成屋齡作為其中一個Attribute。
在空值上;若是Continous的數值資料則填 Median;若是 Categorical 或是 Discrete 的資料則用 Mode 填入

![Label_distribution](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/distribution.png)
在目標Label(每坪價格)的分布上，有許多Outliers，為了讓Model不被影響，我們可以將 Z-score 大於 3 的資料清除掉

如附圖所示
![outlier_cleaning](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/outlier.png)

## EDA

![outlier_cleaning](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/district.png)
可以看到大安區跟中山區的平均房價比其他區高

![outlier_cleaning](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/house_age.png)
屋齡在62~70歲反而平均價格比較高

![outlier_cleaning](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/main_use.png)
不同房屋使用目的有不同的平均價格，然而較常見的住家用、商業用、工業用等差距並不大


## 模型建置

### CatBoost

### XGBoost

### KNN Regressor

### Random Forest

### Linear Regression

### Final Ensemble Model




