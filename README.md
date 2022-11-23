# 台北房價預測

利用Ensemble Learning組成兩層式的架構，第一層由CatBoost、XGBoost、KNN Regressor、Linear Regression、及Random Forest組成。
再將將各個Model的預測結果 Feed in 至第二層的Linear Regression做出最終的預測

模型示意圖
![Model](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/ensemble.png)

資料來源為內政部不動產成交案件資料供應系統之台北市104~109年第2季交易資料
資料共12萬筆 ; 每筆資料具有17種屬性，包含該不動產所處之區、土地面積、建物面積、停車格面積、有無隔間、有無管理室、建物類型、樓層數、用途。。。等

## Table of Contents
1. [使用插件](#lib)
2. [資料清理與工程](#data_cleaning)
3. [EDA](#EDA)
4. [模型建置](#Model)
5. [Evaluation](#evaluation)


<a name="lib"></a>

## 使用插件

![Cleaning](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/library.png)

<a name="data_cleaning"></a>

## 資料清理

![Cleaning](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/cleaning.png)
在房價的預測上，屋齡也是一大影響價格的重要因素，由於原始資料裡只有建造年月日，因此我們將年份提取出來轉化成屋齡作為其中一個Attribute。
在空值上;若是Continous的數值資料則填 Median;若是 Categorical 或是 Discrete 的資料則用 Mode 填入

![Label_distribution](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/distribution.png)
在目標Label(每坪價格)的分布上，有許多Outliers，為了讓Model不被影響，我們可以將 Z-score 大於 3 的資料清除掉

如附圖所示
![outlier_cleaning](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/outlier.png)

<a name="EDA"></a>
## EDA

![district](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/district.png)
可以看到大安區跟中山區的平均房價比其他區高

![housing age](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/house_age.png)
屋齡在62~70歲反而平均價格比較高

![main_use](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/main_use.png)
不同房屋使用目的有不同的平均價格，然而較常見的住家用、商業用、工業用等差距並不大

<a name="Model"></a>

## 模型建置

在cross validation上，我們用80%的資料做training,10%的資料做validation，10%的資料做testing

![performance](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/model_performance.png)
同時透過視覺化的方式透過預測結果與Validation裡Label的差來初步判斷模型的Performance
### CatBoost

![CatBoost_code](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/Catboost_code.png)

利用Catboost的模型，我們給予每個屬性權重，若是要減少Noise及減少後續Model的負擔可以只取其中前幾個分數明顯較大的屬性，這邊我們取前4名; 分別是屋齡，建築的坪數，是否位於大安區，及土地的坪數。
![CatBoost_Att](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/Catboost_Att.png)

預測結果
![Catboost](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/Catboost.png)
Y-axis為Validation set的Label ; X-axis為模型的預測結果
可以從紫色虛線的偏移大致判斷模型的效果


### XGBoost

![XG_code](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/XGBoost_code.png)

可以看到在XGBoost的模型中，對屬性的權重跟CatBoost不同。第一名是土地坪數，再來分別是建築坪數及屋齡。之前CatBoost所認為的是否位於大安區在XGboost中甚至被排到很後面。但是可以發現土地坪數、建築坪數及屋齡對於房價的影響相較於其他屬性多很多。
![XG_Att](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/XGBoost_Att.png)

預測結果
![XGB](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/XGBoost.png)
### KNN Regressor

![KNN_code](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/KNN_code.png)


![KNN_regressor](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/KNN_regressor.png)

### Random Forest
![rf_code](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/random_forest_code.png)

![rf](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/random_forest.png)
可以看到random forest跟KNN Regressor的Performance相較於XGBoost跟CatBoost比紫虛線的偏移較嚴重(Performance較差)

### Linear Regression

![lr](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/linear_regression.png)

### Final Ensemble Model

![df](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/dataframe.png)
將各個模型的預測整合成一個Dataframe作為2層模型的input

![ensemble_code](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/ensemble_code.png)

![ensemble_p](https://raw.githubusercontent.com/eric807791961/Ensemble_Learning-Taipei_Housing_Price_Prediction/main/pics/Ensemble_d.png)
最後Ensemble的預測結果相較於上面單個的Model相比確實有更好的Performance
在每坪價格的median為17萬下，該模型MSE約為4萬。但是如果只考慮某使用目的(如住家用)，則MSE可以降至2萬

<a name="evaluation"></a>

## Evaluation

* 在虛值的填空上可以用Cross-Correlation的方式找出屬性之間的關西，利用額外的資訊去填入虛值。
* 在將第一層的model整合時，可以根據該model的Performance給予權重，再輸入進第2層讓Performance較佳的模型可以對最終的預測結果有更多影響。
* 可以在第二層使用不同的模型。





