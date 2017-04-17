################################################
## 代码说明
* 入口：./Code/Feat/run_all.py，目前已实现的是预处理和模糊特征（出于时间考虑只复现了QRatio和WRatio）
* 预处理：./Code/Feat/preprocess.py，
    输入：./Data/%s.csv % dome （dome指train或test，原始文件）
    输出：./Feat/solution/%s.processed.csv.pkl % dome
    处理：大小写转换，同义替换（目前无拼写纠错）
* fuzzy模糊特征：./Code/Feat/genFeat_fuzzy_feat.py
    输入：./Feat/solution/train.processed.csv.pkl（预处理的结果）
    输出：./Feat/solution/%s/%s/%s.%s.feat.pkl % (path, dome,feat_name)
          （path指All或Run/Fold，All时为全部，否则指交叉验证某个集合；dome指train或valid或test，feat_name指某个详细的特征，比如fuzzy特征里的QRatio特征）
          ./Feat/solution/%s.fuzzy_feature.csv % dome （dome指train或test）
          这里pkl文件保存各个特征结果，用于下一步处理；csv文件保存原数据与特征结果，用于观察分析的。

* 代码路径主要是./Code：
    ./Code/Feat: 生成单特征、组合特征；其他特征的处理可以参考fuzzy特征，而且原项目有很多特征的实现，如tfidf、svd、distance等，可以在其基础上做相应修改。
    ./Code/Model: 训练模型
    ./Code/param_config.py:参数配置，如path、k-fold等

## 数据说明
* ./Data: 原始数据
* ./Feat/solution: 预处理后数据
* ./Feat/solution/All: 训练或测试所有特征数据
* ./Feat/solution/Run/Fold: 交叉验证特征数据
* ./Output:模型结果

* 强烈建议花点时间看文档`./Doc/Kaggle_CrowdFlower_ChenglongChen.pdf'。

## 其他
* python环境要求2.7.x（我是2.7.13）

#################################################
##  原项目说明
1st Place Solution for [Search Results Relevance Competition on Kaggle](https://www.kaggle.com/c/crowdflower-search-relevance)
The best single model we have obtained during the competition was an [XGBoost](https://github.com/dmlc/xgboost) model with linear booster of Public LB score **0.69322** and Private LB score **0.70768**. Our final winning submission was a median ensemble of 35 best Public LB submissions. This submission scored **0.70807** on Public LB and **0.72189** on Private LB.

## Instruction
* download data from the [competition website](https://www.kaggle.com/c/crowdflower-search-relevance/data) and put all the data into folder `./Data`.
* run `python ./Code/Feat/run_all.py` to generate features. This will take a few hours.
* run `python ./Code/Model/generate_best_single_model.py` to generate best single model submission. In our experience, it only takes a few trials to generate model of best performance or similar performance. See the training log in `./Output/Log/[Pre@solution]_[Feat@svd100_and_bow_Jun27]_[Model@reg_xgb_linear]_hyperopt.log` for example.
* run `python ./Code/Model/generate_model_library.py` to generate model library. This is quite time consuming. **But you don't have to wait for this script to finish: you can run the next step once you have some models trained.**
* run `python ./Code/Model/generate_ensemble_submission.py` to generate submission via ensemble selection.
* if you don't want to run the code, just submit the file in `./Output/Subm`.