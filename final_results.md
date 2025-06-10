# Подведение итогов по всем исследованым моделям


| Модель                           | ROC-AUC | F1 / F1_weighted | Гиперпараметры                                      |
|----------------------------------|---------|------------------|-----------------------------------------------------|
| LogReg                           | 0.95    | 0.74             | C=10, penalty='l2'                                  |
| LogReg                           | 0.95    | 0.86             | C=10³, penalty='l2'                                 |
| BERT (дообучение)                | 0.97    | 0.79             | lr=2e-5, scheduler=linearwithwarmup, num_epochs=20, 1 дополнительный линейный слой + Dropout(p=0.2)|
| BERT tokenizer + LogReg          | 0.90    | 0.56             | pretrained tokenizer, log reg params: C=10, penalty='l2'|
| Naive Bayes                      | 0.91    | 0.46             | ngram_range=(1,2), alpha=10e−10                    |
| Linear SVC                       | 0.92    | 0.63             | C=100, gamma=0.01, kernel='rbf'                    |
| Decision Tree                    | 0.66    | 0.51             | criterion='entropy', max_depth=12                  |
| CatBoost (text)                  | 0.96    | 0.76             | depth = 3, terations = 200, l2_leaf_reg=0.01, lr=0.3                                                |
| Random Forest (text)             | 0.95    | 0.76             | max_depth=30, min_samples_leaf=1, min_samples_split=5, n_estimators=100                                                  |
| CatBoost (доп. признаки)         | 0.96    | 0.71             | depth=7,iterations=200,l2_leaf_reg=3,learning_rate=0.3                                      |
| Random Forest (доп. признаки)    | 0.93    | 0.72             | min_samples_leaf=1, min_samples_split=2,n_estimators=500


# Лучшая модель
Исходя из приведенной таблицы, лучшая модель - дообученный на классификацию BERT.
    * ROC-AUC = 0.97
    * F1 weighted = 0.79 
