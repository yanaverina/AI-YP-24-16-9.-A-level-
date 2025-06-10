# Чекпоинт: Нелинейные модели

Проведено 4 эксперимента со следующими моделями:
* CatboostClassifier + Tf-idf (только на текстах) - catboost_text
* RandomForestClassifier + Tf-idf (только на текстах) - random_forest_text
* CatboostClassifier + Tf-idf (тексты + доп. признаки) - catboost_add_feats
* RandomForestClassifier + Tf-idf (тексты + доп. признаки) - randomforest_add_feats

## Сводная таблица метрик моделей основанных на деревьях

| model                  |   f1_macro |   f1_weighted |   roc_auc |
|:-----------------------|-----------:|--------------:|----------:|
| catboost_text          |   0.764633 |      0.763762 |  0.962938 |
| random_forest_text     |   0.75707  |      0.763789 |  0.947208 |
| catboost_add_feats     |   0.743381 |      0.706272 |  0.955698 |
| randomforest_add_feats |   0.746132 |      0.717097 |  0.932667 |

