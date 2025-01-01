# BASELINE
Построенно несколько линейных моделей машинного обучения для предсказания темы вопроса по его текстовому описанию и нескольким другим признакам, а именно:
* page - номер страницы на которой расположен вопрос в экзамене
* score
* year
* qst_len - длина предобработанного вопроса (под предобработкой понимается лемматизация, токенизация, удаление стоп-слов)

Для векторизации вопросов во всех моделях использовался метод TF-IDF для столбца `qst_processed` - лемматизированных, токенизированных вопросов с удаленными стоп-словами.

Все эксперименты находятся в файле `ml_model.ipynb`.

## Описание моделей
##### Note: во всех моделях (кроме случайного предсказания и предсказания по n-грамме) параметры подбирались с помощью GridSearchCV

1) Логистическая регрессия:
1.1) Подобранные параметры: LogisticRegression(C=103, max_iter=10000, multi_class='ovr')
    * f1-score macro на тестовой: 0.64
    * f1-score weighted на тестовой: 0.73
    * Roc-Auc на тестовой: 0.943
      
1.2) Подобранные параметры: LogisticRegression('C': 10, 'penalty': 'l2', 'solver': 'saga')
      Обучение заняло: 12096.86 секунд
      Лучшее качество: 0.774

               precision    recall  f1-score   support
         
                   0.74      0.77      0.76        22
                   1.00      0.50      0.67         6
                   1.00      0.75      0.86         8
                   0.75      0.38      0.50         8
                   0.70      0.90      0.79        29

Accuracy Score: 0.753

Лучшее качество по f1-мере на 3-ем классе (тематика 'the market'), f1 = 0.86.

Наибольшее значение по ROC-AUC достигается на 5-ом классе (тематика 'meeting customer needs').
![Screenshot from 2024-12-11 21-27-29](https://github.com/user-attachments/assets/8cf4f562-7f08-4d13-bb3e-5158ab443fac)


3) N-Grams + Naive bayes 
Подобранные параметры: MultinomialNB(alpha=1e-10)
    * f1-score macro на тестовой: 0.46
    * f1-score weighted на тестовой: 0.56
    * Roc-Auc на тестовой: 0.909

4) Линейный SVM
4.1) Подобранные параметры: LinearSVC(C=1e-05)
    * f1-score macro на тестовой: 0.10
    * f1-score weighted на тестовой: 0.20 
    * Roc-Auc на тестовой: 0.622

5) SVC
   Обучение заняло: 3.92 секунд
   Подобранные параметры:  {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}
   Лучшее качество: 0.744

   Метрики на тесте
                 precision    recall  f1-score   support

           1       0.68      0.77      0.72        22
           2       0.67      0.67      0.67         6
           3       1.00      0.75      0.86         8
           4       0.33      0.12      0.18         8
           5       0.70      0.79      0.74        29

Accuracy Score: 0.699

Лучшее значение по f1-мере достигается на 3-ем классе (тематика 'the market'), f1-мера=0,86.

6) Случайное предсказание класса:
    * f1-score macro на тестовой: 0.12
    * f1-score weighted на тестовой: 0.18 
    * Roc-Auc на тестовой: 0.383
  
7) Дерево решений DecisionTreeClassifier
   Обучение заняло:  3.74 секунд.
   Подобранные параметры: {'criterion': 'entropy', 'max_depth': 15}
   Лучшее качество: Cross-Validated Score of the Best Estimator: 0.661

    Метрики на тесте
                 precision    recall  f1-score   support

           1       0.43      0.45      0.44        22
           2       0.67      0.33      0.44         6
           3       1.00      0.62      0.77         8
           4       0.22      0.25      0.24         8
           5       0.58      0.66      0.61        29

Accuracy Score: 0.521

Лучшее качество на третьем классе (тематика 'the market'), f1-мера = 0.77.

9) Предсказание класса по наиболее частой n-грамме
 
 #### Описание метода: на трейн-выборке получаем из вопросов все n-gramm'ы (n=1, 2, 3)
 Для каждой n-gramm'ы находится класс, в котором она чаще всего встречается на train-выборке
 При получении на вход нового вопроса, модель находит самую часто-встречающуюся на трейн-выборке n-gramm'у  из n-gramm в вопросе и по ней предсказывает класс.

* f1-score macro на тестовой: 0.18
* f1-score weighted на тестовой: 0.28 


По полученым метрикам можно сделать вывод, что лучше всего тему вопроса предсказывает `Логистическая регрессия`.
Пайплайн обучения Логистической регрессии лежит в `pipeline.ipynb`


