# Нейросетевые подходы

Было опробовано несколько нейросетевых подходов для решения задачи многоклассовой классификации вопросов экзамена по темам

1) Дообучение BERT на классификацию
    Была дообучена модель `bert-base-uncased` c добавлением dropout и 1 линейного слоя.
    В качетсве планировщика learning_rate был взят `linear_schedule_with_warmup`:
        * На отрезке 0, numwarmup LR растёт линейно от 0 до initlr.  
        * После шага numwarmup начинается линейный «спуск» от initlr обратно к 0 вплоть до шага num_training.
    
    Полученные метрики на валидации:

        F1 (weighted) = 0.7930099975212757
        ROC-AUC (weighted, ovr) = 0.9742022965620226

2) Токенайзер BERT + логистическая регрессия:
    ROC-AUC = 0.90
    F1 (weighted) = 0.56



# Таблица всех опробованных моделей за все чекпоинты

| Модель | ROC-AUC | F1 | Гиперпараметры |
|------------|------------|------------|------------|
| LogReg   | 0.95   |  0.74  | C=10, penalty=’l2’ |
| LogReg   | 0.95   | 0.86   | C=103, penalty=’l2’|
| BERT (дообучение)| 0.97 | 0.79 |lr=2e-5, shceduler=linear_with_warmup, num_epochs = 20 | 
| BERT tokenizer + LogReg  | 0.90   | 0.56   | ?|
| Naive Bayess   | 0.91   | 0.46   | ngram_range=(1, 2), alpha=10e−10|
| Linear SVC   | 0.92   | 0.63   | ’C’: 100, ’gamma’: 0.01, ’kernel’: ’rbf’|
|Decision Tree   | 0.66   | 0.51   | ’criterion’: ’entropy’, ’max_depth’: 12 |
