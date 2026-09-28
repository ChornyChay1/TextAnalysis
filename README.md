# Лабораторная №1 — Twitter US Airline Sentiment

Выполненная работа находится в [notebook.ipynb](notebook.ipynb): реальная загрузка данных, разбиение Train/Test, исследование обучающей части, очистка, нормализация, расстояние Левенштейна, биграммная модель, перплексия и генерация. Результаты и графики сохранены **внутри ноутбука**. Ячейки запускаются сверху вниз.

```bash
python -m pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Первая ячейка скачивает оригинальный `Tweets.csv` в `data/Tweets.csv` с [общедоступного зеркала](https://github.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/blob/master/Tweets.csv). NLTK при необходимости скачивает WordNet и английский теггер. После первой загрузки CSV и ресурсов работа воспроизводится без сети. Зафиксированы `random_state=42` и начальные состояния для генерации.

[Страница набора на Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment). Лицензия CSV — CC BY-NC-SA 4.0; файл данных в репозиторий не включён.
