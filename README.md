# otus-mla-2-fastapi

## Цель:
В этом домашнем задании, вам нужно создать fastapi-приложение для вашей модели классификации. И развернуть данное приложение локально при помощи Docker. Протестировать get запросы (направляя X-вектор переменных) и получить response в виде целевой переменной (для теста можно использовать Postman).


## Описание/Пошаговая инструкция выполнения домашнего задания:
Воспользуйтесь готовым приложением heart_cleveland_upload.zip
Создайте Dockerfile для запуска этого приложения в контейнере
Последовательностью команд соберите и запустите контейнер с приложением: Опубликуйте порт 8000 контейнера для локального доступа аргументом -p 8000:8000

```
docker build -t myapp .
docker run -p 8000:8000 myapp
```

В отдельном терминале инструментом curl или в браузере с помощью инструментов Insomnia/Postman убедитесь в работоспособности приложения.
Результат ДЗ - ссылка на репозиторий (или каталог в имеющемся репозитории), который содержит:
- собственно, приложение из приложенного архива;
- Dockerfile
- README.md - описание задачи и инструкцию по запуску и проверке контейнера

## Инструкция по запуску

1. Склонировать репозиторий
2. Собрать и запустить контейнер:
    ```
    docker build -t myimage .
    docker run --rm --name mycontainer -p 8000:8000 myimage
    ```
3. Зайти на url: [localhost:8000/docs](http://localhost:8000/docs), запустить endpoint `predict` из документации fastapi