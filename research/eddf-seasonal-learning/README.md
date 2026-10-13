# Франкфурт: сезонная проверка Polynomial Ridge

Независимое обучение на EDDF 07L: по пять погожих дней весны, лета и осени 2025 года. Сравниваются постоянная вертикальная скорость, Ridge и Polynomial Ridge степени 2. Схема повторяет LFPG и EGLL; победитель заранее не назначается. Отдельная ветка `research/eddf-seasonal-learning` создана от `main` dd0d1cf.

В каждом сезоне первые три выбранных по дате дня используются для обучения, четвертый для валидации, пятый для теста. Общая модель обучается на девяти днях. Параметры выбираются только по валидации; все модели проверяются на одинаковых тестовых заходах. Каждый заход получает одинаковый суммарный вес обучения. Прогнозируется поправка к baseline на 30 ± 3 секунды по прежним девяти признакам истории наблюдений.

Погода должна подходить весь интервал 08:00–18:00 Europe/Berlin, не полные сутки. Видимость от 10 км, потолок от 3000 ft либо отсутствие низкого потолка, ветер до 15 kt, порывы до 25 kt, температура от -10 до +30 C, без неблагоприятных явлений и неизвестных условий. Для дня нужно минимум десять пригодных заходов и 300 пар. Назначение 07L проверяется по траекториям, а не по направлению ветра. Северо-западная полоса 07L/25R предназначена для посадок по [описанию Fraport](https://www.fraport.com/en/sustainability/dialog-with-neighbors/noise-and-air/flight-operations/runway-system-and-operating-hours.html).

Высота аэродрома 364 ft задается в отдельной конфигурации EDDF. QNH соединяется с предыдущим METAR с условной задержкой 15 минут; COR исключены как в LFPG. Высота — барометрический аэродромный прокси, не радиовысота и не высота над порогом полосы. Геометрия сохранена в `airports/EDDF/reports/`; историческая эксплуатационная применимость справочника не подтверждена.

## Структура

[Итоговый отчет и сравнение трех аэропортов](reports/RESULTS.md)

- `notebooks/01_seasonal_days.ipynb`: погода и подтвержденные дни
- `notebooks/02_seasonal_training.ipynb`: выполненное сравнение моделей и влияние числа дней
- `reports/RESULTS.md`: итог и сравнение трех аэропортов
- `reports/csv/`: численные результаты отдельно от выводов
- `figures/`: графики
- `data/`, `artifacts/`: локальные источники и модели, исключены из Git

## Воспроизведение

Из корня репозитория:

```powershell
python -m pip install -r research/eddf-seasonal-learning/requirements.txt
python research/eddf-seasonal-learning/scripts/fetch_geometry.py
python research/eddf-seasonal-learning/scripts/select_weather.py 2025
python research/eddf-seasonal-learning/scripts/collect_seasons.py
python research/eddf-seasonal-learning/scripts/verify_results.py
```

Сбор использует подходящие сохраненные суточные архивы LFPG и EGLL, недостающие скачивает из [ADSB.lol](https://github.com/adsblol/globe_history_2025). Каждый архив проверяется по размеру и SHA-256; повторный запуск использует готовые траектории и примеры. METAR: [IEM](https://mesonet.agron.iastate.edu/request/download.phtml). Геометрия: [OurAirports](https://github.com/davidmegginson/ourairports-data). Сбор требует больших суточных архивов; обучение и оформление можно повторять отдельно по сохраненным данным.

Только обучение: `python research/eddf-seasonal-learning/scripts/train_models.py`. Обновление ноутбуков и отчета: `python research/eddf-seasonal-learning/scripts/build_training_notebook.py`.

## Результат

Собрано 15 дней, 2 634 пригодных захода и 96 714 наблюдаемых пар. На 540 заходах трех тестовых дней общая Polynomial Ridge дала MAE 20.87 ft, Ridge 22.23 ft, baseline 72.60 ft. Снижение к baseline 71.3%, к Ridge 6.1%. Общая Polynomial Ridge лучше в каждом сезонном тесте; при обучении только на осенних днях линейная Ridge лучше. Вывод ограничен выбранными днями и проверенными моделями.

Ноутбуки выполнены; итоговые MAE воспроизведены сохраненными моделями, высоты независимо сверены с исходными наблюдениями и привязкой EDDF 364 ft. [Итоговый отчет](reports/RESULTS.md), [выполненное обучение](notebooks/02_seasonal_training.ipynb).
