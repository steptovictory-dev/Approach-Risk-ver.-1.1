"""Оформить результат EDDF без предположения о победителе"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]


def run():
    final = pd.read_csv(BASE / 'reports/csv/final_table.csv', index_col=0)
    days = pd.read_csv(BASE / 'reports/csv/selected_days.csv')
    baseline = final.loc['Constant vertical rate', 'all']
    winner = final['all'].idxmin()
    best = final.loc[winner, 'all']
    models = ['Constant vertical rate', 'Ridge', 'Polynomial Ridge']
    comparison = pd.DataFrame({airport: pd.read_csv(BASE.parent / folder / 'reports/csv/final_table.csv', index_col=0).loc[models, 'all']
        for airport, folder in [('LFPG 09L', 'lfpg-seasonal-learning'), ('EGLL 09L', 'egll-seasonal-learning')]})
    comparison['EDDF 07L'] = final.loc[models, 'all']
    comparison.to_csv(BASE / 'reports/csv/airport_comparison.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True)
    for ax, airport in zip(axes, comparison.columns):
        values = comparison[airport]
        ax.barh(models, values, color=['#ed8975', '#92aaa2', '#33896c'])
        ax.invert_yaxis()
        ax.set(title=airport, xlabel='MAE, ft (lower is better)', xlim=(0, max(comparison.max()) * 1.15))
        ax.spines[['top', 'right']].set_visible(False)
        for y, value in enumerate(values):
            ax.text(value + .4, y, f'{value:.2f}', va='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(BASE / 'figures/03_three_airports.png', dpi=180)
    plt.close(fig)
    table = '\n'.join('| ' + str(model) + ' | ' + ' | '.join(f'{final.loc[model, s]:.2f}' for s in ['spring','summer','autumn','all']) + ' |' for model in models)
    compare = '\n'.join('| ' + model + ' | ' + ' | '.join(f'{comparison.loc[model, a]:.2f}' for a in comparison.columns) + ' |' for model in models)
    test_count = int(days.loc[days.split.eq('test'), 'approaches'].sum())
    learning = pd.read_csv(BASE / 'reports/csv/learning_curve.csv')
    autumn = learning.loc[learning.season.eq('autumn') & learning.train_days.eq(3)].set_index('model')
    approaches = f'{days.approaches.sum():,}'.replace(',', ' ')
    pairs = f'{days.examples.sum():,}'.replace(',', ' ')
    seasons = '\n'.join(f'- {label}: минимальная MAE у {final[s].idxmin()} ({final[s].min():.2f} ft)'
                        for s, label in [('spring', 'Весна'), ('summer', 'Лето'), ('autumn', 'Осень')])
    text = f'''# Франкфурт: сезонная проверка Polynomial Ridge

EDDF 07L, 15 погожих дней 2025 года: по пять весной, летом и осенью. Пригодных заходов {approaches}; наблюдаемых пар {pairs}. Тест: {test_count} заходов трех отложенных дней.

## Итоговая таблица

MAE в ft, среднее сначала внутри захода, затем между заходами. Общие модели обучены на девяти днях; одинаковые тестовые дни для всех моделей. Параметры выбраны по валидации.

| Модель | Весна | Лето | Осень | Все сезоны |
| --- | ---: | ---: | ---: | ---: |
{table}

Наименьшая итоговая ошибка у **{winner}**: **{best:.2f} ft**, baseline **{baseline:.2f} ft**, снижение **{100*(1-best/baseline):.1f}%**. Общая Polynomial Ridge улучшает MAE линейной Ridge на **{100*(1-final.loc['Polynomial Ridge','all']/final.loc['Ridge','all']):.1f}%**.

{seasons}

## Сравнение трех аэропортов

| Модель | LFPG 09L | EGLL 09L | EDDF 07L |
| --- | ---: | ---: | ---: |
{compare}

![Сравнение трех аэропортов](../figures/03_three_airports.png)

Это отдельное обучение одинаковых семейств на каждом аэропорту. Перенос готовой модели не проверялся. Для EDDF намеренно оставлены только Ridge, Polynomial Ridge степени 2 и baseline; вывод о лучшей модели ограничен этим набором. Другие семейства, проверенные ранее в LFPG и EGLL, на EDDF не запускались.

## Число обучающих дней

![Обучение на одном, двух и трех днях сезона](../figures/02_learning.png)

На этом графике модели обучаются отдельно внутри сезона, в отличие от общей модели на девяти днях в итоговой таблице. Добавление дней не всегда улучшает качество. При обучении только на трех осенних днях Ridge дает {autumn.loc['Ridge','macro_mae_ft']:.2f} ft, Polynomial Ridge {autumn.loc['Polynomial Ridge','macro_mae_ft']:.2f} ft: здесь линейная модель лучше. Преимущество Polynomial Ridge по всем сезонным тестам относится к общей модели, обученной на девяти днях. Это различие не позволяет утверждать, что полиномиальная модель выигрывает при любом составе обучения.

## Условия и ограничения

Прогноз наблюдаемой барометрической высоты через 30 ± 3 секунды до прокси 300 ft. Высота аэродрома EDDF 364 ft; это общий аэродромный барометрический прокси, не высота над порогом 07L и не радиовысота. Признаки используют предыдущую историю. QNH берется из предыдущего METAR с условной задержкой доступности 15 минут, COR исключены как в LFPG. Доступность точного времени публикации обычных METAR архивом не доказана.

Погожим считается весь интервал 08:00–18:00 Europe/Berlin по прежнему фильтру, не полные сутки. Назначение 07L определяется по траектории, ветер используется только для очередности поиска. Геометрия OurAirports — текущий исследовательский справочник, историческая эксплуатационная применимость не подтверждена.

Один тестовый день на сезон ограничивает вывод. Выбор даты сделан по погоде, наличию архивов и числу пригодных заходов, без просмотра модельной ошибки. При выборе семейства по итоговому тесту результат остается исследовательским; доверительные интервалы и статистическая значимость не оценивались. Разные даты, объемы и правила COR ограничивают сравнение аэропортов. Борта могут повторяться между днями, их идентификаторы не являются признаками.

Это прогноз высоты, а не полной трехмерной траектории или ухода на второй круг. Больший объем обучения и более сложная модель не гарантируют улучшения во всех сезонах.

Источники: [Fraport: полосы](https://www.fraport.com/en/sustainability/dialog-with-neighbors/noise-and-air/flight-operations/runway-system-and-operating-hours.html), [ADSB.lol](https://github.com/adsblol/globe_history_2025), [IEM METAR](https://mesonet.agron.iastate.edu/request/download.phtml), [OurAirports](https://github.com/davidmegginson/ourairports-data).

[Даты и группы](csv/selected_days.csv), [подбор параметров](csv/model_search.csv), [таблица трех аэропортов](csv/airport_comparison.csv), [выполненный ноутбук](../notebooks/02_seasonal_training.ipynb). Проверка: `python research/eddf-seasonal-learning/scripts/verify_results.py`.
'''
    (BASE / 'reports/RESULTS.md').write_text(text, encoding='utf-8')
    print('EDDF report and three-airport figure ready')


if __name__ == '__main__':
    run()
