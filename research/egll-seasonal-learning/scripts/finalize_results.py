"""Сохранить выводы EGLL и сравнение двух аэропортов"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
LABELS = {'spring': 'Весна', 'summer': 'Лето', 'autumn': 'Осень', 'all': 'Все сезоны'}


def markdown(frame):
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def run():
    days = pd.read_csv(BASE / 'reports/csv/selected_days.csv')
    final = pd.read_csv(BASE / 'reports/csv/final_table.csv', index_col=0)
    paris = pd.read_csv(PROJECT / 'research/lfpg-seasonal-learning/reports/csv/final_table.csv', index_col=0)
    comparison = pd.DataFrame({'LFPG 09L': paris['all'], 'EGLL 09L': final['all']})
    comparison.to_csv(BASE / 'reports/csv/airport_comparison.csv')
    best = final.drop(index='Constant vertical rate')['all'].idxmin()
    paris_best = paris.drop(index='Constant vertical rate')['all'].idxmin()
    baseline = final.loc['Constant vertical rate', 'all']
    score = final.loc[best, 'all']
    text = f'''# Хитроу: сезонный прогноз высоты

15 погожих дней 2025 года, одна полоса EGLL 09L. Пригодных заходов: {days.approaches.sum()}; наблюдаемых пар: {days.examples.sum()}. В каждом сезоне три дня обучения, один валидации и один теста. Общая модель обучается на девяти днях. Даты выбирались по погоде и наличию данных, без просмотра ошибок моделей; сначала использовались уже загруженные архивы LFPG.

## Итоговая таблица

MAE в ft, сначала усреднение внутри каждого захода, затем между заходами. Меньше лучше.

{markdown(final.rename(columns=LABELS).reset_index())}

Наименьшая ошибка среди общих моделей у **{best}**: **{score:.2f} ft**, baseline **{baseline:.2f} ft**, снижение ошибки **{100*(1-score/baseline):.1f}%**. Это наблюдение на отложенных днях; параметры каждого семейства выбраны только по валидации.

## Сравнение с Парижем

{markdown(comparison.reset_index())}

На объединенном тесте LFPG минимальная ошибка у {paris_best}, на объединенном тесте EGLL у {best}. Победитель определяется по наблюдаемому результату этих отложенных дней; универсальная лучшая модель для аэропорта этим не установлена.

Модели обучены отдельно для каждого аэропорта. Это сравнение одинакового метода, а не проверка переноса модели LFPG на EGLL. Состав дней и число заходов различаются; меньшая MAE не доказывает, что один аэропорт проще другого.

## Что измеряется

Прогноз наблюдаемой барометрической высоты через 30 ± 3 секунды, до прокси 300 ft. Четыре простые модели учат поправку к постоянной вертикальной скорости. Признаки траектории используют только предыдущую историю; QNH берется из предыдущего архивного METAR с условной задержкой доступности 15 минут. Это не восстановление полной трехмерной траектории и не классификация ухода на второй круг.

## Ограничения

Погожий день означает весь интервал 08:00–18:00 Europe/London по принятому фильтру METAR. Один тестовый день на сезон, одна полоса, только погожие условия. Борта могут повторяться между днями, идентификаторы не входят в признаки. В EGLL включены архивные METAR с COR, которыми помечено большинство сообщений. Фактическое время публикации исправлений неизвестно; задержка 15 минут является допущением, поэтому историческая доступность QNH не доказана. В LFPG такие сообщения исключались; это различие источников ограничивает сравнение аэропортов. Геометрия OurAirports используется как исследовательский справочник, историческая эксплуатационная применимость не подтверждалась. Больше обучающих дней одновременно меняет объем и состав данных; универсального закона улучшения здесь нет.

Даты и размеры групп: [selected_days.csv](csv/selected_days.csv). Подбор параметров: [model_search.csv](csv/model_search.csv). Влияние числа дней: [learning_curve.csv](csv/learning_curve.csv). Выполненное сравнение: [ноутбук](../notebooks/02_seasonal_training.ipynb).
'''
    (BASE / 'reports/RESULTS.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    run()
