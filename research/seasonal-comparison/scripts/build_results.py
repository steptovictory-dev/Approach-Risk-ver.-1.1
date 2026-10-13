"""Общий отчет из сохраненных результатов двух сезонных экспериментов"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RESEARCH = BASE.parent
FIGURES = BASE / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)
MODELS = ['Constant vertical rate', 'Ridge', 'Polynomial Ridge', 'Huber', 'ElasticNet']
SEASONS = ['spring', 'summer', 'autumn']
frames = {airport: pd.read_csv(RESEARCH / folder / 'reports/csv/final_table.csv').set_index('model')
          for airport, folder in [('LFPG', 'lfpg-seasonal-learning'), ('EGLL', 'egll-seasonal-learning')]}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white'})
colors = ['#ed8975', '#92aaa2', '#33896c', '#92aaa2', '#92aaa2']
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
for ax, (airport, frame) in zip(axes, frames.items()):
    values = frame.loc[MODELS, 'all'].to_numpy()
    ax.barh(MODELS, values, color=colors)
    ax.invert_yaxis()
    ax.set_title(f'{airport} 09L | pooled training, held-out days')
    ax.set_xlabel('MAE, ft (lower is better)')
    ax.set_xlim(0, 64)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    for y, value in enumerate(values):
        ax.text(value + .5, y, f'{value:.2f}', va='center')
fig.tight_layout()
fig.savefig(FIGURES / '01_model_comparison.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
for ax, (airport, frame) in zip(axes, frames.items()):
    x = np.arange(3)
    for offset, model, color, label in [(-.25, MODELS[0], colors[0], 'Baseline'),
                                      (0, 'Ridge', '#92aaa2', 'Ridge'),
                                      (.25, 'Polynomial Ridge', '#33896c', 'Polynomial Ridge')]:
        values = frame.loc[model, SEASONS].to_numpy(dtype=float)
        bars = ax.bar(x + offset, values, width=.24, color=color, label=label)
        ax.bar_label(bars, fmt='%.1f', fontsize=9, padding=3)
    ax.set_xticks(x, ['Spring', 'Summer', 'Autumn'])
    ax.set_title(f'{airport} 09L | same pooled model across seasons')
    ax.set_ylim(0, 76)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
axes[0].set_ylabel('MAE, ft (lower is better)')
axes[1].legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(FIGURES / '02_seasonal_comparison.png', dpi=180)
plt.close(fig)

rows = []
for airport, frame in frames.items():
    baseline = frame.loc[MODELS[0], 'all']
    poly = frame.loc['Polynomial Ridge', 'all']
    ridge = frame.loc['Ridge', 'all']
    assert frame['all'].idxmin() == 'Polynomial Ridge'
    assert (frame.loc[MODELS[1:], SEASONS].lt(frame.loc[MODELS[0], SEASONS], axis=1)).all().all()
    rows.append(f'| {airport} 09L | {baseline:.2f} | {ridge:.2f} | **{poly:.2f}** | {(1-poly/baseline)*100:.1f}% | {(1-poly/ridge)*100:.1f}% |')
season_rows = []
for airport, frame in frames.items():
    for season, label in zip(SEASONS, ['Весна', 'Лето', 'Осень']):
        best = frame[season].idxmin()
        season_rows.append(f'| {airport} | {label} | {frame.loc[MODELS[0], season]:.2f} | {frame.loc["Polynomial Ridge", season]:.2f} | {best} | {frame.loc[best, season]:.2f} |')
model_rows = [f'| {model} | {frames["LFPG"].loc[model, "all"]:.2f} | {frames["EGLL"].loc[model, "all"]:.2f} |' for model in MODELS]
report = '''# Общий результат: сезонный прогноз высоты в Париже и Хитроу

**Polynomial Ridge дала наименьшую итоговую MAE в обоих сезонных исследованиях.** По сравнению с продолжением текущей вертикальной скорости ошибка ниже на 65.2% в Париже и на 72.4% в Хитроу. По сравнению с линейной Ridge выигрыш меньше: 14.3% и 5.4% соответственно. Это эмпирическое подтверждение на выбранных отложенных днях, а не доказательство универсального превосходства модели.

## Условия сравнения

LFPG 09L и EGLL 09L, 2025 год. Для каждого аэропорта взяты по пять погожих дней весны, лета и осени. Зима в общее сравнение не включена. Погожие условия проверялись по принятому фильтру METAR в дневном интервале 08:00–18:00 местного времени, а не круглосуточно.

В каждом сезоне три дня обучения, один день валидации и один тестовый день. Общие модели обучались на девяти днях каждого аэропорта отдельно. Все модели внутри аэропорта проверяются на одинаковых тестовых заходах. Параметры подбирались по валидации, тестовые дни не использовались для настройки параметров. Итоговый выбор лучшего семейства сделан после сравнения тестовых результатов: отдельного нового теста для подтверждения этого выбора пока нет.

Задача: прогноз наблюдаемой барометрической высоты через 30 ± 3 секунды до прокси 300 ft. Baseline продолжает текущую вертикальную скорость. Модели обучают поправку к этому прогнозу по истории движения. Polynomial Ridge добавляет квадраты и попарные взаимодействия признаков, затем применяет регуляризацию Ridge. Это умеренное усложнение, без нейросетей.

| Набор | Погожих дней | Заходов всего | Пар всего | Тестовых дней | Тестовых заходов |
| --- | ---: | ---: | ---: | ---: | ---: |
| LFPG 09L | 15 | 553 | 15 931 | 3 | 55 |
| EGLL 09L | 15 | 5 601 | 166 508 | 3 | 1 136 |

MAE сначала усредняется внутри захода, затем между заходами. Длинная траектория не получает больший вес только из-за количества точек. Итоговая MAE взвешена числом заходов, а не равным весом трех сезонов. Все значения в ft; меньше лучше.

## Итог по двум исследованиям

| Аэропорт | Baseline | Ridge | Polynomial Ridge | Снижение к baseline | Снижение к Ridge |
| --- | ---: | ---: | ---: | ---: | ---: |
''' + '\n'.join(rows) + '''

Снижение ошибки = 100 × (1 − MAE модели / MAE сравнения). Расчет выполнен по исходным значениям CSV, округление применяется только для показа.

![Сравнение всех моделей на объединенном тесте](figures/01_model_comparison.png)

| Модель | LFPG 09L | EGLL 09L |
| --- | ---: | ---: |
''' + '\n'.join(model_rows) + '''

## Проверка по сезонам

Здесь оцениваются общие модели, обученные на девяти днях, а не отдельные модели каждого сезона.

![Baseline, Ridge и Polynomial Ridge на сезонных тестах](figures/02_seasonal_comparison.png)

| Аэропорт | Сезон | Baseline | Polynomial Ridge | Лучшая модель сезона | Ее MAE |
| --- | --- | ---: | ---: | --- | ---: |
''' + '\n'.join(season_rows) + '''

Все четыре обученные модели лучше baseline во всех шести сравнениях «аэропорт × сезон». Polynomial Ridge лучшая в четырех из шести: во всех трех сезонах Хитроу и весной в Париже. Летом в Париже лучше ElasticNet, осенью Huber. В Париже осенью Polynomial Ridge немного уступает даже линейной Ridge: 16.32 против 16.09 ft.

## Вывод для выбора модели

**В обоих исследованиях переход от физического baseline к обучаемой поправке оправдан наблюдаемой точностью.** Polynomial Ridge — лучший общий кандидат среди четырех проверенных семейств для этих данных и условий. Ее выигрыш над линейной Ridge составляет 2.87 ft в Париже и 0.89 ft в Хитроу.

Формулировка «в каждом случае стоит усложнять модель» слишком широкая. Здесь подтверждается польза обучения относительно baseline в обоих аэропортах, но дополнительная нелинейность полезна не в каждом сезоне. Baseline стоит сохранить как контрольный ориентир; Polynomial Ridge можно выбрать как рабочую модель следующего исследования и проверить на новых днях без повторного подбора по ним.

## Границы результата

- Один тестовый день на сезон: много заходов не заменяет разнообразия независимых дней. Статистическая значимость различий и доверительные интервалы здесь не оценивались
- Аэропорты имеют разные даты и размеры выборки. Меньшая ошибка Хитроу не доказывает, что его заходы проще. Перенос готовой модели между аэропортами не проверялся
- В EGLL использованы архивные исправленные METAR (COR) с условной задержкой доступности 15 минут. Реальное время публикации исправлений неизвестно. В LFPG COR исключались; это различие ограничивает сопоставимость
- Геометрия полос используется как исследовательский справочник; историческая эксплуатационная применимость отдельно не подтверждена
- Результат относится к краткосрочному прогнозу высоты при погожих условиях на 09L, а не ко всей трехмерной траектории, плохой погоде или прогнозированию ухода на второй круг

## Источники и воспроизведение

Исходные результаты: [LFPG](../lfpg-seasonal-learning/reports/RESULTS.md), [EGLL](../egll-seasonal-learning/reports/RESULTS.md). Числа для графиков: [LFPG final_table.csv](../lfpg-seasonal-learning/reports/csv/final_table.csv), [EGLL final_table.csv](../egll-seasonal-learning/reports/csv/final_table.csv). Даты и разделение: [LFPG selected_days.csv](../lfpg-seasonal-learning/reports/csv/selected_days.csv), [EGLL selected_days.csv](../egll-seasonal-learning/reports/csv/selected_days.csv).

Отчет и графики собираются из сохраненных CSV без нового обучения: `python research/seasonal-comparison/scripts/build_results.py`. Скрипт проверяет победителя итоговой таблицы и преимущество всех четырех моделей над baseline в каждом сезонном тесте.
'''
(BASE / 'RESULTS.md').write_text(report, encoding='utf-8')
print('Built RESULTS.md and two figures; checked pooled winners and six seasonal baseline comparisons')
