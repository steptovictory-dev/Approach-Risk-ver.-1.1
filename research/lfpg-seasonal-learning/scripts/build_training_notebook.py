"""Выполнить представление результатов простых сезонных моделей"""
from pathlib import Path
import sys
import nbformat as nbf

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parents[1] / 'scripts'))
from execute_notebook import execute
sys.stdout.reconfigure(encoding='utf-8')
cells = [nbf.v4.new_markdown_cell(
    '# Простые модели и сезонное обучение\n\n'
    'LFPG 09L, погожие дни 2025 года. Ridge, Huber, ElasticNet и Polynomial Ridge '
    'учат поправку к прогнозу с постоянной вертикальной скоростью. '
    'Горизонт 30 ± 3 секунды, девять прежних причинных признаков, наблюдения до '
    'барометрического прокси 300 ft. Меток ухода на второй круг здесь нет.\n\n'
    'Последний выбранный день каждого сезона оставлен для теста, предыдущий '
    'для валидации, остальные для обучения. Идентификатор борта не является признаком; '
    'борта могут повторяться между днями. Параметры выбираются по валидации. '
    'Каждый заход имеет одинаковый суммарный обучающий вес')]
cells.append(nbf.v4.new_code_cell('''from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import sys
base = Path.cwd()
if not (base / 'scripts/table_style.py').is_file():
    base = base.parent
sys.path.insert(0, str(base / 'scripts'))
from table_style import display_table
days = pd.read_csv(base / 'reports/csv/selected_days.csv')
metrics = pd.read_csv(base / 'reports/csv/model_metrics.csv')
learning = pd.read_csv(base / 'reports/csv/learning_curve.csv')
table = pd.read_csv(base / 'reports/csv/final_table.csv', index_col=0)
season_labels = {'winter': 'Зима', 'spring': 'Весна', 'summer': 'Лето', 'autumn': 'Осень'}
seasons = [s for s in season_labels if s in table.columns]
labels = [season_labels[s] for s in seasons]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
display_table(display, days.groupby(['season', 'split']).agg(Дней=('day', 'size'),
             Заходов=('approaches', 'sum'), Пар=('examples', 'sum')))
'''))
cells.append(nbf.v4.new_markdown_cell(
    '## Итоговая таблица\n\n'
    'Обучение на всех доступных обучающих днях выбранных сезонов. '
    'Ошибка MAE усредняется сначала внутри каждого захода, затем между заходами; '
    'единица измерения ft. Проверочные дни одинаковы для всех моделей'))
cells.append(nbf.v4.new_code_cell('''shown = table.rename(columns=dict(zip(seasons + ['all'], labels + ['Все сезоны'])))
display_table(display, shown.style.format('{:.2f}').highlight_min(axis=0, color='#B6E66B'))
'''))
cells.append(nbf.v4.new_markdown_cell(
    '## Влияние числа дней\n\n'
    'Четыре модели используют одни и те же проверочные дни при добавлении '
    'обучающих дней по порядку дат. Берем все пригодные примеры включенных дней. '
    'Это первое сравнение: одновременно растут число примеров и разнообразие дней. '
    'Параметры фиксируются после выбора по валидации полного обучающего набора сезона'))
cells.append(nbf.v4.new_code_cell('''fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
for ax, season, label in zip(axes.flat, seasons, labels):
    subset = learning.loc[learning.season.eq(season) & learning.converged]
    for model, color in [('Ridge', '#00897B'), ('Polynomial Ridge', '#D65A4A'),
                         ('Huber', '#25343B'), ('ElasticNet', '#829198')]:
        values = subset.loc[subset.model.eq(model)].sort_values('train_days')
        ax.plot(values.train_days, values.macro_mae_ft, marker='o', color=color, label=model)
    ax.set(title=label, xlabel='Обучающих дней', ylabel='MAE, ft')
    ax.set_xticks(sorted(subset.train_days.unique()))
    ax.grid(axis='y', alpha=.15)
axes[0, 0].legend(frameon=False)
for ax in list(axes.flat)[len(seasons):]:
    ax.set_visible(False)
fig.suptitle('Добавление обучающих дней')
fig.tight_layout()
fig.savefig(base / 'figures/02_learning.png', dpi=180, bbox_inches='tight')
plt.show()
display_table(display, learning[['season', 'model', 'train_days', 'train_approaches', 'macro_mae_ft']]
        .style.format({'macro_mae_ft': '{:.2f}'}).hide(axis='index'))
'''))
import numpy as np
import pandas as pd

final = pd.read_csv(BASE / 'reports/csv/final_table.csv', index_col=0)
learning = pd.read_csv(BASE / 'reports/csv/learning_curve.csv')
winner = final.drop(index='Constant vertical rate')['all'].idxmin()
baseline = final.loc['Constant vertical rate', 'all']
best = final.loc[winner, 'all']
lines = [f'Общая модель {winner}: MAE {best:.2f} ft против {baseline:.2f} ft '
         f'у прогноза с постоянной вертикальной скоростью '
         f'(снижение ошибки на {100 * (1 - best / baseline):.1f}%).']
labels = {'spring': 'Весна', 'summer': 'Лето', 'autumn': 'Осень'}
for season, group in learning.groupby('season', sort=False):
    monotonic = sum(np.diff(values.sort_values('train_days').macro_mae_ft).max() <= 0
                    for _, values in group.groupby('model'))
    lines.append(f'{labels.get(season, season)}: при добавлении 1, 2 и 3 дней '
                 f'ошибка последовательно снижалась у {monotonic} из 4 моделей.')
lines.append('Больше дней не гарантирует меньшую ошибку. Здесь одновременно меняются '
             'размер обучающего набора и состав дней; в каждом сезоне есть только один '
             'тестовый день. Результаты относятся к краткосрочному прогнозу высоты, '
             'а не к прогнозированию ухода на второй круг.')
cells.append(nbf.v4.new_markdown_cell('## Выводы\n\n' + '\n\n'.join(lines)))
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
execute(notebook, BASE / 'notebooks/02_seasonal_training.ipynb', BASE)
print('Сезонный ноутбук обучения выполнен')
if {'spring', 'summer', 'autumn'}.issubset(final.columns):
    import build_days_notebook
    from finalize_results import run as finalize
    finalize()
