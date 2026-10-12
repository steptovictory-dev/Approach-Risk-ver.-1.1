"""Выполнить обзор погоды перед сезонным обучением"""
from pathlib import Path
import sys
import nbformat

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parents[1] / 'scripts'))
from execute_notebook import execute
sys.stdout.reconfigure(encoding='utf-8')
cells = [
    nbformat.v4.new_markdown_cell(
        '# Погожие дни для сезонного эксперимента\n\n'
        'LFPG 09L, только 2025 год. Ищем 9 зимних дней и по 10 дней других сезонов. '
        'Месяц внутри сезона любой. Здесь показан предварительный отбор по METAR; '
        'фактические заходы на полосу проверяются отдельно.\n\n'
        'Дневной интервал 08:00–18:00 по времени Парижа должен целиком проходить '
        'при подходящих условиях: видимость от 10 км, BKN/OVC/VV от 3000 ft, '
        'ветер до 15 kt, порывы до 25 kt, температура от -10 до +30 C, '
        'без осадков, тумана, гроз и сдвига ветра. Неизвестные условия исключены. '
        'Это исследовательский отбор, а не эксплуатационные минимумы'),
    nbformat.v4.new_code_cell('''from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

base = Path.cwd()
candidates = pd.read_csv(base / 'data/weather_candidates_2025.csv')
seasons = ['winter', 'spring', 'summer', 'autumn']
labels = ['Зима', 'Весна', 'Лето', 'Осень']
counts = candidates.groupby('season').size().reindex(seasons, fill_value=0)
table = pd.DataFrame({'Сезон': labels, 'Погодных кандидатов': counts.to_numpy(),
                      'Нужно подтвержденных дней': [9, 10, 10, 10]})
display(table.style.hide(axis='index'))
'''),
    nbformat.v4.new_code_cell('''plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'text.color': '#25343B', 'axes.labelcolor': '#25343B'})
fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(labels, counts.to_numpy(), color=['#D65A4A', '#00897B', '#00897B', '#00897B'], width=0.55)
ax.scatter(labels, [9, 10, 10, 10], color='#25343B', marker='D', s=40,
           zorder=3, label='Нужно подтвержденных дней')
ax.bar_label(bars, padding=5)
ax.set(title='Погодные кандидаты по сезонам, 2025', ylabel='Дней')
ax.set_ylim(0, counts.max() * 1.2)
ax.legend(frameon=False, loc='upper left')
fig.tight_layout()
(base / 'figures').mkdir(exist_ok=True)
fig.savefig(base / 'figures/01_weather_candidates.png', dpi=180, bbox_inches='tight')
plt.show()
'''),
    nbformat.v4.new_markdown_cell(
        'Для зимы погодных кандидатов десять. '
        'Проверка архива за 1 декабря нашла 69 геометрических заходов на 26L '
        'и ни одного на 09L в дневном интервале. Поэтому зимних дней остается '
        'максимум девять еще до проверки остальных кандидатов. '
        'По решению пользователя ищем девять зимних дней и по десять дней других сезонов. '
        'Числа в таблице пока не означают готовую обучающую выборку. '
        'Берем дни с десятками пригодных заходов на 09L.\n\n'
        'В дальнейшем сравниваем Ridge, Huber, ElasticNet и Polynomial Ridge второй степени '
        'с прогнозом по постоянной вертикальной скорости. '
        'При изменении числа обучающих дней проверочные дни сохраняем одинаковыми. '
        'Обучение пока не запускалось: еще проверяются заходы на 09L в зимних кандидатах')]
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
path = BASE / 'notebooks/01_seasonal_days.ipynb'
execute(notebook, path, BASE)
print(path)
