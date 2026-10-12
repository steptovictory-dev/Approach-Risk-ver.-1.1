"""Собрать и исполнить локальную инвентаризацию источников EGLL"""
from pathlib import Path
import nbformat as nbf
from execute_notebook import execute

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'airports/EGLL'
cells = []
def md(value):
    cells.append(nbf.v4.new_markdown_cell(value))
def code(value):
    cells.append(nbf.v4.new_code_cell(value))

md('''# EGLL – локальная доступность траекторий

**Этап 01 · 12 октября 2026**

Вопрос: содержат ли сохраненные локальные источники последовательные наблюдения, из которых можно построить пилот конечного захода EGLL?

Область аудита: исходная европейская таблица версии 1.0 и 11 CSV из прежнего манифеста траекторий. Проверяются фактическая схема, контрольные суммы и наличие координат в широком поисковом прямоугольнике вокруг EGLL. Это ограниченная инвентаризация, не полный поиск всех возможных источников и не разметка заходов.''')
code('''from pathlib import Path
import hashlib
import json
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from IPython.display import display

# Находим внешний исторический источник через общий реестр
BASE = Path.cwd()
if BASE.name == 'notebooks':
    BASE = BASE.parent
PROJECT = BASE.parents[1]
config = json.loads((PROJECT / 'configs/sources.json').read_text(encoding='utf-8'))
OLD = (PROJECT / config['airport_summary']).resolve().parents[2]
for name in ['reports', 'figures']:
    (BASE / name).mkdir(exist_ok=True)
RED, GREEN = '#D65A4A', '#00897B'
plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 10})
manifest_path = OLD / 'research/final-approach-dynamics/reports/00_inventory_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

# Читаем только метаданные Parquet без загрузки миллионов строк
parquet_path = OLD / 'europe/data/source/osn23_landings_merged.parquet.gzip'
with pq.ParquetFile(parquet_path) as parquet:
    schema = pd.DataFrame([{'field': field.name, 'type': str(field.type)}
                           for field in parquet.schema_arrow])
    parquet_rows = parquet.metadata.num_rows
schema.to_csv(BASE / 'reports/01_source_schema.csv', index=False)
print('Строк в исходном Parquet:', parquet_rows)
display(schema)''')
md('''## Проверка сохраненных временных рядов

Поисковый прямоугольник **51,2–51,8° N; 0,9–0,1° W** намеренно широк. Координаты в нем означают только пространственных кандидатов, включая возможные пролеты и соседние аэропорты. Нулевое число кандидатов относится только к проверенным файлам. Порог не используется для определения ВПП, захвата ILS или события GoA.

Контрольная сумма каждого CSV сравнивается с прежним манифестом. Метки GoA не выводятся из самого факта наличия траектории.''')
code('''rows = []
fields = []
fragments = []
for item in manifest['sources']:
    path = OLD / item['source']
    if not path.is_file():
        rows.append({'source': item['source'], 'status': 'Файл отсутствует'})
        continue
    # Проверяем происхождение до анализа содержимого
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    frame = pd.read_csv(path, dtype={'icao24': str})
    # Считаем пространственных кандидатов без присвоения аэропорта
    candidate = frame['lat'].between(51.2, 51.8) & frame['lon'].between(-0.9, -0.1)
    # Группируем кандидатов по файлу и борту без присвоения попытки захода
    for aircraft, group in frame.loc[candidate].groupby('icao24'):
        ordered = group.sort_values('time')
        gaps = ordered.time.diff()
        fragments.append({'source': item['source'], 'icao24': aircraft,
                          'rows': len(group),
                          'utc_start': str(pd.to_datetime(group.time.min(), unit='s', utc=True)),
                          'utc_end': str(pd.to_datetime(group.time.max(), unit='s', utc=True)),
                          'span_s': group.time.max() - group.time.min(),
                          'max_gap_s': gaps.max(),
                          'baroaltitude_min_raw': group.baroaltitude.min(),
                          'baroaltitude_max_raw': group.baroaltitude.max(),
                          'geoaltitude_min_raw': group.geoaltitude.min(),
                          'geoaltitude_max_raw': group.geoaltitude.max()})
    label_fields = [name for name in frame.columns
                    if name.lower() in {'goa', 'has_ga', 'go_around', 'go_arounds'}]
    rows.append({'source': item['source'], 'status': 'Проверен',
                 'sha256': digest, 'matches_manifest': digest == item['sha256'],
                 'rows': len(frame), 'candidate_rows': int(candidate.sum()),
                 'candidate_aircraft': frame.loc[candidate, 'icao24'].nunique(),
                 'label_fields': ', '.join(label_fields) or 'Отсутствуют'})
    for name in frame.columns:
        fields.append({'source': item['source'], 'field': name,
                       'nonmissing_rows': int(frame[name].notna().sum())})
inventory = pd.DataFrame(rows)
inventory.to_csv(BASE / 'reports/01_local_trajectory_inventory.csv', index=False)
pd.DataFrame(fields).to_csv(BASE / 'reports/01_trajectory_fields.csv', index=False)
display(inventory[['source', 'status', 'matches_manifest', 'rows',
                   'candidate_rows', 'candidate_aircraft', 'label_fields']])
assert inventory['status'].eq('Проверен').all(), 'Инвентаризация неполна'
assert inventory['matches_manifest'].all(), 'Исходники изменились относительно манифеста'
print('Проверено файлов:', len(inventory))
print('Строк в поисковом прямоугольнике:', int(inventory.candidate_rows.sum()))''')
md('''## Характер пространственных фрагментов

Группы ниже определены по файлу и ICAO24. Это не выделенные попытки захода: временной размах может включать разрывы, а файлы могут перекрываться. Высоты показаны в исходных значениях; система отсчета и единицы должны быть подтверждены по источнику до применения границы 300 ft.''')
code('''# Сохраняем диапазоны времени и высоты без исправления исходных значений
fragment_table = pd.DataFrame(fragments)
fragment_table.to_csv(BASE / 'reports/01_candidate_fragments.csv', index=False)
display(fragment_table)
print('Групп файл / ICAO24:', len(fragment_table))
print('Уникальных ICAO24:', fragment_table.icao24.nunique())''')
code('''# Сравниваем объем файлов с пространственными кандидатами
labels = [str(i + 1) + ': ' + Path(p).parent.name for i, p in enumerate(inventory.source)]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].barh(labels, inventory.rows, color=GREEN)
axes[0].set(title='Объем сохраненных CSV', xlabel='Строки')
axes[1].barh(labels, inventory.candidate_rows, color=RED)
axes[1].set(title='Кандидаты в районе EGLL', xlabel='Строки в поисковом прямоугольнике')
if inventory.candidate_rows.sum() == 0:
    axes[1].set_xlim(0, 1)
    axes[1].text(0.5, 0.5, 'Кандидаты отсутствуют', ha='center', transform=axes[1].transAxes)
for ax in axes:
    ax.invert_yaxis()
fig.tight_layout()
for ext in ['png', 'svg']:
    fig.savefig(BASE / f'figures/01_local_inventory.{ext}', bbox_inches='tight')
plt.show()''')
md('''## Интерпретация и следующий шаг

Проверены 11 файлов; все контрольные суммы совпали с прежним манифестом. В поисковой области EGLL найдены 239 строк в 7 файлах. Поля меток GoA в этих CSV отсутствуют. Пространственное совпадение само по себе не подтверждает конечный заход в аэропорт.

Кандидаты образуют 12 групп файл / ICAO24. Размах времени каждой группы – от 0 до 350 секунд, минимальное исходное значение барометрической высоты – 10668. В принятой для OpenSky системе единиц это 10,668 км, что указывает на высотные пролеты, а не конечный участок. Единицы высоты в метрах описаны в [официальной документации OpenSky](https://openskynetwork.github.io/opensky-api/rest.html); локальный скрипт прежнего проекта analyze_public_pilot.py также обозначает baroaltitude как метры. Это не высота над аэродромом.

Основной Parquet содержит flight_id, icao24, landing_time, attempt_times, ILS, n_attempts, GoA и метаданные C40. Его схема не содержит последовательного ряда координат, высоты, скорости и вертикальной скорости. Аэропортовые записи с метаданными C40 нельзя автоматически считать полными конечными траекториями.

Необходимо сопоставить результат пространственной проверки с наличием меток и покрытием конечного участка. До появления подходящих рядов нельзя обоснованно выбрать направление ВПП и проверить мониторинг до 300 ft.

В проверенном наборе пригодные ряды конечного захода EGLL не найдены. Этот вывод не распространяется на другие локальные файлы или внешние архивы.

Для следующего этапа требуется реестр источников с условиями доступа, датами, полями и возможностью связать наблюдения с попыткой захода. Массовая загрузка не входит в текущую инвентаризацию. Тема возможного доклада формируется после первого подтвержденного результата о качестве данных или поведении baseline.''')
nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
path = BASE / 'notebooks/01_local_data_inventory.ipynb'
execute(nb, path, BASE)
print(path)
