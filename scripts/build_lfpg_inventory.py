"""Собрать и исполнить локальную инвентаризацию источников LFPG"""
from pathlib import Path
import nbformat as nbf
from execute_notebook import execute

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'airports/LFPG'
cells = []
def md(value):
    cells.append(nbf.v4.new_markdown_cell(value))
def code(value):
    cells.append(nbf.v4.new_code_cell(value))

md('''# LFPG – локальная доступность траекторий

**Этап 01 · 12 октября 2026**

Вопрос: содержат ли сохраненные локальные источники последовательные наблюдения, из которых можно построить пилот конечного захода LFPG?

Область аудита: исходная европейская таблица версии 1.0 и 11 CSV из прежнего манифеста траекторий. Проверяются фактическая схема, контрольные суммы и наличие координат в широком поисковом прямоугольнике вокруг LFPG. Это ограниченная инвентаризация, не полный поиск всех возможных источников и не разметка заходов.''')
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

Поисковый прямоугольник **48,7–49,3° N; 2,0–3,0° E** намеренно широк. Координаты в нем означают только пространственных кандидатов, включая возможные пролеты и соседние аэропорты. Нулевое число кандидатов относится только к проверенным файлам. Порог не используется для определения ВПП, захвата ILS или события GoA.

Контрольная сумма каждого CSV сравнивается с прежним манифестом. Метки GoA не выводятся из самого факта наличия траектории.''')
code('''rows = []
fields = []
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
    candidate = frame['lat'].between(48.7, 49.3) & frame['lon'].between(2.0, 3.0)
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
code('''# Сравниваем объем файлов с пространственными кандидатами
labels = [str(i + 1) + ': ' + Path(p).parent.name for i, p in enumerate(inventory.source)]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].barh(labels, inventory.rows, color=GREEN)
axes[0].set(title='Объем сохраненных CSV', xlabel='Строки')
axes[1].barh(labels, inventory.candidate_rows, color=RED)
axes[1].set(title='Кандидаты в районе LFPG', xlabel='Строки в поисковом прямоугольнике')
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

Проверены 11 файлов; все контрольные суммы совпали с прежним манифестом. В широком поисковом прямоугольнике LFPG найдено 0 строк. Поля меток GoA в этих CSV отсутствуют.

Основной Parquet содержит flight_id, icao24, landing_time, attempt_times, ILS, n_attempts, GoA и метаданные C40. Его схема не содержит последовательного ряда координат, высоты, скорости и вертикальной скорости. Аэропортовые записи с метаданными C40 нельзя автоматически считать полными конечными траекториями.

Сохраненные пилоты не дают данных для эксперимента LFPG. Необходимо изучить происхождение основной таблицы и возможность получить ограниченный пилот временных рядов для выбранных полетов. До появления рядов нельзя обоснованно выбрать направление ВПП и проверить мониторинг до 300 ft.

Для следующего этапа требуется реестр источников с условиями доступа, датами, полями и возможностью связать наблюдения с попыткой захода. Массовая загрузка не входит в текущую инвентаризацию. Тема возможного доклада формируется после первого подтвержденного результата о качестве данных или поведении baseline.''')
nb = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
path = BASE / 'notebooks/01_local_data_inventory.ipynb'
execute(nb, path, BASE)
print(path)
