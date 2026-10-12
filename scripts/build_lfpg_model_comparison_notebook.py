"""Создать и исполнить презентационный эксперимент сравнения моделей LFPG"""
from pathlib import Path
import sys
import nbformat as nbf
from execute_notebook import execute

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / 'airports/LFPG'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    cells.append(nbf.v4.new_code_cell(source))


md('''# Конечный заход: какая модель уточняет прогноз?
## Париж, Шарль-де-Голль, LFPG, 09L

**Исследование 05, 15 января 2025, Approach Risk 1.1**

**37 кандидатов, 35 бортов, 633 пары наблюдений, горизонт 30 ± 3 с**

Продолжение текущей Vz уже дает физический ориентир, а Ridge снижает ошибку. Следующий вопрос: помогают ли робастная линейная модель, нелинейное ядро или ансамбль деревьев на той же группе?

Сначала фиксируем пространство моделей и выбор по validation. Затем сравниваем прогнозы и исследуем, как модели реагируют на изменение текущей вертикальной скорости. Высотная точность и обнаружение нестабильного захода остаются разными задачами.

**Статус:** исследовательское расширение. Результаты этих test-заходов уже известны из этапа 04. Повторная проверка на них не является новым независимым подтверждением. Подтверждение переносимости требует другого дня.

**Маршрут:** baseline → общий протокол → обучение → выбор на validation → реакция на Vz → сравнение на test → разбор заходов → вывод.''')
code('''from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from IPython.display import display
from IPython.display import HTML

BASE = Path.cwd()
if BASE.name == 'notebooks':
    BASE = BASE.parent
PROJECT = BASE.parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
from lfpg_training import event_errors, summarize
from lfpg_model_comparison import (make_model, fit_model, forecast, paired_intervals,
                                   current_rate_response)
REPORTS, FIGURES = BASE / 'reports', BASE / 'figures'
ARTIFACTS = BASE / 'artifacts/training_05'
ARTIFACTS.mkdir(parents=True, exist_ok=True)
plan_path = PROJECT / 'configs/lfpg_09l_model_comparison.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
prior = json.loads((REPORTS / '04_training_manifest.json').read_text(encoding='utf-8'))
inherited = json.loads((PROJECT / plan['inherited_protocol']).read_text(encoding='utf-8'))

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

examples_path = BASE / plan['examples']
cohort_path = BASE / plan['split']
assert digest(examples_path) == prior['examples_sha256']
for relative, expected in prior['inputs_sha256'].items():
    assert digest(BASE / relative) == expected
examples = pd.read_parquet(examples_path)
cohort = pd.read_csv(cohort_path)
FEATURES = plan['features']
assert FEATURES == inherited['features']
assert len(examples) == 633 and len(cohort) == 37 and cohort.icao24.nunique() == 35
for key in ['icao24', 'trace_id', 'event_id']:
    assert examples.groupby(key).split.nunique().eq(1).all()
assert examples.origin_metar_available_time.le(examples.origin_time).all()
assert examples.target_time.gt(examples.origin_time).all()
assert examples.horizon_s.between(27, 33).all()
assert np.allclose(examples.baseline_ft,
                   examples.height_ft + examples.vertical_rate_fps * examples.horizon_s)
assert not set(FEATURES) & {'target_height_ft', 'residual_ft', 'target_time', 'event_id', 'icao24', 'boundary_time'}
train = examples.loc[examples.split.eq('train')].copy()
validation = examples.loc[examples.split.eq('validation')].copy()
weights = 1 / train.groupby('event_id').event_id.transform('size')
weights *= len(train) / weights.sum()
assert np.allclose(weights.groupby(train.event_id).sum(), len(train) / train.event_id.nunique())

# Общая палитра продолжает оформление предыдущего эксперимента
RED, GREEN, INK, GREY = '#D65A4A', '#00897B', '#25343B', '#829198'
plt.rcParams.update({'figure.dpi': 140, 'font.size': 10, 'axes.titlesize': 13,
    'axes.labelcolor': INK, 'text.color': INK, 'axes.edgecolor': '#B7C3C8',
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.hashsalt': 'lfpg-model-comparison-05'})

def table(frame, decimals=2):
    formats = {col: '{:.0f}' if pd.api.types.is_integer_dtype(frame[col])
               else '{:.' + str(decimals) + 'f}'
               for col in frame.select_dtypes(include='number').columns}
    display(frame.style.format(formats).hide(axis='index')
        .set_properties(**{'text-align': 'left', 'padding': '8px'})
        .set_table_styles([{'selector': 'th', 'props': [('background-color', '#E8F3F1'),
                                                     ('color', INK), ('padding', '8px')]}]))

def panel(title, body, color=GREEN):
    display(HTML(f'<div style="border-left:5px solid {color};padding:14px 20px;'
        f'background:#F4F8F7;margin:12px 0;color:{INK}"><b>{title}</b><br>{body}</div>'))

def savefig(fig, name):
    fig.savefig(FIGURES / (name + '.png'), dpi=180, bbox_inches='tight')
    svg = FIGURES / (name + '.svg')
    fig.savefig(svg, bbox_inches='tight', metadata={'Date': None})
    svg.write_text(chr(10).join(line.rstrip() for line in
        svg.read_text(encoding='utf-8').splitlines()) + chr(10), encoding='utf-8')
    plt.show()

# Сохраняем протокол и контрольные суммы до запуска сетки моделей
frozen_plan = {**plan, 'plan_sha256': digest(plan_path),
    'examples_sha256': digest(examples_path), 'cohort_sha256': digest(cohort_path),
    'prior_manifest_sha256': digest(REPORTS / '04_training_manifest.json')}
(REPORTS / '05_comparison_protocol.json').write_text(
    json.dumps(frozen_plan, ensure_ascii=False, indent=2), encoding='utf-8')
print('Группа, признаки, веса и пространство моделей зафиксированы')''')
md(r'''## 01 Что показал baseline

Физический прогноз продолжает текущую барометрическую вертикальную скорость на заданный горизонт:

$$\widehat h_{CV}(t+\tau)=h_{Q(t)}(t)+V_{z,Q(t)}(t)\tau.$$

Высота и скорость согласованы по QNH, доступному в момент прогноза; скорость в формуле измеряется в ft/s. Цель – наблюдаемая будущая барометрическая высота на той же настройке, без интерполяции. Фактический горизонт 27–33 с наследуется из этапа 04.

На восьми test-кандидатах этапа 04 macro MAE составила **67.75 ft**, macro RMSE **96.17 ft**, построчный 95-й процентиль абсолютной ошибки **182.38 ft**. Ridge дала **23.71 ft** macro MAE. Эта поправка может ослаблять реакцию на текущую Vz; это проверяем отдельно.

Знаковое смещение baseline отрицательное: в среднем он прогнозировал высотный прокси ниже наблюдаемой цели. 633 строки принадлежат 37 заходам и не являются 633 независимыми полетами.''')
code('''# Показываем уже полученный результат этапа 04
previous_metrics = pd.read_csv(REPORTS / '04_test_metrics.csv')
table(previous_metrics[['model', 'events', 'macro_mae_ft', 'macro_rmse_ft',
                        'point_p95_abs_ft', 'macro_bias_ft']])
split_summary = examples.groupby('split').agg(examples=('event_id', 'size'),
    events=('event_id', 'nunique'), aircraft=('icao24', 'nunique'))
table(split_summary.reindex(['train', 'validation', 'test']).reset_index())
panel('Предел исходного результата',
      'Один день, одно направление ВПП, 8 test-заходов и 7 независимых адресов ICAO24. '
      'Это точность высотного прогноза, а не вероятность GoA')''')
md('''## 02 Что сравниваем и что фиксируем

Все обучаемые модели получают те же девять признаков и обучаются одной величине: ошибке физического прогноза. Итог – baseline плюс поправка в ft. Суммарный вес каждого train-захода одинаков. Масштабирование для линейных моделей и SVR обучается только на train.

Каждая семья имеет небольшую фиксированную сетку. Лучший вариант внутри семьи и общий выбранный прогноз определяются по **validation macro MAE**. Все варианты видят одинаковые данные; validation не добавляется к обучению. В бустинге отключена автоматическая остановка, чтобы внутри него не возникало случайного разделения коррелированных строк.

Дополнительный физический ориентир использует скорость изменения высоты между текущей и предыдущей точкой, уже сохраненную как причинный признак. Он не обучается и проверяет, не сводится ли выигрыш ML к замене текущей Vz прошлой высотной динамикой.

Документация методов: [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html), [Huber](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html), [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), [Extra Trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html), [Hist Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).

Результаты test уже просмотрены в предыдущем эксперименте. Здесь они служат общей исследовательской проверкой. Расширять сетки по этим результатам в этом запуске не будем.''')
code('''# Фиксируем число вариантов каждой семьи до обучения
roles = {'ridge': 'Линейная поправка с L2', 'elasticnet': 'Линейная поправка с L1 и L2',
    'huber': 'Линейная поправка с робастной потерей', 'svr': 'Нелинейное RBF-ядро',
    'randomforest': 'Усреднение деревьев', 'extratrees': 'Дополнительная случайность деревьев',
    'histgb': 'Последовательная поправка деревьями'}
family_table = pd.DataFrame([{'model': spec['name'], 'meaning': roles[spec['kind']],
    'candidates': len(ParameterGrid(spec['grid'])),
    'grid': json.dumps(spec['grid'], ensure_ascii=False)} for spec in plan['model_families']])
table(family_table)
print('Всего обучаемых вариантов:', family_table.candidates.sum())''')
md('''## 03 Обучение и выбор параметров

В каждой строке журнала – один реально обученный вариант. Сохраняем train и validation ошибки, параметры и предупреждения оптимизатора. Предупреждение о сходимости делает вариант непригодным для выбора в этом запуске; неудачный вариант не исчезает из журнала.

Проверка общей семьи не требует, чтобы более сложная модель победила. Физические ориентиры участвуют в итоговом выборе на validation наравне с обучаемыми прогнозами.''')
code('''# Перебираем фиксированную сетку без доступа к test в цикле обучения
selected_models, selection_rows, tuning_rows = {}, [], []
for spec in plan['model_families']:
    fitted = {}
    for index, variable in enumerate(ParameterGrid(spec['grid'])):
        parameters = {**spec['fixed'], **variable}
        model = make_model(spec['kind'], parameters, plan['seed'])
        started = time.perf_counter()
        with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit_model(model, train, FEATURES, weights)
            train_result = summarize(train, forecast(model, train, FEATURES), spec['name'])
            result = summarize(validation, forecast(model, validation, FEATURES), spec['name'])
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        result.update({'candidate_index': index,
            'parameters': json.dumps(parameters, sort_keys=True),
            'train_macro_mae_ft': train_result['macro_mae_ft'],
            'converged': converged, 'fit_seconds': time.perf_counter() - started,
            'warnings': ' | '.join(str(w.message) for w in caught)})
        tuning_rows.append(result)
        fitted[index] = model
    family_results = pd.DataFrame([row for row in tuning_rows if row['model'] == spec['name']])
    usable = family_results.loc[family_results.converged]
    assert len(usable), f"Нет сошедшихся вариантов {spec['name']}"
    best = usable.sort_values(['macro_mae_ft', 'candidate_index']).iloc[0]
    selected_models[spec['name']] = fitted[int(best.candidate_index)]
    selection_rows.append({'model': spec['name'], 'kind': spec['kind'],
        'candidate_index': int(best.candidate_index), 'parameters': json.loads(best.parameters),
        'validation_macro_mae_ft': float(best.macro_mae_ft)})
    print(f"{spec['name']}: validation MAE {best.macro_mae_ft:.2f} ft")

tuning = pd.DataFrame(tuning_rows)
tuning.to_csv(REPORTS / '05_validation_tuning.csv', index=False)
all_models = {name: name for name in plan['baselines']}
all_models.update(selected_models)
validation_metrics = pd.DataFrame([summarize(validation,
    forecast(model, validation, FEATURES), name) for name, model in all_models.items()])
validation_metrics = validation_metrics.sort_values(['macro_mae_ft', 'model']).reset_index(drop=True)
winner = str(validation_metrics.iloc[0].model)
best_ml = str(validation_metrics.loc[validation_metrics.model.isin(selected_models)].iloc[0].model)
selection = {'winner_by_validation': winner, 'best_ml_by_validation': best_ml,
    'selection_metric': plan['selection_metric'], 'test_previously_inspected': True,
    'refit_on_validation': False, 'selected_models': selection_rows,
    'plan_sha256': digest(plan_path), 'examples_sha256': digest(examples_path)}
(REPORTS / '05_model_selection.json').write_text(
    json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
validation_metrics.to_csv(REPORTS / '05_validation_metrics.csv', index=False)
joblib.dump({'models': selected_models, 'features': FEATURES, 'selection': selection},
            ARTIFACTS / 'selected_models.joblib')
table(validation_metrics[['model', 'events', 'macro_mae_ft', 'macro_rmse_ft', 'point_p95_abs_ft']])
panel('Выбор зафиксирован до повторной оценки test',
      f'Выбранный прогноз: {winner}. Лучший обучаемый вариант: {best_ml}. '
      'Validation определяет выбор, test показывает его поведение')''')
md('''## 04 Цена сложности на validation

Показаны только лучшие варианты каждой семьи, выбранные по validation. Разрыв train и validation помогает заметить подгонку; это диагностическое сравнение, не независимый тест. Объем данных определяется числом бортов и заходов.

Правая панель показывает все варианты сетки. Семья с большим числом проб получает больше возможностей подстроиться к validation. При семи validation-бортах такой выбор может быть неустойчивым.''')
code('''# Сопоставляем train и validation до просмотра новых прогнозов test
train_metrics = pd.DataFrame([summarize(train, forecast(model, train, FEATURES), name)
    for name, model in all_models.items()])
train_metrics.to_csv(REPORTS / '05_train_metrics.csv', index=False)
order = validation_metrics.model.tolist()
train_values = train_metrics.set_index('model').loc[order, 'macro_mae_ft']
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
y = np.arange(len(order))
axes[0].barh(y - .18, train_values, height=.34, color='#80CBC4', label='train')
axes[0].barh(y + .18, validation_metrics.macro_mae_ft, height=.34, color=GREEN, label='validation')
axes[0].set(yticks=y, yticklabels=order, xlabel='Macro MAE, ft',
            title='Лучший вариант каждой семьи')
axes[0].invert_yaxis()
axes[0].legend()
families = family_table.model.tolist()
for i, name in enumerate(families):
    values = tuning.loc[tuning.model.eq(name)]
    axes[1].scatter(values.macro_mae_ft, np.full(len(values), i),
                    color=GREEN, s=35, alpha=.65)
    best = validation_metrics.loc[validation_metrics.model.eq(name), 'macro_mae_ft'].iloc[0]
    axes[1].scatter(best, i, marker='D', color=RED, s=55)
axes[1].set(yticks=np.arange(len(families)), yticklabels=families,
            xlabel='Validation macro MAE, ft', title='Все варианты; ромб – выбор семьи')
axes[1].invert_yaxis()
fig.tight_layout()
savefig(fig, '05_validation_comparison')''')
md(r'''## 05 Реакция прогноза на текущую Vz

Проверяем локальную реакцию полной формулы прогноза при изменении только текущей барометрической Vz на ±1 ft/s. Остальные признаки оставляем неизменными. Это диагностика модели в исходных точках validation, а не физически согласованная имитация аномального захода.

$$S_{Vz}=\frac{\widehat h(V_z+\delta)-\widehat h(V_z-\delta)}{2\delta\tau}.$$

У продолжения текущей Vz отношение равно 1. Значение около 0 означает, что поправка почти подавляет ее прямое влияние. У прогноза по прошлой высотной динамике это влияние равно 0 по определению. Малое значение само по себе не доказывает, что модель пропустит уход, но объясняет, почему одной MAE недостаточно для выбора будущего индикатора риска.''')
code('''# Используем validation для диагностического изменения текущей Vz
response = pd.DataFrame([{'model': name, **current_rate_response(model, validation, FEATURES,
    plan['sensitivity_delta_vertical_rate_fps'])} for name, model in all_models.items()])
response.to_csv(REPORTS / '05_vz_response.csv', index=False)
table(response)
view = response.set_index('model').loc[order].reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
for i, row in view.iterrows():
    ax.plot([row.p10_vz_response_ratio, row.p90_vz_response_ratio], [i, i], color=GREY, lw=3)
    ax.scatter(row.median_vz_response_ratio, i, c=GREEN, s=55)
ax.axvline(0, color=GREY, ls='--', lw=1)
ax.axvline(1, color=RED, ls='--', lw=1)
ax.set(yticks=np.arange(len(view)), yticklabels=view.model,
       xlabel='Реакция относительно физического продолжения Vz',
       title='Медиана и 10–90 процентили по точкам validation')
ax.invert_yaxis()
fig.tight_layout()
savefig(fig, '05_vz_response')''')
md('''## 06 Повторное сравнение на test

Параметры всех семейств и выбранный прогноз уже записаны. Теперь вычисляем результаты на прежних восьми test-заходах, принадлежащих семи ICAO24. Модели не переобучаются.

Основная метрика – среднее MAE заходов с равным весом каждого. Macro RMSE, построчный 95-й процентиль абсолютной ошибки и смещение показывают другие стороны результата. Положительная ошибка означает завышение прогнозируемой высоты. Рейтинг test является описательным; выбор модели сохраняется по validation.''')
code('''# Повторно оцениваем общую test-группу после сохранения выбора
assert (REPORTS / '05_model_selection.json').exists()
test = examples.loc[examples.split.eq('test')].copy()
assert test.event_id.nunique() == 8 and test.icao24.nunique() == 7
test_forecasts = {name: forecast(model, test, FEATURES) for name, model in all_models.items()}
test_metrics = pd.DataFrame([summarize(test, test_forecasts[name], name) for name in all_models])
test_metrics = test_metrics.sort_values(['macro_mae_ft', 'model']).reset_index(drop=True)
test_metrics['selected_on_validation'] = test_metrics.model.eq(winner)
baseline_mae = float(test_metrics.loc[test_metrics.model.eq('Constant vertical rate'), 'macro_mae_ft'].iloc[0])
test_metrics['improvement_vs_current_vz_pct'] = 100 * (1 - test_metrics.macro_mae_ft / baseline_mae)
test_metrics.to_csv(REPORTS / '05_test_metrics.csv', index=False)
per_event = pd.concat([event_errors(test, test_forecasts[name], name)
    for name in all_models], ignore_index=True)
per_event.to_csv(REPORTS / '05_test_event_metrics.csv', index=False)
prediction_rows = pd.concat([test.assign(model=name, prediction_ft=test_forecasts[name])
    for name in all_models], ignore_index=True)
prediction_rows.to_parquet(BASE / 'data/2025-01-15/training_05_predictions.parquet', index=False)

# Ridge этапа 05 обязана воспроизвести уже опубликованную модель этапа 04
old_ridge = joblib.load(BASE / 'artifacts/training_04/altitude_residual_ridge.joblib')
assert np.allclose(test_forecasts['Ridge'],
    test.baseline_ft + old_ridge['model'].predict(test[FEATURES]))
table(test_metrics[['model', 'selected_on_validation', 'macro_mae_ft', 'macro_rmse_ft',
                    'point_p95_abs_ft', 'macro_bias_ft', 'improvement_vs_current_vz_pct']])

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
test_order = test_metrics.model.tolist()
bars = axes[0].barh(test_order, test_metrics.macro_mae_ft,
    color=[GREEN if name == winner else '#80CBC4' if name in selected_models else RED
           for name in test_order])
for bar, value in zip(bars, test_metrics.macro_mae_ft):
    axes[0].text(value + 1, bar.get_y() + bar.get_height()/2, f'{value:.2f}', va='center')
axes[0].set(xlabel='Macro MAE, ft', title='Test: темный зеленый – выбор validation',
            xlim=(0, test_metrics.macro_mae_ft.max()*1.25))
axes[0].invert_yaxis()
matrix = per_event.pivot(index='event_id', columns='model', values='mae_ft')[test_order]
error_colors = LinearSegmentedColormap.from_list('approach_errors', ['#E8F3F1', GREEN, RED])
image = axes[1].imshow(matrix.to_numpy(), aspect='auto', cmap=error_colors, vmin=0)
axes[1].set(xticks=np.arange(len(test_order)), xticklabels=test_order,
    yticks=np.arange(len(matrix)), yticklabels=[name[:8] for name in matrix.index],
    title='Каждый заход и каждый прогноз')
plt.setp(axes[1].get_xticklabels(), rotation=50, ha='right', fontsize=8)
fig.colorbar(image, ax=axes[1], label='MAE отдельного захода, ft')
fig.tight_layout()
savefig(fig, '05_test_comparison')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
comparison = pd.concat([train_metrics.assign(split='train'),
    validation_metrics.assign(split='validation'), test_metrics.assign(split='test')], ignore_index=True)
comparison.to_csv(REPORTS / '05_split_metrics.csv', index=False)
for shift, split, color in [(-.25, 'train', '#80CBC4'), (0, 'validation', GREEN), (.25, 'test', RED)]:
    values = comparison.loc[comparison.split.eq(split)].set_index('model').loc[order, 'macro_mae_ft']
    axes[0].barh(np.arange(len(order)) + shift, values, height=.23, color=color, label=split)
axes[0].set(yticks=np.arange(len(order)), yticklabels=order,
            xlabel='Macro MAE, ft', title='Ошибка по частям одной группы')
axes[0].invert_yaxis()
axes[0].legend(fontsize=8)
for name, color in [('Constant vertical rate', RED), ('Ridge', GREY), (winner, GREEN)]:
    if name == winner and name in {'Constant vertical rate', 'Ridge'}:
        continue
    absolute = np.sort(np.abs(test_forecasts[name] - test.target_height_ft.to_numpy()))
    axes[1].step(absolute, np.arange(1, len(absolute)+1)/len(absolute), label=name, color=color)
axes[1].set(xlabel='Абсолютная ошибка, ft', ylabel='Доля строк',
            title='Построчное распределение, без независимости строк', ylim=(0, 1.02))
axes[1].legend(fontsize=8)
fig.tight_layout()
savefig(fig, '05_generalization_and_errors')''')
md('''## 07 Парное сравнение целыми бортами

В каждой bootstrap-выборке повторно выбираются ICAO24; все заходы одного выбранного борта остаются вместе. Для каждой модели сравниваем MAE с текущей Vz и отдельно с Ridge. Отрицательная разность означает меньшую ошибку.

Интервалы условны для фиксированных моделей и этой группы. Они не учитывают поиск по семействам, множественные сравнения, неопределенность выбора на validation или изменчивость других дней. Это диагностические интервалы, а не основание объявить модель универсально лучшей.''')
code('''# Сохраняем обе точки сравнения с одинаковыми единицами повторного выбора
intervals = pd.concat([paired_intervals(per_event, reference, plan['seed'] + 5,
    plan['bootstrap_repetitions']) for reference in ['Constant vertical rate', 'Ridge']],
    ignore_index=True)
intervals.to_csv(REPORTS / '05_paired_intervals.csv', index=False)
ridge_intervals = intervals.loc[intervals.reference.eq('Ridge')].set_index('model').loc[order].reset_index()
table(ridge_intervals[['model', 'delta_macro_mae_ft', 'interval_2_5_ft',
                       'interval_97_5_ft', 'independent_aircraft']])
fig, ax = plt.subplots(figsize=(10, 5))
for i, row in ridge_intervals.iterrows():
    ax.plot([row.interval_2_5_ft, row.interval_97_5_ft], [i, i], color=GREY, lw=3)
    ax.scatter(row.delta_macro_mae_ft, i,
               color=GREEN if row.delta_macro_mae_ft < 0 else RED, s=55)
ax.axvline(0, color=INK, lw=1)
ax.set(yticks=np.arange(len(order)), yticklabels=order,
       xlabel='Разность macro MAE относительно Ridge, ft',
       title='Парный bootstrap семью бортами, условный интервал 95%')
ax.invert_yaxis()
fig.tight_layout()
savefig(fig, '05_paired_vs_ridge')''')
md('''## 08 Как ведут себя отдельные заходы

Для иллюстрации выбираем три захода по ошибке исходного физического baseline: максимальную, среднюю по рангу и минимальную. Так выбор примеров не зависит от того, на каком заходе новая модель выглядит удачнее.

По оси времени показано время целевой точки, по вертикали – будущий высотный прокси. Черные точки – наблюдаемые цели на QNH момента соответствующего прогноза. Каждая цветная точка – прогноз, сделанный за 27–33 с до цели. Линии только соединяют последовательные значения.''')
code('''# Фиксируем правило выбора примеров по физическому baseline
baseline_events = per_event.loc[per_event.model.eq('Constant vertical rate')].sort_values('mae_ft')
case_ids = baseline_events.iloc[[len(baseline_events)-1, len(baseline_events)//2, 0]].event_id.tolist()
case_rows = []
fig, axes = plt.subplots(3, 1, figsize=(12, 11))
shown = list(dict.fromkeys(['Constant vertical rate', 'Ridge', winner]))
colors = {'Constant vertical rate': RED, 'Ridge': GREY, winner: GREEN}
for ax, event_id in zip(axes, case_ids):
    mask = test.event_id.eq(event_id).to_numpy()
    group = test.loc[mask]
    x = (group.target_time - group.origin_time.min()).to_numpy()
    ax.scatter(x, group.target_height_ft, c=INK, s=24, label='Наблюдаемая цель', zorder=4)
    for name in shown:
        ax.plot(x, test_forecasts[name][mask], '.-', c=colors[name], label=name, lw=1.5)
    event_values = per_event.loc[per_event.event_id.eq(event_id)].set_index('model')
    ax.set(title=f"{event_id[:8]}: baseline {event_values.loc['Constant vertical rate', 'mae_ft']:.1f} ft, "
                 f"{winner} {event_values.loc[winner, 'mae_ft']:.1f} ft",
           xlabel='Время целевой точки от первого прогноза, с', ylabel='Высотный прокси, ft')
    ax.legend(fontsize=8, ncol=2)
    case_rows.append({'event_id': event_id,
        'baseline_mae_ft': float(event_values.loc['Constant vertical rate', 'mae_ft']),
        'selected_mae_ft': float(event_values.loc[winner, 'mae_ft']), 'selected_model': winner})
fig.tight_layout()
savefig(fig, '05_approach_cases')
pd.DataFrame(case_rows).to_csv(REPORTS / '05_case_selection.csv', index=False)''')
md('''## 09 Результат и авиационная интерпретация

Выбор по validation и описательный лидер test могут различаться. Ниже они названы отдельно. Снижение высотной ошибки не превращается в оценку риска или вероятность ухода на второй круг.

Геометрическая привязка 09L остается предварительной; захват ILS, истинная высота и исход захода не подтверждены. Исторический комплект NOTAM на 15.01.2025 не получен. [A3237/26](../reports/04_NOTAM_09L_A3237-26.md) относится к другому периоду и не входит в признаки.

Группа отобрана по наблюдаемому покрытию до прокси 300 ft; ранние уходы и неполные траектории могут в нее не попасть. Модель, успешно описывающая обычное снижение, требует отдельной проверки реакции на нестабильные профили. Локальная чувствительность к Vz помогает сформулировать такую проверку, но не заменяет ее.''')
code('''# Формируем вывод из реально выполненных расчетов
selected_test = test_metrics.loc[test_metrics.model.eq(winner)].iloc[0]
ridge_test = test_metrics.loc[test_metrics.model.eq('Ridge')].iloc[0]
descriptive_test_leader = str(test_metrics.iloc[0].model)
chosen_interval = intervals.loc[intervals.reference.eq('Ridge') & intervals.model.eq(winner)].iloc[0]
panel('Результат сравнения',
    f'Выбор validation: {winner}. Test macro MAE: {selected_test.macro_mae_ft:.2f} ft.<br>'
    f'Текущая Vz: {baseline_mae:.2f} ft; Ridge: {ridge_test.macro_mae_ft:.2f} ft.<br>'
    f'Описательный лидер test: {descriptive_test_leader}; выбор по нему не меняется.<br>'
    f'Разность выбранной модели и Ridge: {chosen_interval.delta_macro_mae_ft:+.2f} ft; '
    f'условный интервал [{chosen_interval.interval_2_5_ft:+.2f}; {chosen_interval.interval_97_5_ft:+.2f}] ft')

manifest = {'study': plan['study'], 'data_date': plan['data_date'],
    'candidate_events': len(cohort), 'aircraft': int(cohort.icao24.nunique()), 'observed_pairs': len(examples),
    'split': split_summary.reset_index().to_dict(orient='records'),
    'tested_parameter_variants': len(tuning), 'trained_families': len(selected_models),
    'winner_by_validation': winner, 'best_ml_by_validation': best_ml,
    'descriptive_test_leader': descriptive_test_leader,
    'baseline_macro_mae_ft': baseline_mae, 'ridge_macro_mae_ft': float(ridge_test.macro_mae_ft),
    'selected_test_macro_mae_ft': float(selected_test.macro_mae_ft),
    'test_previously_inspected': True, 'independent_new_test': False,
    'selection_uses_test_scores': False, 'refit_on_validation': False,
    'future_weather_used': False, 'target_interpolation': False, 'feature_selection': False,
    'goaround_model_trained': False, 'historical_notam_verified': False,
    'input_sha256': {'examples': digest(examples_path), 'cohort': digest(cohort_path),
        'prior_manifest': digest(REPORTS / '04_training_manifest.json'), 'plan': digest(plan_path)},
    'selection_sha256': digest(REPORTS / '05_model_selection.json'),
    'model_sha256': digest(ARTIFACTS / 'selected_models.joblib'),
    'predictions_sha256': digest(BASE / 'data/2025-01-15/training_05_predictions.parquet'),
    'implementation_sha256': {str(path.relative_to(PROJECT)).replace(chr(92), '/'): digest(path)
        for path in [PROJECT / 'scripts/lfpg_model_comparison.py',
                     PROJECT / 'scripts/build_lfpg_model_comparison_notebook.py',
                     PROJECT / 'scripts/execute_notebook.py']},
    'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                    'numpy': np.__version__, 'pandas': pd.__version__}}
(REPORTS / '05_comparison_manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
report = ['# LFPG 09L – сравнение моделей прогноза высотной динамики', '',
    '37 кандидатов, 35 бортов, 633 пары наблюдений за 15.01.2025. '
    'Разделение этапа 04 сохранено: 21 train, 8 validation, 8 test; test содержит 7 бортов.', '',
    f'Обучение выполнено для {len(tuning)} вариантов в {len(selected_models)} семействах. '
    'Признаки, target, физический baseline и веса заходов одинаковы. '
    'Параметры и выбранный прогноз определены по validation macro MAE.', '',
    f'Выбор validation: **{winner}**, test macro MAE **{selected_test.macro_mae_ft:.2f} ft**. '
    f'Описательный лидер test: **{descriptive_test_leader}**; выбор по нему не меняется.', '',
    '| Прогноз | Validation MAE, ft | Test MAE, ft | Test RMSE, ft |',
    '|---|---:|---:|---:|']
for name in order:
    val = validation_metrics.loc[validation_metrics.model.eq(name)].iloc[0]
    result = test_metrics.loc[test_metrics.model.eq(name)].iloc[0]
    report.append(f'| {name} | {val.macro_mae_ft:.2f} | {result.macro_mae_ft:.2f} | {result.macro_rmse_ft:.2f} |')
report.extend(['', f'Выбранная модель минус Ridge: {chosen_interval.delta_macro_mae_ft:+.2f} ft; '
    f'условный парный интервал [{chosen_interval.interval_2_5_ft:+.2f}; '
    f'{chosen_interval.interval_97_5_ft:+.2f}] ft. Bootstrap целыми бортами.', '',
    'Результаты этих test-заходов уже просмотрены в этапе 04. Это исследовательское '
    'сравнение на повторно используемой группе; интервал не учитывает выбор модели и '
    'множественные сравнения. Подтверждение требует другого дня.', '',
    'Высотная точность не измеряет способность обнаружить нестабильность или GoA. '
    'Реакция на текущую Vz проверена отдельно на validation путем изолированного изменения признака. '
    'Это диагностика формулы, не имитация физически согласованного полета.', '',
    'Захват ILS, истинная высота, исходы GoA и исторические NOTAM не подтверждены. '
    'Контрольные 300 ft являются исследовательским барометрическим прокси.', '',
    '[Выполненный ноутбук](../notebooks/05_model_comparison.ipynb), '
    '[фиксированный протокол](05_comparison_protocol.json), '
    '[выбор по validation](05_model_selection.json), '
    '[манифест](05_comparison_manifest.json).', ''])
(REPORTS / '05_model_comparison_result.md').write_text(chr(10).join(report), encoding='utf-8')
print('Модели, прогнозы, таблицы, рисунки и контрольные суммы сохранены')''')
md('''## 10 Следующий проверяемый шаг

Сохранить выбранную на validation модель и компактную сетку без расширения по test. Проверить зафиксированный прогноз на другом дне LFPG 09L с теми же причинными правилами и отдельно рассмотреть профили с изменением вертикальной динамики. Только затем обсуждать переносимость на EGLL или индикатор риска.

**Воспроизведение:** из корня проекта `python scripts/build_lfpg_model_comparison_notebook.py`. Проверка сериализованных моделей, прогнозов, выбора и сохраненных выводов: `python scripts/verify_lfpg_model_comparison.py`.

Данные и модели находятся в игнорируемых `data` и `artifacts`; таблицы и рисунки – в `reports/05_*` и `figures/05_*`. Этап 04 сохраняется как исходный эксперимент.

**Происхождение:** [ADSB.lol, архив 2025](https://github.com/adsblol/globe_history_2025), [формат trace readsb](https://github.com/wiedehopf/readsb/blob/dev/README-json.md), [OurAirports](https://ourairports.com/data/), [IEM METAR](https://mesonet.agron.iastate.edu/request/download.phtml?network=FR__ASOS). Сырые данные ADSB.lol опубликованы под ODbL 1.0; этот этап наследует их происхождение и контрольные суммы из этапов 02–04.''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'}})
path = BASE / 'notebooks/05_model_comparison.ipynb'
execute(notebook, path, BASE)
print(path)
print((BASE / 'reports/05_model_comparison_result.md').read_text(encoding='utf-8'))
