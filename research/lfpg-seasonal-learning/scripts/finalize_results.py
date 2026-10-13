"""Сохранить краткие выводы завершенного сезонного эксперимента"""
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
LABELS = {'spring': 'Весна', 'summer': 'Лето', 'autumn': 'Осень', 'all': 'Все сезоны'}


def markdown(frame):
    names = [str(name) for name in frame.columns]
    rows = ['| ' + ' | '.join(names) + ' |', '| ' + ' | '.join(['---'] * len(names)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        rows.append('| ' + ' | '.join(f'{value:.2f}' if isinstance(value, float) else str(value)
                                      for value in row) + ' |')
    return '\n'.join(rows)


def run():
    days = pd.read_csv(BASE / 'reports/selected_days.csv')
    assert days.groupby('season').size().to_dict() == {'autumn': 5, 'spring': 5, 'summer': 5}
    metrics = pd.read_csv(BASE / 'reports/model_metrics.csv')
    learning = pd.read_csv(BASE / 'reports/learning_curve.csv')
    final = pd.read_csv(BASE / 'reports/final_table.csv', index_col=0)
    assert len(learning) == 36
    assert learning.converged.all()
    assert metrics.macro_mae_ft.notna().all()
    notebook = json.loads((BASE / 'notebooks/02_seasonal_training.ipynb').read_text(encoding='utf-8'))
    code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert all(cell['execution_count'] and cell['outputs'] for cell in code)
    assert not any(output['output_type'] == 'error' for cell in code for output in cell['outputs'])
    best_model = final.drop(index='Constant vertical rate')['all'].idxmin()
    best = final.loc[best_model, 'all']
    baseline = final.loc['Constant vertical rate', 'all']
    shown = final.rename(columns=LABELS).rename(index={'Constant vertical rate': 'Постоянная вертикальная скорость'})
    shown.index.name = 'Модель'
    summary = (f'Завершено: 15 погожих дней 2025 года на LFPG 09L, '
               f'Число пригодных заходов: {days.approaches.sum()}; учебных пар: {days.examples.sum()}. '
               'В каждом сезоне 3 дня обучения, 1 день валидации и 1 день теста. '
               'Общая модель обучается на 9 днях. Зима оставлена в прежнем виде.')
    text = '# Итоги сезонного эксперимента\n\n' + summary + '\n\n'
    text += '## Общие модели\n\nMAE в ft, среднее по заходам. Тестовые дни одинаковы для всех моделей.\n\n'
    text += markdown(shown.reset_index()) + '\n\n'
    text += (f'Лучшая общая модель: {best_model}, {best:.2f} ft против {baseline:.2f} ft '
             f'у baseline; ошибка ниже на {100 * (1 - best / baseline):.1f}%.\n\n')
    text += '## Даты и размер групп\n\n'
    selected = days.rename(columns={'day': 'Дата', 'season': 'Сезон', 'approaches': 'Заходов',
                                    'examples': 'Пар', 'split': 'Назначение'})
    selected['Сезон'] = selected['Сезон'].map(LABELS)
    selected['Назначение'] = selected['Назначение'].map({'train': 'Обучение', 'validation': 'Валидация', 'test': 'Тест'})
    text += markdown(selected) + '\n\n## Влияние числа дней\n\n'
    curves = learning.pivot(index=['season', 'model'], columns='train_days', values='macro_mae_ft').reset_index()
    curves['season'] = curves['season'].map(LABELS)
    curves = curves.rename(columns={'season': 'Сезон', 'model': 'Модель', 1: '1 день', 2: '2 дня', 3: '3 дня'})
    text += markdown(curves) + '\n\n'
    for season, group in learning.groupby('season', sort=False):
        monotonic = sum(values.sort_values('train_days').macro_mae_ft.diff().dropna().le(0).all()
                        for _, values in group.groupby('model'))
        text += f'{LABELS[season]}: последовательное снижение ошибки у {monotonic} из 4 моделей.\n\n'
    text += ('Больше дней не гарантирует улучшение. Одновременно меняются число примеров и состав дней; '
             'каждый сезон представлен одним тестовым днем. Это приблизительное сравнение '
             'краткосрочного прогноза высоты, а не оценка прогнозирования ухода на второй круг.\n\n'
             'Выполненный ноутбук: [02_seasonal_training.ipynb](../notebooks/02_seasonal_training.ipynb). '
             'Полные результаты переноса всех четырех моделей: `model_metrics.csv`.\n')
    (BASE / 'reports/RESULTS.md').write_text(text, encoding='utf-8')
    readme = BASE / 'README.md'
    previous = readme.read_text(encoding='utf-8')
    previous = previous.replace('будущего общего обучения', 'общего обучения')
    previous = previous.replace('Новый набор из 15 дней пока собирается; полный расчет трех сезонов еще не запускался. '
                                'Осенняя группа собрана и обучена отдельно.',
                                'Новый набор из 15 дней собран; полный расчет трех сезонов выполнен. '
                                'Осенний пилот также сохранен отдельно.')
    marker = '\n## Итог нового сравнения\n'
    previous = previous.split(marker)[0]
    readme.write_text(previous + marker + '\n' + summary + '\n\n' + markdown(shown.reset_index()) +
                      '\n\nДаты, влияние числа дней и выводы: [итоги эксперимента](reports/RESULTS.md). '
                      'Таблицы и рисунки: [выполненный ноутбук](notebooks/02_seasonal_training.ipynb).\n',
                      encoding='utf-8')
    print('FINALIZED', len(days), 'days;', days.approaches.sum(), 'approaches;', best_model, round(best, 2), flush=True)


if __name__ == '__main__':
    run()
