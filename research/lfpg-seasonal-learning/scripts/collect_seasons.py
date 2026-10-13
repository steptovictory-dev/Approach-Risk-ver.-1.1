"""Собрать по пять готовых дней весны, лета и осени и выполнить простое обучение"""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import pandas as pd
from collect_days import BASE, analyze
from prepare_examples import prepare

TARGET = 5


def collect(season):
    candidates = pd.read_csv(BASE / 'data/weather_candidates_2025.csv')
    candidates = candidates.loc[candidates.season.eq(season)].sort_values(
        ['east_fair_fraction', 'date'], ascending=[False, True])
    ready, failures = [], []
    for date in candidates.date:
        folder = BASE / 'data' / date
        try:
            examples_file = folder / 'examples.parquet'
            if examples_file.exists():
                examples = pd.read_parquet(examples_file)
                result = {'date': date, 'qualified_09l': int(examples.event_id.nunique()),
                          'examples': len(examples)}
            else:
                if not (folder / 'day_result.json').exists():
                    analyze(date)
                result = prepare(date)
            if result['qualified_09l'] >= 10 and result['examples'] >= 300:
                ready.append(result)
            print(f'SEASON_PROGRESS {season} {len(ready)} / {TARGET} {date} '
                  f'{result["qualified_09l"]}', flush=True)
            if len(ready) == TARGET:
                break
        except Exception as error:
            failures.append({'date': date, 'error': str(error)})
            print('DAY_ERROR', season, date, str(error), flush=True)
    status = {'season': season, 'ready': ready, 'failures': failures, 'needed': TARGET}
    (BASE / 'data' / f'{season}_collection.json').write_text(
        json.dumps(status, indent=2), encoding='utf-8')
    print(f'SEASON_DONE {season} {len(ready)} / {TARGET}', flush=True)
    return status


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with ThreadPoolExecutor(max_workers=3) as pool:
        status = list(pool.map(collect, ['spring', 'summer', 'autumn']))
    if any(len(item['ready']) < TARGET for item in status):
        raise SystemExit('Не все сезоны набрали пять дней, обучение не запускалось')
    from train_models import run
    run(seasons=['spring', 'summer', 'autumn'], days_per_season=TARGET)
    import build_training_notebook
    print('EXPERIMENT_DONE reports/final_table.csv notebooks/02_seasonal_training.ipynb', flush=True)
