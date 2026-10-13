"""Собрать по десять готовых дней весны, лета и осени из погодного отбора"""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import pandas as pd
from collect_days import BASE, analyze
from prepare_examples import prepare


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
            if result['qualified_09l'] >= 15:
                ready.append(result)
            print('SEASON_PROGRESS', season, len(ready), '/ 10', date,
                  result['qualified_09l'], flush=True)
            if len(ready) == 10:
                break
        except Exception as error:
            failures.append({'date': date, 'error': str(error)})
            print('DAY_ERROR', season, date, str(error), flush=True)
    status = {'season': season, 'ready': ready, 'failures': failures, 'needed': 10}
    (BASE / 'data' / f'{season}_collection.json').write_text(
        json.dumps(status, indent=2), encoding='utf-8')
    print('SEASON_DONE', season, len(ready), '/ 10', flush=True)
    return status


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with ThreadPoolExecutor(max_workers=3) as pool:
        status = list(pool.map(collect, ['spring', 'summer', 'autumn']))
    if any(len(item['ready']) < 10 for item in status):
        raise SystemExit('Не все сезоны набрали десять дней, обучение не запускалось')
