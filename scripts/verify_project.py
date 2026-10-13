"""Проверяет публичные артефакты проекта и сохраненные сезонные прогнозы LFPG"""
from pathlib import Path
import ast
import json
import re
import subprocess
import sys
import joblib
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding='utf-8')
tracked = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode('utf-8').split('\0')
files = [ROOT / p for p in tracked if p]
assert not any(set(p.relative_to(ROOT).parts) & {'Articles', 'Checkpoints', 'data', 'artifacts', '.venv'} for p in files)
assert all(p.exists() for p in files)
notebooks = links = scripts = 0
for path in files:
    if path.suffix == '.py':
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        scripts += 1
    elif path.suffix == '.ipynb':
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code' and cell.source.strip():
                assert cell.execution_count is not None, path
                assert not any(o.output_type == 'error' for o in cell.outputs), path
        notebooks += 1
    if path.suffix == '.md':
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            target = target.strip('<>').split('#', 1)[0]
            if target and not re.match(r'^[a-z]+:', target):
                assert (path.parent / target).exists(), (path, target)
                links += 1
for relative in ('Articles/article_05/article_05_draft.md', 'Checkpoints/closure.md', 'research/lfpg-seasonal-learning/data/probe', 'research/eddf-seasonal-learning/artifacts/probe'):
    assert subprocess.run(['git', 'check-ignore', '-q', relative], cwd=ROOT).returncode == 0
base = ROOT / 'research/lfpg-seasonal-learning'
days = pd.read_csv(base / 'reports/csv/selected_days.csv')
assert len(days) == 15 and days.day.is_unique
assert days.groupby('season').size().eq(5).all()
assert days.groupby(['season', 'split']).size().unstack().eq(pd.Series({'train': 3, 'validation': 1, 'test': 1})).all().all()
weather = pd.read_csv(base / 'data/weather_candidates_2025.csv').set_index('date')
assert weather.loc[days.day, 'fair_fraction'].eq(1).all()
snapshots = []
for row in days.itertuples():
    examples = pd.read_parquet(base / 'data' / row.day / 'examples.parquet')
    assert len(examples) == row.examples and examples.event_id.nunique() == row.approaches
    assert examples.origin_metar_available_time.le(examples.origin_time).all()
    assert examples.target_time.lt(examples.boundary_time).all()
    assert examples.horizon_s.between(27, 33).all()
    snapshots.append(examples.assign(season=row.season, split=row.split))
all_examples = pd.concat(snapshots, ignore_index=True)
assert all_examples.groupby('event_id').split.nunique().eq(1).all()
test = all_examples.loc[all_examples.split.eq('test')]
assert not test.duplicated(['event_id', 'origin_time', 'target_time']).any()
saved = joblib.load(base / 'artifacts/models.joblib')
final = pd.read_csv(base / 'reports/csv/final_table.csv').set_index('model')
for model in final.index:
    predicted = test.baseline_ft.to_numpy()
    if model != 'Constant vertical rate':
        predicted = predicted + saved['models']['pooled', model].predict(test[saved['features']])
    errors = pd.DataFrame({'event': test.event_id, 'season': test.season,
                            'error': abs(predicted-test.target_height_ft.to_numpy())})
    for season in final.columns:
        part = errors if season == 'all' else errors.loc[errors.season.eq(season)]
        assert np.isclose(part.groupby('event').error.mean().mean(), final.loc[model, season], rtol=0, atol=1e-9)
print(json.dumps({'status': 'passed', 'tracked_notebooks': notebooks, 'parsed_scripts': scripts,
                  'valid_local_links': links, 'lfpg_days': len(days), 'lfpg_saved_predictions': True,
                  'private_material_ignored': True}, ensure_ascii=False))
