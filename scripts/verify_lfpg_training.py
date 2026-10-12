"""Проверить сохраненный эксперимент обучения и воспроизводимость прогнозов"""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tokenize
import joblib
import nbformat
import numpy as np
import pandas as pd
from lfpg_training import summarize

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / 'airports/LFPG'
REPORTS = BASE / 'reports'
manifest = json.loads((REPORTS / '04_training_manifest.json').read_text(encoding='utf-8'))
protocol = json.loads((PROJECT / 'configs/lfpg_09l_training.json').read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert digest(PROJECT / 'configs/lfpg_09l_training.json') == manifest['protocol_sha256']
for relative, expected in manifest['inputs_sha256'].items():
    assert digest(BASE / relative) == expected == protocol['inputs_sha256'][relative]
for relative, expected in manifest['implementation_sha256'].items():
    assert digest(PROJECT / relative) == expected
model_path = BASE / 'artifacts/training_04/altitude_residual_ridge.joblib'
assert digest(model_path) == manifest['model_sha256']
assert digest(BASE / 'data/2025-01-15/training_04_examples.parquet') == manifest['examples_sha256']

cohort = pd.read_csv(REPORTS / '04_cohort_and_split.csv')
examples = pd.read_parquet(BASE / 'data/2025-01-15/training_04_examples.parquet')
predictions = pd.read_parquet(BASE / 'data/2025-01-15/training_04_test_predictions.parquet')
bundle = joblib.load(model_path)
assert len(cohort) == 37 and examples.event_id.nunique() == 37
assert set(cohort.event_id) == set(examples.event_id)
assert cohort.groupby('icao24').split.nunique().eq(1).all()
assert examples.groupby('trace_id').split.nunique().eq(1).all()
assert len(predictions) == len(examples.loc[examples.split.eq('test')])
assert predictions.event_id.nunique() == 8
replayed = predictions.baseline_ft + bundle['model'].predict(predictions[bundle['features']])
assert np.allclose(replayed, predictions.ridge_ft)
metrics = pd.read_csv(REPORTS / '04_test_metrics.csv')
replayed_metrics = summarize(predictions, replayed, 'Ridge correction')
for metric in ['macro_mae_ft', 'macro_rmse_ft', 'point_mae_ft', 'point_p95_abs_ft', 'macro_bias_ft']:
    saved = float(metrics.loc[metrics.model.eq('Ridge correction'), metric].iloc[0])
    assert np.isclose(saved, replayed_metrics[metric])
tuning = pd.read_csv(REPORTS / '04_validation_tuning.csv').sort_values(['macro_mae_ft', 'alpha'])
assert float(tuning.alpha.iloc[0]) == bundle['selected_alpha'] == manifest['selected_alpha']

# Проверяем сохраненные выводы и переносимые представления рисунков
notebook_path = BASE / 'notebooks/04_altitude_dynamics_training.ipynb'
notebook = nbformat.read(notebook_path, as_version=4)
nbformat.validate(notebook)
code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
assert all(cell.outputs for cell in code_cells)
assert not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs)
images = sum('image/png' in output.get('data', {}) for cell in code_cells for output in cell.outputs)
assert images == 7
assert len(list((BASE / 'figures').glob('04_*.png'))) == 7
assert len(list((BASE / 'figures').glob('04_*.svg'))) == 7

# Проверяем правила текста и комментариев в исполняемых исходниках
files = [PROJECT / 'scripts/lfpg_training.py', PROJECT / 'scripts/build_lfpg_training_notebook.py',
         PROJECT / 'scripts/verify_lfpg_training.py', PROJECT / 'configs/lfpg_09l_training.json',
         PROJECT / 'README.md', BASE / 'README.md', *REPORTS.glob('04_*.md'), *REPORTS.glob('04_*.json')]
texts = [path.read_text(encoding='utf-8') for path in files]
texts += [cell.source for cell in notebook.cells]
for value in texts:
    assert not set(value) & {chr(1105), chr(1025), chr(8212)}
    reserved = ''.join(map(chr, [99, 111, 100, 101, 120]))
    assert reserved not in value.lower()
for path in [item for item in files if item.suffix == '.py']:
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
for source in [item.read_text(encoding='utf-8') for item in files if item.suffix == '.py'] + [cell.source for cell in code_cells]:
    comments = [item.string for item in tokenize.generate_tokens(io.StringIO(source).readline)
                if item.type == tokenize.COMMENT]
    assert all(not comment.rstrip().endswith('.') for comment in comments)

# Убеждаемся, что локальные данные, модель и чекпоинты исключены из публикации
ignored_paths = [
    'airports/LFPG/data/2025-01-15/training_04_examples.parquet',
    'airports/LFPG/artifacts/training_04/altitude_residual_ridge.joblib',
    'airports/LFPG/Checkpoints/verification.md']
for relative in ignored_paths:
    ignored = subprocess.run(['git', 'check-ignore', '--quiet', relative], cwd=PROJECT)
    assert ignored.returncode == 0
tracked = subprocess.run(['git', 'ls-files'], cwd=PROJECT, check=True,
                         capture_output=True, text=True).stdout.splitlines()
assert not any(set(path.lower().split('/')) & {'data', 'artifacts', 'checkpoint', 'checkpoints'}
               for path in tracked)
assert not manifest['historical_notam_verified'] and not manifest['goaround_model_trained']
print(json.dumps({'verified': True, 'cohort': len(cohort), 'pairs': len(examples),
                  'code_cells': len(code_cells), 'embedded_figures': images,
                  'serialized_predictions_reproduced': True}, indent=2))
