"""Проверить сохраненные модели, независимость групп и выбор по validation"""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import tokenize
import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
from lfpg_training import summarize
from lfpg_model_comparison import forecast

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / 'airports/LFPG'
REPORTS = BASE / 'reports'
manifest = json.loads((REPORTS / '05_comparison_manifest.json').read_text(encoding='utf-8'))
plan_path = PROJECT / 'configs/lfpg_09l_model_comparison.json'
plan = json.loads(plan_path.read_text(encoding='utf-8'))
selection_path = REPORTS / '05_model_selection.json'
selection = json.loads(selection_path.read_text(encoding='utf-8'))
frozen = json.loads((REPORTS / '05_comparison_protocol.json').read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


for key, path in {'examples': BASE / plan['examples'], 'cohort': BASE / plan['split'],
                  'prior_manifest': REPORTS / '04_training_manifest.json', 'plan': plan_path}.items():
    assert digest(path) == manifest['input_sha256'][key]
for relative, expected in manifest['implementation_sha256'].items():
    assert digest(PROJECT / relative) == expected
assert digest(selection_path) == manifest['selection_sha256']
model_path = BASE / 'artifacts/training_05/selected_models.joblib'
prediction_path = BASE / 'data/2025-01-15/training_05_predictions.parquet'
assert digest(model_path) == manifest['model_sha256']
assert digest(prediction_path) == manifest['predictions_sha256']
assert frozen['plan_sha256'] == manifest['input_sha256']['plan']
assert frozen['examples_sha256'] == manifest['input_sha256']['examples']
assert frozen['cohort_sha256'] == manifest['input_sha256']['cohort']
for key in plan:
    assert plan[key] == frozen[key]

examples = pd.read_parquet(BASE / plan['examples'])
assert len(examples) == 633 and examples.event_id.nunique() == 37
for key in ['event_id', 'icao24', 'trace_id']:
    assert examples.groupby(key).split.nunique().eq(1).all()
train = examples.loc[examples.split.eq('train')]
validation = examples.loc[examples.split.eq('validation')]
test = examples.loc[examples.split.eq('test')]
assert test.event_id.nunique() == 8 and test.icao24.nunique() == 7
bundle = joblib.load(model_path)
assert bundle['selection'] == selection and bundle['features'] == plan['features']
assert len(bundle['models']) == 7
all_models = {name: name for name in plan['baselines']}
all_models.update(bundle['models'])
saved_predictions = pd.read_parquet(prediction_path)
assert len(saved_predictions) == len(test) * len(all_models)
metrics = pd.read_csv(REPORTS / '05_split_metrics.csv')

# Воспроизводим прогнозы сериализованными моделями без нового обучения
with threadpool_limits(limits=2):
    for name, model in all_models.items():
        predicted = forecast(model, test, plan['features'])
        saved = saved_predictions.loc[saved_predictions.model.eq(name)]
        assert saved.event_id.tolist() == test.event_id.tolist()
        assert np.allclose(saved.origin_time, test.origin_time)
        assert np.allclose(saved.target_time, test.target_time)
        assert np.allclose(saved.prediction_ft, predicted)
        for split, frame in [('train', train), ('validation', validation), ('test', test)]:
            replay = summarize(frame, forecast(model, frame, plan['features']), name)
            stored = metrics.loc[metrics.model.eq(name) & metrics.split.eq(split)].iloc[0]
            for field in ['macro_mae_ft', 'macro_rmse_ft', 'point_mae_ft', 'point_p95_abs_ft', 'macro_bias_ft']:
                assert np.isclose(replay[field], stored[field], atol=1e-8)

# Выбор каждой семьи восстанавливается только из журнала validation
tuning = pd.read_csv(REPORTS / '05_validation_tuning.csv')
assert len(tuning) == sum(len(ParameterGrid(spec['grid'])) for spec in plan['model_families']) == 32
for spec in plan['model_families']:
    rows = tuning.loc[tuning.model.eq(spec['name'])].sort_values('candidate_index')
    variables = list(ParameterGrid(spec['grid']))
    assert rows.candidate_index.tolist() == list(range(len(variables)))
    for row, variable in zip(rows.itertuples(index=False), variables):
        assert json.loads(row.parameters) == {**spec['fixed'], **variable}
    best = rows.loc[rows.converged].sort_values(['macro_mae_ft', 'candidate_index']).iloc[0]
    chosen = next(item for item in selection['selected_models'] if item['model'] == spec['name'])
    assert chosen['candidate_index'] == best.candidate_index
    assert chosen['parameters'] == json.loads(best.parameters)
    assert np.isclose(chosen['validation_macro_mae_ft'], best.macro_mae_ft)
validation_metrics = metrics.loc[metrics.split.eq('validation')].sort_values(['macro_mae_ft', 'model'])
assert selection['winner_by_validation'] == validation_metrics.iloc[0].model == manifest['winner_by_validation']
assert manifest['test_previously_inspected'] and not manifest['independent_new_test']
assert not manifest['selection_uses_test_scores'] and not manifest['refit_on_validation']
assert not manifest['goaround_model_trained'] and not manifest['historical_notam_verified']

# Проверяем единицу повторного выбора и фактические разности MAE
intervals = pd.read_csv(REPORTS / '05_paired_intervals.csv')
test_metrics = metrics.loc[metrics.split.eq('test')].set_index('model')
for row in intervals.itertuples(index=False):
    expected = test_metrics.loc[row.model, 'macro_mae_ft'] - test_metrics.loc[row.reference, 'macro_mae_ft']
    assert np.isclose(row.delta_macro_mae_ft, expected)
    assert row.independent_aircraft == 7 and row.events == 8 and row.resampling_unit == 'icao24'
    assert row.interval_2_5_ft <= row.interval_97_5_ft

notebook_path = BASE / 'notebooks/05_model_comparison.ipynb'
notebook = nbformat.read(notebook_path, as_version=4)
nbformat.validate(notebook)
code_cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells)+1))
assert all(cell.outputs for cell in code_cells)
assert not any(output.output_type == 'error' for cell in code_cells for output in cell.outputs)
images = sum('image/png' in output.get('data', {}) for cell in code_cells for output in cell.outputs)
assert images == 6
assert len(list((BASE / 'figures').glob('05_*.png'))) == images
assert len(list((BASE / 'figures').glob('05_*.svg'))) == images
files = [PROJECT / 'scripts/lfpg_model_comparison.py',
         PROJECT / 'scripts/build_lfpg_model_comparison_notebook.py', Path(__file__),
         plan_path, *REPORTS.glob('05_*.md'), *REPORTS.glob('05_*.json')]
texts = [path.read_text(encoding='utf-8') for path in files] + [cell.source for cell in notebook.cells]
for text in texts:
    assert not set(text) & {chr(1105), chr(1025), chr(8212), chr(183)}
    assert ''.join(map(chr, [99, 111, 100, 101, 120])) not in text.lower()
for source in [path.read_text(encoding='utf-8') for path in files if path.suffix == '.py'] + [cell.source for cell in code_cells]:
    comments = [token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT]
    assert all(not comment.rstrip().endswith('.') for comment in comments)
for relative in ['airports/LFPG/data/2025-01-15/training_05_predictions.parquet',
                 'airports/LFPG/artifacts/training_05/selected_models.joblib']:
    assert subprocess.run(['git', 'check-ignore', '--quiet', relative], cwd=PROJECT).returncode == 0
print(json.dumps({'verified': True, 'families': len(bundle['models']), 'variants': len(tuning),
    'predictions_replayed': len(saved_predictions), 'code_cells': len(code_cells),
    'embedded_figures': images, 'winner_by_validation': selection['winner_by_validation'],
    'test_status': 'exploratory_reuse'}, indent=2))
