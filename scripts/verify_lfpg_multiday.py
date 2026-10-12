"""Проверить выполненный ноутбук и воспроизвести сохраненные test-прогнозы"""
from pathlib import Path
import sys
import joblib
import nbformat
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from lfpg_multiday import load_data, split_data, predict, BASELINES
from lfpg_training import summarize

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / 'airports/LFPG'
sys.stdout.reconfigure(encoding='utf-8')
path = BASE / 'notebooks/06_multiday_training.ipynb'
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
code = [cell for cell in notebook.cells if cell.cell_type == 'code']
assert [cell.execution_count for cell in code] == list(range(1,len(code)+1))
assert not any(output.output_type == 'error' for cell in code for output in cell.outputs)
images = [output for cell in code for output in cell.outputs if 'image/png' in output.get('data',{})]
assert len(images) == 6
for cell in notebook.cells:
    assert '\u2014' not in cell.source and '\u00b7' not in cell.source
    assert '\u0451' not in cell.source and '\u0401' not in cell.source
    assert 'когорт' not in cell.source.lower()
data, features = load_data(BASE)
artifacts = BASE / 'artifacts/training_06'
bundle = joblib.load(artifacts / 'models.joblib')
assert bundle['features'] == features
predictions = pd.read_parquet(artifacts / 'test_predictions.parquet')
splits = pd.read_parquet(artifacts / 'splits.parquet')
metrics = pd.read_csv(BASE / 'reports/06_metrics.csv')
selected = pd.read_csv(BASE / 'reports/06_selected.csv')
search = pd.read_csv(BASE / 'reports/06_search.csv')
assert len(search) == 184 and len(metrics) == 48 and len(selected) == 44
assert len(predictions) == 24*(872+715)
for scheme, models in bundle['models'].items():
    frame = split_data(data,scheme)
    expected_splits = frame.set_index('example_id').split.sort_index()
    actual_splits = splits.loc[splits.scheme.eq(scheme)].set_index('example_id').split.sort_index()
    assert expected_splits.equals(actual_splits)
    test = frame.loc[frame.split.eq('test')].set_index('example_id')
    choice = selected.loc[selected.scheme.eq(scheme)].sort_values(['validation_mae_ft','model']).iloc[0]['model']
    assert choice == bundle['winner_by_validation'][scheme]
    for name,item in models.items():
        saved = predictions.loc[predictions.scheme.eq(scheme) & predictions.model.eq(name)]
        assert saved.example_id.is_unique and set(saved.example_id) == set(test.index)
        examples = test.loc[saved.example_id]
        with threadpool_limits(limits=1):
            replay = predict(item['model'],examples,features,item['mode'])
        assert np.allclose(replay,saved.prediction_ft,rtol=1e-10,atol=1e-8)
        assert np.allclose(examples.target_height_ft,saved.target_height_ft)
        calculated = summarize(examples,replay,name)
        recorded = metrics.loc[metrics.scheme.eq(scheme) & metrics.model.eq(name)].iloc[0]
        for field in ['macro_mae_ft','macro_rmse_ft','point_p95_abs_ft','macro_bias_ft']:
            assert np.isclose(calculated[field],recorded[field],rtol=1e-10,atol=1e-8)
        if name not in BASELINES:
            row = selected.loc[selected.scheme.eq(scheme) & selected.model.eq(name)].iloc[0]
            candidates = search.loc[search.scheme.eq(scheme) & search.family.eq(item['family'])
                                    & search['mode'].eq(item['mode']) & search.converged]
            assert np.isclose(row.validation_mae_ft,candidates.validation_mae_ft.min())
final = pd.read_csv(BASE / 'reports/06_final_table.csv')
assert len(final) == 13
for row in final.itertuples(index=False):
    for scheme,field in [('next_day','macro_mae_ft'),('new_aircraft','new_aircraft_mae_ft')]:
        expected = metrics.loc[metrics.scheme.eq(scheme) & metrics.family.eq(row.family)].sort_values(['validation_mae_ft','model']).iloc[0]
        assert np.isclose(getattr(row,field),expected.macro_mae_ft)
print(f'Проверены {len(code)} выполненных ячеек, {len(images)} рисунков и {len(predictions)} сохраненных прогнозов')
print('Разделения, выбор на validation и итоговая таблица согласованы')
