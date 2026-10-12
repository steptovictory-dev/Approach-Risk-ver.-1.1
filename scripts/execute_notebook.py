"""Последовательно исполнить ячейки и сохранить таблицы и рисунки"""
import base64
import contextlib
import io
import os
from pathlib import Path
import nbformat

def execute(notebook, path, working_directory):
    previous = Path.cwd()
    namespace = {'__name__': '__main__'}
    try:
        os.chdir(working_directory)
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            count += 1
            outputs = []
            # Сохраняем представление таблиц в выводах ячейки
            def display(value):
                data = {'text/plain': repr(value)}
                if hasattr(value, '_repr_html_'):
                    data['text/html'] = value._repr_html_()
                outputs.append(nbformat.v4.new_output('display_data', data=data))
            import matplotlib.pyplot as plt
            original_show = plt.show
            # Сохраняем рисунки внутри ноутбука
            def show():
                buffer = io.BytesIO()
                plt.gcf().savefig(buffer, format='png', bbox_inches='tight')
                outputs.append(nbformat.v4.new_output('display_data', data={
                    'image/png': base64.b64encode(buffer.getvalue()).decode('ascii')}))
                plt.close('all')
            namespace['display'] = display
            plt.show = show
            try:
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    exec(compile(cell.source.replace('from IPython.display import display', ''),
                                 str(path), 'exec'), namespace)
                if stream.getvalue():
                    outputs.append(nbformat.v4.new_output('stream', name='stdout', text=stream.getvalue()))
                cell.execution_count = count
                cell.outputs = outputs
            finally:
                plt.show = original_show
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
    finally:
        os.chdir(previous)
