"""Белый текст таблиц и выделение лучших значений без заливки"""
import re
import pandas as pd
from IPython.display import HTML

GREEN = '#B6E66B'


def table_html(html):
    if 'class="lfpg-table"' in html:
        return html
    html = re.sub(r'color:\s*#18221B\s*!important;', '', html, flags=re.IGNORECASE)
    html = re.sub(r'background-color:\s*#[0-9a-fA-F]{6};(?:\s*color:\s*#18221B\s*!important;)?',
                  f'color: {GREEN} !important; font-weight: bold;', html, flags=re.IGNORECASE)
    css = ('<style>.lfpg-table th, .lfpg-table td {color: #FFFFFF !important;} '
           '.lfpg-table th, .lfpg-table td {padding: 6px 10px;}</style>')
    return css + '<div class="lfpg-table">' + html + '</div>'


def display_table(display, value):
    if isinstance(value, pd.DataFrame):
        value = value.style
    if hasattr(value, 'to_html'):
        display(HTML(table_html(value.to_html())))
    else:
        display(value)
