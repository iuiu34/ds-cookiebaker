"""App config."""
import inspect
import os

import streamlit as st

import edo.cookiebaker

font_family = 'Roboto'
package_path = os.path.dirname(inspect.getfile(edo.cookiebaker))
filename = os.path.join(package_path, 'app', '_utils', 'theme.css')
with open(filename) as f:
    theme_css = f.read()

primary_color = st.get_option('theme.primaryColor')[:7]
# secondary_color = color_variant(primary_color, -110)
# secondary_color = '#054a05'
secondary_color = '#293273'
opacity = 1

plotly_config = {'displaylogo': False,
                 'modeBarButtonsToRemove': [
                     'lasso2d', 'autoScale2d', 'select2d', 'resetScale2d', 'zoom2d']}


def human_format(num, round_to=2, suffix=None):
    if suffix is None:
        suffix = ''
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num = round(num / 1000.0, round_to)
    if round_to == 0 and abs(num) < 1:
        num = 0
    magnitude = ['', 'K', 'M', 'G', 'T', 'P'][magnitude]
    return f'{num:.{round_to}f}{magnitude}{suffix}'


def update_layout_default(g):
    # margin = 100
    g.update_layout(
        font_family=font_family,
        title_font_family=font_family,
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        title={
            "x": .04
        },
        margin=dict(l=80, r=80, t=100, b=50),

    )
