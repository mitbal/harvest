from harvest import page_ui as ui


def test_esc_escapes_markup_ampersand_and_quotes():
    assert ui.esc('<b>&"\'') == '&lt;b&gt;&amp;&quot;&#x27;'


def test_esc_handles_none_and_numbers():
    assert ui.esc(None) == ''
    assert ui.esc(6.35) == '6.35'


def test_chip_escapes_text_and_emits_tone_class():
    markup = ui.chip('<PTBA & co>', tone='up')
    assert '&lt;PTBA &amp; co&gt;' in markup
    assert '<PTBA' not in markup
    assert 'class="pdx-chip pdx-chip-up"' in markup


def test_chip_plain_and_unknown_tone_render_plain():
    assert 'class="pdx-chip"' in ui.chip('x')
    assert 'class="pdx-chip"' in ui.chip('x', tone='"><script>')


def test_kpi_escapes_label_value_and_note():
    markup = ui.kpi('<label>', 'IDR <1>', note='a & b', tone='up')
    assert '&lt;label&gt;' in markup
    assert 'IDR &lt;1&gt;' in markup
    assert 'a &amp; b' in markup
    assert '<label>' not in markup


def test_kpi_emits_tone_class_on_note():
    assert 'pdx-kpi-note pdx-tone-up' in ui.kpi('L', 'V', note='n', tone='up')
    assert 'pdx-kpi-note pdx-tone-muted' in ui.kpi('L', 'V', note='n', tone='muted')
    assert 'pdx-tone-' not in ui.kpi('L', 'V', note='n', tone='bogus')


def test_kpi_without_note_has_no_note_span():
    assert 'pdx-kpi-note' not in ui.kpi('L', 'V')


def test_kpi_row_html_wraps_tiles():
    markup = ui.kpi_row_html([ui.kpi('A', '1'), ui.kpi('B', '2')])
    assert markup.startswith('<div class="pdx-kpis">')
    assert markup.count('class="pdx-kpi"') == 2


def test_styles_use_prefix_and_load_font():
    assert ui.STYLES.strip().startswith('<style>')
    assert 'Manrope' in ui.STYLES
    for cls in ('.pdx-kpi', '.pdx-chip', '.pdx-note', '.pdx-cal', '.pdx-dow', '.pdx-day', '.pdx-ev',
                'st-key-pdx-card-', 'st-key-pdx-split-', 'st-key-pdx-pane-left-',
                'st-key-pdx-pane-right-'):
        assert cls in ui.STYLES
