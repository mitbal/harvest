"""Shared, dependency-free chart color tokens."""

CHART_PRIMARY = '#0F766E'
CHART_PRIMARY_DARK = '#115E59'
CHART_BLUE = '#2563EB'
CHART_AMBER = '#D97706'
CHART_PURPLE = '#7C3AED'
CHART_POSITIVE = '#15803D'
CHART_NEGATIVE = '#DC2626'
CHART_NEUTRAL = '#64748B'
CHART_TEXT = '#334155'
CHART_MUTED_TEXT = '#64748B'
CHART_GRID = '#CBD5E1'
CHART_SURFACE = '#EFF1EF'

CHART_CATEGORY_RANGE = (
    CHART_PRIMARY,
    CHART_BLUE,
    CHART_AMBER,
    CHART_PURPLE,
    '#0891B2',
    '#BE123C',
    '#4D7C0F',
    '#A16207',
    '#475569',
    '#C026D3',
)

# Fixed per-entity hues for comparison line charts (indices, currencies).
# Slots 1-3 stay distinguishable for colour-blind readers when shown together;
# no red or green, which this app reserves for losses and gains.
CHART_COMPARISON_RANGE = (
    '#2A78D6',  # blue
    '#EB6834',  # orange
    '#1BAF7A',  # aqua
    '#EDA100',  # yellow
    '#E87BA4',  # magenta
    '#4A3AA7',  # violet
    '#64748B',  # slate
    '#8B5E3C',  # brown
    '#0891B2',  # cyan
    CHART_PRIMARY,
)

MARKET_HEATMAP_RANGE = (
    '#7F1D1D',
    '#B91C1C',
    '#DC2626',
    '#475569',
    '#16A34A',
    '#15803D',
    '#14532D',
)

# ColorBrewer RdYlGn, the same ramp as Vega's 'redyellowgreen' scheme.
RED_YELLOW_GREEN = (
    '#A50026',
    '#D73027',
    '#F46D43',
    '#FDAE61',
    '#FEE08B',
    '#FFFFBF',
    '#D9EF8B',
    '#A6D96A',
    '#66BD63',
    '#1A9850',
    '#006837',
)


def _hex_to_rgb(color):
    color = color.lstrip('#')
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def diverging_color(value, low, mid, high, ramp=RED_YELLOW_GREEN):
    """Interpolate `ramp` so `mid` lands on its centre, `low` on its start and `high` on its end."""
    if value <= mid:
        position = 0.5 * (value - low) / (mid - low) if mid > low else 0.5
    else:
        position = 0.5 + 0.5 * (value - mid) / (high - mid) if high > mid else 0.5
    position = min(max(position, 0.0), 1.0) * (len(ramp) - 1)
    start = int(position)
    end = min(start + 1, len(ramp) - 1)
    fraction = position - start
    rgb = [
        round(a + (b - a) * fraction)
        for a, b in zip(_hex_to_rgb(ramp[start]), _hex_to_rgb(ramp[end]))
    ]
    return '#{:02X}{:02X}{:02X}'.format(*rgb)


def _relative_luminance(color):
    channels = []
    for channel in _hex_to_rgb(color):
        c = channel / 255
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def readable_text_color(background, dark=CHART_TEXT, light='#FFFFFF'):
    """Return whichever of `dark` or `light` has the higher WCAG contrast on `background`."""
    bg = _relative_luminance(background)

    def contrast(color):
        fg = _relative_luminance(color)
        return (max(fg, bg) + 0.05) / (min(fg, bg) + 0.05)

    return dark if contrast(dark) >= contrast(light) else light
