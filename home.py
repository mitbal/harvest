import base64
from pathlib import Path

import streamlit as st


ASSET_DIR = Path(__file__).parent / 'asset'


def image_data_uri(filename):
    image_data = (ASSET_DIR / filename).read_bytes()
    encoded = base64.b64encode(image_data).decode('ascii')
    return f'data:image/png;base64,{encoded}'


if 'porto_file' not in st.session_state:
    st.session_state['porto_file'] = 'EMPTY'
if 'history_file' not in st.session_state:
    st.session_state['history_file'] = 'EMPTY'


logo = image_data_uri('favicon.png')
portfolio_preview = image_data_uri('home/portfolio_table.png')
ranking_preview = image_data_uri('home/screener_table.png')
calendar_preview = image_data_uri('home/dividend_monthly_calendar.png')
compounding_preview = image_data_uri('home/compounding_simulator.png')
market_preview = image_data_uri('home/heatmap_1d_price_return.png')
comparison_preview = image_data_uri('home/stock_comparison_scatterplot.png')
timing_preview = image_data_uri('home/best_timing_best_time_to_buy.png')


st.html(
    f'''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        :root {{
            --pd-ink: #10251d;
            --pd-ink-soft: #3c5148;
            --pd-forest: #073f2e;
            --pd-forest-deep: #052d22;
            --pd-green: #066642;
            --pd-green-bright: #16a96b;
            --pd-lime: #d7f35a;
            --pd-mint: #e7f4ec;
            --pd-canvas: #eff1ef;
            --pd-surface: #fbfdfc;
            --pd-line: #cbd9d2;
            --pd-image-line: #aebdb5;
            --pd-white: #f9fcfa;
        }}

        [data-testid="stAppViewContainer"] {{
            background: var(--pd-canvas);
        }}

        [data-testid="stMainBlockContainer"] {{
            max-width: 1240px;
            padding: 1.25rem clamp(1rem, 3vw, 2.5rem) 4rem;
        }}

        [data-testid="stHeader"] {{
            background: color-mix(in srgb, var(--pd-canvas) 86%, transparent);
        }}

        .footer {{
            display: none !important;
        }}

        .pd-home,
        .pd-home * {{
            box-sizing: border-box;
        }}

        .pd-home {{
            container-type: inline-size;
            color: var(--pd-ink);
            font-family: 'Manrope', sans-serif;
        }}

        .pd-home a {{
            color: inherit;
        }}

        .pd-masthead {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.7rem 0 1.15rem;
        }}

        .pd-brand {{
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}

        .pd-brand img {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
        }}

        .pd-masthead-note {{
            color: var(--pd-ink-soft);
            font-size: 0.82rem;
            font-weight: 600;
        }}

        .pd-hero {{
            display: grid;
            grid-template-columns: minmax(0, 0.9fr) minmax(360px, 1.1fr);
            min-height: 610px;
            overflow: hidden;
            background: var(--pd-forest);
            border-radius: 16px;
            color: var(--pd-white);
        }}

        .pd-hero-copy {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: clamp(2.25rem, 5vw, 5.25rem);
        }}

        .pd-hero-kicker {{
            display: inline-flex;
            align-items: center;
            align-self: flex-start;
            gap: 0.55rem;
            margin: 0 0 1.45rem;
            color: #d3e9de;
            font-size: 0.86rem;
            font-weight: 700;
        }}

        .pd-hero-kicker::before {{
            width: 8px;
            height: 8px;
            background: var(--pd-lime);
            border-radius: 50%;
            content: '';
        }}

        .pd-hero h1 {{
            max-width: 11ch;
            margin: 0;
            color: var(--pd-white) !important;
            font-size: clamp(2.8rem, 5.6vw, 5.6rem);
            font-weight: 700;
            letter-spacing: -0.04em;
            line-height: 0.98;
            text-wrap: balance;
        }}

        .pd-hero-copy > p {{
            max-width: 58ch;
            margin: 1.7rem 0 0;
            color: #cde0d7;
            font-size: clamp(1rem, 1.35vw, 1.18rem);
            line-height: 1.75;
            text-wrap: pretty;
        }}

        .pd-actions {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            margin-top: 2.1rem;
        }}

        .pd-button {{
            display: inline-flex;
            min-height: 48px;
            align-items: center;
            justify-content: center;
            gap: 0.65rem;
            padding: 0.75rem 1.15rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 800;
            text-decoration: none !important;
            transition: transform 180ms cubic-bezier(0.22, 1, 0.36, 1), background 180ms ease;
        }}

        .pd-home .pd-button-primary {{
            background-color: var(--pd-lime);
            color: #17351f !important;
        }}

        .pd-button-secondary {{
            border: 1px solid #6c9783;
            color: var(--pd-white) !important;
        }}

        .pd-button:hover {{
            transform: translateY(-2px);
        }}

        .pd-home .pd-button-primary:hover {{
            background-color: #e2f77f;
        }}

        .pd-button-secondary:hover {{
            background: #145740;
        }}

        .pd-button:focus-visible,
        .pd-tool-link:focus-visible,
        .pd-text-link:focus-visible {{
            outline: 3px solid var(--pd-forest);
            outline-offset: 3px;
            box-shadow: 0 0 0 5px var(--pd-lime);
        }}

        .pd-hero-visual {{
            position: relative;
            display: flex;
            min-width: 0;
            align-items: center;
            padding: clamp(2.5rem, 5vw, 5rem) 0 clamp(2.5rem, 5vw, 5rem) clamp(1rem, 3vw, 2.5rem);
            background: var(--pd-forest-deep);
        }}

        .pd-preview {{
            position: relative;
            width: 100%;
            overflow: hidden;
            border: 1px solid #3f7661;
            border-right: 0;
            border-radius: 12px 0 0 12px;
            background: var(--pd-white);
        }}

        .pd-preview-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            min-height: 50px;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #d5e0da;
            color: var(--pd-ink);
            font-size: 0.8rem;
            font-weight: 800;
        }}

        .pd-preview-status {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            color: #46705f;
            font-size: 0.72rem;
            font-weight: 700;
        }}

        .pd-preview-status::before {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--pd-green-bright);
            content: '';
        }}

        .pd-preview img {{
            display: block;
            width: 145%;
            max-width: none;
            height: 400px;
            border: 1px solid var(--pd-image-line);
            object-fit: cover;
            object-position: left top;
        }}

        .pd-hero-chip {{
            position: absolute;
            right: clamp(1rem, 3vw, 2.5rem);
            bottom: clamp(1.5rem, 4vw, 3.5rem);
            max-width: 220px;
            padding: 0.9rem 1rem;
            border-radius: 10px;
            background: var(--pd-lime);
            color: #17351f;
            font-size: 0.78rem;
            font-weight: 800;
            line-height: 1.45;
        }}

        .pd-context {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 0.65rem 1.5rem;
            padding: 1rem;
            color: var(--pd-ink-soft);
            font-size: 0.8rem;
            font-weight: 700;
        }}

        .pd-context span {{
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
        }}

        .pd-context span::before {{
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: var(--pd-green-bright);
            content: '';
        }}

        .pd-section {{
            padding: clamp(4.5rem, 9vw, 8rem) 0;
        }}

        .pd-section-header {{
            display: grid;
            grid-template-columns: minmax(0, 0.9fr) minmax(280px, 0.6fr);
            gap: 3rem;
            align-items: end;
            margin-bottom: clamp(2.5rem, 5vw, 4.25rem);
        }}

        .pd-section h2 {{
            max-width: 14ch;
            margin: 0;
            color: var(--pd-ink) !important;
            font-size: clamp(2.2rem, 4.2vw, 4.6rem);
            font-weight: 700;
            letter-spacing: -0.04em;
            line-height: 1.02;
            text-wrap: balance;
        }}

        .pd-section-header p {{
            max-width: 56ch;
            margin: 0;
            color: var(--pd-ink-soft);
            font-size: 1rem;
            line-height: 1.75;
            text-wrap: pretty;
        }}

        .pd-workflow {{
            display: grid;
            grid-template-columns: repeat(12, minmax(0, 1fr));
            gap: 1rem;
        }}

        .pd-workflow-item {{
            overflow: hidden;
            background: var(--pd-surface);
            border-radius: 14px;
        }}

        .pd-workflow-item:first-child {{
            grid-column: span 7;
        }}

        .pd-workflow-item:nth-child(2) {{
            grid-column: span 5;
            background: var(--pd-mint);
        }}

        .pd-workflow-item:nth-child(3) {{
            display: grid;
            grid-column: span 12;
            grid-template-columns: 0.8fr 1.2fr;
            align-items: center;
            background: var(--pd-forest);
            color: var(--pd-white);
        }}

        .pd-workflow-copy {{
            padding: clamp(1.5rem, 3vw, 2.4rem);
        }}

        .pd-step {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            margin-bottom: 2rem;
            border-radius: 50%;
            background: var(--pd-ink);
            color: var(--pd-white);
            font-size: 0.75rem;
            font-weight: 800;
        }}

        .pd-workflow-item:nth-child(3) .pd-step {{
            background: var(--pd-lime);
            color: #17351f;
        }}

        .pd-workflow h3 {{
            margin: 0;
            color: inherit;
            font-size: clamp(1.5rem, 2.6vw, 2.5rem);
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1.08;
            text-wrap: balance;
        }}

        .pd-workflow p {{
            max-width: 52ch;
            margin: 1rem 0 1.4rem;
            color: var(--pd-ink-soft);
            line-height: 1.7;
            text-wrap: pretty;
        }}

        .pd-workflow-item:nth-child(3) p {{
            color: #cde0d7;
        }}

        .pd-text-link {{
            display: inline-flex;
            min-height: 44px;
            align-items: center;
            gap: 0.5rem;
            padding-block: 0.55rem;
            color: var(--pd-green) !important;
            font-size: 0.88rem;
            font-weight: 800;
            text-decoration: none !important;
        }}

        .pd-workflow-item:nth-child(3) .pd-text-link {{
            color: var(--pd-lime) !important;
        }}

        .pd-text-link::after {{
            content: '→';
            transition: transform 160ms ease;
        }}

        .pd-text-link:hover::after {{
            transform: translateX(4px);
        }}

        .pd-workflow-image {{
            overflow: hidden;
            border-top: 1px solid var(--pd-line);
        }}

        .pd-workflow-image img {{
            display: block;
            width: 100%;
            height: 260px;
            border: 1px solid var(--pd-image-line);
            object-fit: cover;
            object-position: left top;
        }}

        .pd-workflow-item:nth-child(3) .pd-workflow-image {{
            height: 100%;
            border-top: 0;
            border-left: 1px solid #3f7661;
        }}

        .pd-workflow-item:nth-child(3) .pd-workflow-image img {{
            height: 100%;
            min-height: 360px;
        }}

        .pd-latest {{
            margin-inline: calc(clamp(1rem, 3vw, 2.5rem) * -1);
            padding: clamp(4rem, 8vw, 7rem) clamp(1rem, 3vw, 2.5rem);
            background: #dfe9e3;
        }}

        .pd-latest-inner {{
            max-width: 1160px;
            margin: 0 auto;
        }}

        .pd-latest-heading {{
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 2rem;
            padding-bottom: 2rem;
            border-bottom: 1px solid #a8bcb2;
        }}

        .pd-latest-heading h2 {{
            max-width: 13ch;
        }}

        .pd-new-label {{
            flex: 0 0 auto;
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            background: var(--pd-lime);
            color: #17351f;
            font-size: 0.72rem;
            font-weight: 800;
        }}

        .pd-latest-row {{
            display: grid;
            grid-template-columns: 0.2fr minmax(180px, 0.65fr) minmax(280px, 1.35fr) auto;
            gap: clamp(1rem, 3vw, 2.5rem);
            align-items: start;
            padding: 2rem 0;
            border-bottom: 1px solid #a8bcb2;
        }}

        .pd-latest-index {{
            color: #557166;
            font-size: 0.78rem;
            font-weight: 800;
        }}

        .pd-latest-row h3 {{
            margin: 0;
            color: var(--pd-ink);
            font-size: 1.1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}

        .pd-latest-row p {{
            max-width: 64ch;
            margin: 0;
            color: var(--pd-ink-soft);
            font-size: 0.92rem;
            line-height: 1.7;
        }}

        .pd-latest-row .pd-text-link {{
            white-space: nowrap;
        }}

        .pd-insights {{
            padding: clamp(4.5rem, 9vw, 8rem) 0;
        }}

        .pd-insight-lead {{
            display: grid;
            grid-template-columns: minmax(280px, 0.7fr) minmax(0, 1.3fr);
            overflow: hidden;
            border-radius: 14px;
            background: var(--pd-forest);
            color: var(--pd-white);
        }}

        .pd-insight-copy {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: clamp(1.75rem, 4vw, 3.5rem);
        }}

        .pd-insight-copy h3,
        .pd-insight-card h3 {{
            margin: 0;
            color: inherit;
            font-size: clamp(1.45rem, 2.5vw, 2.4rem);
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1.08;
            text-wrap: balance;
        }}

        .pd-insight-copy p,
        .pd-insight-card p {{
            margin: 1rem 0 1.25rem;
            color: #cde0d7;
            font-size: 0.92rem;
            line-height: 1.7;
            text-wrap: pretty;
        }}

        .pd-insight-lead figure,
        .pd-insight-card figure {{
            margin: 0;
            overflow: hidden;
        }}

        .pd-insight-lead figure {{
            min-height: 420px;
            border-left: 1px solid #3f7661;
        }}

        .pd-insight-lead img {{
            display: block;
            width: 100%;
            height: 100%;
            min-height: 420px;
            border: 1px solid var(--pd-image-line);
            object-fit: cover;
            object-position: center;
        }}

        .pd-insight-lead .pd-text-link {{
            color: var(--pd-lime) !important;
        }}

        .pd-insight-pair {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}

        .pd-insight-card {{
            display: flex;
            min-width: 0;
            overflow: hidden;
            border-radius: 14px;
            background: var(--pd-surface);
            color: var(--pd-ink);
            flex-direction: column;
        }}

        .pd-insight-card figure {{
            display: flex;
            min-height: 270px;
            align-items: center;
            border-bottom: 1px solid var(--pd-line);
            background: #eef1ef;
        }}

        .pd-insight-card img {{
            display: block;
            width: 100%;
            height: 270px;
            border: 1px solid var(--pd-image-line);
            object-fit: contain;
        }}

        .pd-insight-card .pd-insight-copy {{
            padding: clamp(1.5rem, 3vw, 2.25rem);
        }}

        .pd-insight-card p {{
            color: var(--pd-ink-soft);
        }}

        .pd-tools-header {{
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 2rem;
            margin-bottom: 3rem;
        }}

        .pd-tools-header p {{
            max-width: 46ch;
            margin: 0;
            color: var(--pd-ink-soft);
            line-height: 1.7;
        }}

        .pd-tool-groups {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: clamp(1.5rem, 4vw, 4rem);
        }}

        .pd-tool-group {{
            min-width: 0;
        }}

        .pd-tool-group-title {{
            margin: 0 0 1rem;
            color: var(--pd-green);
            font-size: 0.8rem;
            font-weight: 800;
        }}

        .pd-tool-link {{
            display: block;
            padding: 1.15rem 0;
            border-top: 1px solid var(--pd-line);
            text-decoration: none !important;
            transition: color 150ms ease;
        }}

        .pd-tool-link:last-child {{
            border-bottom: 1px solid var(--pd-line);
        }}

        .pd-tool-link strong {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            color: var(--pd-ink);
            font-size: 0.98rem;
        }}

        .pd-tool-link strong::after {{
            color: var(--pd-green);
            content: '↗';
            font-size: 0.88rem;
            transition: transform 160ms ease;
        }}

        .pd-tool-link span {{
            display: block;
            margin-top: 0.45rem;
            color: var(--pd-ink-soft);
            font-size: 0.8rem;
            line-height: 1.55;
        }}

        .pd-tool-link:hover strong {{
            color: var(--pd-green);
        }}

        .pd-tool-link:hover strong::after {{
            transform: translate(2px, -2px);
        }}

        .pd-closing {{
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: clamp(2rem, 6vw, 6rem);
            align-items: center;
            padding: clamp(2.5rem, 5vw, 4.5rem);
            border-radius: 16px;
            background: var(--pd-lime);
            color: #17351f;
        }}

        .pd-closing h2 {{
            max-width: 13ch;
            color: #17351f !important;
        }}

        .pd-closing-copy p {{
            max-width: 48ch;
            margin: 0 0 1.5rem;
            color: #38543f;
            line-height: 1.7;
        }}

        .pd-home .pd-closing .pd-button-primary {{
            background-color: var(--pd-forest);
            color: var(--pd-white) !important;
        }}

        .pd-footer {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 2.25rem 0 0;
            color: #587066;
            font-size: 0.76rem;
            line-height: 1.6;
        }}

        .pd-footer-links {{
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .pd-footer a {{
            color: #335c4b !important;
            font-weight: 700;
            text-decoration: none !important;
        }}

        .pd-footer a:hover {{
            text-decoration: underline !important;
            text-underline-offset: 3px;
        }}

        @media (max-width: 900px) {{
            .pd-hero {{
                grid-template-columns: 1fr;
            }}

            .pd-hero-copy {{
                min-height: 520px;
            }}

            .pd-hero-visual {{
                padding: 2rem 0 3rem 1.5rem;
            }}

            .pd-preview img {{
                height: 340px;
            }}

            .pd-section-header,
            .pd-closing {{
                grid-template-columns: 1fr;
            }}

            .pd-workflow-item:first-child,
            .pd-workflow-item:nth-child(2) {{
                grid-column: span 6;
            }}

            .pd-workflow-item:nth-child(3) {{
                grid-template-columns: 1fr;
            }}

            .pd-workflow-item:nth-child(3) .pd-workflow-image {{
                border-top: 1px solid #3f7661;
                border-left: 0;
            }}

            .pd-workflow-item:nth-child(3) .pd-workflow-image img {{
                min-height: 280px;
            }}

            .pd-latest-row {{
                grid-template-columns: 45px minmax(160px, 0.7fr) minmax(240px, 1.3fr);
            }}

            .pd-latest-row .pd-text-link {{
                grid-column: 2 / -1;
            }}

            .pd-tools-header {{
                align-items: start;
                flex-direction: column;
            }}

            .pd-tool-groups {{
                grid-template-columns: 1fr 1fr;
            }}
        }}

        @container (max-width: 900px) {{
            .pd-hero {{
                grid-template-columns: 1fr;
            }}

            .pd-hero-copy {{
                min-height: 520px;
            }}

            .pd-hero-visual {{
                padding: 2rem 0 3rem 1.5rem;
            }}

            .pd-preview img {{
                height: 340px;
            }}

            .pd-section-header,
            .pd-closing {{
                grid-template-columns: 1fr;
            }}

            .pd-workflow-item:first-child,
            .pd-workflow-item:nth-child(2) {{
                grid-column: span 6;
            }}

            .pd-workflow-item:nth-child(3) {{
                grid-template-columns: 1fr;
            }}

            .pd-workflow-item:nth-child(3) .pd-workflow-image {{
                border-top: 1px solid #3f7661;
                border-left: 0;
            }}

            .pd-workflow-item:nth-child(3) .pd-workflow-image img {{
                min-height: 280px;
            }}

            .pd-latest-row {{
                grid-template-columns: 45px minmax(160px, 0.7fr) minmax(240px, 1.3fr);
            }}

            .pd-latest-row .pd-text-link {{
                grid-column: 2 / -1;
            }}

            .pd-tools-header {{
                align-items: start;
                flex-direction: column;
            }}

            .pd-tool-groups {{
                grid-template-columns: 1fr 1fr;
            }}

            .pd-insight-lead {{
                grid-template-columns: 1fr;
            }}

            .pd-insight-lead figure {{
                min-height: 320px;
                border-top: 1px solid #3f7661;
                border-left: 0;
            }}

            .pd-insight-lead img {{
                min-height: 320px;
            }}
        }}

        @media (max-width: 640px) {{
            [data-testid="stMainBlockContainer"] {{
                padding-inline: 0.75rem;
            }}

            .pd-masthead-note {{
                display: none;
            }}

            .pd-hero {{
                min-height: auto;
            }}

            .pd-hero-copy {{
                min-height: auto;
                padding: 2.75rem 1.35rem 2.25rem;
            }}

            .pd-hero h1 {{
                max-width: 10ch;
                font-size: clamp(2.65rem, 15vw, 4.2rem);
            }}

            .pd-actions {{
                align-items: stretch;
                flex-direction: column;
            }}

            .pd-button {{
                width: 100%;
            }}

            .pd-hero-visual {{
                padding: 1.25rem 0 2.5rem 1.25rem;
            }}

            .pd-preview img {{
                width: 175%;
                height: 260px;
            }}

            .pd-hero-chip {{
                right: 0.85rem;
                bottom: 1rem;
                max-width: 190px;
            }}

            .pd-context {{
                align-items: flex-start;
                flex-direction: column;
                padding-inline: 0.5rem;
            }}

            .pd-section {{
                padding: 4.5rem 0;
            }}

            .pd-section-header {{
                gap: 1.5rem;
            }}

            .pd-workflow {{
                display: block;
            }}

            .pd-workflow-item + .pd-workflow-item {{
                margin-top: 1rem;
            }}

            .pd-workflow-image img {{
                height: 220px;
            }}

            .pd-latest-heading {{
                align-items: flex-start;
                flex-direction: column;
            }}

            .pd-latest {{
                margin-inline: 0;
                padding-inline: 1.25rem;
            }}

            .pd-latest-row {{
                grid-template-columns: 34px 1fr;
                gap: 0.75rem 1rem;
            }}

            .pd-latest-row p,
            .pd-latest-row .pd-text-link {{
                grid-column: 2;
            }}

            .pd-tool-groups {{
                grid-template-columns: 1fr;
                gap: 2.5rem;
            }}

            .pd-closing {{
                padding: 2.25rem 1.25rem;
            }}

            .pd-footer {{
                align-items: flex-start;
                flex-direction: column;
            }}
        }}

        @container (max-width: 640px) {{
            .pd-hero {{
                min-height: auto;
            }}

            .pd-hero-copy {{
                min-height: auto;
                padding: 2.75rem 1.35rem 2.25rem;
            }}

            .pd-hero h1 {{
                max-width: 10ch;
                font-size: clamp(2.65rem, 15vw, 4.2rem);
            }}

            .pd-actions {{
                align-items: stretch;
                flex-direction: column;
            }}

            .pd-button {{
                width: 100%;
            }}

            .pd-hero-visual {{
                padding: 1.25rem 0 2.5rem 1.25rem;
            }}

            .pd-preview img {{
                width: 175%;
                height: 260px;
            }}

            .pd-hero-chip {{
                right: 0.85rem;
                bottom: 1rem;
                max-width: 190px;
            }}

            .pd-context {{
                align-items: flex-start;
                flex-direction: column;
                padding-inline: 0.5rem;
            }}

            .pd-workflow {{
                display: block;
            }}

            .pd-workflow-item + .pd-workflow-item {{
                margin-top: 1rem;
            }}

            .pd-workflow-image img {{
                height: 220px;
            }}

            .pd-latest {{
                margin-inline: 0;
                padding-inline: 1.25rem;
            }}

            .pd-latest-heading {{
                align-items: flex-start;
                flex-direction: column;
            }}

            .pd-latest-row {{
                grid-template-columns: 34px 1fr;
                gap: 0.75rem 1rem;
            }}

            .pd-latest-row p,
            .pd-latest-row .pd-text-link {{
                grid-column: 2;
            }}

            .pd-tool-groups {{
                grid-template-columns: 1fr;
                gap: 2.5rem;
            }}

            .pd-insight-pair {{
                grid-template-columns: 1fr;
            }}

            .pd-insight-lead figure,
            .pd-insight-lead img {{
                min-height: 250px;
            }}

            .pd-insight-card figure {{
                min-height: 220px;
            }}

            .pd-insight-card img {{
                height: 220px;
            }}

            .pd-closing {{
                padding: 2.25rem 1.25rem;
            }}

            .pd-footer {{
                align-items: flex-start;
                flex-direction: column;
            }}
        }}

        @media (prefers-reduced-motion: no-preference) {{
            .pd-hero-copy > * {{
                animation: pd-enter 650ms cubic-bezier(0.22, 1, 0.36, 1) both;
            }}

            .pd-hero-copy > :nth-child(2) {{
                animation-delay: 80ms;
            }}

            .pd-hero-copy > :nth-child(3) {{
                animation-delay: 150ms;
            }}

            .pd-hero-copy > :nth-child(4) {{
                animation-delay: 220ms;
            }}

            .pd-preview {{
                animation: pd-preview-enter 800ms 140ms cubic-bezier(0.22, 1, 0.36, 1) both;
            }}

            @keyframes pd-enter {{
                from {{ opacity: 0; transform: translateY(12px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}

            @keyframes pd-preview-enter {{
                from {{ opacity: 0.75; transform: translateX(28px); }}
                to {{ opacity: 1; transform: translateX(0); }}
            }}
        }}
    </style>

    <main class="pd-home">
        <header class="pd-masthead">
            <div class="pd-brand">
                <img src="{logo}" alt="Panen Dividen logo">
                <span>Panen Dividen</span>
            </div>
            <div class="pd-masthead-note">Ruang kerja investasi berbasis data</div>
        </header>

        <section class="pd-hero" aria-labelledby="pd-hero-title">
            <div class="pd-hero-copy">
                <div class="pd-hero-kicker">Dibangun untuk investor Indonesia</div>
                <h1 id="pd-hero-title">Data jernih. Keputusan lebih tenang.</h1>
                <p>
                    Satukan riset saham dividen, analisis portofolio, timing historis,
                    simulasi compounding, dan pengujian strategi dalam satu ruang kerja.
                </p>
                <div class="pd-actions">
                    <a class="pd-button pd-button-primary" href="./stock_picker" target="_self">
                        Mulai dari Ranking Screener <span aria-hidden="true">→</span>
                    </a>
                    <a class="pd-button pd-button-secondary" href="./porto_overview" target="_self">
                        Analisis portofolio
                    </a>
                </div>
            </div>
            <div class="pd-hero-visual" aria-label="Portfolio Analytics preview">
                <div class="pd-preview">
                    <div class="pd-preview-bar">
                        <span>Portfolio Analytics</span>
                        <span class="pd-preview-status">Preview dashboard</span>
                    </div>
                    <img
                        src="{portfolio_preview}"
                        alt="Portfolio Analytics yang menampilkan annual dividend income, target progress, yield on cost, nilai investasi, dan holdings"
                        decoding="async"
                        fetchpriority="high"
                    >
                </div>
                <div class="pd-hero-chip">
                    Dari yield on cost hingga target income, lihat portofolio sebagai satu cerita yang utuh.
                </div>
            </div>
        </section>

        <div class="pd-context" aria-label="Product coverage">
            <span>Fokus IDX dan investor ritel</span>
            <span>Riset JKSE dan S&amp;P 500</span>
            <span>9 alat analisis yang saling terhubung</span>
        </div>

        <section class="pd-section" aria-labelledby="pd-workflow-title">
            <div class="pd-section-header">
                <h2 id="pd-workflow-title">Satu alur dari ide hingga evaluasi.</h2>
                <p>
                    Mulai dari pertanyaan sederhana, telusuri data yang relevan, lalu uji dampaknya
                    terhadap tujuan income dan strategi Anda.
                </p>
            </div>

            <div class="pd-workflow">
                <article class="pd-workflow-item">
                    <div class="pd-workflow-copy">
                        <span class="pd-step">1</span>
                        <h3>Temukan kandidat dengan konteks, bukan sekadar yield.</h3>
                        <p>
                            Ranking Screener merangkum kualitas dividen, valuasi, pertumbuhan,
                            profitabilitas, dan risiko untuk membantu menyusun shortlist yang lebih tajam.
                        </p>
                        <a class="pd-text-link" href="./stock_picker" target="_self">Buka screener</a>
                    </div>
                    <div class="pd-workflow-image">
                        <img
                            src="{ranking_preview}"
                            alt="Ranking Screener terbaru dengan peringkat saham, dividend yield, Dividend Score, valuasi, profit margin, dan return"
                            loading="lazy"
                            decoding="async"
                        >
                    </div>
                </article>

                <article class="pd-workflow-item">
                    <div class="pd-workflow-copy">
                        <span class="pd-step">2</span>
                        <h3>Pahami kapan income historis terbentuk.</h3>
                        <p>
                            Jelajahi pola pembayaran dividen per bulan, bandingkan yield historis,
                            dan lanjutkan ke analisis timing sebelum serta sesudah ex-date.
                        </p>
                        <a class="pd-text-link" href="./calendar" target="_self">Lihat payout calendar</a>
                    </div>
                    <div class="pd-workflow-image">
                        <img
                            src="{calendar_preview}"
                            alt="Payout Calendar bulanan dengan tanggal dividen dan saham ber-yield tertinggi pada bulan April"
                            loading="lazy"
                            decoding="async"
                        >
                    </div>
                </article>

                <article class="pd-workflow-item">
                    <div class="pd-workflow-copy">
                        <span class="pd-step">3</span>
                        <h3>Uji dampaknya terhadap portofolio Anda.</h3>
                        <p>
                            Pantau income, diversifikasi sektor, dan target portofolio. Bandingkan juga
                            skenario DRIP dan tanpa DRIP sebelum membuat rencana jangka panjang.
                        </p>
                        <a class="pd-text-link" href="./simulator" target="_self">Jalankan simulasi</a>
                    </div>
                    <div class="pd-workflow-image">
                        <img
                            src="{compounding_preview}"
                            alt="Compounding Simulator yang membandingkan total investasi, nilai portofolio, dividen, yield on cost, dan pertumbuhan tahunan"
                            loading="lazy"
                            decoding="async"
                        >
                    </div>
                </article>
            </div>
        </section>

        <section class="pd-latest" aria-labelledby="pd-latest-title">
            <div class="pd-latest-inner">
                <div class="pd-latest-heading">
                    <h2 id="pd-latest-title">Kemampuan terbaru untuk riset yang lebih dalam.</h2>
                    <span class="pd-new-label">Latest tools</span>
                </div>

                <article class="pd-latest-row">
                    <span class="pd-latest-index">01</span>
                    <h3>Historical DRIP Simulator</h3>
                    <p>
                        Bandingkan hasil reinvestasi dan tanpa reinvestasi memakai histori satu saham
                        atau beberapa saham, lengkap dengan alokasi, residual cash, dan transaction log.
                    </p>
                    <a class="pd-text-link" href="./simulator" target="_self">Coba simulator</a>
                </article>

                <article class="pd-latest-row">
                    <span class="pd-latest-index">02</span>
                    <h3>Best Timing Analysis</h3>
                    <p>
                        Pelajari bulan yang relatif murah, distribusi titik rendah sebelum ex-date,
                        serta pola recovery setelah ex-date dengan data historis yang dapat diaudit.
                    </p>
                    <a class="pd-text-link" href="./best_timing" target="_self">Pelajari timing</a>
                </article>

                <article class="pd-latest-row">
                    <span class="pd-latest-index">03</span>
                    <h3>Growth at a Discount</h3>
                    <p>
                        Saring perusahaan IDX yang tetap profitable dan bertumbuh saat multiple P/E
                        serta P/S berada di bawah rentang historisnya. Hasil ditampilkan sebagai skenario, bukan target harga.
                    </p>
                    <a class="pd-text-link" href="./backtester" target="_self">Buka long-term screen</a>
                </article>

                <article class="pd-latest-row">
                    <span class="pd-latest-index">04</span>
                    <h3>Short-Term Swing Trading Lab</h3>
                    <p>
                        Riset strategi mean-reversion, pullback, relative strength, dan breakout
                        dengan holdout validation untuk horizon sekitar 1–10 sesi.
                    </p>
                    <a class="pd-text-link" href="./day_trading" target="_self">Masuk ke lab</a>
                </article>
            </div>
        </section>

        <section class="pd-insights" aria-labelledby="pd-insights-title">
            <div class="pd-section-header">
                <h2 id="pd-insights-title">Baca pasar dari lebih dari satu sudut.</h2>
                <p>
                    Beralih dari gambaran pasar ke perbandingan saham dan pola historis tanpa kehilangan
                    konteks di sepanjang proses riset.
                </p>
            </div>

            <article class="pd-insight-lead">
                <div class="pd-insight-copy">
                    <h3>Lihat arah pasar dalam satu pandangan.</h3>
                    <p>
                        Market Heatmap menyusun return harian berdasarkan sektor, industri, dan market cap
                        agar breadth serta pergerakan yang paling menonjol lebih cepat terbaca.
                    </p>
                    <a class="pd-text-link" href="./market_watch" target="_self">Buka market heatmap</a>
                </div>
                <figure>
                    <img
                        src="{market_preview}"
                        alt="Market Heatmap IDX berwarna merah dan hijau yang menampilkan return satu hari berdasarkan market cap"
                        loading="lazy"
                        decoding="async"
                    >
                </figure>
            </article>

            <div class="pd-insight-pair">
                <article class="pd-insight-card">
                    <figure>
                        <img
                            src="{comparison_preview}"
                            alt="Scatter plot perbandingan saham berdasarkan dividend yield dan PE ratio"
                            loading="lazy"
                            decoding="async"
                        >
                    </figure>
                    <div class="pd-insight-copy">
                        <h3>Bandingkan trade-off, bukan satu angka.</h3>
                        <p>
                            Tempatkan beberapa saham pada dua metrik sekaligus untuk melihat posisi relatif,
                            outlier, dan kandidat yang lebih sesuai dengan fokus riset Anda.
                        </p>
                        <a class="pd-text-link" href="./stock_comparison" target="_self">Bandingkan saham</a>
                    </div>
                </article>

                <article class="pd-insight-card">
                    <figure>
                        <img
                            src="{timing_preview}"
                            alt="Analisis Best Timing dengan pola harga bulanan dan distribusi hari sebelum ex-date"
                            loading="lazy"
                            decoding="async"
                        >
                    </figure>
                    <div class="pd-insight-copy">
                        <h3>Ubah timing menjadi distribusi yang dapat diuji.</h3>
                        <p>
                            Pelajari seasonality dan jarak historis menuju ex-date sebagai pola data,
                            bukan kepastian waktu beli.
                        </p>
                        <a class="pd-text-link" href="./best_timing" target="_self">Pelajari timing historis</a>
                    </div>
                </article>
            </div>
        </section>

        <section class="pd-section" aria-labelledby="pd-tools-title">
            <div class="pd-tools-header">
                <h2 id="pd-tools-title">Semua alat, tersusun sesuai cara Anda bekerja.</h2>
                <p>
                    Pilih titik masuk yang paling dekat dengan pertanyaan Anda. Setiap halaman dibuat
                    untuk menjawab satu keputusan riset dengan lebih jelas.
                </p>
            </div>

            <div class="pd-tool-groups">
                <div class="pd-tool-group">
                    <p class="pd-tool-group-title">Dividend research</p>
                    <a class="pd-tool-link" href="./stock_picker" target="_self">
                        <strong>Ranking Screener</strong>
                        <span>Ranking dividen, valuasi, fundamental, dan risiko dalam satu riset saham.</span>
                    </a>
                    <a class="pd-tool-link" href="./calendar" target="_self">
                        <strong>Payout Calendar</strong>
                        <span>Pola pembayaran historis dan yield per bulan untuk JKSE atau S&amp;P 500.</span>
                    </a>
                    <a class="pd-tool-link" href="./best_timing" target="_self">
                        <strong>Best Timing</strong>
                        <span>Seasonality, pola sebelum ex-date, dan recovery setelah pembayaran.</span>
                    </a>
                    <a class="pd-tool-link" href="./porto_overview" target="_self">
                        <strong>Portfolio Analytics</strong>
                        <span>Income, yield on cost, target, diversifikasi, dan timeline dividen.</span>
                    </a>
                    <a class="pd-tool-link" href="./simulator" target="_self">
                        <strong>Compounding Simulator</strong>
                        <span>Proyeksi sederhana dan historical replay dengan atau tanpa DRIP.</span>
                    </a>
                </div>

                <div class="pd-tool-group">
                    <p class="pd-tool-group-title">Market intelligence</p>
                    <a class="pd-tool-link" href="./market_watch" target="_self">
                        <strong>Market Heatmap</strong>
                        <span>Snapshot breadth, sektor, return, valuasi, yield, dan indikator makro.</span>
                    </a>
                    <a class="pd-tool-link" href="./stock_comparison" target="_self">
                        <strong>Stock Comparison</strong>
                        <span>Bandingkan 2–5 saham pada income, quality, growth, return, dan risk.</span>
                    </a>
                </div>

                <div class="pd-tool-group">
                    <p class="pd-tool-group-title">Strategy research</p>
                    <a class="pd-tool-link" href="./backtester" target="_self">
                        <strong>Position Trading</strong>
                        <span>Growth at a Discount screen untuk kandidat IDX berorientasi jangka panjang.</span>
                    </a>
                    <a class="pd-tool-link" href="./day_trading" target="_self">
                        <strong>Short-Term Swing Trading Lab</strong>
                        <span>Optimasi, validasi, backtest, dan scanner setup berbasis daily close.</span>
                    </a>
                </div>
            </div>
        </section>

        <section class="pd-closing" aria-labelledby="pd-closing-title">
            <h2 id="pd-closing-title">Mulai dari satu saham yang ingin Anda pahami.</h2>
            <div class="pd-closing-copy">
                <p>
                    Buka Ranking Screener, pilih pasar, lalu telusuri dividen, fundamental,
                    valuasi, price action, dan simulasi compounding dari satu halaman.
                </p>
                <a class="pd-button pd-button-primary" href="./stock_picker" target="_self">
                    Mulai riset saham <span aria-hidden="true">→</span>
                </a>
            </div>
        </section>

        <footer class="pd-footer">
            <span>Data dapat terlambat atau mengandung kesalahan. Verifikasi dengan sumber resmi; Panen Dividen bukan rekomendasi investasi atau jaminan hasil.</span>
            <nav class="pd-footer-links" aria-label="Community and support links">
                <a href="https://www.reddit.com/r/panendividen" target="_blank" rel="noreferrer">Community</a>
                <a href="https://github.com/mitbal/harvest/issues" target="_blank" rel="noreferrer">Feedback</a>
                <a href="https://blog.panendividen.com?utm_source=pd_web" target="_blank" rel="noreferrer">Blog</a>
                <a href="https://trakteer.id/mitbal" target="_blank" rel="noreferrer">Support</a>
            </nav>
        </footer>
    </main>
    '''
)
