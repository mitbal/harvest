import html

import streamlit as st

from harvest.premium import BENEFITS, BMC_URL, TIERS, TRAKTEER_URL, WHY, current_email, is_premium


# Served by Streamlit static serving (static/home/), so the browser can cache
# and lazy-load them instead of receiving inline base64 on every rerun.
# Regenerate from asset/home/*.png as WebP at 1600px wide when screenshots change.
STATIC_HOME = 'app/static/home'


if 'porto_file' not in st.session_state:
    st.session_state['porto_file'] = 'EMPTY'
if 'history_file' not in st.session_state:
    st.session_state['history_file'] = 'EMPTY'


logo = f'{STATIC_HOME}/favicon.webp'
portfolio_preview = f'{STATIC_HOME}/portfolio_table.webp'
ranking_preview = f'{STATIC_HOME}/screener_table.webp'
calendar_preview = f'{STATIC_HOME}/dividend_monthly_calendar.webp'
compounding_preview = f'{STATIC_HOME}/compounding_simulator.webp'
market_preview = f'{STATIC_HOME}/heatmap_1d_price_return.webp'
comparison_preview = f'{STATIC_HOME}/stock_comparison_scatterplot.webp'
timing_preview = f'{STATIC_HOME}/best_timing_best_time_to_buy.webp'
# Dividend Play walkthrough (Today cards -> See evidence), a muted loop played only while on screen.
# The WebP is the poster and the still shown when the visitor asks for reduced motion.
trade_demo_poster = f'{STATIC_HOME}/dividend_play_demo.webp'
trade_demo_webm = f'{STATIC_HOME}/dividend_play_demo.webm'
trade_demo_mp4 = f'{STATIC_HOME}/dividend_play_demo.mp4'


# Supporter copy and prices come from harvest.premium, so the homepage pitch says
# the same thing as the paywall on the trading pages.
def _support_tiers() -> str:
    monthly = TIERS[0]['min_idr']
    rows = []
    for tier in TIERS:
        months = max(1, round(tier['days'] / 30.5))
        saving = 1 - tier['min_idr'] / months / monthly
        note = f'save {saving:.0%}' if saving > 0.005 else 'no auto-renew'
        best = ' pd-tier-best' if tier is TIERS[-1] and saving > 0.005 else ''
        rows.append(
            f'<div class="pd-tier{best}"><span class="pd-tier-name">{html.escape(tier["name"])}</span>'
            f'<strong>{html.escape(tier["price"])}</strong>'
            f'<span class="pd-tier-note">{html.escape(tier["duration"])} · {html.escape(note)}</span></div>'
        )
    return ''.join(rows)


def _support_panel(supporter: bool) -> str:
    if supporter:
        return (
            '<section class="pd-support pd-support-thanks" id="pd-support" aria-labelledby="pd-support-title">'
            '<div class="pd-support-copy">'
            '<span class="pd-support-eyebrow">Supporter access active</span>'
            '<h2 id="pd-support-title">Thank you for keeping Panen Dividen running.</h2>'
            '<p>Every signal, the evidence for any stock, the research Lab and your trade journal are open to you. '
            'Tonight\'s picks are waiting in My Trades.</p>'
            '<div class="pd-actions"><a class="pd-button pd-button-primary" href="./my_trades" target="_self">'
            'Open My Trades <span aria-hidden="true">→</span></a></div>'
            '</div></section>'
        )
    perks = ''.join(f'<li>{html.escape(b)}</li>' for b in BENEFITS)
    return (
        '<section class="pd-support" id="pd-support" aria-labelledby="pd-support-title">'
        '<div class="pd-support-copy">'
        '<span class="pd-support-eyebrow">Become a supporter</span>'
        '<h2 id="pd-support-title">Unlock every signal for less than a coffee a week.</h2>'
        f'<p>{html.escape(WHY)}</p>'
        f'<ul class="pd-support-perks">{perks}</ul>'
        '</div>'
        '<div class="pd-support-offer">'
        f'<div class="pd-tiers">{_support_tiers()}</div>'
        '<ol class="pd-support-steps">'
        '<li>Log in with Google on any trading page.</li>'
        '<li>Donate on Trakteer or Buy Me a Coffee with that same email.</li>'
        '<li>Access unlocks within minutes, on every device you log in from.</li>'
        '</ol>'
        '<div class="pd-actions">'
        f'<a class="pd-button pd-button-primary" href="{TRAKTEER_URL}" target="_blank" rel="noreferrer">'
        'Support on Trakteer <span aria-hidden="true">→</span></a>'
        f'<a class="pd-button pd-button-secondary" href="{BMC_URL}" target="_blank" rel="noreferrer">'
        'Buy Me a Coffee</a>'
        '</div>'
        '</div></section>'
    )


supporter = is_premium(current_email())
support_panel = _support_panel(supporter)
support_pill = (
    '' if supporter else
    '<a class="pd-support-pill" href="#pd-support">Become a supporter</a>'
)


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
            /* Streamlit's header is fixed and 3.75rem tall: start the masthead just below it */
            padding: 4.25rem clamp(1rem, 3vw, 2.5rem) 4rem;
        }}

        [data-testid="stHeader"] {{
            background: color-mix(in srgb, var(--pd-canvas) 86%, transparent);
        }}

        /* Keep in-page jumps (the Become a supporter pill) from landing under the header */
        .pd-home [id] {{
            scroll-margin-top: 4.5rem;
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

        /* --- Trading tools and supporter pitch --- */
        .pd-masthead-actions {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .pd-home .pd-support-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            min-height: 36px;
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            background: var(--pd-forest);
            color: var(--pd-white) !important;
            font-size: 0.78rem;
            font-weight: 800;
            text-decoration: none !important;
        }}

        .pd-home .pd-support-pill::before {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--pd-lime);
            content: '';
        }}

        .pd-trading {{
            padding: clamp(4.5rem, 9vw, 8rem) 0 0;
        }}

        .pd-trade-demo {{
            display: grid;
            grid-template-columns: minmax(260px, 0.62fr) minmax(0, 1.38fr);
            margin-bottom: 1rem;
            overflow: hidden;
            border-radius: 14px;
            background: var(--pd-forest);
            color: var(--pd-white);
        }}

        .pd-trade-demo-copy {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: clamp(1.75rem, 3.5vw, 3rem);
        }}

        .pd-trade-demo-copy .pd-trade-kicker {{
            color: var(--pd-lime);
        }}

        .pd-trade-demo-copy h3 {{
            margin: 0.6rem 0 0;
            color: inherit;
            font-size: clamp(1.4rem, 2.3vw, 2.1rem);
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1.1;
            text-wrap: balance;
        }}

        .pd-trade-demo-steps {{
            display: grid;
            gap: 0.7rem;
            margin: 1.25rem 0 1.4rem;
            padding: 0;
            list-style: none;
            counter-reset: pd-demo;
            color: #cde0d7;
            font-size: 0.9rem;
            line-height: 1.55;
        }}

        .pd-trade-demo-steps li {{
            display: grid;
            grid-template-columns: 1.6rem 1fr;
            gap: 0.6rem;
            counter-increment: pd-demo;
        }}

        .pd-trade-demo-steps li::before {{
            display: grid;
            width: 1.6rem;
            height: 1.6rem;
            place-items: center;
            border: 1px solid #5d8f7a;
            border-radius: 50%;
            color: var(--pd-lime);
            content: counter(pd-demo);
            font-size: 0.72rem;
            font-weight: 800;
        }}

        .pd-trade-demo-copy .pd-text-link {{
            color: var(--pd-lime) !important;
        }}

        .pd-trade-demo figure {{
            position: relative;
            margin: 0;
            padding: clamp(1rem, 2.5vw, 2rem) 0 clamp(1rem, 2.5vw, 2rem) clamp(0.5rem, 1.5vw, 1rem);
            border-left: 1px solid #3f7661;
        }}

        .pd-trade-demo-frame {{
            position: relative;
            overflow: hidden;
            border: 1px solid #3f7661;
            border-right: 0;
            border-radius: 12px 0 0 12px;
            background: var(--pd-white);
        }}

        .pd-trade-demo video {{
            display: block;
            width: 100%;
            height: auto;
            aspect-ratio: 16 / 10;
            object-fit: cover;
            object-position: left top;
            background: var(--pd-canvas);
        }}

        .pd-trade-demo-toggle {{
            position: absolute;
            right: 0.75rem;
            bottom: 0.75rem;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.4rem 0.75rem;
            border: 0;
            border-radius: 999px;
            background: rgba(5, 45, 34, 0.82);
            color: var(--pd-white);
            cursor: pointer;
            font: inherit;
            font-size: 0.74rem;
            font-weight: 700;
        }}

        .pd-trade-demo-toggle:focus-visible {{
            outline: 2px solid var(--pd-lime);
            outline-offset: 2px;
        }}

        .pd-trade-demo-note {{
            margin: 0.6rem 0 0;
            color: #a9c7ba;
            font-size: 0.74rem;
        }}

        @media (max-width: 900px) {{
            .pd-trade-demo {{
                grid-template-columns: 1fr;
            }}

            .pd-trade-demo figure {{
                padding: 0 0 1.25rem 1.25rem;
                border-left: 0;
            }}
        }}

        .pd-trade-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
        }}

        .pd-trade-card {{
            display: flex;
            flex-direction: column;
            padding: clamp(1.5rem, 3vw, 2.25rem);
            border-radius: 14px;
            background: var(--pd-surface);
        }}

        .pd-trade-card:first-child {{
            background: var(--pd-mint);
        }}

        .pd-trade-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.25rem;
        }}

        .pd-trade-kicker {{
            color: var(--pd-green);
            font-size: 0.78rem;
            font-weight: 800;
        }}

        .pd-trade-icon {{
            font-size: 1.4rem;
            line-height: 1;
        }}

        .pd-trade-card h3 {{
            margin: 0;
            color: var(--pd-ink);
            font-size: clamp(1.35rem, 2vw, 1.7rem);
            font-weight: 800;
            letter-spacing: -0.03em;
        }}

        .pd-trade-card > p {{
            margin: 0.75rem 0 1.25rem;
            color: var(--pd-ink-soft);
            font-size: 0.92rem;
            line-height: 1.7;
        }}

        .pd-trade-access {{
            display: grid;
            gap: 0.55rem;
            margin: auto 0 0.75rem;
            padding: 1rem 0 0;
            border-top: 1px solid var(--pd-line);
            list-style: none;
            font-size: 0.82rem;
            line-height: 1.55;
        }}

        .pd-trade-access li {{
            display: grid;
            grid-template-columns: 86px 1fr;
            gap: 0.75rem;
            margin: 0;
            color: var(--pd-ink-soft);
        }}

        .pd-trade-access b {{
            color: var(--pd-ink);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding-top: 0.1rem;
        }}

        .pd-trade-access li.pd-plus b {{
            color: var(--pd-green);
        }}

        .pd-support {{
            display: grid;
            grid-template-columns: minmax(0, 1.05fr) minmax(300px, 0.95fr);
            gap: clamp(2rem, 5vw, 4.5rem);
            margin-top: 1rem;
            padding: clamp(2rem, 5vw, 4rem);
            border-radius: 16px;
            background: var(--pd-forest);
            color: var(--pd-white);
            scroll-margin-top: 5rem;
        }}

        .pd-support-thanks {{
            grid-template-columns: 1fr;
        }}

        .pd-support-eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            color: #d3e9de;
            font-size: 0.82rem;
            font-weight: 700;
        }}

        .pd-support-eyebrow::before {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--pd-lime);
            content: '';
        }}

        .pd-support h2 {{
            max-width: 16ch;
            margin: 1rem 0 0;
            color: var(--pd-white) !important;
            font-size: clamp(2rem, 3.6vw, 3.4rem);
            font-weight: 700;
            letter-spacing: -0.04em;
            line-height: 1.04;
            text-wrap: balance;
        }}

        .pd-support-copy > p {{
            max-width: 52ch;
            margin: 1.25rem 0 0;
            color: #cde0d7;
            line-height: 1.75;
        }}

        .pd-support-perks {{
            display: grid;
            gap: 0.7rem;
            margin: 1.75rem 0 0;
            padding: 0;
            list-style: none;
        }}

        .pd-support-perks li {{
            position: relative;
            margin: 0;
            padding-left: 1.75rem;
            color: var(--pd-white);
            font-size: 0.92rem;
            line-height: 1.55;
        }}

        .pd-support-perks li::before {{
            position: absolute;
            top: 0;
            left: 0;
            color: var(--pd-lime);
            content: '✓';
            font-weight: 800;
        }}

        .pd-support-offer {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-width: 0;
        }}

        .pd-tiers {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.6rem;
        }}

        .pd-tier {{
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
            padding: 1rem;
            border: 1px solid #3f7661;
            border-radius: 12px;
            background: var(--pd-forest-deep);
        }}

        .pd-tier-best {{
            border-color: var(--pd-lime);
            background: var(--pd-lime);
            color: #17351f;
        }}

        .pd-tier-name {{
            font-size: 0.76rem;
            font-weight: 800;
            opacity: 0.85;
        }}

        .pd-tier strong {{
            font-size: 1.3rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}

        .pd-tier-note {{
            font-size: 0.72rem;
            line-height: 1.45;
            opacity: 0.85;
        }}

        .pd-support-steps {{
            display: grid;
            gap: 0.45rem;
            margin: 1.5rem 0 0;
            padding-left: 1.2rem;
            color: #cde0d7;
            font-size: 0.84rem;
            line-height: 1.55;
        }}

        .pd-support-steps li {{
            margin: 0;
        }}

        .pd-support .pd-actions {{
            margin-top: 1.5rem;
        }}

        @media (max-width: 900px) {{
            .pd-support {{
                grid-template-columns: 1fr;
            }}
        }}

        @container (max-width: 900px) {{
            .pd-support {{
                grid-template-columns: 1fr;
            }}
        }}

        @media (max-width: 640px) {{
            .pd-trade-grid {{
                grid-template-columns: 1fr;
            }}

            .pd-tiers {{
                grid-template-columns: 1fr;
            }}

            .pd-support {{
                padding: 2.25rem 1.25rem;
            }}

            .pd-trade-access li {{
                grid-template-columns: 1fr;
                gap: 0.15rem;
            }}
        }}

        @container (max-width: 640px) {{
            .pd-trade-grid {{
                grid-template-columns: 1fr;
            }}

            .pd-tiers {{
                grid-template-columns: 1fr;
            }}

            .pd-support {{
                padding: 2.25rem 1.25rem;
            }}

            .pd-trade-access li {{
                grid-template-columns: 1fr;
                gap: 0.15rem;
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
            <div class="pd-masthead-actions">
                <div class="pd-masthead-note">A data-driven investing workspace</div>
                {support_pill}
            </div>
        </header>

        <section class="pd-hero" aria-labelledby="pd-hero-title">
            <div class="pd-hero-copy">
                <div class="pd-hero-kicker">Built for Indonesian investors</div>
                <h1 id="pd-hero-title">Clear data. Calmer decisions.</h1>
                <p>
                    Dividend research, portfolio tracking, market heatmaps and backtested trading
                    signals for IDX and the S&amp;P 500, in one workspace. Free to use, with no ads.
                </p>
                <div class="pd-actions">
                    <a class="pd-button pd-button-primary" href="./stock_picker" target="_self">
                        Start with the Screener <span aria-hidden="true">→</span>
                    </a>
                    <a class="pd-button pd-button-secondary" href="./porto_overview" target="_self">
                        Analyze your portfolio
                    </a>
                </div>
            </div>
            <div class="pd-hero-visual" aria-label="Portfolio preview">
                <div class="pd-preview">
                    <div class="pd-preview-bar">
                        <span>Portfolio</span>
                        <span class="pd-preview-status">Dashboard preview</span>
                    </div>
                    <img
                        src="{portfolio_preview}"
                        alt="Portfolio showing annual dividend income, target progress, yield on cost, investment value, and holdings"
                        decoding="async"
                        fetchpriority="high"
                    >
                </div>
                <div class="pd-hero-chip">
                    From yield on cost to your income goal, saved to your account when you log in.
                </div>
            </div>
        </section>

        <div class="pd-context" aria-label="Product coverage">
            <span>Focused on IDX and retail investors</span>
            <span>Research for JKSE and S&amp;P 500</span>
            <span>11 tools across Dividend, Market and Trading</span>
        </div>

        <section class="pd-section" aria-labelledby="pd-workflow-title">
            <div class="pd-section-header">
                <h2 id="pd-workflow-title">One flow from idea to evaluation.</h2>
                <p>
                    Start with a simple question, dig into the data that matters, then test its impact
                    on your income goals and strategy.
                </p>
            </div>

            <div class="pd-workflow">
                <article class="pd-workflow-item">
                    <div class="pd-workflow-copy">
                        <span class="pd-step">1</span>
                        <h3>Find candidates with context, not just yield.</h3>
                        <p>
                            Rank stocks by Balanced, Income, Growth or Potential, filter for Syariah or
                            non-state-owned names, then open one stock for its dividends, financials,
                            multiples, intrinsic value, price and buy timing.
                        </p>
                        <a class="pd-text-link" href="./stock_picker" target="_self">Open the screener</a>
                    </div>
                    <div class="pd-workflow-image">
                        <img
                            src="{ranking_preview}"
                            alt="Screener with stock rankings, dividend yield, Dividend Score, valuation, profit margin, and return"
                            loading="lazy"
                            decoding="async"
                        >
                    </div>
                </article>

                <article class="pd-workflow-item">
                    <div class="pd-workflow-copy">
                        <span class="pd-step">2</span>
                        <h3>Understand when historical income shows up.</h3>
                        <p>
                            See a full year of payouts at a glance, zoom into a single month, or list each
                            month's payers by yield. Share the Full Year view as an image.
                        </p>
                        <a class="pd-text-link" href="./calendar" target="_self">View the payout calendar</a>
                    </div>
                    <div class="pd-workflow-image">
                        <img
                            src="{calendar_preview}"
                            alt="Dividend Calendar month view with dividend dates and the highest-yielding stocks in April"
                            loading="lazy"
                            decoding="async"
                        >
                    </div>
                </article>

                <article class="pd-workflow-item">
                    <div class="pd-workflow-copy">
                        <span class="pd-step">3</span>
                        <h3>Test the impact on your portfolio.</h3>
                        <p>
                            Track income, sectors and your income goal in Portfolio. Then simulate a fixed
                            yield, a single stock or a whole portfolio, with and without DRIP.
                        </p>
                        <a class="pd-text-link" href="./simulator" target="_self">Run a simulation</a>
                    </div>
                    <div class="pd-workflow-image">
                        <img
                            src="{compounding_preview}"
                            alt="Compounding Simulator comparing total invested, portfolio value, dividends, yield on cost, and annual growth"
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
                    <h2 id="pd-latest-title">New capabilities for deeper research.</h2>
                    <span class="pd-new-label">Latest tools</span>
                </div>

                <article class="pd-latest-row">
                    <span class="pd-latest-index">01</span>
                    <h3>Dividend Play: waiting for the dip</h3>
                    <p>
                        See which stocks are inside their entry window but still above the buy-at price,
                        and which announced dividends are still worth buying before the cum date.
                    </p>
                    <a class="pd-text-link" href="./dividend_play" target="_self">Open Dividend Play</a>
                </article>

                <article class="pd-latest-row">
                    <span class="pd-latest-index">02</span>
                    <h3>Market Sell-off Dip</h3>
                    <p>
                        A new Swing Play setup that buys stocks pulled down with the JKSE on a sell-off day,
                        with a limit below the close. Like every published setup, it passed a walk-forward test.
                    </p>
                    <a class="pd-text-link" href="./day_trading" target="_self">See tonight's swings</a>
                </article>

                <article class="pd-latest-row">
                    <span class="pd-latest-index">03</span>
                    <h3>Performance and preset groups in Comparison</h3>
                    <p>
                        Compare total return with dividends on one rebased chart, and load ready-made groups
                        such as the big banks in one click.
                    </p>
                    <a class="pd-text-link" href="./stock_comparison" target="_self">Compare stocks</a>
                </article>

                <article class="pd-latest-row">
                    <span class="pd-latest-index">04</span>
                    <h3>Global indices and currencies</h3>
                    <p>
                        The Heatmap now covers world stock indices and currencies against the rupiah,
                        next to JKSE and the S&amp;P 500.
                    </p>
                    <a class="pd-text-link" href="./market_watch" target="_self">Open the Heatmap</a>
                </article>

                <article class="pd-latest-row">
                    <span class="pd-latest-index">05</span>
                    <h3>Intrinsic value in the Screener</h3>
                    <p>
                        A dividend discount model for each dividend payer, with a low and high range so you can see
                        how much the estimate depends on its assumptions.
                    </p>
                    <a class="pd-text-link" href="./stock_picker" target="_self">Open the Screener</a>
                </article>
            </div>
        </section>

        <section class="pd-section pd-trading" aria-labelledby="pd-trading-title">
            <div class="pd-section-header">
                <h2 id="pd-trading-title">Trading signals, published every evening.</h2>
                <p>
                    Each tool backtests its rules on IDX history, holds out the latest trades, and only
                    publishes setups that pass. Everyone sees the top pick; supporters see all of them.
                </p>
            </div>

            <article class="pd-trade-demo" aria-labelledby="pd-trade-demo-title">
                <div class="pd-trade-demo-copy">
                    <span class="pd-trade-kicker">See it in action</span>
                    <h3 id="pd-trade-demo-title">From tonight's signal to its track record in one click.</h3>
                    <ol class="pd-trade-demo-steps">
                        <li>After the close, Dividend Play lists the stocks inside their entry window.</li>
                        <li>Each card gives the buy-at price, the sell-by date and the dip rule to wait for.</li>
                        <li>See evidence opens every past trade on that stock, with its win rate and returns.</li>
                    </ol>
                    <a class="pd-text-link" href="./dividend_play" target="_self">Open Dividend Play</a>
                </div>
                <figure>
                    <div class="pd-trade-demo-frame">
                        <video
                            muted
                            loop
                            playsinline
                            preload="none"
                            poster="{trade_demo_poster}"
                            aria-label="Dividend Play walkthrough: the entry window cards for tonight, then the backtest evidence for one stock"
                        >
                            <source src="{trade_demo_webm}" type="video/webm">
                            <source src="{trade_demo_mp4}" type="video/mp4">
                        </video>
                        <button class="pd-trade-demo-toggle" type="button" hidden>Pause</button>
                    </div>
                    <figcaption class="pd-trade-demo-note">Supporter view. Free visitors see the top window and the evidence for BBRI.</figcaption>
                </figure>
            </article>

            <div class="pd-trade-grid">
                <article class="pd-trade-card">
                    <div class="pd-trade-top">
                        <span class="pd-trade-kicker">Around the ex-date</span>
                        <span class="pd-trade-icon" aria-hidden="true">🎣</span>
                    </div>
                    <h3>Dividend Play</h3>
                    <p>
                        Buy a dividend payer on a dip ahead of its ex-date, then sell into the run-up. Every
                        past ex-date is one trade, and announced dividends get a buy-before-cum verdict.
                    </p>
                    <ul class="pd-trade-access">
                        <li><b>Free</b><span>The top open entry window and the full evidence for BBRI.</span></li>
                        <li class="pd-plus"><b>Supporters</b><span>Every window, the buy-at price, best exit and past record for each dividend.</span></li>
                    </ul>
                    <a class="pd-text-link" href="./dividend_play" target="_self">Open Dividend Play</a>
                </article>

                <article class="pd-trade-card">
                    <div class="pd-trade-top">
                        <span class="pd-trade-kicker">1 to 10 sessions</span>
                        <span class="pd-trade-icon" aria-hidden="true">⚡</span>
                    </div>
                    <h3>Swing Play</h3>
                    <p>
                        Short swings on liquid IDX stocks, confirmed at the close and bought in the next session.
                        Gorengan stays out, and every setup is backtested on a point-in-time universe.
                    </p>
                    <ul class="pd-trade-access">
                        <li><b>Free</b><span>The top plan for the next session and the evidence for BBRI.</span></li>
                        <li class="pd-plus"><b>Supporters</b><span>Every plan, the evidence for any stock and the research Lab.</span></li>
                    </ul>
                    <a class="pd-text-link" href="./day_trading" target="_self">Open Swing Play</a>
                </article>

                <article class="pd-trade-card">
                    <div class="pd-trade-top">
                        <span class="pd-trade-kicker">Growth at a discount</span>
                        <span class="pd-trade-icon" aria-hidden="true">📈</span>
                    </div>
                    <h3>Value Play</h3>
                    <p>
                        Profitable IDX businesses whose P/E and P/S sit below their own history while revenue
                        and earnings keep growing. Shown as scenarios, not price targets.
                    </p>
                    <ul class="pd-trade-access">
                        <li><b>Free</b><span>The top candidate of the valuation screen.</span></li>
                        <li class="pd-plus"><b>Supporters</b><span>Every candidate, the evidence for any company and the screen Lab.</span></li>
                    </ul>
                    <a class="pd-text-link" href="./backtester" target="_self">Open Value Play</a>
                </article>

                <article class="pd-trade-card">
                    <div class="pd-trade-top">
                        <span class="pd-trade-kicker">Your trading hub</span>
                        <span class="pd-trade-icon" aria-hidden="true">📒</span>
                    </div>
                    <h3>My Trades</h3>
                    <p>
                        Tonight's signals from every tool and the dates coming up, in one place. Track the trades
                        you take and see your closed results.
                    </p>
                    <ul class="pd-trade-access">
                        <li><b>Free</b><span>The top signal from each tool and the upcoming dates.</span></li>
                        <li class="pd-plus"><b>Supporters</b><span>A trade journal with sell-by countdowns and Telegram reminders.</span></li>
                    </ul>
                    <a class="pd-text-link" href="./my_trades" target="_self">Open My Trades</a>
                </article>
            </div>

            {support_panel}
        </section>

        <section class="pd-insights" aria-labelledby="pd-insights-title">
            <div class="pd-section-header">
                <h2 id="pd-insights-title">Read the market from more than one angle.</h2>
                <p>
                    Move from the market-wide picture to stock comparisons and historical patterns
                    without losing context along the way.
                </p>
            </div>

            <article class="pd-insight-lead">
                <div class="pd-insight-copy">
                    <h3>See where the market is heading at a glance.</h3>
                    <p>
                        The Heatmap arranges returns by sector, industry and market cap for JKSE and the
                        S&amp;P 500, and tracks global indices and currencies against the rupiah.
                    </p>
                    <a class="pd-text-link" href="./market_watch" target="_self">Open the Heatmap</a>
                </div>
                <figure>
                    <img
                        src="{market_preview}"
                        alt="Red and green IDX Market Heatmap showing one-day returns by market cap"
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
                            alt="Scatter plot comparing stocks by dividend yield and PE ratio"
                            loading="lazy"
                            decoding="async"
                        >
                    </figure>
                    <div class="pd-insight-copy">
                        <h3>Compare trade-offs, not a single number.</h3>
                        <p>
                            Put up to five stocks side by side in a table, distribution or scatter, then
                            check their dividend history and total return over the same period.
                        </p>
                        <a class="pd-text-link" href="./stock_comparison" target="_self">Compare stocks</a>
                    </div>
                </article>

                <article class="pd-insight-card">
                    <figure>
                        <img
                            src="{timing_preview}"
                            alt="Best Timing analysis with monthly price patterns and the distribution of days before the ex-date"
                            loading="lazy"
                            decoding="async"
                        >
                    </figure>
                    <div class="pd-insight-copy">
                        <h3>Turn timing into a distribution you can test.</h3>
                        <p>
                            Study seasonality and the historical distance to the ex-date as data patterns,
                            not as a guaranteed time to buy.
                        </p>
                        <a class="pd-text-link" href="./best_timing" target="_self">Explore historical timing</a>
                    </div>
                </article>
            </div>
        </section>

        <section class="pd-section" aria-labelledby="pd-tools-title">
            <div class="pd-tools-header">
                <h2 id="pd-tools-title">Every tool, arranged around how you work.</h2>
                <p>
                    Pick the entry point closest to your question. Each page is built to answer
                    one research decision more clearly.
                </p>
            </div>

            <div class="pd-tool-groups">
                <div class="pd-tool-group">
                    <p class="pd-tool-group-title">Dividend</p>
                    <a class="pd-tool-link" href="./stock_picker" target="_self">
                        <strong>Screener</strong>
                        <span>Balanced, Income, Growth and Potential rankings, Syariah and ownership filters, and a full research page per stock.</span>
                    </a>
                    <a class="pd-tool-link" href="./calendar" target="_self">
                        <strong>Calendar</strong>
                        <span>Full year, single month and monthly tables of payouts for JKSE or the S&amp;P 500.</span>
                    </a>
                    <a class="pd-tool-link" href="./porto_overview" target="_self">
                        <strong>Portfolio</strong>
                        <span>Income, yield on cost, your income goal, sectors and an allocation map.</span>
                    </a>
                    <a class="pd-tool-link" href="./simulator" target="_self">
                        <strong>Simulator</strong>
                        <span>Compounding with a fixed yield, a single stock or a portfolio, with or without DRIP.</span>
                    </a>
                </div>

                <div class="pd-tool-group">
                    <p class="pd-tool-group-title">Market</p>
                    <a class="pd-tool-link" href="./market_watch" target="_self">
                        <strong>Heatmap</strong>
                        <span>Returns, valuation and yield by sector for JKSE and the S&amp;P 500, plus global indices and currencies.</span>
                    </a>
                    <a class="pd-tool-link" href="./stock_comparison" target="_self">
                        <strong>Comparison</strong>
                        <span>Compare 2–5 stocks on income, quality, growth and total return, or start from a preset group.</span>
                    </a>
                    <a class="pd-tool-link" href="./best_timing" target="_self">
                        <strong>Best Timing</strong>
                        <span>Seasonality, pre-ex-date patterns, and recovery after the payout.</span>
                    </a>
                </div>

                <div class="pd-tool-group">
                    <p class="pd-tool-group-title">Trading</p>
                    <a class="pd-tool-link" href="./my_trades" target="_self">
                        <strong>My Trades</strong>
                        <span>Tonight's signals from every tool, upcoming dates, and your trade journal.</span>
                    </a>
                    <a class="pd-tool-link" href="./dividend_play" target="_self">
                        <strong>Dividend Play</strong>
                        <span>Buy ahead of the ex-date and sell into the run-up, backtested on every past dividend.</span>
                    </a>
                    <a class="pd-tool-link" href="./day_trading" target="_self">
                        <strong>Swing Play</strong>
                        <span>Walk-forward tested swing setups for the next 1–10 sessions, plus a research Lab.</span>
                    </a>
                    <a class="pd-tool-link" href="./backtester" target="_self">
                        <strong>Value Play</strong>
                        <span>A Growth at a Discount screen: quality IDX companies trading below their own historical valuation.</span>
                    </a>
                </div>
            </div>
        </section>

        <section class="pd-closing" aria-labelledby="pd-closing-title">
            <h2 id="pd-closing-title">Start with one stock you want to understand.</h2>
            <div class="pd-closing-copy">
                <p>
                    Open the Screener, pick a market, then explore dividends, financials, multiples,
                    intrinsic value, price action and compounding from a single page.
                </p>
                <a class="pd-button pd-button-primary" href="./stock_picker" target="_self">
                    Start researching stocks <span aria-hidden="true">→</span>
                </a>
            </div>
        </section>

        <footer class="pd-footer">
            <span>Data may be delayed or contain errors. Verify with official sources; Panen Dividen is not investment advice and does not guarantee results.</span>
            <nav class="pd-footer-links" aria-label="Community and support links">
                <a href="https://www.reddit.com/r/panendividen" target="_blank" rel="noreferrer">Community</a>
                <a href="https://github.com/mitbal/harvest/issues" target="_blank" rel="noreferrer">Feedback</a>
                <a href="https://blog.panendividen.com?utm_source=pd_web&amp;utm_medium=home_footer" target="_blank" rel="noreferrer">Blog</a>
                <a href="https://trakteer.id/mitbal" target="_blank" rel="noreferrer">Support</a>
            </nav>
        </footer>
    </main>
    '''
)

# Plain <a href="./page"> links force a full reload, which starts a new Streamlit
# session (losing session_state and minting a new visitor_id). Route in-app links
# through the matching sidebar nav link so Streamlit navigates client-side.
# Falls back to the normal reload when no matching nav link is in the DOM.
st.html(
    '''
    <script>
    if (!window.__pdHomeNavBound) {
        window.__pdHomeNavBound = true;
        const slugOf = (href) => new URL(href, window.location.href).pathname.split('/').filter(Boolean).pop() || '';
        document.addEventListener('click', (event) => {
            if (event.defaultPrevented || event.button !== 0 ||
                event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
                return;
            }
            // st.html's sanitizer drops target attributes, so match relative hrefs.
            const link = event.target.closest('.pd-home a[href^="./"]');
            if (!link) {
                return;
            }
            const slug = slugOf(link.getAttribute('href'));
            const navLink = Array.from(document.querySelectorAll('[data-testid="stSidebarNavLink"]'))
                .find((a) => slugOf(a.getAttribute('href')) === slug);
            if (!navLink) {
                return;
            }
            event.preventDefault();
            navLink.click();
        });
    }

    // Trading demo clip: play only while on screen, never for reduced motion (the poster stays),
    // and keep a visible pause button. Bound per element since reruns replace the DOM.
    (function bindTradeDemo(tries) {
        const video = document.querySelector('.pd-home .pd-trade-demo video');
        if (!video) {
            if (tries < 30) setTimeout(() => bindTradeDemo(tries + 1), 100);
            return;
        }
        if (video.dataset.bound) return;
        video.dataset.bound = '1';
        const toggle = video.parentElement.querySelector('.pd-trade-demo-toggle');
        const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
        let userPaused = reduced.matches;
        let onScreen = false;
        video.muted = true;
        const label = () => {
            if (!toggle) return;
            toggle.hidden = false;
            toggle.textContent = video.paused ? 'Play' : 'Pause';
            toggle.setAttribute('aria-label', (video.paused ? 'Play' : 'Pause') + ' the Dividend Play walkthrough');
        };
        const sync = () => {
            if (onScreen && !userPaused) {
                video.play().then(label).catch(label);
            } else if (!video.paused) {
                video.pause();
            }
            label();
        };
        video.addEventListener('play', label);
        video.addEventListener('pause', label);
        if (toggle) {
            toggle.addEventListener('click', () => {
                userPaused = !video.paused;
                if (!userPaused) onScreen = true;
                sync();
            });
        }
        reduced.addEventListener('change', () => { userPaused = reduced.matches; sync(); });
        if ('IntersectionObserver' in window) {
            new IntersectionObserver((entries) => {
                onScreen = entries[0].isIntersecting;
                sync();
            }, { threshold: 0.4 }).observe(video);
        } else {
            onScreen = true;
            sync();
        }
    })(0);
    </script>
    ''',
    unsafe_allow_javascript=True,
)
