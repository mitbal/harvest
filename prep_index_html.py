import shutil
import pathlib

import streamlit as st
from bs4 import BeautifulSoup

GA_ID = 'google_analytics'
GA_TAG_ID = 'G-V434YW6FCJ'
GA_SCRIPT = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_TAG_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'GA_TAG_ID');
</script>
"""

META_TAG = """
<html lang="en" prefix="og: http://ogp.me/ns#">
<meta property="og:image" content="https://github.com/mitbal/harvest/blob/master/asset/calendar.png?raw=true" />

<meta name="description" content="Panen Dividen is a web application to help you build, track, and monitor your investment portfolio." />
<meta name="keywords" content="investment, portfolio, tracking, monitoring, dividend, stock, mutual fund, ETF, index fund, bond, real estate, cryptocurrency, forex, commodity" />
"""

FAVICON_TAG = """
<link rel="icon" href="https://github.com/mitbal/harvest/blob/master/asset/favicon.png?raw=true" type="image/png">
"""

def inject_ga():
    
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")
    if not soup.find(id=GA_ID):
        bck_index = index_path.with_suffix('.bck')
        if bck_index.exists():
            shutil.copy(bck_index, index_path)
        else:
            shutil.copy(index_path, bck_index)
        html = str(soup)
        new_html = html.replace('<head>', '<head>\n' + GA_SCRIPT.replace('GA_TAG_ID',GA_TAG_ID) + '\n' + FAVICON_TAG)
        new_html = new_html.replace('<html lang="en">', META_TAG)
        index_path.write_text(new_html)

inject_ga()
