import os
import toml

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
<html lang="id-ID" prefix="og: http://ogp.me/ns#">
<meta property="og:image" content="https://github.com/mitbal/harvest/blob/master/asset/calendar.png?raw=true" />

<meta name="description" content="Panen Dividen - Pantau portofolio investasi dan saham dividen Anda. Dapatkan informasi riwayat dividen, jadwal pembayaran, dan analisis kinerja portofolio secara gratis." />
<meta name="description" lang="en" content="Panen Dividen - Track your dividend stocks and investment portfolio with our free tools. Get insights on dividend history, payment schedules, and portfolio performance analysis." />

<meta name="keywords" content="investasi, portofolio, saham, dividen, reksa dana, ETF, obligasi, cryptocurrency, forex, komoditas, analisis saham, pasar modal indonesia, idx, bei, bursa efek indonesia, saham dividen" />
<meta name="keywords" lang="en" content="investment, portfolio, tracking, monitoring, dividend, stock, mutual fund, ETF, index fund, bond, real estate, cryptocurrency, forex, commodity" />

<meta property="og:title" content="Panen Dividen - Aplikasi Monitoring Portofolio & Saham Dividen" />
<meta property="og:description" content="Pantau portofolio investasi dan saham dividen Anda. Dapatkan informasi riwayat dividen, jadwal pembayaran, dan analisis kinerja portofolio secara gratis." />
<meta property="og:locale" content="id_ID" />
<meta property="og:locale:alternate" content="en_US" />
<meta property="og:type" content="website" />
<meta property="og:url" content="https://panendividen.com" />
"""

FAVICON_TAG = """
<link rel="icon" href="https://github.com/mitbal/harvest/blob/master/asset/favicon.png?raw=true" type="image/png">
"""


def inject_ga():
    
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")
    
    soup.find('title').string = 'Panen Dividen - Aplikasi Monitoring Portofolio & Saham Dividen'
    soup.find('noscript').string = 'Panen dividen adalah aplikasi monitoring portofolio dan saham dividen. Pantau portofolio investasi Anda dan dapatkan informasi riwayat dividen, jadwal pembayaran, dan analisis kinerja portofolio secara gratis.'

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


def populate_secret():

    secrets = {

        'auth': {
            'google': {
                'client_id': os.environ['GOOGLE_CLIENT_ID'],
                'client_secret': os.environ['GOOGLE_CLIENT_SECRET'],
                'server_metadata_url': 'https://accounts.google.com/.well-known/openid-configuration'
            },
            'redirect_uri': os.environ['REDIRECT_URI'],
            'cookie_secret': os.environ['COOKIE_SECRET'],
        },

        'connections': {
            'supabase': {
                'SUPABASE_URL': os.environ['SUPABASE_URL'],
                'SUPABASE_KEY': os.environ['SUPABASE_KEY'],
                'EMAIL_ADDRESS': os.environ['SUPABASE_EMAIL'],
                'PASSWORD': os.environ['SUPABASE_PASSWORD'],
            }
        }
    }

    with open('.streamlit/secrets.toml', 'w') as f:
        toml.dump(secrets, f)

inject_ga()
populate_secret()
