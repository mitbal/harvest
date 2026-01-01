import os
import toml
import shutil
import pathlib

import streamlit as st
from bs4 import BeautifulSoup

GA_ID = "google_analytics"
GA_TAG_ID = "G-V434YW6FCJ"

# 1) Put ONLY the <html ...> tag here (no meta tags)
HTML_TAG = '<html lang="id-ID" prefix="og: http://ogp.me/ns#">'

# 2) Put meta tags inside HEAD (valid HTML)
HEAD_META_TAGS = """
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

# 3) GA4 snippet: force page_location to include query params (UTM)
#    Also: turn off auto page_view and send our own explicit page_view.
GA_SCRIPT = f"""
<!-- Google tag (gtag.js) -->
<script id="{GA_ID}">
  (function() {{
    var s = document.createElement('script');
    s.async = true;
    s.src = 'https://www.googletagmanager.com/gtag/js?id={GA_TAG_ID}';
    document.head.appendChild(s);
  }})();
</script>

<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());

  // Disable auto page_view; we will send one with full URL
  gtag('config', '{GA_TAG_ID}', {{
    send_page_view: false
  }});

  // IMPORTANT: include full URL (with utm_*)
  var fullUrl = window.location.href;

  // Send a reliable page_view with UTM in page_location
  gtag('event', 'page_view', {{
    page_location: fullUrl,
    page_path: window.location.pathname + window.location.search,
    page_title: document.title
  }});
</script>

<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-5027985566203504"
     crossorigin="anonymous"></script>
"""

def inject_ga():
    index_path = pathlib.Path(st.__file__).parent / "static" / "index.html"
    soup = BeautifulSoup(index_path.read_text(), features="html.parser")

    # Set title + noscript
    if soup.find("title"):
        soup.find("title").string = "Panen Dividen - Aplikasi Monitoring Portofolio & Saham Dividen"
    if soup.find("noscript"):
        soup.find("noscript").string = (
            "Panen dividen adalah aplikasi monitoring portofolio dan saham dividen. "
            "Pantau portofolio investasi Anda dan dapatkan informasi riwayat dividen, "
            "jadwal pembayaran, dan analisis kinerja portofolio secara gratis."
        )

    # Backup original once
    bck_index = index_path.with_suffix(".bck")
    if not bck_index.exists():
        shutil.copy(index_path, bck_index)

    html = str(soup)

    # Replace <html lang="en"> (or whatever exists) with a clean html tag
    # Do not inject meta tags here.
    html = html.replace("<html lang=\"en\">", HTML_TAG)
    html = html.replace("<html lang='en'>", HTML_TAG)

    # Inject into <head> exactly once
    injection_block = "\n".join([GA_SCRIPT, FAVICON_TAG, HEAD_META_TAGS])
    if f'id="{GA_ID}"' not in html:
        html = html.replace("<head>", "<head>\n" + injection_block + "\n", 1)

    index_path.write_text(html)


def populate_secret():
    secrets = {
        "auth": {
            "google": {
                "client_id": os.environ["GOOGLE_CLIENT_ID"],
                "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
                "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
            },
            "redirect_uri": os.environ["REDIRECT_URI"],
            "cookie_secret": os.environ["COOKIE_SECRET"],
        },
        "connections": {
            "supabase": {
                "SUPABASE_URL": os.environ["SUPABASE_URL"],
                "SUPABASE_KEY": os.environ["SUPABASE_KEY"],
                "EMAIL_ADDRESS": os.environ["SUPABASE_EMAIL"],
                "PASSWORD": os.environ["SUPABASE_PASSWORD"],
            }
        },
    }

    os.makedirs(".streamlit", exist_ok=True)
    with open(".streamlit/secrets.toml", "w") as f:
        toml.dump(secrets, f)

inject_ga()
populate_secret()
