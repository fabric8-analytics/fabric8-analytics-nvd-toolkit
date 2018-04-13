"""Configurations."""

import os

GITHUB_BASE_URL = u"https://github.com/"

# token for GitHub authentication
OAUTH_TOKEN = os.environ.get("OAUTH_TOKEN", None)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/51.0.2704.103 Safari/537.36',
    'Authorization': 'token %s' % OAUTH_TOKEN,
}

