"""
OAuth downloader middleware
"""

from scrapy import signals


class GithubOAuthMiddleware(object):
    """Set OAuth header
    (oauth_token spider class attributes)"""

    auth = None

    @classmethod
    def from_crawler(cls, crawler):
        o = cls()
        crawler.signals.connect(o.spider_opened, signal=signals.spider_opened)
        return o

    def spider_opened(self, spider):
        token = getattr(spider, 'oauth_token', '')
        if token:
            self.auth = 'token {}'.format(token).encode()

    def process_request(self, request, spider):
        auth = getattr(self, 'auth', None)
        if auth and b'Authorization' not in request.headers:
            request.headers[b'Authorization'] = auth