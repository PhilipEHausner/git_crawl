from scrapy import signals
from scrapy.exceptions import NotConfigured


class GithubRateLimitWatcher(object):

    def __init__(self, crawler):
        self.crawler = crawler

        self.close_on = {
            'limitcount': crawler.settings.getint('GITHUB_RATE_LIMIT_WATCHER_LIMITCOUNT'),
            }

        if not any(self.close_on.values()):
            raise NotConfigured

        if self.close_on.get('limitcount'):
            crawler.signals.connect(self.rate_limit, signal=signals.response_received)

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)

    def rate_limit(self, response, request, spider):
        remain = response.headers["X-RateLimit-Remaining"] if "X-RateLimit-Remaining" in response.headers else ""
        if int(remain) < self.close_on['limitcount']:
            self.crawler.engine.close_spider(spider, 'closespider_github_rate_limit')