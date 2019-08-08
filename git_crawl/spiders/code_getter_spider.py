import scrapy
import os
import random
import time


class CodeGetterSpider(scrapy.Spider):
    name = 'code_getter'
    folder = 'output'

    def start_requests(self):
        # specify input file with -a input='xyz'
        url_file = getattr(self, 'input', None)
        # File with given urls is specified in inputs folder of project
        with open(url_file, 'r') as f:
            urls = [x.strip() for x in f.readlines()]
        for url in urls:
            if not url:
                continue
            time.sleep(random.uniform(0.2, 0.5))
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        if response.url.endswith('.py'):
            language = 'python'
        elif response.url.endswith('.m'):
            language = 'matlab'
        else:
            raise ValueError('URL with wrong language detected! {}'.format(response.url))

        filename = os.path.join(self.folder, language, response.url.split('/')[3], response.url.split('/')[4],
                                '_'.join(response.url.split('/')[5:]) + '.dat')
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        with open(filename, 'wb') as f:
            f.write(response.body)
