import scrapy
import os


class MatlabDownloadSpider(scrapy.Spider):
    name = 'matlab_downloader'
    folder = 'output'

    def start_requests(self):
        urls = [
            'https://github.com/dennlinger/quantAFM/raw/master/DNA.m',
            'https://raw.githubusercontent.com/dennlinger/quantAFM/master/DNA.m',
        ]
        for url in urls:
            if url.endswith('.m'):
                yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        filename = os.path.join(self.folder, response.url.split('/')[3], response.url.split('/')[4],
                                '_'.join(response.url.split('/')[5:]) + '.dat')
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)

