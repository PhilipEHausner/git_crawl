import scrapy
import json
import os


class CreateDownloadLinkSpider(scrapy.Spider):
    name = 'create_download_links'
    filename = os.path.join('output', 'download_links.txt')
    with open('secret.json', 'r') as f:
        oauth_token = json.load(f)['credentials']

    def start_requests(self):
        # specify input file with -a input='xyz'
        url_file = getattr(self, 'input', None)
        # File with given urls is specified in inputs folder of project
        with open(url_file, 'r') as f:
            urls = [x.strip() for x in f.readlines()]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # github api return json objects
        data = json.loads(response.body)
        for item in data:
            if item['type'] == 'file':
                # can be extended to further file formats (e.g. .py, .cpp, ...)
                if item['name'].endswith('.m'):
                    with open(self.filename, 'a') as f:
                        f.write(item['download_url'] + '\n')
            elif item['type'] == 'dir':
                yield response.follow(item['url'], callback=self.parse)
                continue

