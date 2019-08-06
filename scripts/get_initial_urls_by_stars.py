import json
import subprocess
import tempfile
import time

with open('../secret.json', 'r') as f:
    password = json.load(f)['credentials']

with open('start_urls.txt', 'r') as f:
    urls = set([line.strip() for line in f.readlines()])

for j in range(10):
    for i in range(30):
        temp = tempfile.NamedTemporaryFile()
        temp.flush()
        query = 'curl -H "Authorization: token {}" -G https://api.github.com/search/repositories --data-urlencode ' \
                '"sort=stars" --data-urlencode "order=desc" --data-urlencode "q=language:python" ' \
                '> {}'.format(password, temp.name)
        subprocess.run(query, shell=True)
        with open(temp.name, 'r') as f:
            data = json.load(f)
        for dat in data["items"]:
            if dat["stargazers_count"] > 10:
                urls.add(dat["url"])
        time.sleep(1)
    time.sleep(20)

with open('start_urls.txt', 'w') as f:
    f.write('\n'.join(urls))


