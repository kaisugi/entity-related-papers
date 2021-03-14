import requests
from bs4 import BeautifulSoup

urls = [
    ("ACL 2020", "https://www.aclweb.org/anthology/events/acl-2020/"),
    ("TACL 2020", "https://www.aclweb.org/anthology/events/tacl-2020/"),
    ("LREC 2020", "https://www.aclweb.org/anthology/events/lrec-2020/"),
    ("ACL 2019", "https://www.aclweb.org/anthology/events/acl-2019/"),
    ("TACL 2019", "https://www.aclweb.org/anthology/events/tacl-2019/"),
    ("NAACL 2019", "https://www.aclweb.org/anthology/events/naacl-2019/"),
    ("EMNLP-IJCNLP 2019", "https://www.aclweb.org/anthology/events/emnlp-2019/"),
    ("CoNLL 2019", "https://www.aclweb.org/anthology/events/conll-2019/"),
    ("ACL 2018", "https://www.aclweb.org/anthology/events/acl-2018/"),
    ("TACL 2018", "https://www.aclweb.org/anthology/events/tacl-2018/"),
    ("NAACL 2018", "https://www.aclweb.org/anthology/events/naacl-2018/"),
    ("EMNLP 2018", "https://www.aclweb.org/anthology/events/emnlp-2018/"),
    ("COLING 2018", "https://www.aclweb.org/anthology/events/coling-2018/"),
    ("CoNLL 2018", "https://www.aclweb.org/anthology/events/conll-2018/"),
    ("LREC 2018", "https://www.aclweb.org/anthology/events/lrec-2018/"),
    ("PACLIC 2018", "https://www.aclweb.org/anthology/events/paclic-2018/"),
    ("ACL 2017", "https://www.aclweb.org/anthology/events/acl-2017/"),
    ("TACL 2017", "https://www.aclweb.org/anthology/events/tacl-2017/"),
    ("EMNLP 2017", "https://www.aclweb.org/anthology/events/emnlp-2017/"),
    ("EACL 2017", "https://www.aclweb.org/anthology/events/eacl-2017/"),
    ("IJCNLP 2017", "https://www.aclweb.org/anthology/events/ijcnlp-2017/"),
    ("CoNLL 2017", "https://www.aclweb.org/anthology/events/conll-2017/"),
    ("PACLIC 2017", "https://www.aclweb.org/anthology/events/paclic-2017/"),
    ("ACL 2016", "https://www.aclweb.org/anthology/events/acl-2016/"),
    ("TACL 2016", "https://www.aclweb.org/anthology/events/tacl-2016/"),
    ("NAACL 2016", "https://www.aclweb.org/anthology/events/naacl-2016/"),
    ("EMNLP 2016", "https://www.aclweb.org/anthology/events/emnlp-2016/"),
    ("COLING 2016", "https://www.aclweb.org/anthology/events/coling-2016/"),
    ("CoNLL 2016", "https://www.aclweb.org/anthology/events/conll-2016/"),
    ("LREC 2016", "https://www.aclweb.org/anthology/events/lrec-2016/"),
    ("PACLIC 2016", "https://www.aclweb.org/anthology/events/paclic-2016/"),
    ("ACL-IJCNLP 2015", "https://www.aclweb.org/anthology/events/acl-2015/"),
    ("TACL 2015", "https://www.aclweb.org/anthology/events/tacl-2015/"),
    ("NAACL 2015", "https://www.aclweb.org/anthology/events/naacl-2015/"),
    ("EMNLP 2015", "https://www.aclweb.org/anthology/events/emnlp-2015/"),
    ("CoNLL 2015", "https://www.aclweb.org/anthology/events/conll-2015/"),
    ("PACLIC 2015", "https://www.aclweb.org/anthology/events/paclic-2015/")
]

txt = "# entity-related-papers\n\n"

for name, url in urls:
    t = f"### [{name}]({url})\n\n"
    t_rec = "\n#### NER\n"
    t_link = "\n#### EL\n"
    t_type = "\n#### Entity Typing\n"
    t_others = "\n#### misc\n"

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"})
    soup = BeautifulSoup(r.text, 'lxml')

    divs = soup.find_all('span', {'class': 'd-block'})

    for div in divs:
        res = div.find('a', {'class': 'align-middle'})
        link = res.get('href')
        text = res.get_text()

        item = f"- [{text}](https://www.aclweb.org{link})\n"

        found = "Entity" in text or "entity" in text or "Entities" in text or "entities" in text or " NER " in text or " NEL " in text or " EL " in text 
        found_rec = "Recogn" in text or "recogn" in text or " NER " in text
        found_link = "Link" in text or "link" in text or " NEL " in text or " EL " in text 
        found_type = "Entity Typing" in text or "Entity Type" in text

        if found:
            if found_rec:
                t_rec += item
            elif found_link:
                t_link += item
            elif found_type:
                t_type += item
            else:
                t_others += item

    t += (t_rec + t_link + t_type + t_others)
    txt += t

    
with open("README.md", "w") as f:
    f.write(txt)