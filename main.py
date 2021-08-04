import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

urls = [
    ("TACL 2021", "https://www.aclweb.org/anthology/events/tacl-2021/", None),
    ("ACL-IJCNLP 2021", "https://aclanthology.org/events/acl-2021/", "2021/08/01 ~ 2021/08/06"),
    ("NAACL 2021", "https://www.aclweb.org/anthology/events/naacl-2021/", "2021/06/06 ~ 2021/06/11"),
    ("EACL 2021", "https://www.aclweb.org/anthology/events/eacl-2021/", "2021/04/19 ~ 2021/04/23"),
    ("TACL 2020", "https://www.aclweb.org/anthology/events/tacl-2020/", None),
    ("COLING 2020", "https://www.aclweb.org/anthology/events/coling-2020/", "2020/12/08 ~ 2020/12/13"),
    ("CoNLL 2020", "https://www.aclweb.org/anthology/events/conll-2020/", "2020/11/19 ~ 2020/11/20"),
    ("EMNLP 2020", "https://www.aclweb.org/anthology/events/emnlp-2020/", "2020/11/16 ~ 2020/11/20"),
    ("EMNLP Findings 2020", "https://www.aclweb.org/anthology/events/findings-2020/", "2020/11/16 ~ 2020/11/20"),
    ("ACL 2020", "https://www.aclweb.org/anthology/events/acl-2020/", "2020/07/05 ~ 2020/07/10"),
    ("LREC 2020", "https://www.aclweb.org/anthology/events/lrec-2020/", None),
    ("TACL 2019", "https://www.aclweb.org/anthology/events/tacl-2019/", None),
    ("EMNLP-IJCNLP 2019", "https://www.aclweb.org/anthology/events/emnlp-2019/", "2019/11/03 ~ 2019/11/07"),
    ("CoNLL 2019", "https://www.aclweb.org/anthology/events/conll-2019/", "2019/11/03 ~ 2019/11/04"),
    ("ACL 2019", "https://www.aclweb.org/anthology/events/acl-2019/", "2019/07/28 ~ 2019/08/02"),
    ("NAACL 2019", "https://www.aclweb.org/anthology/events/naacl-2019/", "2019/06/02 ~ 2019/06/07"),
    ("TACL 2018", "https://www.aclweb.org/anthology/events/tacl-2018/", None),
    ("PACLIC 2018", "https://www.aclweb.org/anthology/events/paclic-2018/", "2018/12/01 ~ 2018/12/03"),
    ("EMNLP 2018", "https://www.aclweb.org/anthology/events/emnlp-2018/", "2018/10/31 ~ 2018/11/04"),
    ("CoNLL 2018", "https://www.aclweb.org/anthology/events/conll-2018/", "2018/10/31 ~ 2018/11/01"),
    ("COLING 2018", "https://www.aclweb.org/anthology/events/coling-2018/", "2018/08/20 ~ 2018/08/26"),
    ("ACL 2018", "https://www.aclweb.org/anthology/events/acl-2018/", "2018/07/15 ~ 2018/07/20"),
    ("NAACL 2018", "https://www.aclweb.org/anthology/events/naacl-2018/", "2018/06/01 ~ 2018/06/06"),
    ("LREC 2018", "https://www.aclweb.org/anthology/events/lrec-2018/", "2018/05/07 ~ 2018/05/12"),
    ("TACL 2017", "https://www.aclweb.org/anthology/events/tacl-2017/", None),
    ("IJCNLP 2017", "https://www.aclweb.org/anthology/events/ijcnlp-2017/", "2017/11/27 ~ 2017/12/01"),
    ("PACLIC 2017", "https://www.aclweb.org/anthology/events/paclic-2017/", "2017/11/16 ~ 2017/11/18"),
    ("EMNLP 2017", "https://www.aclweb.org/anthology/events/emnlp-2017/", "2017/09/07 ~ 2017/09/11"),
    ("CoNLL 2017", "https://www.aclweb.org/anthology/events/conll-2017/", "2017/08/03 ~ 2017/08/04"),
    ("ACL 2017", "https://www.aclweb.org/anthology/events/acl-2017/", "2017/07/30 ~ 2017/08/04"),
    ("EACL 2017", "https://www.aclweb.org/anthology/events/eacl-2017/", "2017/04/03 ~ 2017/04/07"),
    ("TACL 2016", "https://www.aclweb.org/anthology/events/tacl-2016/", None),
    ("COLING 2016", "https://www.aclweb.org/anthology/events/coling-2016/", "2016/12/11 ~ 2016/12/16"),
    ("EMNLP 2016", "https://www.aclweb.org/anthology/events/emnlp-2016/", "2016/11/01 ~ 2016/11/05"),
    ("PACLIC 2016", "https://www.aclweb.org/anthology/events/paclic-2016/", "2016/10/28 ~ 2016/10/30"),
    ("CoNLL 2016", "https://www.aclweb.org/anthology/events/conll-2016/", "2016/08/11 ~ 2016/08/12"),
    ("ACL 2016", "https://www.aclweb.org/anthology/events/acl-2016/", "2016/08/07 ~ 2016/08/12"),
    ("NAACL 2016", "https://www.aclweb.org/anthology/events/naacl-2016/", "2016/06/12 ~ 2016/06/17"),
    ("LREC 2016", "https://www.aclweb.org/anthology/events/lrec-2016/", "2016/05/23 ~ 2016/05/28"),
    ("TACL 2015", "https://www.aclweb.org/anthology/events/tacl-2015/", None),
    ("PACLIC 2015", "https://www.aclweb.org/anthology/events/paclic-2015/", "2015/10/30 ~ 2015/11/01"),
    ("EMNLP 2015", "https://www.aclweb.org/anthology/events/emnlp-2015/", "2015/09/17 ~ 2015/09/21"),
    ("CoNLL 2015", "https://www.aclweb.org/anthology/events/conll-2015/", "2015/07/30 ~ 2015/07/31"),
    ("ACL-IJCNLP 2015", "https://www.aclweb.org/anthology/events/acl-2015/", "2015/07/26 ~ 2015/07/31"),
    ("NAACL 2015", "https://www.aclweb.org/anthology/events/naacl-2015/", "2015/05/31 ~ 2015/06/05"),
]

txt = "# entity-related-papers\n\n"

for name, url, date in tqdm(urls):
    if date is None:
        t = f"### [{name}]({url})\n\n"
    else:
        t = f"### [{name}]({url}) ({date})\n\n"

    t_rec = "\n#### NER\n"
    t_link = "\n#### EL\n"
    t_type = "\n#### Entity Typing\n"
    t_repre = "\n#### Entity Representations / Embeddings\n"
    t_others = "\n#### misc\n"

    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"})
    soup = BeautifulSoup(r.text, 'lxml')

    divs = soup.find_all('span', {'class': 'd-block'})

    for div in divs:
        res = div.find('a', {'class': 'align-middle'})
        link = res.get('href')
        text = res.get_text()

        item = f"- [{text}](https://aclanthology.org{link})\n"

        found = "Entity" in text or "entity" in text or "Entities" in text or "entities" in text or " NER " in text or " NEL " in text or " EL " in text 
        found_rec = "Recogn" in text or "recogn" in text or " NER " in text
        found_link = "Link" in text or "link" in text or " NEL " in text or " EL " in text 
        found_type = "Entity Typing" in text or "Entity Type" in text
        found_repre = "Representation" in text or "representation" in text or "Embedding" in text or "embedding" in text

        if found:
            if found_rec:
                t_rec += item
            elif found_link:
                t_link += item
            elif found_type:
                t_type += item
            elif found_repre:
                t_repre += item
            else:
                t_others += item

    t += (t_rec + t_link + t_type + t_repre + t_others)
    txt += t

    
with open("README.md", "w") as f:
    f.write(txt)