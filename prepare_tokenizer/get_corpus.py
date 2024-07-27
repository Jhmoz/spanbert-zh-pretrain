import json
import re
import os


from tqdm import tqdm


os.chdir('/home/yingying/sigir2024/spanbert-zh/pretrain_spanbert_zh')


discussion_corpus_filepath = '../data/disscussion/discussion/Discussion_FinalVersion.json'
baike_para_corpus_filepath = '../data/baike_para/para_dict_fixed.json'
baike_summary_corpus_filepath = '../data/baike_summary/tag&summary_dict.json'
wiki_zh_corpus_dir = '../data/chinese_wiki/token_cleaned_plain_files'
output_dir = './data'
puncs_set = set("﹒﹔﹖﹗．；。！？’”」』》><《‘“「『")

SENT_SPLIT_REGEX = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』》>]{0,2}|：(?=[<《"‘“「『]{1,2}|$))')



def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        d=json.load(f)
    return d

def save_corpus(file_name, paragraphs:list):
    file_path = os.path.join(output_dir, file_name)
    print(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        for paragraph in paragraphs:
            f.write(paragraph + '\n')

def get_wiki_zh_corpus(file_dir):
    files = os.listdir(file_dir)
    contents=[]
    for file in tqdm(files):
        file_path = os.path.join(file_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            _ = f.readline()
            content = f.read()
        contents.append(content.strip())
    return contents

def get_discussion_corpus(discussion:dict):
    res=[]
    for k in tqdm(discussion):
        assert isinstance(discussion[k],list)
        for text in discussion[k]:
            assert isinstance(text,str)
            text = re.sub("\n+","",text)
            res.append(text)
    return res

def get_baike_summary_corpus(summarys:dict):
    res=[]
    for lemma_id in tqdm(summarys):
        res.append(summarys[lemma_id]["Summary"].strip())
    return res

def get_baike_paragraphs_corpus(baike_paragraphs:dict):
    res=[]
    for lemma_id in tqdm(baike_paragraphs):
        paras = get_nested_paras(baike_paragraphs[lemma_id]["paragragh"])
        paras = [re.sub("\n+"," ",para).strip() for para in paras]
        res.extend(paras)
    return res

def get_nested_paras(paragraphs:dict):
    texts=[]
    for title in paragraphs:
        if isinstance(paragraphs[title],dict):
            texts.extend(get_nested_paras(paragraphs[title]))
        if isinstance(paragraphs[title],str):
            texts.append(paragraphs[title])
    return texts

def is_punc(str):
    return set(str).issubset(puncs_set)
def get_texts_with_valid_length(texts):
    res=[]
    for text in tqdm(texts):
        sentences = SENT_SPLIT_REGEX.split(text.strip())
        for i in range(0, len(sentences), 2):
            cur_sentences=re.sub("\n+"," ",sentences[i]).strip()
            if i < len(sentences):
                if res == []:
                    res.append(cur_sentences)
                else:
                    if is_punc(cur_sentences):
                        res[-1] += cur_sentences
                    else:
                        if len(res[-1]) + len(cur_sentences) < 510:
                            res[-1] += cur_sentences
                        else:
                            res.append(cur_sentences)
    return res

def get_corpus():
    print("=============================== Loading data from disscussion data ===============================")
    raw_discussion = read_json(discussion_corpus_filepath)
    print("=============================== Loading data from baike_para data ===============================")
    raw_baike_paras = read_json(baike_para_corpus_filepath)
    print("=============================== Loading data from baike_summary data ===============================")
    raw_baike_summarys = read_json(baike_summary_corpus_filepath)
    print("=============================== parsing data from baike_summary data ===============================")
    baike_summary_texts = get_baike_summary_corpus(raw_baike_summarys)
    print("=============================== parsing data from baike_para data ===============================")
    baike_para_texts = get_baike_paragraphs_corpus(raw_baike_paras)
    print("=============================== parsing data from discussion data ===============================")
    discussion_texts = get_discussion_corpus(raw_discussion)
    print("=============================== parsing data from wikipidea_zh data ===============================")
    wiki_zh_texts = get_wiki_zh_corpus(wiki_zh_corpus_dir)
    raw_texts = baike_para_texts + discussion_texts + wiki_zh_texts +baike_summary_texts
    corpus = get_texts_with_valid_length(raw_texts)
    del raw_texts
    save_corpus("corpus_v3.txt", corpus)





if __name__ == "__main__":
    get_corpus()

