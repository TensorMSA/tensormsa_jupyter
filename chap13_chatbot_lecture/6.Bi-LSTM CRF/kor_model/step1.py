from kor_model.data_crawler import crawler
from kor_model.data_crawler import mecab
from kor_model.data_embed_model import build_data
from kor_model.config import config

if(config.use_crawler == True) :
    # (1) get some korean texts for embedding models by using WebCrawler
    crawler.spider(2, 'https://ko.wikipedia.org/wiki/',
                   path=config.crawler_path ,
                   file_name= config.crawler_file,
                   reg_exp='[가-히\s]{1,}')

    # (2) Meacb Tockenizer Preprocess Text
    mecab.tockenizer(config.crawler_path + config.crawler_file,
                     config.pos_path)

else :
    # (2) Meacb Tockenizer Preprocess Text
    mecab.tockenizer(config.train_filename,
                     config.pos_path)

# (3) build data (Mecab, Embedding, Char Embedding)
build_data.build_data(config)




