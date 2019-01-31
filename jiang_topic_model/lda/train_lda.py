#!/usr/bin/env python 
# encoding: utf-8 

"""
@author: zkjiang
@site: https://www.github.com
@software: PyCharm
@file: train_lda.py
@time: 2019/1/23 20:00
"""
import time

import jieba
import lda
import pickle
import sys
import os
from gensim import corpora

ROOT = "/home/zkjiang/projects/nlp_learn/"
sys.path.append(ROOT)
from src.KnowledgeCommon import Common
from conf.config import get_config
__config = get_config()
from src.acquisition_common_lib.file.stop_words import StopWords
sw = StopWords()
import numpy as np
from src.acquisition_common_lib.mongo.mongo_data import MyMongoClient
from gensim.models import LdaModel, LdaMulticore
from gensim.corpora.dictionary import Dictionary
from src.acquisition_common_lib.similarity.word2vec_similarity import cos_similary
import logging

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
           )





ROOT_DATA = __config["path"]["root"]
ABSTRACT_LIST = __config["path"]["abstract_list"]
NUM_ABSTRACT_LIST = __config["path"]["num_abstract_list"]
WORD_TO_ID = __config["path"]["word_to_id"]
REVERSE_WORD_TO_ID = __config["path"]["reverse_word_to_id"]
# mongo = MyMongoClient()
# mongo_cli = mongo.get_cli()



def pre_data():
    abstract_list_path = ROOT_DATA + ABSTRACT_LIST
    num_abstract_list_path = ROOT_DATA + NUM_ABSTRACT_LIST
    word_to_id_file_path = ROOT_DATA + WORD_TO_ID
    reverse_word_to_id_file_path = ROOT_DATA + REVERSE_WORD_TO_ID
    if os.path.exists(abstract_list_path) and os.path.exists(word_to_id_file_path) and os.path.exists(reverse_word_to_id_file_path) and os.path.exists(num_abstract_list_path):
        print("从文件读取数据")
        abstract_list_file = open(abstract_list_path, "rb")
        num_abstract_list_file = open(num_abstract_list_path, "rb")
        word_to_id_file = open(word_to_id_file_path, "rb")
        reverse_word_to_id_file = open(reverse_word_to_id_file_path, "rb")
        abstract_list = pickle.load(abstract_list_file)
        num_abstract_list = pickle.load(num_abstract_list_file)
        word2id = pickle.load(word_to_id_file)
        rever_word2id = pickle.load(reverse_word_to_id_file)
    else:
        print("从数据库读取数据")
        abstract_list = []
        entity_info = mongo_cli.offline_sznlp.ocp_word_tbl.find({"wordtype": 1, "is_del": 0})
        for i in  entity_info:
            abstract = i["abstract"].strip("\n")
            if abstract != "":
                sw_abstract = sw.apply(jieba.lcut(abstract))
                # print(sw_abstract)
                if sw_abstract:
                    abstract_list.append(sw_abstract)
        word2id = {}
        rever_word2id = {}
        i = 0
        num_abstract_list = []
        for abstract in abstract_list:
            num_abstract = []
            for word in abstract:
                if word not in word2id:
                    word2id[word] = i
                    rever_word2id[i] = word
                    i += 1
                num_abstract.append(word2id[word])
            num_abstract_list.append(num_abstract)
        abstract_list_file = open(ROOT_DATA + ABSTRACT_LIST, "wb")
        pickle.dump(abstract_list, abstract_list_file)
        num_abstract_list_file = open(num_abstract_list_path, "wb")
        pickle.dump(num_abstract_list, num_abstract_list_file)
        word_to_id_file = open(ROOT_DATA + WORD_TO_ID, "wb")
        pickle.dump(word2id, word_to_id_file)
        reverse_word_to_id_file = open(ROOT_DATA + REVERSE_WORD_TO_ID, "wb")
        pickle.dump(rever_word2id, reverse_word_to_id_file)
    return abstract_list, num_abstract_list, word2id, rever_word2id


def train_lda():
    """
    训练lda模型
    :return:lda 模型
    """
    # print("正在训练lda模型")
    abstract_list, num_abstract_list, word2id, rever_word2id = pre_data()
    # abstract_list = abstract_list[:1000]
    common_dictionary = corpora.Dictionary(abstract_list)
    common_corpus = [common_dictionary.doc2bow(text) for text in abstract_list]
    print("开始训练lda模型")
    start = time.time()
    lda_model = LdaMulticore(common_corpus, id2word=common_dictionary, num_topics= 50, workers = 1, minimum_probability = 0)
    end = time.time()
    print("模型训练完毕，消耗的时间是{}".format(str(end - start)))
    vector = lda_model[common_corpus[0]]
    print(vector)
    lda_model_path = Common().Config().lda_model_path
    lda_model.save(lda_model_path)
    common_dictionary_file_path = Common().Config().common_dictionary_file_path
    common_dictionary_file = open(common_dictionary_file_path, "wb")
    pickle.dump(common_dictionary, common_dictionary_file)
    return lda_model, common_dictionary
lda_model = None
def get_vector(sentence_list):
    global lda_model, common_dictionary
    if not lda_model or not common_dictionary:
        lda_model_path = Common().Config().lda_model_path
        common_dictionary_file_path = Common().Config().common_dictionary_file_path
        # lda = LdaModel(minimum_probability = 0)
        if os.path.exists(lda_model_path) and 1 == 1 and os.path.exists(common_dictionary_file_path):
            print("从文件读取模型")
            lda_model = LdaMulticore.load(lda_model_path)
            common_dictionary_file = open(common_dictionary_file_path, "rb")
            common_dictionary = pickle.load(common_dictionary_file)
        else:
            print("未找到模型文件")
            lda_model, common_dictionary = train_lda()
    doc = common_dictionary.doc2bow(sentence_list)
    vector = lda_model[doc]
    return vector

def lda_similary(list1, list2, remove_tag = ""):
    s = jieba.lcut(list1)
    z = jieba.lcut(list2)
    # if remove_tag:
    s = [x for x in s if x != remove_tag ]
    z = [x for x in z if x != remove_tag]
    s = np.array([i[1] for i in get_vector(s)])
    z = np.array([i[1] for i in get_vector(z)])
    return cos_similary(z, s)

if __name__ == '__main__':

    """
    # t = jieba.lcut("英雄萨姆2（Serious_SamII）为克罗地亚开发公司Croteam开发的一款带有浓厚的幽默气息的第一人称射击类游戏，目前已经开发有四个版本，在《英雄萨姆2》中，我们的俏皮英雄萨姆·斯通将展开新的冒险。这次他孤军奋战的对手是一伙长得奇形怪状的外星侵略者。游戏一共设计了七个不同的世界，让我们来看看萨姆在其中的一个世界历险的片断：这是一个被丛林和草原覆盖的绿色星球，阳光明媚的大地上纵横散布着一些高科技设施基地。")
    # t = jieba.lcut(
    #     "小说有多少章节")
    # s = jieba.lcut(
    #     "新版越剧《》创作于1999年，首演于同年8月。它从调整戏剧结构入手，别样营造大悲大喜、大实大虚的舞台意境，并提高舞美空间层次，丰富音乐形象，整合流派表演，精缩演出时间，实现了一次富有创意的新编。它对原版既有承传，又有创新，是一个注入现代审美意识的新时期版本，被称为“展示上海文化风采的标志之作”。首演后，评论普遍认为：“《》是越剧的经典之作，这次演出不是简单的复排，而是以现代审美观点进行加工修改。新版越剧《》在剧本_、编导演、舞美、音乐等方面进行的可贵探索，使传统剧目焕发出新的光彩。")
    # z = jieba.lcut("《》，中国古代章回体长篇小说，又名《石头记》等，被列为中国古典四大名著之首，一般认为是清代作家曹雪芹所著。小说以贾、史、王、薛四大家族的兴衰为背景，以富贵公子贾宝玉为视角，描绘了一批举止见识出于须眉之上的闺阁佳人们的人生百态，展现了正邪两赋有情人的人性美和悲剧美，可以说是一部从各个角度展现女性美的史诗。_《》分为120回“程本”和80回“脂本”两种版本系统，程本为程伟元排印的印刷本，脂本为脂砚斋在不同时期抄评的早期手抄本，脂本是程本的底本。此书新版通行本前80回据脂本汇校，后40回据程本汇校，署名“曹雪芹著，无名氏续，程伟元、高鹗整理”。_《》是一部具有世界影响力的人情小说，举世公认的中国古典小说巅峰之作，中国封建社会的百科全书，传统文化的集大成者。小说以“大旨谈情，实录其事”自勉，只按自己的事体情理，按迹循踪，摆脱旧套，新鲜别致，取得了非凡的艺术成就。“真事隐去，假语村言”的特殊笔法更是令后世读者脑洞大开，揣测之说久而遂多。后世围绕《》的品读研究形成了一门显学——红学。")

    t = jieba.lcut(
        "的导演是谁呢")
    s = jieba.lcut(
        "《》是由中国电影股份有限公司、乐视影业、传奇影业、环球影业联合出品，由中国导演张艺谋执导，马特·达蒙、景甜、佩德罗·帕斯卡、威廉·达福、刘德华、张涵予等联合主演的奇幻动作片。_影片故事背景设定在中国宋朝时期，讲述了欧洲雇佣兵威廉·加林在被囚禁在期间，发现可怕的掠食怪兽将这座巨型城墙重重围困之时，他决定加入了一支由中国精英勇士们组成的大军，共同对抗怪兽饕餮的故事。_影片在2016年12月15日晚19点在中国350家IMAX影院超前上映，16日以3D、IMAX3D、中国巨幕3D、杜比视界、杜比全景声、Auro格式在中国全面上映")
    z = jieba.lcut(
        "（Great_Wall），又称万里，是中国古代的军事防御工程，是一道高大、坚固而连绵不断的长垣，用以限隔敌骑的行动。不是一道单纯孤立的城墙，而是以城墙为主体，同大量的城、障、亭、标相结合的防御体系。_修筑的历史可上溯到西周时期，发生在首都镐京（今陕西西安）的著名的典故“烽火戏诸侯”就源于此。春秋战国时期列国争霸，互相防守，修筑进入第一个高潮，但此时修筑的长度都比较短。秦灭六国统一天下后，秦始皇连接和修缮战国，始有万里之称。明朝是最后一个大修的朝代，今天人们所看到的多是此时修筑。_资源主要分布在河北、北京、天津、山西、陕西、甘肃、内蒙古、黑龙江、吉林、辽宁、山东、河南、青海、宁夏、新疆等15个省区市。期中陕西省是中国资源最为丰富的省份，境内长度达1838千米。根据文物和测绘部门的全国性资源调查结果，明总长度为8851.8千米，秦汉及早期超过1万千米，总长超过2.1万千米。_1961年3月4日，被国务院公布为第一批全国重点文物保护单位。1987年12月，被列入世界文化遗产。")
    """


    t = jieba.lcut(
        "打网球的李娜今年多大了")
    s = jieba.lcut(
        "李娜，1982年2月26日出生于湖北省武汉市，中国女子网球运动员。2008年北京奥运会女子单打第四名，2011年法国网球公开赛、2014年澳大利亚网球公开赛女子单打冠军，亚洲第一位大满贯女子单打冠军，亚洲历史上女单世界排名最高选手。毕业于华中科技大学。_1989年，6岁的李娜开始练习网球。1999年转为职业选手。2002年年底，李娜前往华中科技大学新闻专业就读。2004年，在丈夫姜山的鼓励和支持下选择了复出。2008年，在北京奥运会上，李娜获得女子单打第四名。2011年，李娜在澳大利亚网球公开赛上个人第一次打进大满贯单打决赛并夺得亚军；同年，在法国网球公开赛女单比赛中登顶封后。2013年，在WTA年终总决赛中获得亚军。_2014年1月25日，第三次跻身澳大利亚网球公开赛决赛并最终收获女单冠军。同年7月31日，李娜通过个人微博宣布自己将退出包括美网在内的北美赛季。9月18日，李娜经纪公司确认其退出武汉和中网的比赛，并将正式退役。9月19日，亚洲首位网球大满贯得主李娜正式宣布退役。12月15日，李娜被英国《金融时报》评选出为2014年年度女性人物。12月23日，李娜入围“2014CCTV体坛风云人物年度评选”的年度最佳女运动员。_在李娜十五年的职业生涯里，21次打入WTA女单赛事决赛，并共获得了9个WTA和19个ITF单打冠军，职业生涯总战绩为503胜188负，并以排名世界第六的身份退役")

    z = jieba.lcut("原名牛志红，出生于河南省郑州市，毕业于河南省戏曲学校，曾是中国大陆女歌手，出家后法名释昌圣。毕业后曾从事于豫剧演出，1997年皈依佛门，法号“昌圣”。从《好人一生平安》到《青藏高原》，再到荣获1995年罗马尼亚MTV国际大奖的《嫂子颂》，踏入音乐界十年间，为160多部影视剧配唱200多首歌。")

    t = jieba.lcut(
        "丑小鸭的作者是谁")
    s = jieba.lcut(
        "李娜，1982年2月26日出生于湖北省武汉市，中国女子网球运动员。2008年北京奥运会女子单打第四名，2011年法国网球公开赛、2014年澳大利亚网球公开赛女子单打冠军，亚洲第一位大满贯女子单打冠军，亚洲历史上女单世界排名最高选手。毕业于华中科技大学。_1989年，6岁的李娜开始练习网球。1999年转为职业选手。2002年年底，李娜前往华中科技大学新闻专业就读。2004年，在丈夫姜山的鼓励和支持下选择了复出。2008年，在北京奥运会上，李娜获得女子单打第四名。2011年，李娜在澳大利亚网球公开赛上个人第一次打进大满贯单打决赛并夺得亚军；同年，在法国网球公开赛女单比赛中登顶封后。2013年，在WTA年终总决赛中获得亚军。_2014年1月25日，第三次跻身澳大利亚网球公开赛决赛并最终收获女单冠军。同年7月31日，李娜通过个人微博宣布自己将退出包括美网在内的北美赛季。9月18日，李娜经纪公司确认其退出武汉和中网的比赛，并将正式退役。9月19日，亚洲首位网球大满贯得主李娜正式宣布退役。12月15日，李娜被英国《金融时报》评选出为2014年年度女性人物。12月23日，李娜入围“2014CCTV体坛风云人物年度评选”的年度最佳女运动员。_在李娜十五年的职业生涯里，21次打入WTA女单赛事决赛，并共获得了9个WTA和19个ITF单打冠军，职业生涯总战绩为503胜188负，并以排名世界第六的身份退役")

    z = jieba.lcut(
        "原名牛志红，出生于河南省郑州市，毕业于河南省戏曲学校，曾是中国大陆女歌手，出家后法名释昌圣。毕业后曾从事于豫剧演出，1997年皈依佛门，法号“昌圣”。从《好人一生平安》到《青藏高原》，再到荣获1995年罗马尼亚MTV国际大奖的《嫂子颂》，踏入音乐界十年间，为160多部影视剧配唱200多首歌。")

    t = np.array([i[1] for i in get_vector(t)])
    s = np.array([i[1] for i in get_vector(s)])
    z = np.array([i[1] for i in get_vector(z)])
    print(1)
    print(cos_similary(t, s))
    print(cos_similary(t, z))
