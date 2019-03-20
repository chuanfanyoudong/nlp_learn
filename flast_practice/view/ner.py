import logging
import time

from flask import Blueprint, request, render_template
from jiang_ner.crf_ner.api import recognize

ner = Blueprint('ner', __name__)

@ner.route('/ner_demo', methods=['GET', 'POST'])
def ner_method():
    """
    命名实体识别接口
    :return: 输入要识别的句子，输出识别的结果
    """

    # print(username)
    split_result = {}
    if request.method == 'POST':
        # print("POST")
        ner_sentence = request.form.get("sentence")
        # print(ner_sentence)
        if ner_sentence.strip() == "":
            split_result = ""
        else:
            crf_start = time.time()
            crf_result = recognize(ner_sentence)
            crf_end = time.time()
            split_result["crf_result"] = " ".join(crf_result[0])
            split_result["crf_cost"] = str(crf_end - crf_start)[:5] + "s"
            print(recognize(ner_sentence))
        return render_template("ner.html", split_result = split_result,
                               lstm_split_result = "NER结果")
        # print(entity_info)
    # result = test_qm.qa(qs)
    return render_template("ner.html", split_result = "NER结果",
                           lstm_split_result = "NER结果")

