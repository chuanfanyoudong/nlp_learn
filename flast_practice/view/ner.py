import logging
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
    if request.method == 'POST':
        # print("POST")
        ner_sentence = request.form.get("ner_sentence")
        # print(ner_sentence)
        if ner_sentence.strip() == "":
            split_result = ""
        else:
            if request.form['submit'] == 'CRF':
                split_result = recognize(ner_sentence)
            # if request.form['submit'] == 'hmm分词':
            #     split_result = hmm_sg.cut(sentence)
            #     # print(type(split_result))
            # if request.form['submit'] == 'crf分词':
            #     split_result = " ".join(mm.main(sentence))
        return render_template("ner.html", split_result = split_result[0],
                               lstm_split_result = "NER结果")
        # print(entity_info)
    # result = test_qm.qa(qs)
    return render_template("ner.html", split_result = "NER结果",
                           lstm_split_result = "NER结果")

