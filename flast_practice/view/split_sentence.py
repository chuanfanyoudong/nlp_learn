from flask import Blueprint, request, render_template
from jiang_fenci.max_fre_segment.main import TokenGet
from jiang_fenci.hmm_segment.segment.model import Segment
from jiang_fenci.lstm_crf.simple_model_test import ModelMain

hmm_sg = Segment()
# hmm_sg = TokenGet()
tg = TokenGet()
mm = ModelMain()





split = Blueprint('split', __name__)

@split.route('/split_demo', methods=['GET', 'POST'])
def split_method():
    """
    分词接口
    :return: 输入要分词的句子，输出分词的结果
    """

    # print(username)
    if request.method == 'POST':
        # print("POST")
        sentence = request.form.get("sentence")
        if sentence.strip() == "":
            split_result = ""
        else:
            split_result = tg.main(sentence)
            if request.form['submit'] == '最大频率分词':
                split_result = " ".join(tg.main(sentence))
            if request.form['submit'] == 'hmm分词':
                split_result = hmm_sg.cut(sentence)
                # print(type(split_result))
            if request.form['submit'] == 'crf分词':
                result = mm.main(sentence)
                # print("crf结果", result)
                split_result = " ".join(mm.main(sentence))
        return render_template("split_sentence.html", split_result = split_result,
                               lstm_split_result = "分词结果")
    return render_template("split_sentence.html", split_result = "分词结果",
                           lstm_split_result = "分词结果")

