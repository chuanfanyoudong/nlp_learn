from flask import Blueprint, request, render_template
from jiang_fenci.max_fre_segment.max_fre import TokenGet
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
    split_result = {}
    if request.method == 'POST':
        # print("POST")
        sentence = request.form.get("sentence")
        if sentence.strip() == "":
            split_result = ""
        else:
            split_result["max_fre_result"] = " ".join(tg.main(sentence))
            split_result["hmm_result"] = hmm_sg.cut(sentence)
            # split_result["crf_result"] = " ".join(mm.main(sentence))
        return render_template("split_sentence.html", split_result = split_result,
                               lstm_split_result = "分词结果")
    return render_template("split_sentence.html", split_result = "分词结果",
                           lstm_split_result = "分词结果")

