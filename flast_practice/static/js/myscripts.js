//start==============progress bar

const fake_target = 85;
let log_id = "-1";

function Bar(bar_id) {
    return $("#" + bar_id);
}

function set_bar_percent_to(bar_id, percent) {
    Bar(bar_id).attr("data-transitiongoal", percent.toString());
    Bar(bar_id).progressbar({transition_delay: 0});
}

function get_current_percent(bar_id) {
    return parseInt(Bar(bar_id).attr("data-transitiongoal"));
}

function bar_start(bar_id) {
    Bar(bar_id).replaceWith(" <div class=\"progress-bar\" id=\"" + bar_id + "\"  aria-valuemin=\"0\" aria-valuemax=\"100\" role=\"progressbar\" data-transitiongoal=\"0\"></div>\n");
    set_bar_percent_to(bar_id, 0);
    let bar_intervl_id = setInterval(bar_running, 100, bar_id);
    Bar(bar_id).attr("bar_intervl_id", bar_intervl_id);
}

function bar_running(bar_id) {
    let current = get_current_percent(bar_id);
    let diff = fake_target - current;
    set_bar_percent_to(bar_id, current + (diff / 10) + 1);
}

function bar_stop(bar_id) {
    let bar_intervl_id = Bar(bar_id).attr("bar_intervl_id");
    clearInterval(bar_intervl_id);
    set_bar_percent_to(bar_id, 800);
}

//end==============progress bar


function get_query() {
    return $("#query-input").val()
}

function add_to_history(query) {
    let hlist = $("#history-list");

    let $hcell = $('<button type="button" class="list-group-item list-group-item-action">' + query + '</button>');

    hlist.prepend($hcell);
    if (hlist.children().length>10){
        hlist.children().last().remove();
    }
    $hcell.on("click", function () {
        $("#query-input").val(query);
        do_query(query)
    })

}

function insert_cell(row, cell_ele) {
    let cell = row.insertCell(-1);
    $(cell).append(cell_ele);
    return cell;
}

function filterHTMLTag(msg) {
    if (msg == null) {
        return "";
    }
    msg = msg.replace(/</g, "&lt");
    msg = msg.replace(/>/g, "&gt");
    msg = msg.replace(/\n/g, "<br/>");
    msg = msg.replace(/<\/?[^>]*>/g, ''); //去除HTML Tag
    msg = msg.replace(/[|]*\n/, '') //去除行尾空格
    msg = msg.replace(/&npsp;/ig, ''); //去掉npsp
    return msg;
}

function isNUmber(o) {
    return typeof o == "number";
}

function hover_show(content) {
    let ele = $("<p title='全部内容' data-trigger='click hover' data-toggle='popover'  data-html=false ></p>");

    let abstract = "";

    if (isNUmber(content)) {
        return content.toFixed(3);
    }


    if (content.length > 200) {
        content = filterHTMLTag(content);
        abstract = content.slice(0, 90) + "......";
        ele.text(abstract);
        ele.attr("data-content", content);
    } else {
        // let text_ele = $("<p></p>");
        // text_ele.text(content);
    	content = content.replace(/</g, "&lt");
    	content = content.replace(/>/g, "&gt");
    	content = content.replace(/\n/g, "<br/>");
        return content;
    }

    return ele;
}

function addRateSelect(row, row_id, data_name) {
    let options = [];
    let selector = $("<select></select>");
    selector.attr('id', row_id);
    switch (data_name) {
        case 'retrival_qa':
            options = [[0, '未评价'], [1, '问题匹配'], [2, '问题不匹配']];
            break;
        case 'doc_qa':
            options = [[0, '未评价'], [1, '抽取答案质量好'], [2, '抽取答案质量一般'], [3, '抽取答案质量差']];
            break;
        case 'kg_qa':
            options = [[0, '未评价'], [1, '回答正确'], [2, '实体识别错误'], [3, '回答错误'], [4, '找不到答案']];
            break;
        case 'gossip_qa':
            options = [[0, '未评价'], [1, '回答质量好'], [2, '回答质量较好'], [3, '回答质量一般'], [4, '回答质量较差'], [5, '回答质量差']];
            break;
    }
    options.forEach(function (item) {
        let op = $('<option></option>');
        op.attr('value', item[0]);
        op.text(item[1]);
        selector.append(op);
    });
    insert_cell(row, selector)
}

function set_data(data) {
    let row_id_index = 0;
    $("#" + data['name'] + "_elapsed").text(data['elapsed'].toFixed(3) + "秒");
    $("#" + data['name']).empty();
    let t = document.getElementById(data['name']);
    let results = data['result'];
    if (results == null) {
        return -1;
    }
    let flag = 0;
    let first = results[0];
    if(t == null){
        return -1;
    }
    let header = t.createTHead();
    let head_row = header.insertRow(0);
    for (key in first) {

        if (key != "row_id") {
            flag += 1;
            let cell = insert_cell(head_row, key);
            row_id_index += 1;
        }
    }
    if (flag != 0) {
        insert_cell(head_row, "rate");
        if (data['name'] == "gossip_qa") {
            $("#gossip_input").parent().show();
        }
    } else {
        insert_cell(head_row, "此模块未找到合适答案");
    }

    for (j = 0; j < results.length; j++) {

        row = t.insertRow(-1);
        let temp = row_id_index + 1;
        for (key in results[j]) {
            temp -= 1;
            if (temp == 0) {
                continue;
            }
            insert_cell(row, hover_show(results[j][key]));
        }
        addRateSelect(row, results[j]['row_id'], data['name']);
    }

}


$(document).ready(function () {

    $("#query-input").bind('keypress', function (event) {
        if (event.keyCode == 13) {
            $("#do-query").click();
        }
    });


    $("#do-query").click(function () {
        log_id = "-1";
        let query_str = $("#query-input").val();
        do_query(query_str);
        add_to_history(query_str);
        $("#gossip_input").val("");
    });

    $("#rate").click(function () {
        if (log_id == "-1") {
            alert("请先进行查询再点击评分按钮");
            return 0;
        }
        let rate_result = {'log_id': log_id, 'rate_data': []};
        $("select").each(function (index, item) {
            rate_result['rate_data'].push({
                'row_id': $(this).attr('id'),
                'score': $(this).val()
            })
        });
        if ($("#gossip_input").val() != "") {
            rate_result['rate_data'].push({"gossip_input": $("#gossip_input").val()});
        }
        send_rate_result(rate_result);
    })

    $(".btn-link").click(function () {
        $("#query-input").val($(this).text());
        $("#do-query").click();
    })

});



function do_query(query_str) {

    //启动进度条
    bar_start("re-bar");
    //向后台发送请求


    $.post("/query", {
        "query_str": query_str
    }, function (data, status) {
        data = JSON.parse(data);
        log_id = data['log_id'];
        if (data["query_result"] !== undefined && data["query_result"].length > 0 && data["query_result"][0]["name"] == "kg_qa") {
            fill_mykgqa_data(data["query_result"][0])
        }
        data['query_result'].forEach(function (item) {
            set_data(item);
        });
        bar_stop("re-bar");
        //启用弹出框
        $(function () {
            $("[data-toggle='popover']").popover();
        });
    },);

}

function fill_mykgqa_data(data) {
    if (data["result"] !== undefined && data["result"].length > 0 ) {
        var item = data['result'][0]
        var body = $($( "#mykgqa" ).find( ".card-body" )[0])
        var badge = $($( "#mykgqa" ).find( ".badge" )[0])
        body.empty()
        badge.empty()
        var show_list = ["Query","ner_merge","NER","NER_time","Template","SPARQL","Answer","Reply"]
        $($( "#mykgqa" ).find( ".badge" )[0]).text(data['elapsed'].toFixed(3) + "秒");
        var content = ""
        for (var index in show_list){
            var tmp = show_list[index]
            if (item.hasOwnProperty(show_list[index])){
                content = content + '<div class="row">\n' +
                    '                                <div class="col-sm-2">'+filterHTMLTag1(tmp) +'</div>\n' +
                    '                                <div class="col-sm-10">'+filterHTMLTag1(item[tmp])+'</div>\n' +
                    '                            </div>'
            }
        }
        // for (var prop in item) {
        //     content = content + '<div class="row">\n' +
        //         '                                <div class="col-sm-2">'+filterHTMLTag1(prop) +'</div>\n' +
        //         '                                <div class="col-sm-10">'+filterHTMLTag1(item[prop])+'</div>\n' +
        //         '                            </div>'
        // }
        body.html(content)
    }
}

function filterHTMLTag1(content) {
    content = content.replace(/</g, "&lt");
    content = content.replace(/>/g, "&gt");
    content = content.replace(/\n/g, "<br/>");
    content = content.replace(/ /g, "&nbsp");
    return content
}

function send_rate_result(rate_result) {

    $.post("/rate", {'data': JSON.stringify(rate_result)}, function (data) {
        alert(data);
    })
}
