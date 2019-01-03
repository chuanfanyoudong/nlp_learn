#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/20 15:38
# @Author  : cbdeng
# @Software: PyCharm
import logging
from flask import Blueprint, request
from website.model.EntAttrRel import EntAttrRel
from website.model.Attr import Attr
from website.model.EntEntRel import EntEntRel
from website.model.Entity import Entity
from website.model.Relation import Relation
from common_lib.utils.utils import ret_in_json

ent_ent_rel = Blueprint('ent_ent_rel', __name__)


@ent_ent_rel.route('/get_rel')
def get_rel():
    insert_by_flag(3,3)
    get_by_flag()
    return "111"


@ent_ent_rel.route('/insert_ent_ent', methods=["GET","POST"])
def insert_ent_ent():
    err_code = 0
    result_json = {"entity_list":[], "relation_list":[]}
    try:
        query_json = request.get_json(force=True)
    except Exception as e:
        logging.error("[input error]%s"%e)
        return ret_in_json(result_json, err_code=-1, msg="input format err")
    if "mainEntityId" in query_json and "logic" in query_json and "relationId" in query_json \
            and  "subEntityIds" in query_json:
        logic = query_json["logic"]
        entity_id = query_json["mainEntityId"]
        relation_id = query_json["relationId"]
        sub_entity_ids = query_json["subEntityIds"]
        if not(logic == 1 or logic == 0) or entity_id == "" or relation_id == "" or sub_entity_ids == []:
            return ret_in_json(result_json, err_code= -1, msg="input format err")
        if not isinstance(sub_entity_ids, list) or not isinstance(entity_id, str) or not isinstance(relation_id, str):
            return ret_in_json(result_json, err_code= -1, msg="input format err")
        all_entity_id = sub_entity_ids + [entity_id]
        query_entity = Entity.select().where(Entity.id.in_(all_entity_id))
        db_list = [t.id for t in query_entity]
        query_relation = Relation.select().where(Relation.id == relation_id)
        if not query_relation.exists():
            result_json["relation_list"].append(relation_id)
            err_code = 1
        new_sub_entity_ids = []
        for sub_entity_id in sub_entity_ids:
            if sub_entity_id not in db_list:
                result_json["entity_list"].append(sub_entity_id)
            else:
                new_sub_entity_ids.append(sub_entity_id)
        if entity_id not in db_list:
            result_json["entity_list"].append(entity_id)
            err_code = 1
        if err_code == 1:
            return ret_in_json(result_json, err_code=err_code, msg="info err")
        if len(new_sub_entity_ids) != 0:
            for new_sub_entity_id in new_sub_entity_ids:
                try:
                    EntEntRel.create(entity_id=entity_id, relation_id=relation_id, sub_entity_id=new_sub_entity_id,
                                     logic=logic)
                    # insert_by_flag(entity_id, new_sub_entity_id, relation_id, logic)
                except Exception as e:
                    # err_code = 1
                    logging.error("wrong insert%s"%e)
                    EntEntRel._meta.database.rollback()
            return ret_in_json(result_json)
        else:
            return ret_in_json(result_json, err_code = 1, msg="info err")
    else:
        err_code = -1
        return ret_in_json(result_json, err_code= err_code, msg="input format err")


def get_by_flag():
    for i in EntAttrRel.select():
        print(i)


def insert_by_flag(entity_id, sub_entity_id, relation_id, logic):
    # for sub_entity_id in sub_entity_ids:
    EntEntRel.create(entity_id = entity_id, relation_id = relation_id, sub_entity_id = sub_entity_id, logic = logic)
    # p.create()