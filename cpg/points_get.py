import os
import re
import pickle


def get_pointers_node(node_dict):
    pointer_node_list = []
    # identifier_list = []
    identifier_node_type = ['Identifier', 'MethodParameterIn', 'Field_Identifier']
    for node in node_dict:
        node_type = node_dict[node].node_type
        if node_type in identifier_node_type:
            node_code = node.code
            # 不存在指针则返回-1
            indx_1 = node_code.find("*")
            if indx_1 != -1:
                pointer_node_list.append(node_dict[node])
            
    pointer_node_list = list(set(pointer_node_list))
    return pointer_node_list

'''
<operator>.indirectIndexAccess
'''

def get_all_array(node_dict):
    array_node_list = []
    identifier_list = []
    identifier_node_type = ['indirectIndexAccess', 'MethodParameterIn']
    for node in node_dict:
        node_type = node_dict[node].node_type
        if node_type in identifier_node_type:
            
            identifier_list.append(node_dict[node])
    for node in identifier_list:
        node_code = node.code
        if node_code.find("[") != -1:
            array_node_list.append(node)
    array_node_list = list(set(array_node_list))
    return array_node_list
    
def get_all_sensitiveAPI(node_dict):
    with open("/home/itachi/A/VulDet/CDDFVUL/pdg/sensitive_func.pkl", "rb") as fin:
        list_sensitive_funcname = pickle.load(fin)
    call_node_list = []
    call_type = "Call"   
    for func_name in list_sensitive_funcname:
        for node in node_dict:
            node_type = node_dict[node].node_type
            node_code = node_dict[node].code.split("(")
            if node_type == call_type:
                for c in node_code:
                    if func_name in c:
                        call_node_list.append(node_dict[node])
                        
    return call_node_list

def get_all_integeroverflow_point(node_dict):
    interoverflow_list = []
    exp_type = 'assignment'
    for node in node_dict:
        node_type = node_dict[node].node_type
        if node_type == exp_type:
            node_code = node_dict[node].code
            if node_code.find("="):
                code = node_code.split('=')[-1].strip()
                pattern = re.compile("((?:_|[A-Za-z])\w*(?:\s(?:\+|\-|\*|\/)\s(?:_|[A-Za-z])\w*)+)")
            else:
                code = node_code
                pattern = re.compile("(?:\s\/\s(?:_|[A-Za-z])\w*\s)")
            results = re.search(pattern, code)
            if results != None:
                interoverflow_list.append(node_dict[node])
            
    return interoverflow_list