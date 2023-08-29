import os
# 导入必要的库
import numpy as np


def calculate_metrics(true_positives, false_positives, false_negatives):

    if (true_positives + false_positives)!=0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision=0
    if (true_positives + false_negatives)!=0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall=0
    if (precision + recall)!=0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score=0

    return precision, recall, f1_score


# 用于模拟的测试数据，0表示未检测到目标，1表示检测到目标
ground_truth = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
predictions = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 1])

# 统计真正例、假正例和假负例
true_positives = np.sum(np.logical_and(predictions == 1, ground_truth == 1))
false_positives = np.sum(np.logical_and(predictions == 1, ground_truth == 0))
false_negatives = np.sum(np.logical_and(predictions == 0, ground_truth == 1))

# 计算指标
precision, recall, f1_score = calculate_metrics(true_positives, false_positives, false_negatives)

# 打印结果
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)


# 读取测试集标签统计信息的函数
def get_detect_TestSample_true_info(origin_test_sample_path=r'D:\LuChang_Program_Total\ADAS_DMS项目\项目测试数据集\DMSTestDatabase\DMS_test_save'):
    testsample_true_info_dict={}
    detect_classify_name_list=os.listdir(origin_test_sample_path)
    for detect_classify_name in detect_classify_name_list:
        if detect_classify_name not in testsample_true_info_dict:
            testsample_true_info_dict[detect_classify_name]={'true_image_names':[],'true_image_number':0}
        secend_single_classify_image_path = os.path.join(origin_test_sample_path,detect_classify_name)
        single_classify_image_name_list = os.listdir(secend_single_classify_image_path)
        for classify_image_name in single_classify_image_name_list:
            testsample_true_info_dict[detect_classify_name]['true_image_names'].append(classify_image_name)
            testsample_true_info_dict[detect_classify_name]['true_image_number']+=1
    return testsample_true_info_dict

testsample_true_info_dict=get_detect_TestSample_true_info()
print(testsample_true_info_dict)
for keys,value in testsample_true_info_dict.items():
    print(keys,value)

# 读取模型预测标签统计信息的函数
def get_model_predict_label_info(predict_label_file_path=r"D:\LuChang_Program_Total\ADAS_DMS项目\DMS_Result.txt"):
    testsample_predict_info_dict = {}
    predict_lable_cur=open(predict_label_file_path,'r')
    all_predict_info_lines=predict_lable_cur.readlines()
    for single_info_line in all_predict_info_lines:
        single_info = single_info_line.strip().split('  ')

        if single_info[1] not in testsample_predict_info_dict:
            testsample_predict_info_dict[single_info[1]]=[]
            testsample_predict_info_dict[single_info[1]].append(single_info[0])
        else:
            testsample_predict_info_dict[single_info[1]].append(single_info[0])

    return testsample_predict_info_dict

testsample_predict_info_dict = get_model_predict_label_info()
for keys,value in testsample_predict_info_dict.items():
    print(keys,value)

# 统计统计真正例、假正例和假负例等指标数据
def calculate_TP_FP_FN_values(testsample_true_info_dict,testsample_predict_info_dict):
    calculate_TP_FP_FN_values_dict={}
    for single_detect_classify in testsample_true_info_dict:
        if single_detect_classify not in testsample_predict_info_dict: #如果这个目标类，完全没有检测出来的话：
            testsample_predict_info_dict[single_detect_classify]=[]
        if single_detect_classify not in calculate_TP_FP_FN_values_dict:
            # print(single_detect_classify,'!!!!!!!!!!!!!!!!!\n\n')
            calculate_TP_FP_FN_values_dict[single_detect_classify]={'true_positives':0,'false_positives':0,'false_negatives':0}
            for image_name in testsample_predict_info_dict[single_detect_classify]:
                if image_name in testsample_true_info_dict[single_detect_classify]['true_image_names']:
                    calculate_TP_FP_FN_values_dict[single_detect_classify]['true_positives']+=1
                else:
                    calculate_TP_FP_FN_values_dict[single_detect_classify]['false_positives'] += 1
            for image_name in testsample_true_info_dict[single_detect_classify]['true_image_names']:
                if image_name not in testsample_predict_info_dict[single_detect_classify]:
                    calculate_TP_FP_FN_values_dict[single_detect_classify]['false_negatives'] += 1
            # print(calculate_TP_FP_FN_values_dict[single_detect_classify])

    return calculate_TP_FP_FN_values_dict

calculate_TP_FP_FN_values_dict = calculate_TP_FP_FN_values(testsample_true_info_dict,testsample_predict_info_dict)
print(calculate_TP_FP_FN_values_dict)
for keys,value in calculate_TP_FP_FN_values_dict.items():
    print(keys,value)

# 现在获取所有目标检测类的精确率、召回率等准确性指标结果

def create_final_effect_result(calculate_TP_FP_FN_values_dict):
    final_effect_result={}
    for single_classify in calculate_TP_FP_FN_values_dict:
        final_effect_result[single_classify]={"Precision":0,"Recall":0,"F1 Score":0,}
        final_effect_result[single_classify]["Precision"],\
        final_effect_result[single_classify]["Recall"],\
        final_effect_result[single_classify]["F1 Score"]=calculate_metrics(calculate_TP_FP_FN_values_dict[single_classify]['true_positives'],calculate_TP_FP_FN_values_dict[single_classify]['false_positives'],calculate_TP_FP_FN_values_dict[single_classify]['false_negatives'])

    return final_effect_result

final_effect_result = create_final_effect_result(calculate_TP_FP_FN_values_dict)
print(final_effect_result)
for keys,value in final_effect_result.items():
    print(keys,value)