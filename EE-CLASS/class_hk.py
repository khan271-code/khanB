# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os
import csv
import shutil
import sys
import time

def main():
    """
    根据上一个脚本生成的 CSV 预测结果，
    将 DICOM 文件复制到对应的分类文件夹中。
    """
    
    # --- 1. 配置路径 ---
    
    # ！！重要！！
    # 这个 model_name_str 必须与上一个脚本 (run_classification.py) 中的 'model_name_str' 完全一致
    # 我从您的上一个脚本中提取了这个值：
    model_name_str = "view_23_e5_class_11-Mar-2018" 

    # 包含 CSV 文件的目录 (来自上一个脚本)
    csv_dir = "/hpc/khan271/project_2025/echo_class/echocv/test_hk/txt"
    
    # CSV 文件的完整路径
    csv_filename = "{}_all_probabilities.csv".format(model_name_str)
    csv_filepath = os.path.join(csv_dir, csv_filename)

    # 分类后 DICOM 文件的【根目录】
    base_class_dir = "/hpc/khan271/project_2025/echo_class/echocv/test_hk/class"

    # (来自您的请求) 所有可能的分类文件夹名称
    # 确保这与 CSV 文件中的 'prob_...' 列名 (去掉 'prob_') 匹配
    # 这应该也与 'viewclasses_...txt' 文件中的视图列表一致
    view_folders = [
        "plax_far", "plax_plax", "plax_laz", "psax_az", "psax_mv", "psax_pap",
        "a2c_lvocc_s", "a2c_laocc", "a2c", "a3c_lvocc_s", "a3c_laocc", "a3c",
        "a4c_lvocc_s", "a4c_laocc", "a4c", "a5c", "other", "rvinf",
        "psax_avz", "suprasternal", "subcostal", "plax_lac", "psax_apex"
    ]
    
    # 我们将查找这些列名
    prob_columns = ["prob_" + v for v in view_folders]

    print("--- 开始 DICOM 分类复制任务 ---")
    start_time = time.time()

    # --- 2. 检查输入的 CSV 文件 ---
    if not os.path.exists(csv_filepath):
        print("错误: 找不到输入的 CSV 文件:")
        print(csv_filepath)
        print("请先运行上一个分类脚本 (run_classification.py) 来生成此文件。")
        sys.exit(1)
    
    print("将从 {} 读取预测结果...".format(csv_filepath))

    # --- 3. 创建所有目标文件夹 ---
    print("正在创建目标文件夹于 {}...".format(base_class_dir))
    if not os.path.exists(base_class_dir):
        os.makedirs(base_class_dir)
        
    for view in view_folders:
        folder_path = os.path.join(base_class_dir, view)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    print("所有目标文件夹准备就绪。")

    # --- 4. 读取 CSV 并复制文件 ---
    copy_count = 0
    error_count = 0
    skipped_count = 0
    
    # 适应 Python 2.7，使用 'rb' 模式读取 CSV
    try:
        with open(csv_filepath, 'rb') as f:
            # 使用 DictReader 可以方便地通过列名访问数据
            reader = csv.DictReader(f)
            
            # 检查表头是否正确
            headers = reader.fieldnames
            
            # --- 修改开始 (1/2): 不再查找 'full_path'，而是获取前两列的名称 ---
            if len(headers) < 2:
                print("错误: CSV 文件至少需要2列 (第一列为路径, 第二列为文件名)。")
                sys.exit(1)
            
            # 根据您的请求，第一列是路径，第二列是文件名
            path_col_name = headers[0]
            file_col_name = headers[1]
            print("信息: 将使用列 '{}' (路径) 和 '{}' (文件名) 来组合完整路径。".format(
                path_col_name, file_col_name))
            # --- 修改结束 (1/2) ---
            
            
            # 循环处理 CSV 中的每一行
            for i, row in enumerate(reader):
                
                # --- 修改开始 (2/2): 从前两列组合完整路径 ---
                try:
                    dcm_dir = row[path_col_name]
                    dcm_file = row[file_col_name]
                    # 如果 dcm_dir 或 dcm_file 为 None (空值), 可能会出错
                    if dcm_dir is None or dcm_file is None:
                        raise TypeError("路径或文件名为 None")
                    original_dcm_path = os.path.join(dcm_dir, dcm_file)
                except (KeyError, TypeError) as e:
                    print("警告: 无法在行 {} (0-based) 组合路径 (列: '{}', '{}')。错误: {}。跳过。".format(
                        i, path_col_name, file_col_name, e))
                    skipped_count += 1
                    continue
                # --- 修改结束 (2/2) ---
                
                # 检查源文件是否存在
                if not os.path.exists(original_dcm_path):
                    print("警告: 源 DICOM 文件未找到，跳过: {}".format(original_dcm_path))
                    skipped_count += 1
                    continue

                # 找出概率最高的视图
                best_view = None
                max_prob = -1.0 # 概率最小为 0

                for col_name in prob_columns:
                    try:
                        # CSV 中的值是字符串，需要转换为浮点数
                        prob = float(row[col_name])
                        if prob > max_prob:
                            max_prob = prob
                            # 从 "prob_a4c" 中提取 "a4c"
                            best_view = col_name[5:] # 去掉 "prob_" 前缀
                    
                    except (ValueError, TypeError, KeyError):
                        # 如果值无效或列名不存在
                        pass 
                
                if best_view is None:
                    print("警告: 无法确定 {} 的最佳视图，跳过。".format(original_dcm_path))
                    skipped_count += 1
                    continue
                    
                # --- 准备复制 ---
                # 目标文件夹
                dest_folder = os.path.join(base_class_dir, best_view)
                
                # 目标文件路径 (保留原始文件名)
                dcm_filename = os.path.basename(original_dcm_path)
                dest_filepath = os.path.join(dest_folder, dcm_filename)
                
                # --- 执行复制 ---
                try:
                    shutil.copy(original_dcm_path, dest_filepath)
                    copy_count += 1
                    
                    # (可选) 打印进度
                    if (i + 1) % 100 == 0:
                        print("... 已处理 {} 个文件，已复制 {} 个...".format(i + 1, copy_count))
                        
                except Exception as e:
                    print("错误: 复制文件 {} 到 {} 时出错: {}".format(original_dcm_path, dest_filepath, e))
                    error_count += 1

    except IOError as e:
        print("错误: 打开 CSV 文件失败: {}".format(e))
        sys.exit(1)
    except Exception as e:
        print("发生意外错误: {}".format(e))
        sys.exit(1)

    end_time = time.time()
    print("\n--- 任务完成 ---")
    print("成功复制: {} 个文件".format(copy_count))
    print("跳过 (源文件丢失或无法分类/组合路径): {} 个文件".format(skipped_count))
    print("复制出错: {} 个文件".format(error_count))
    print("总耗时: {:.2f} 秒。".format(end_time - start_time))

# --- 脚本入口 ---
if __name__ == '__main__':
    main()