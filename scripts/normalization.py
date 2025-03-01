# coding=utf-8

# coding=utf-8
import os
import re
import argparse
from clean_gadget import clean_gadget
from tqdm import tqdm
import signal
import os


def parse_options():
    parser = argparse.ArgumentParser(description='Normalization.')
    parser.add_argument('-i', '--input', help='The dir path of input dataset', type=str, required=True)
    args = parser.parse_args()
    return args


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException


def pro_one_file(filepath):
    # Set the timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # Timeout in 10 seconds

    try:
        with open(filepath, "r") as file:
            code = file.read()

        code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
        with open(filepath, "w") as file:
            file.write(code.strip())

        with open(filepath, "r") as file:
            org_code = file.readlines()
            nor_code = clean_gadget(org_code)

        with open(filepath, "w") as file:
            file.writelines(nor_code)
        # Disable the alarm after successful processing
        signal.alarm(0)
    except TimeoutException:
        print(f"Processing of {filepath} exceeded time limit and will be deleted.")
        os.remove(filepath)
    except Exception as e:
        print(f"An error occurred: {e}")
        os.remove(filepath)
        # Consider whether to delete the file in case of other exceptions
    finally:
        signal.alarm(0)  # Ensure the alarm is disabled
        

def normalize(path):
    setfolderlist = os.listdir(path)
    for setfolder in setfolderlist:
        catefolderlist = os.listdir(path + "/" + setfolder)
        # print(catefolderlist)
        for catefolder in tqdm(catefolderlist):
            filepath = path + "/" + setfolder + "/" + catefolder
            # print(catefolder)
            pro_one_file(filepath)

def main():
    args = parse_options()
    normalize(args.input)


if __name__ == '__main__':
    print("Start Normalization...")
    main()
    print("Normalization Done!")


# import os
# import re
# import argparse
# import glob
# from clean_gadget import clean_gadget
# from tqdm import tqdm

# def parse_options():
#     parser = argparse.ArgumentParser(description='Normalization.')
#     parser.add_argument('-i', '--input', help='The dir path of input dataset', type=str, required=True)
#     args = parser.parse_args()
#     return args

# def normalize(path):
#     cpp_files = glob.glob(path + "/*.c")
#     for cpp_file in tqdm(cpp_files):
#         try:
#             pro_one_file(cpp_file)
#         except:
#             # 删除该文件
#             os.remove(cpp_file)
                

# def pro_one_file(filepath):
#     with open(filepath, "r") as file:
#         code = file.read()

#     file.close()
#     code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
#     # print(code)
#     with open(filepath, "w") as file:
#         file.write(code.strip())
#     file.close()

#     with open(filepath, "r") as file:
#         org_code = file.readlines()
#         nor_code = clean_gadget(org_code)
#     file.close()
#     with open(filepath, "w") as file:
#         file.writelines(nor_code)
#     file.close()

# def main():
#     args = parse_options()
#     normalize(args.input)
    
# if __name__ == '__main__':
#     print("Start Normalization...")
#     main()
#     print("Normalization Done!")
    
    