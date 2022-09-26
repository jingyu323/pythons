# -*- coding: utf-8 -*-

import difflib




def get_filename_arr(file_path):
    file_name_arr = []

    with open(file_path, 'r',encoding='UTF-8') as f:
        while True:
            str = f.readline()  # 每次读取一行
            if not str:
                break
            tep_filename=str.split()
            file_name_arr.append(tep_filename[len(tep_filename)-1])

    return  file_name_arr


def get_filename_map(file_path):
    file_name_map = {}

    with open(file_path, 'r',encoding='UTF-8') as f:
        while True:
            str = f.readline()  # 每次读取一行
            if not str:
                break
            tep_filename=str.split()

            file_name=tep_filename[len(tep_filename)-1]
            file_name_map[file_name]=tep_filename[4]

    return  file_name_map


def compare_file_arr():
    print("s")

if __name__ == '__main__':
    pt_file = "pt_file.txt"
    pt_file_names =get_filename_arr(pt_file)
    print(len(pt_file_names))
    print(pt_file_names)

    gw_file = "gw_file.txt"
    gw_file_names = get_filename_arr(gw_file)
    print(gw_file_names)
    print(len(gw_file_names))

    file_unfind = []
    for name in gw_file_names:
        print(name)

        if name  in pt_file_names:
            print(name + " in pt, index is:"+ str(gw_file_names.index(name)) )
        else:
            print(name + " not in pt, index is:" + str(gw_file_names.index(name)) )

            file_unfind.append(name)

    print(file_unfind)
    print(len(file_unfind))


    ## file_size

    pt_file_size = get_filename_map(pt_file)
    print(len(pt_file_size))
    print(pt_file_size)

    gw_file_size = get_filename_map(gw_file)
    print(len(gw_file_size))
    print(gw_file_size)
    count = 0
    not_eq_count = 0
    not_eq_count_filenam=[]
    for key in pt_file_size:
        pt_f_size = pt_file_size[key]
        gw_f_size = gw_file_size[key]
        count=count+1
        if gw_f_size != pt_f_size:
            print(gw_f_size +"not eq "+gw_f_size+",filename is:"+key)
            not_eq_count = not_eq_count + 1
            not_eq_count_filenam.append(key)
        else:
            print(gw_f_size + " eq " + gw_f_size + ",filename is:" + key)
    print("compare count is:"+str(count))
    print(" not eq count is:"+str(not_eq_count))
    print(" not eq file count file name is:"+str(not_eq_count_filenam))