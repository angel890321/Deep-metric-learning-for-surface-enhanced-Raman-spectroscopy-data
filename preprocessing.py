# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 11:54:15 2018

@author: HW def2864
0-0-6 add standard deviation calculation, average on single spectrum file, fit average function on circular mapping 190619

0-0-4 change part of code into function can be called by other
0-0-3 change the way of import and export data form csv moldue to numpy function
0-0-2 can remove the baselie of mutliple sigle and mapping spectra data in the same folder
"""
from time import time, strftime, localtime
initial_time = time()
from traceback import print_exc
import os
import numpy as np
from scipy.interpolate import interp1d

os.chdir(os.path.split(__file__)[0])

FWHM = 50 # FWHM
step = 0.5
m = round((FWHM -1) / 2) # clipping window
m = 50
folder_name = '{} {} {}'.format(os.path.basename(__file__)[:-3], strftime("%y%m%d %H%M", localtime()), m)
no_brain_avg = True
delimiter = "\t"

def clip_baseline(y_data, m):
        background = np.copy(y_data)
        n = len(background)
        z = np.zeros(n)

        for p in range(m, 0, -1):  # 从 m 到 1
            for i in range(p, n - p):
                a1 = background[i]
                lowerB = max(0, i - p)
                upperB = min(n - 1, i + p)

                a2 = (background[lowerB] + background[upperB]) / 2

                z[i] = min(a1, a2)

            # 更新 background
            background[p:n - p] = np.minimum(background[p:n - p], z[p:n - p])

        return background

def savedata(subfolder, outputname, nparray, file_name):
    # 確保新的文件夾路徑是正確的

    # 構建文件的完整路徑
    file_path = os.path.join(f"{os.path.splitext(file_name)[0]}_{outputname}_window{m}.txt")

    # 保存文件
    np.savetxt(file_path, nparray, delimiter=delimiter, fmt='%g')
    print(f"文件保存到: {file_path}")


def save_result(x_data, y_data, backgrounds,count,file_name, name = ''):
    debaselined = y_data-backgrounds
    debaselined[0] = x_data
    if count == 0 and len(y_data) <= 2:
        y_data = y_data.transpose()
        backgrounds = backgrounds.transpose()
        debaselined = debaselined.transpose()
        
        
    savedata('interpolate', name + 'interpolate', y_data, file_name)
    savedata('background', name + 'background', backgrounds, file_name)
    savedata('debaselined', name + 'debaselined', debaselined, file_name)


def remove_baseline(file_name, m, step):
    # check whether the exist of result dictionary
    delimiter = "\t"
    single_spec = False
    
    
    # import spectra
    data_in = np.genfromtxt(file_name, delimiter=delimiter)
    count = 0
    while np.isnan(data_in[0, count]):     # cehck mapping 第一列前幾行會是空格
        count += 1
    
    if count == 0 and len(data_in[0]) == 2: # 只有一條光譜的話，labspc存檔會是直的須轉置
        rawdata = data_in.transpose()
        if rawdata[0,0] > rawdata[0,-1]:
            rawdata = np.flip(rawdata, 1) #labspec 存檔是由大到小
        single_spec = True

    else:
        rawdata = data_in[:,count:]
        if rawdata[0,0] > rawdata[0,-1]:
            rawdata = np.flip(rawdata, axis=1)

        rawdata = rawdata[np.any(rawdata, axis=1)] # 把全為零的光譜刪掉

    
    # 內插光譜
    x_data = np.arange(int(rawdata[0,0])+1, int(rawdata[0,-1]) + step , step)
    
    y_data = np.zeros((len(rawdata), len(x_data)), dtype=np.float32)
    y_data[0] = x_data
    
    #print('讀取處理結束')
    
    for spectrum in range(1,len(rawdata)):
        
        lin_f = interp1d(rawdata[0], rawdata[spectrum], kind='linear')
        y_data[spectrum] = lin_f(x_data)
    
    print('內插光譜結束')


    #backgrounds = np.copy(y_data)
    #for spectrum in range(1,len(backgrounds)): 
        #backgrounds[spectrum] = clip_baseline(y_data[spectrum], m)

    #print('背景逼近結束')

    #debaselined = y_data-backgrounds
    #debaselined[0] = x_data
    if count == 0 and len(y_data) <= 2:
        y_data = y_data.transpose()
        #backgrounds = backgrounds.transpose()
        #debaselined = debaselined.transpose()
        
    #return debaselined,single_spec
    return y_data,single_spec

    
    
    #save_result(x_data, y_data, backgrounds, count,file_name,new_folder_name, name = '')




    
