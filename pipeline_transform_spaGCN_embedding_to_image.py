import pandas as pd
from pipeline_sparse_expression_to_image import save_transformed_RGB_to_image_and_csv, scale_to_RGB
import json, os
import numpy as np
import pickle as pkl
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt



def get_ground_truth(anndata,sample_name,img_folder, img_type, scale_factor_file= None):
    print('start ploting!')
    count=0
    label_index = anndata.obs['label']
    #import matplotlib.pyplot as plt
    for x,y in zip(anndata.obs["array_row"],anndata.obs["array_col"]):
        c = color_get(int(label_index[count]))
        #print("c: ", c)
        plt.scatter(x/100,y/100,color=c,s=75)
        #print("Scatter updated")
        count+=1
    print("Out of for loop")

    """
    # Adjust y-axis
    axes = plt.gca()
    print(axes.get_ylim())
    y_ax = list(axes.get_ylim())
    axes.set_ylim([y_ax[0]-0.25,y_ax[1]+0.25])
    print(axes.get_ylim())
    """

    # print
    filename="ground_truth_"+sample_name
    filename=str(filename)+'.png'
    plt.savefig(f"{img_folder}"+filename,format='png')
    print('image saved')
    
    return None,None

def get_clusters(anndata,sample_name,img_folder, img_type, scale_factor_file= None):
    print('start ploting!')
    count=0
    label_index = anndata.obs['layer']
    #import matplotlib.pyplot as plt
    for x,y in zip(anndata.obs["array_row"],anndata.obs["array_col"]):
        c = color_get(int(label_index[count]))
        #print("c: ", c)
        plt.scatter(x/100,y/100,color=c,s=75)
        #print("Scatter updated")
        count+=1
    print("Out of for loop")

    """
    # Adjust y-axis
    axes = plt.gca()
    print(axes.get_ylim())
    y_ax = list(axes.get_ylim())
    axes.set_ylim([y_ax[0]-0.25,y_ax[1]+0.25])
    print(axes.get_ylim())
    """

    # Print
    filename=sample_name + "_clusters"
    filename=str(filename)+'.png'
    plt.savefig(f"{img_folder}"+filename,format='png')
    print('image saved')
    
    return None,None

def transform_embedding_to_image(anndata,sample_name,img_folder, img_type, scale_factor_file= None):
    X_transform = anndata.obsm["embedding"]
    print(X_transform.shape)
    # embedding_data = pd.read_csv("LogCPM_151507_humanBrain_128_pca_0.2_res.csv")
    # X_transform = embedding_data.loc[:, ['embedding0', 'embedding1', 'embedding2']].values
    full_data = anndata.obs
    X_transform[:, 0] = scale_to_RGB(X_transform[:, 0], 100)/255
    X_transform[:, 1] = scale_to_RGB(X_transform[:, 1], 100)/255
    X_transform[:, 2] = scale_to_RGB(X_transform[:, 2], 100)/255
    print('start ploting!')
    count=0
    #import matplotlib.pyplot as plt
    for x,y in zip(anndata.obs["array_row"],anndata.obs["array_col"]):
        c=(X_transform[count][0],X_transform[count][1],X_transform[count][2])
        #print("c: ", c)
        plt.scatter(x/100,y/100,color=c,s=75)
        #print("Scatter updated")
        count+=1
    print("Out of for loop")

    # Adjust y-axis
    axes = plt.gca()
    print(axes.get_ylim())
    y_ax = list(axes.get_ylim())
    axes.set_ylim([y_ax[0]-0.25,y_ax[1]+0.25])
    print(axes.get_ylim())
    #y_ax[0] = y_ax[0]+1
    #y_ax[1] = y_ax[1]+1
    #plt.ylim(y_ax[0],y_ax[1])   # Adjusts chart y axis limits

    # Print
    filename=sample_name + "_embedding"
    filename=str(filename)+'.png'
    plt.savefig(f"{img_folder}"+filename,format='png')
    print('image saved')
    
    return None,None
    X_transform[:, 0] = scale_to_RGB(X_transform[:, 0], 100)
    X_transform[:, 1] = scale_to_RGB(X_transform[:, 1], 100)
    X_transform[:, 2] = scale_to_RGB(X_transform[:, 2], 100)

    if scale_factor_file is not None:   # uns

        radius = int(0.5 *  anndata.uns['fiducial_diameter_fullres'] + 1)
        # radius = int(scaler['spot_diameter_fullres'] + 1)
        max_row = max_col = int((2000 / anndata.uns['tissue_hires_scalef']) + 1)


    else:  # no 10x
        radius = 100 #
        max_row = np.int(np.max(full_data['pxl_col_in_fullres'].values + 1) + radius)   #
        max_col = np.int(np.max(full_data['pxl_row_in_fullres'].values + 1) + radius)

    high_img, low_img= save_transformed_RGB_to_image_and_csv(full_data['pxl_col_in_fullres'].values,
                                          full_data['pxl_row_in_fullres'].values,
                                          max_row,
                                          max_col,
                                          X_transform,
                                          sample_name,  # file name  default/define
                                          img_type,
                                          img_folder,
                                          plot_spot_radius = radius
                                          )

    del full_data, X_transform
    return high_img, low_img



def color_get(val):
    color_list = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4', 
                '#469990', '#dcbeff', '#9A6324', '#fffac8', '#800000', 
                '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', 
                '#ffffff', '#000000']
    if val == 0:
        return color_list[0]
    elif val == 1:
        return color_list[1]
    elif val == 2:
        return color_list[2]
    elif val == 3:
        return color_list[3]
    elif val == 4:
        return color_list[4]
    elif val == 5:
        return color_list[5]
    elif val == 6:
        return color_list[6]
    elif val == 7:
        return color_list[7]
    elif val == 8:
        return color_list[8]
    elif val == 9:
        return color_list[9]
    elif val == 10:
        return color_list[10]
    elif val == 11:
        return color_list[11]
    elif val == 12:
        return color_list[12]
    elif val == 13:
        return color_list[13]
    elif val == 14:
        return color_list[14]
    elif val == 15:
        return color_list[15]
    elif val == 16:
        return color_list[16]

