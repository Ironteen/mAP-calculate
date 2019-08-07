#!/usr/bin/env python
# -*- coding:utf-8 -*-
# title            : main.py
# description      : compute the mAP of detection results
# author           : Zhijun Tu
# email            : tzj19970116@163.com
# date             : 2017/07/28
# version          : 1.0
# notes            : 
# python version   : 2.7.12 which is also applicable to 3.5
############################################################### 
import os
import cv2
import json
import numpy as np 
from collections import defaultdict
# basic info
data_dir       = './data'
detections_dir = os.path.join(data_dir,'predictions')
gt_dir         = os.path.join(data_dir,'ground_truth')
images_dir     = os.path.join(data_dir,'images')
# save .json file
det_key = ['confidence','image_id','bbox']
gt_key = ['category','bbox','detected']
restore_dir    = './restore'
min_iou = 0.5
if not os.path.exists(restore_dir):
	os.mkdir(restore_dir)

def PraseGroundtruth(file_dir):
	'''
		prase ground truth data and save in .json format
	'''
	# obtain all the text file name
	gt_counter_per_class = dict()
	file_list = os.listdir(file_dir)
	for file_idx in file_list:
		txt_path = os.path.join(file_dir,file_idx)
		image_id = file_idx.strip('.txt')
		img_path = os.path.join(restore_dir,image_id+'_ground_truth.json')
		bboxes_list = []
		content = open(txt_path,'r').readlines()
		# process the bbox line by line
		for line in content:
			cls,xmin,ymin,xmax,ymax = line.split(' ')
			if cls not in gt_counter_per_class.keys():
				gt_counter_per_class[cls] = 1
			else:
				gt_counter_per_class[cls] += 1
			bbox = [float(xmin),float(ymin),float(xmax),float(ymax)]
			gt_dict = dict(zip(gt_key,[cls,bbox,0]))
			bboxes_list.append(gt_dict)
		with open(img_path,'w') as gt_file:
			json.dump(bboxes_list,gt_file)
	return gt_counter_per_class

def PraseDetections(file_dir,CategoryList):
	'''
		prase detection data and save in .json format
	'''
	CategoryDict = defaultdict(list)
	# obtain all the text file name
	file_list = os.listdir(file_dir)
	for file_idx in file_list:
		txt_path = os.path.join(file_dir,file_idx)
		image_id = file_idx.strip('.txt')
		content = open(txt_path,'r').readlines()
		# process the bbox line by line
		for line in content:
			cls,conf,xmin,ymin,xmax,ymax = line.split(' ')
			conf,bbox = float(conf),[float(xmin),float(ymin),float(xmax),float(ymax)]
			det_dict = dict(zip(det_key,[conf,image_id,bbox]))
			CategoryDict[cls].append(det_dict)
	# save all the info by category
	for cls,bboxs in CategoryDict.items():
		if cls in CategoryList:
			cls_dir = os.path.join(restore_dir,cls+'.json')
			bboxs.sort(key=lambda x:x['confidence'], reverse=True)
			with open(cls_dir,'w') as det_file:
				json.dump(bboxs,det_file)
	return CategoryDict

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    ap = 0.0
    # print('mrec:',mrec,len(mrec))
    # print('mpre:',mpre,len(mpre))
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def cal_iou(gt_bbox,det_bbox):
	'''
		compute the IoU of one detect bbox and 
		all the groundtruth in the image
	'''
	xmin = np.maximum(gt_bbox[:,0], det_bbox[0])
	ymin = np.maximum(gt_bbox[:,1], det_bbox[1])
	xmax = np.minimum(gt_bbox[:,2], det_bbox[2])
	ymax = np.minimum(gt_bbox[:,3], det_bbox[3])
	w = np.maximum(xmax - xmin + 1. ,0.0)
	h = np.maximum(ymax - ymin + 1. ,0.0)
	inters = w * h

	area1 = (det_bbox[2] - det_bbox[0] + 1. ) * (det_bbox[3] - det_bbox[1] + 1. )
	area2 = (gt_bbox[:,2] - gt_bbox[:,0] + 1. ) *(gt_bbox[:,3] - gt_bbox[:,1] + 1. )
	iou = inters /(area1+area2-inters)

	return iou

def computeAP(category,count_tp):
	'''
		compute the AP of one category
	'''
	# load detection 
	count_tp[category] = 0
	cate_dir = os.path.join(restore_dir,category+'.json')
	DetInfo = json.load(open(cate_dir,'r'))
	tp = [0]*len(DetInfo)
	fp = [0]*len(DetInfo)
	for idx,info in enumerate(DetInfo):
		ImgId,DetBbox = info['image_id'],info['bbox']
		img_dir = os.path.join(restore_dir,ImgId+'_ground_truth.json')
		GTinfo = json.load(open(img_dir,'r'))
		gtbboxes = []
		for gt in GTinfo:
			if gt['category']==category:
				gtbboxes.append(gt['bbox'])
			else:
				gtbboxes.append([0,0,0,0])
		ious = cal_iou(np.array(gtbboxes),DetBbox)
		index = np.argmax(ious)
		iou,cate = ious[index],GTinfo[index]['category']
		if iou>=min_iou:
			if GTinfo[index]['detected']==0:
				tp[idx]=1
				count_tp[category]+=1
				GTinfo[index]['detected'] = 1
				with open(img_dir,'w') as gt_file:
					json.dump(GTinfo,gt_file)
			else:
				fp[idx] = 1
		else:
			fp[idx] = 1
	cumsum = 0
	for idx, val in enumerate(fp):
	    fp[idx] += cumsum
	    cumsum += val
	cumsum = 0
	for idx, val in enumerate(tp):
	    tp[idx] += cumsum
	    cumsum += val
	#print(tp)
	rec = tp[:]
	for idx, val in enumerate(tp):
		rec[idx] = float(tp[idx]) / gt_counter_per_class[category]
	#print(rec)
	prec = tp[:]
	for idx, val in enumerate(tp):
	    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
	ap, mrec, mprec = voc_ap(rec[:], prec[:])
	print("{0:.2f}%".format(ap*100) + " = " + category + " AP ") #class_name + " AP = {0:.2f}%".format(ap*100)
	return ap

if __name__=='__main__':
	# prase ground truth info
	gt_counter_per_class = PraseGroundtruth(gt_dir)
	CategoryList = list(gt_counter_per_class.keys())
	# print('gt_counter',CategoryList,len(CategoryList),type(CategoryList))
	# prase detections info
	dt_counter_per_class = PraseDetections(detections_dir,CategoryList)
	DetCategoryList = list(dt_counter_per_class.keys())
	# calculate the mp
	count_tp = dict()
	sum_AP = 0
	for category in CategoryList:
		if category in DetCategoryList:
			ap = computeAP(category,count_tp)
			sum_AP +=ap
	mAP = sum_AP/len(CategoryList)
	print("mAP = {0:.2f}%".format(mAP*100))






	

