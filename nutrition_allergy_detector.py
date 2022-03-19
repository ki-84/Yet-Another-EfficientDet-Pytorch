#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob 
import os
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess


compound_coef = 0
force_input_size = None  # set None to use default size
#files = glob.glob("datasets/food_label/val_test/*.jpg")
files = glob.glob("datasets/genzairyo/val/*.jpg")
print("imgnum=",len(files))
for img_path in files:
    print(img_path)



    threshold = 0.5
    iou_threshold = 0.1

    use_cuda = True
    use_float16 = False
    cudnn.fastest = False
    cudnn.benchmark = False

    #obj_list = ['allergy','eiyou','object']
    obj_list = ['genzairyo','object']
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
    full_size_image = cv2.imread(img_path)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                                 # replace this part with your project's anchor config
                                 ratios=[(1.0, 1.0), (1.3, 0.8), (1.9, 0.5)],
                                 scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    model.load_state_dict(torch.load('logs/genzairyo/efficientdet-d0_268_10207.pth'))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)


    out = invert_affine(framed_metas, out)
    #print("out",out)
    #print("x",x)
    #plt.imshow(ori_imgs[0])
    #print("len_imgs",len(ori_imgs))
    
    for i in range(len(ori_imgs)):
        #print("len_out",len(out[i]['rois']))
        if len(out[i]['rois']) == 0:
            cv2.imwrite("res/res_th_"+str(threshold)+"_"+os.path.basename(img_path),ori_imgs[i])
            continue
        ori_imgs[i] = ori_imgs[i].copy()

        for j in range(len(out[i]['rois'])):
            (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
            #print("out",out[i]['rois'][j].astype(np.int))
            obj = obj_list[out[i]['class_ids'][j]]
            
            cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255*obj_list.index(obj), 128, 64), 2)
            crop_img = full_size_image[y1:y2, x1:x2]
            cv2.imwrite("res/res_crop_"+obj+"_"+str(j)+"_"+os.path.basename(img_path),crop_img)
            
            #result = reader.readtext(crop_img, detail=0)
            
            score = float(out[i]['scores'][j])
            print(obj)
            print(obj_list.index(obj))

            cv2.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255*obj_list.index(obj), 128, 64), 2)
            cv2.imwrite("res/res_th_"+str(threshold)+"_"+os.path.basename(img_path),ori_imgs[i])
    

            #plt.imshow(ori_imgs[i])


# %%





# %%





# %%





# %%





# %%





# %%




