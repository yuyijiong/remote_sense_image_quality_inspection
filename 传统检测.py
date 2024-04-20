import cv2
import numpy as np
import torch
import PIL
from PIL import Image as PILImage
from matplotlib import pyplot as plt
class TraditionalDetection:
    def __init__(self,missing_pixel_label_id=10,stripe_noise_label_id=9):
        self.missing_pixel_label_id=missing_pixel_label_id
        self.stripe_noise_label_id=stripe_noise_label_id
    #提取像素缺失
    @staticmethod
    def extract_missing_pixel(img,label_id=10):
        #将彩图图片二值化，值若为[0,0,0]或[255,255,255]则为缺失像素，设为255，否则为0
        #若三个通道都为0或255，则为缺失像素
        mask1=np.where((img==0).all(axis=2),255,0)
        mask2=np.where((img==255).all(axis=2),255,0)
        mask=np.where((mask1==255)|(mask2==255),255,0)

        #将缺失像素的值设为label_id
        mask[mask==255]=label_id
        return mask

    #提取条纹状噪声,单通道
    @staticmethod
    def extract_stripe_noise_one_channel(img,label_id):
        #对做otscu阈值分割
        binary_img=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        # cv2.imshow('binary_img',binary_img)
        # cv2.waitKey(0)
        mask=np.zeros(img.shape,dtype=np.uint8)
        # 连通域检测，并提取长宽比大于5或者小于0.2的连通域，将这些连通域设为label_id，否则为0
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, None, None, None, 8, cv2.CV_32S)
        for i in range(1, nlabels):
            # 过滤长宽比
            if stats[i][2] / stats[i][3] > 5 or stats[i][2] / stats[i][3] < 0.2:
                #mask中将这些连通域设为label_id
                x, y, w, h, area = stats[i]
                mask[y:y+h,x:x+w]=label_id

        return mask

    #提取条纹状噪声
    @classmethod
    def extract_stripe_noise(cls,img,label_id=11):
        #分离三个通道，分别提取条纹状噪声
        img = np.array(img)
        mask_r=cls.extract_stripe_noise_one_channel(img[:,:,0],label_id)
        mask_g=cls.extract_stripe_noise_one_channel(img[:,:,1],label_id)
        mask_b=cls.extract_stripe_noise_one_channel(img[:,:,2],label_id)
        #将三个通道的mask合并，取最大值
        mask=np.maximum(mask_r,mask_g)
        mask=np.maximum(mask,mask_b)
        # plt.imshow(mask)
        # plt.show()
        return mask

    #预测一张图片
    def predict_one_image(self, img:PILImage.Image):
        img=np.array(img)
        #提取缺失像素
        missing_pixel_mask=self.extract_missing_pixel(img,self.missing_pixel_label_id)
        #提取条纹状噪声
        stripe_noise_mask=self.extract_stripe_noise(img,self.stripe_noise_label_id)
        #将两个mask合并，取最大值
        mask=np.maximum(missing_pixel_mask,stripe_noise_mask)
        return mask


if __name__ == '__main__':
    #测试
    img=PILImage.open('../测试用图片/条纹噪声.png')
    model=TraditionalDetection()
    mask=model.predict_one_image(img)
    #mask=PILImage.fromarray(mask)
    plt.imshow(mask)
    plt.axis('off')
    plt.savefig('../测试用图片/条纹噪声_传统检测.png')
    plt.show()




