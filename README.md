# Deep Homography Estimation- PyTorch Implementation
[**Deep Image Homography Estimation**](https://arxiv.org/pdf/1606.03798.pdf)<br>
Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich
      
## Generate training dataset
```bash
cd data/
python gen_data.py
```
## Training
```bash 
python train.py
```
## Test
Download pre-trained weights
```bash 
链接：https://pan.baidu.com/s/10HXNthOBhlZbrtvIkolxKw 	提取码：l9l8 
```
Store the model to checkpoints/ folder
```bash 
python test.py
```

results | 
---   | 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210323211344844.png?x-oss-process)
 | 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210323211415816.png?x-oss-process)
 | 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210323211439899.png?x-oss-process)
 | 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210323211457964.png?x-oss-process)
 | 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210323211530847.png?x-oss-process)
 | 

##  Reference
[https://github.com/mazenmel/Deep-homography-estimation-Pytorch](https://github.com/mazenmel/Deep-homography-estimation-Pytorch)
