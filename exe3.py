# 实验目的：实现基于BoW特征的物体识别，学习物体识别过程
# 实验要求：自选一种公开图像数据集，实现基于BoW特征的物体识别，在测试图像上的识别精度不低于70%
# wubbalubbadubdub
import cv2
import numpy as np



# 获取不同类别图像的路径
def path(cls, i):
    return "%s/%s%d.pgm" % (datapath, cls, i + 1)


# 以灰度格式读取图像,并从图像中提取SIFT特征，然后返回描述符
def extract_sift(fn):
    im = cv2.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]


# 返回基于BOW描述符提取器计算得到的描述符
def bow_features(fn):
    im = cv2.imread(fn, 0)
    return extract_bow.compute(im, detect.detect(im))

# 显示predict方法结果，并返回结果信息
def predict(fn):
    f = bow_features(fn);
    p = svm.predict(f)
    print(fn, "\t", p[1][0][0])
    return p

# 声明训练图像的基础路径
datapath = "./CarData/TrainImages/"

# 查看下载的数据素材，发现汽车数据集中图像按：pos-x.pgm 和 neg-x.pgm 命名，其中x是一个数字。
# 这样从path读取图像时，只需传递变量i的值即可
pos, neg = "pos-", "neg-"


# 创建基于FLANN匹配器实例
flann_params = dict(algorithm=1, trees=5)
flann = cv2.FlannBasedMatcher(flann_params, {})

# 创建BOW训练器，指定簇数为40
bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)

# 创建两个SIFT实例，一个提取关键点，一个提取特征；
detect = cv2.SIFT_create()
extract = cv2.SIFT_create()

# 初始化BOW提取器，视觉词汇将作为BOW类输入，在测试图像中会检测这些视觉词汇
extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)

# 从每个类别中读取50个正样本和50个负样本，并增加到训练集的描述符
for i in range(50):
    bow_kmeans_trainer.add(extract_sift(path(pos, i)))
    bow_kmeans_trainer.add(extract_sift(path(neg, i)))

# cluster()函数执行k-means分类并返回词汇
# 并为BOWImgDescriptorExtractor指定返回的词汇，以便能从测试图像中提取描述符
voc = bow_kmeans_trainer.cluster()
extract_bow.setVocabulary(voc)



# 创建两个数组，分别存放训练数据和标签
# 调用BOWImgDescriptorExtractor产生的描述符填充两个数组，生成正负样本图像的标签
traindata, trainlabels = [], []
for i in range(20):
    traindata.extend(bow_features(path(pos, i)));
    trainlabels.append(1)  # 1表示正匹配

    traindata.extend(bow_features(path(neg, i)));
    trainlabels.append(-1)  # -1表示负匹配

# 创建一个svm实例
svm = cv2.ml.SVM_create()

# 通过将训练数据和标签放到NumPy数组中来进行训练
svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

'''
以上设置都是用于训练好的SVM，剩下要做的是给SVM一些样本图像
'''



def Car(car):
    # 定义两个样本图像的路径，并读取样本图像信息
    # notcar = 'dilireba.png'
    car_img = cv2.imread(car)

    # 将图像传给已经训练好的SVM，并获取检测结果
    car_predict = predict(car)

    # 以下用于屏幕上显示识别的结果和图像
    font = cv2.FONT_HERSHEY_SIMPLEX

    if (car_predict[1][0][0] == 1.0):
        cv2.putText(car_img, 'Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(car_img, 'Car Not Detected', (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return car_img

if __name__=="__main__":
    Path = "E:/PythonFile/R-C.jpg"
    Car(Path)