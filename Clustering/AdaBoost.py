from numpy import *
import pandas as pd
from sklearn.model_selection import KFold

dataSet = array(pd.read_csv(r'F:\DataSet\Blood_transfusion\transfusion.csv'))
X, y = dataSet[:, :4], dataSet[:, 4]
# 平均误差
cvError = 0
# k-fold交叉验证
k = 10
kf = KFold(n_splits=k)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 装载数据
    def loadData():
        dataMat = mat(X_train)
        Label = y_train
        print(dataMat)
        print(Label)
        return dataMat, Label

    # ----------------------------------单层决策树生成函数---------------------------------
    # 通过阈值比较对数据分类
    def stumpClassify(dataMat, dimen, threshold, threshIneq):
        # dimen: 维度
        # threshValue：阈值
        # threshIneq: 'lt':lower than;  'gt':greater than
        # 将返回数组所有元素置1
        retArray = ones((shape(dataMat)[0], 1))
        if threshIneq == 'lt':
            retArray[dataMat[:, dimen] <= threshold] = -1.0
        else:
            retArray[dataMat[:, dimen] > threshold] = -1.0
        return retArray

    # 遍历所有输入，找到最佳决策树
    # D是权重向量
    # dataSet: 数据集
    def buildSingleTree(dataSet, Label, D):
        dataMat = mat(dataSet)
        # 标签转为列向量
        labelMat = mat(Label).T
        m, n = shape(dataMat)
        #numSteps = 10.0
        # bestSingleTree: 存储给定权重向量D得到的最佳单层决策树信息
        bestSingleTree = {}
        bestClassify = mat(zeros((m, 1)))
        # 最佳决策树的错误率，初始化为正无穷
        minError = float('inf')

        # 遍历所有特征
        for i in range(n):
            featureValueList = sorted(dataMat[:, i])
            # 遍历特征的每个值
            for j in featureValueList:
                # 遍历条件
                for inequal in ['lt', 'gt']:
                    threshold = float(j)
                    # 预测dataMat的第i个特征的分类
                    predictedLable = stumpClassify(dataMat, i, threshold, inequal)
                    # predictError: 预测分类与真实分类的差别向量
                    # 当预测分类不等于真实分类时，predictError置1
                    predictError = mat(ones((m, 1)))
                    # 预测对的设为0
                    predictError[predictedLable == labelMat] = 0
                    # weightedError：加权误差,等于predictError与权重向量D相乘
                    weightedError = D.T * predictError

                    print("维数: %d, 阈值: %.2f, 大小关系: %s, 加权误差: %.3f"
                          % (i, threshold, inequal, weightedError))

                    if weightedError < minError:
                        minError = weightedError
                        bestClassify = predictedLable.copy()
                        bestSingleTree['dim'] = i
                        bestSingleTree['threshold'] = threshold
                        bestSingleTree['inequal'] = inequal
        return bestSingleTree, minError, bestClassify
    #-------------------------------------------------------------------------------------------------

    # 训练分类器
    # numIt：迭代次数
    # 输出多个弱分类器组成的分类器集合
    def adaBoostTrain(dataSet, Label, numIt):
        weakClassArr = []
        m = shape(dataSet)[0]
        D = mat(ones((m, 1))/m)
        # 初始化每个样本预估值为0
        estimated_value = mat(zeros((m, 1)))
        for i in range(numIt):
            # 构建一棵单层决策树，返回最好的树，错误率和分类结果
            bestSingleTree, error, classEst = buildSingleTree(dataSet, Label, D)
            print("D:", D.T)
            # 计算分类器权重
            alpha = float(0.5 * log((1.0-error) / max(error, 1e-16)))
            # 将alpha值也加入最佳树字典
            bestSingleTree['alpha'] = alpha
            # 将弱分类器加入数组
            weakClassArr.append(bestSingleTree)
            print("classEst: ", classEst.T)
            # 迭代计算D
            expon = multiply(-1 * alpha * mat(Label).T, classEst)
            D = multiply(D, exp(expon))
            D = D/D.sum()
            # 错误率累加，直到错误率为0或迭代次数
            estimated_value += alpha * classEst
            print("estimated_value: ", estimated_value.T)
            aggErrors = multiply(sign(estimated_value) != mat(Label).T, ones((m, 1)))
            errorRate = aggErrors.sum()/m
            print("total error: ", errorRate)
            # 错误率为0就返回
            if errorRate == 0.0:
                break
        return weakClassArr


    # datToClass: 测试集
    # classifierArr：弱分类器数组
    def adaClassify(datToClass, classifierArr):
        dataMat = mat(datToClass)
        m = shape(dataMat)[0]
        # estimated_value: 输出结果向量，先初始化为0
        estimated_value = mat(zeros((m, 1)))
        for i in range(len(classifierArr)):
            classEst = stumpClassify(dataMat, classifierArr[i]['dim'],\
                                     classifierArr[i]['threshold'],\
                                     classifierArr[i]['inequal'])
            # 将弱分类器结果加权求和
            estimated_value += classifierArr[i]['alpha'] * classEst
            print(estimated_value)
        return sign(estimated_value)

    # main函数
    if __name__ == "__main__":
        # 加载数据集
        dataMat, Label = loadData()
        print("dataMat:", dataMat)
        print("Label:", Label)
        # 基于单层决策树的Adaboost训练过程
        classifierArray = adaBoostTrain(dataMat, Label, 1)
        print(classifierArray)
        # 预测
        predict = adaClassify(X_test, classifierArray)
        # 误差
        predictError = mat(ones((len(y_test), 1)))
        predictError = predictError[predict != mat(y_test).T].sum() / len(y_test)
        print("predictError: ", predictError)
        cvError += predictError
        print("cverror:", cvError)

print("10折交叉验证平均误差 :", cvError / k)
