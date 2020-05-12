# @Time    : 2020/5/11 15:17
# @Author  : Libuda
# @FileName: fizz_buzz_game.py
# @Software: PyCharm
import torch
import numpy as np

# 判断当前机器是否支持cuda 如果支持可以将一些tensor转为cuda支持的
print(torch.cuda.is_available())


# 小游戏  从1开始数数字  数到3的倍数说fizz  5的倍数buzz  3和5的倍数fizzbuzz
def fizz_buzz_encode(i):
    """
    3的倍数retunr fizz(1)  5的倍数buzz(2)  3和5的倍数fizzbuzz(3)
    :param i:
    :return:
    """
    if i%15 ==0:
        return 3
    elif i%5==0:
        return 2
    elif i%3==0:
        return 1
    else:
        return 0

def fizz_buzz_decode(i,prediction):
    return [str(i),"fizz","buzz","fizzbuzz"][prediction]


def helper(i):
    print(fizz_buzz_decode(i,fizz_buzz_encode(i)))

def binary_encode(i,num_digits=10):
    """
    数字十进制转换二进制  最多表示1024个数字
    还有这种操作？？
    网络训练时输入一个数字没有输入十个数字训练更好
    :param i:
    :param num_digits:
    :return:
    """
    res = np.array([i>>d&1 for d in range(num_digits)][::-1])
    return res

def main():
    NUM_DIGITS = 10
    # 用pytorch 定义模型
    # 为了让我们的模型学会这个游戏 我们需要定义一个损失函数和一个优化算法
    # 这个优化算法会不断优化（降低）损失函数 使得模型在该任务上取得更可能低的损失
    # 损失值低往往代表我们的模型表现好 损失值高代表我们的模型表现差
    # 由于fizzbuzz 本质是一个四分类问题  所以我们选用Cross Entropyy loss函数
    # 优化函数我们选用Stochastic Gradient Descent

    # 训练数据
    trX = torch.Tensor([binary_encode(i) for i in range(101,2**NUM_DIGITS)])
    # y只能是 0 1 2 3 这种数字 而不是浮点型
    trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101,2**NUM_DIGITS)])


    print("train x shape:{}".format(trX.shape))
    print("train y shape:{}".format(trY.shape))

    # 定义模型
    NUM_HIDDEN = 100
    model = torch.nn.Sequential(
        # 线性变换  10维转100维
        torch.nn.Linear(NUM_DIGITS,NUM_HIDDEN),
        torch.nn.ReLU(),
        # 输出4分类 返回4个向量的相似度 如果在softmax后得到概率分布
        torch.nn.Linear(NUM_HIDDEN,4)
    )


    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    # 优化函数
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

    BATCH_SIZE = 128

    # 训练多少个epoch
    END_EPOCH = 1000

    for epoch in range(END_EPOCH):
        for start in range(0,len(trX),BATCH_SIZE):
            end = start + BATCH_SIZE
            batchX = trX[start:end]
            batchY = trY[start:end]


            y_pred = model(batchX)
            loss = loss_fn(y_pred,batchY)

            print("Epoch:{},loss:{}".format(epoch,loss.item()))

            # 清空梯度
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 测试模型
    testX = torch.Tensor([binary_encode(i) for i in range(0,101)])
    with torch.no_grad():
        testY = model(testX)

    predictions = zip(range(1,101),list(testY.max(1)[1].data.tolist()))

    print([fizz_buzz_decode(i,x) for (i,x) in predictions])


if __name__ == '__main__':

    main()


