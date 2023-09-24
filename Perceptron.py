#公式介紹-------------------------

#公式1 計算輸出值並放入激勵函數去做運算
#Y = sign( Σ(Xi * Wi) + b )
#解釋 : sign為激勵函數的，統一以sign來稱呼，作用是判斷()內計算後的結果是否大於小於等於0，
#然後根據判斷而給出1或0之類的結果，以此程式為例，當()內計算後的結果>0時 Y = 1 否則 Y = 0
#Σ為求和，Xi為訊號源，Wi為權重也可稱"權向量"，有幾個訊號源就有幾個權重，b則為偏移因子 b也等於W0
#需特別注意的是不管訊號源有幾個都會有一個X0，這樣才能對應上一行所解釋的"有幾個訊號源就有幾個權重"，然而X0的數值恆為1
#以此程式為例，撇除X0及W0，訊號源有兩個，分別是X1,X2，所以理所當然也會有W1,W2，將公式展開計算則等於 Y = sign(X1 * W1 + X2 * W2 + b)

#公式2 閥值與權重的變化量
#∆Wi = η(t - Y) * Xi
#解釋 : ∆唸做delta，意思是變化量，η為學習率，通常依照個人喜好自己去做設定及更改，數值為0~1之間，
#變更學習率會影響模型的訓練效率，★撰寫該程式時還未學習如何找出最佳學習率，
#t為target的縮寫，意思是期望or目標輸出值(簡單來說就是正確答案)，Y則為公式1運算後的結果，Xi一樣為訊號源
#需特別注意的是(t - Y)稱為損失函數也可稱為差距量，前者比較正式
#以此程式為例，學習率設定為0.1，將公式展開計算則等於
#閥值變化 : ∆W0 = η(t - Y) * 1 因為X0恆為1，看更改閥值還是權重就必須乘以相對應的訊號源
#權重1的變化 : ∆W1 = η(t - Y) * X1
#權重2的變化 : ∆W2 = η(t - Y) * X2

#公式3 閥值與權重的更新
#new_Wi = old_Wi + ∆Wi
#解釋 : 由公式2可得出∆Wi，通常依照個人喜好先設定好起始的閥值與權重，觀察文章與影片通常將起始的閥值與權重皆設定為0
#以此程式為例，假設目前閥值與權重皆為0.1，將公式展開計算則等於 
#閥值的更新 : new_W0 = 0.1 + ∆W0
#權重1的更新 : new_W1 = 0.1 + ∆W1
#權重2的更新 : new_W2 = 0.1 + ∆W2

import numpy as np

class Perceptron: #創建類為Perceptron(感知器)
    #初始化，self是拿來引用屬性和方法的，預設學習率為0.1，迭代次數為100次
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate 
        self.n_iterations = n_iterations 

    def operation(self, X, y): #定義operation方法用於訓練模型
        self.weights = np.zeros(X.shape[1] + 1)
        #設定W0, W1, W2起始為 0, 0, 0 ， 其中W0為公式中的b(偏移因子)，W1, W2則為權向量
        self.errors = [] #記錄每次迭代判斷錯誤次數
        time = 0 #紀錄跌代次數
        convergence = 0 #用於判斷是否達到收斂

        #下面三個變數用於判斷閥值及權重是否還有變化
        threshold = 0 
        weight1 = 0
        weight2 = 0

        for n in range(self.n_iterations):
            errors = 0 #記錄錯誤次數
            for xi, target in zip(X, y): #xi負責接收訓練數據 target負責接收正確的結果
                delta_w = self.learning_rate * (target - self.predict_y(xi)) #應用公式2，計算閥值及權重的變化量
                loss = delta_w / self.learning_rate #計算差距量(損失函數)
                print("差距量:",loss)
                print("學習率:",self.learning_rate)
                
                self.weights[1:] += delta_w * xi #應用公式3，更新權重
                self.weights[0] += delta_w #應用公式3，更新閥值
                if int(delta_w != 0.0): #判斷delta_w是否有值(變化量)
                    errors += 1 #有值(變化量)表示該筆資料判斷有錯，才會需要計算閥值或權重的變化量
                    print("此判斷為錯誤")
                print("-------------分隔線-------------")
            self.errors.append(errors) #記錄每次迭代的錯誤次數
            time += 1 #記錄迭代次數
            #判斷模型是否達到收斂
            if threshold == self.weights[0] and weight1 == self.weights[1] and weight2 == self.weights[2]: 
                convergence += 1 #閥值與權重不再變化則convergence變數+1

            else: #否則更新下列三個變數等繼續判斷閥值與權重是否還有變化
                threshold = self.weights[0]
                weight1 = self.weights[1]
                weight2 = self.weights[2]
                convergence = 0 #有變化時convergence變數歸0

            if convergence >= 3: #判斷convergence變數是否連續三次迭代中都不再變化
                print("閥值與權重已不再改變，判斷此模型已達到收斂，中斷迭代")
                break #convergence變數連續三次不再變化，判斷該模型已達到收斂，終止迭代訓練
            print("------------------第%d次迭代結束------------------"%time)
        print("總錯誤次數:",self.errors)
        print("-----------------------完成-----------------------")
        return self #最後返回self，因為已將內容皆存在該參數內

    #定義公式1不含激勵函數
    def net_input(self, X):
        Y = np.dot(X, self.weights[1:]) + self.weights[0] 
        print("輸入值:",X)
        print("權重值:",self.weights[1:])
        print("閥值:",self.weights[0])
        return Y

    #定義激勵函數
    def predict_y(self, X):
        sign = np.where(self.net_input(X) > 0.0, 1, 0)
        print("輸出值:",sign)
        return sign

X = np.array([[-1, -1], [-0.8, -0.3], [-0.4, -0.6], [-1, 1], [-0.3, 0.7], [-0.2, 0.1], [1, -1]]) #訓練數據
y = np.array([0, 0, 0, 1, 1, 1, 1]) #運算結果

prediction_X = np.array([[1, 1], [0.6, -0.3], [0.1, 0.3], [-1, 1], [-0.3, 0.7], [-0.2, -0.9], [-1, -0.1]]) #實測案例

#創建模型並進行訓練
perceptron = Perceptron(learning_rate=0.1, n_iterations=10)
perceptron.operation(X, y)

#進行預測
predictions = perceptron.predict_y(prediction_X)

#輸出預測結果
print("预测结果:", predictions)