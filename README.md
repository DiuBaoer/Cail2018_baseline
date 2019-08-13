#该项目主要记录一下Cail2018的baseline模型SVM的运行机制

#说一下各个文件的使用基本情况，主要根据训练模型，预测测试集，计算预测准确率三个方面。

#数据集主要包括训练集data_train.json和测试集input_path/data_test.json文件

#accu.txt文件是用来存放罪名的，其中读取的时候是按照每一行从0开始编码的。

#law.txt文件主要是用来存放判决所依据的法律条款，读取的时候也是从0开始编码的。




###svm_train.py文件是训练模型的主入口，右键点击运行即可。
	
    #主要逻辑就是调用训练数据集data_train.json，accu.txt，law.txt文件，以及处理数据的predictor/data.py文件。最终将训练好的模型存放到predictor/model文件夹中，一般会生成四个.model模型文件，其中三个主要是用于预测结果的。



###main_predictor.py文件是预测测试集的主入口，右键点击运行即可。

    #还是调用predictor/model文件夹中生成的模型文件，然后取测试input_path/data_test.json文件，最后将预测结果存放在output_path文件中，也会自动生成一个与input_path/data_test.json文件相同名称的json文件，不过存放的内容是三项预测的结果。


###judger_score.py文件是计算最终的准确率的主入口，右键点击运行即可，计算结果会打印在输出台上，不会生成文件。



##先写到这里，以后有补充的再写，如有问题欢迎与我讨论，QQ：1272857192.