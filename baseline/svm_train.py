from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import json
from predictor import data
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import pickle
import thulac


dim = 5000


def cut_text(alltext):
	count = 0	
	cut = thulac.thulac(seg_only = True)
	train_text = []
	for text in alltext:
		count += 1
		if count % 2000 == 0:
			print(count)
		train_text.append(cut.cut(text, text = True))
	
	return train_text


def train_tfidf(train_data):
	tfidf = TFIDF(
			min_df = 5,
			max_features = dim,
			ngram_range = (1, 3),
			use_idf = 1,
			smooth_idf = 1
			)
	tfidf.fit(train_data)
	
	return tfidf


def read_trainData(path):
	fin = open(path, 'r', encoding = 'utf8')
	
	alltext = []
	
	accu_label = []
	law_label = []
	time_label = []

	line = fin.readline()
	while line:
		d = json.loads(line)
		alltext.append(d['fact'])
		accu_label.append(data.getlabel(d, 'accu'))
		law_label.append(data.getlabel(d, 'law'))
		time_label.append(data.getlabel(d, 'time'))
		line = fin.readline()
	fin.close()

	return alltext, accu_label, law_label, time_label


def train_SVC(vec, label):
	SVC = LinearSVC()
	SVC.fit(vec, label)
	return SVC


if __name__ == '__main__':
	print('reading...')
	alltext, accu_label, law_label, time_label = read_trainData('data_train.json')
	print('cut text...')
	train_data = cut_text(alltext)
	print('train tfidf...')
	tfidf = train_tfidf(train_data)
	
	vec = tfidf.transform(train_data)
	
	print('accu SVC')
	accu = train_SVC(vec, accu_label)
	print('law SVC')
	law = train_SVC(vec, law_label)
	print('time SVC')
	time = train_SVC(vec, time_label)
	
	print('saving model')
	joblib.dump(tfidf, 'predictor/model/tfidf.model')
	joblib.dump(accu, 'predictor/model/accu.model')
	joblib.dump(law, 'predictor/model/law.model')
	joblib.dump(time, 'predictor/model/time.model')


 # 下面是数据集中的一行数据，json数据集中一行存放一个案件信息，包括"fact"中的案件详情，以及"meta"中的案件判决结果，案件判决结果中又分为下面注释所列的各种详情。
 # {
 #  "fact": "公诉机关起诉指控，被告人张某某秘密窃取他人财物，价值2210元，××数额较大，其行为已触犯《中华人民共和国刑法》××之规定，
 #                应当以××罪追究其刑事责任。建议判处被告人张某某××以下刑罚，并处罚金。",
 #  "meta": {
 #           "relevant_articles": [ 264 ],   # 相关法律条文
 #           "accusation": [ "盗窃" ],       # 罪名
 #           "punish_of_money": 0,           # 处罚金额
 #           "criminals": [ "张某某" ],      # 罪犯
 # 		     "term_of_imprisonment":         # 刑期信息
 #              {
 #               "death_penalty": false,     # 是否死刑
 #               "imprisonment": 2,          # 刑期时间
 #               "life_imprisonment": false  # 是否无期徒刑
 #              }
 #          }
 # }



