import pandas as pd
import numpy as np

import sys
from collections import Counter
from string import punctuation
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')


def count_coinc(text: str, synonyms: list) -> int:
	'''
	:param text: Текст (песни)
	:param synonyms: Список синонимов категории для сравнения
	:return: Общее количество совпадений слов текста с синонимами
	'''
	coinc = 0
	for syn in synonyms:
		coinc += text.count(syn)
	return coinc


def stemline(line: str) -> str:
	'''
    :param line: Строка слов
    :return: Строка слов после стемминга
    '''

	ps = PorterStemmer()
	token_words = line.split()
	stem_sentence = []
	for word in token_words:
		stem_sentence.append(ps.stem(word))
		stem_sentence.append(" ")
	return "".join(stem_sentence)


def title_without_stopwords(text: str, stoplist: list) -> str:
	'''
	:param text: Исходный текст
	:param stoplist: Список стоп-слов
	:return: Текст без стоп-слов
	'''
	text = text.lower()
	text_split = text.split(' ')
	# print(text_split)
	res = [word for word in text_split if word not in stoplist]
	# print(res)
	return ' '.join(res)


def get_mxm_text_songs(file_name: str, stoplist: list) -> pd.DataFrame:
	'''
	Данные 'track_id', 'mxm_track_id', 'text' из файла -> датафрейм. Из
	'text' удаляются стоп-слова.
	:param file_name: Файл с данными
	:param stoplist: Список стоп-слов
	:return: датафрейм с данными полей 'track_id', 'mxm_track_id', 'text'
	'''
	res = pd.DataFrame(columns=['track_id', 'mxm_track_id', 'text'])
	with open(file_name, 'r') as f:
		lines = f.readlines()
		words = lines[17].replace('%', '').split(',')
		for i, line in list(enumerate(lines))[18:]:
			song_info = line.split(',')
			res.at[i - 18, 'track_id'] = song_info[0]
			res.at[i - 18, 'mxm_track_id'] = song_info[1].replace(
				'\n', '')
			song_word_info = [x.split(':') for x in song_info[2:]]
			song_dict = {}
			for word, word_count in song_word_info:
				song_dict[int(word)] = int(word_count.replace('\n', ''))
			word_lists = [[words[word - 1]] * song_dict[word] for word in
						  song_dict.keys()]
			song = [word for word_list in word_lists for word in word_list]
			song = [w for w in song if w not in stoplist]
			res.at[i - 18, 'text'] = ' '.join(song).replace('\n', '')
	return res


def get_tag_genre_songs(file_name: str) -> pd.DataFrame:
	'''
	Данные 'track_id', 'majority_genre','minority_genre' из файла ->
	датафрейм. Отсутствующие значения заменяются на Nan
	:param file_name: Файл с данными
	:return: датафрейм с данными полей 'track_id', 'majority_genre',
	'minority_genre'
	'''
	res = pd.DataFrame(columns=['track_id', 'majority_genre', 'minority_genre'])
	with open(file_name, 'r') as f:
		lines = f.readlines()
		for i in range(7, len(lines)):
			line = lines[i].split('\t')
			res.at[i - 8, 'track_id'] = line[0]
			res.at[i - 8, 'majority_genre'] = line[1].replace('\n', '')
			try:
				res.at[i - 8, 'minority_genre'] = line[2].replace('\n', '')
			except:
				res.at[i - 8, 'minority_genre'] = np.nan
	return res


# def title_without_stopwords(text: str) -> str:
#     '''
# 	:param text:
# 	:param vect:
# 	:return:
# 	'''
#     text_tokens = word_tokenize(text)
#     return str([word for word in text_tokens if not word in stopwords.words()])


def get_tag_artist_title(file_name: str) -> pd.DataFrame:
	'''
	Данные 'track_id', 'song_id', 'artist','title' из файла -> датафрейм.
	:param file_name: Файл с данными
	:return: датафрейм с данными полей 'track_id', 'artist',
	'minority_genre'
	'''
	res = pd.DataFrame(columns=['track_id', 'song_id', 'artist', 'title'])
	with open(file_name, 'r') as f:
		lines = f.readlines()
		for i in range(18, len(lines)):
			line = lines[i].split('<SEP>')
			res.at[i - 19, 'track_id'] = line[0].replace('\n', '')
			res.at[i - 19, 'song_id'] = line[1].replace('\n', '')
			res.at[i - 19, 'artist'] = line[2].replace('\n', '')
			res.at[i - 19, 'title'] = line[3].replace('\n', '')
	return res


def get_voc(text: str) -> dict:
	return dict(
		Counter(text.translate(str.maketrans('', '', punctuation)).split()))


def calc_num_word(text: str, word: str) -> int:
	return str(text).count(word)


def collections_baseline_(word: str, threshold=1, syn='n') -> pd.DataFrame:
	'''

	:param syn: Индикатор, рассматривается слово или его синонимы ('y' -
	синонимы, 'n' - слово)
	:param threshold: количество вхождений (нижняя грань)
	:param word: Слово (название категории)
	:return: первые 50 эл-ов датафрейма
	'''
	res_df = pd.DataFrame()
	try:
		if type(syn) != 'str' and (syn not in ['n', 'y']):
			print(f"third parameter syn should be str, and one of ['n', 'y']. Now "
				  f"using default value 'n'.")
		if type(threshold) != 'ínt' and threshold < 1:
			print(f'second parameter threshold should be int, and > 0. Now '
				  f'using default value 1.')
		# with open('data/bow.txt', 'r') as file:
		# 	bow = file.readline()
		# 	file.close()
		if word in ['love', 'war', 'money', 'happiness', 'loneliness']:
			df_collections = pd.read_csv('data/df_collections.csv')
			if syn == 'y':
				word += '_syn'
			res_df = df_collections.loc[df_collections[word] >= threshold]
			res_df = res_df[['artist', 'title', 'play_count']]
		else:
			print(
				f"Sorry, no such word {word} in song's categories. Try another "
				f"one.")
	except Exception as e:
		print(f'{e.__doc__}')
		sys.exit(0)
	return res_df[:50].reset_index(drop=True)
