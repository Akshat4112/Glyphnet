import time, multiprocessing
from multiprocessing import Process, Manager
import pandas as pd
import re, random
import argparse
import os

parser = argparse.ArgumentParser(description="Parameters while pasing the argument..")
parser.add_argument("--path_data", type=str, help="Define path for the data")
path_arg  = parser.parse_args().path_data


homo_1 = []
homo_2 = []

glyphs = {
		'2': ['ƻ'],
		'5': ['ƽ'],
		'a': ['à', 'á', 'à', 'â', 'ã', 'ä', 'å', 'ɑ', 'ạ', 'ǎ', 'ă', 'ȧ', 'ą'],
		'b': ['d', 'lb', 'ʙ', 'ɓ', 'ḃ', 'ḅ', 'ḇ', 'ƅ'],
		'c': ['e', 'ƈ', 'ċ', 'ć', 'ç', 'č', 'ĉ', 'ᴄ'],
		'd': ['b', 'cl', 'dl', 'ɗ', 'đ', 'ď', 'ɖ', 'ḑ', 'ḋ', 'ḍ', 'ḏ', 'ḓ'],
		'e': ['c', 'é', 'è', 'ê', 'ë', 'ē', 'ĕ', 'ě', 'ė', 'ẹ', 'ę', 'ȩ', 'ɇ', 'ḛ'],
		'f': ['ƒ', 'ḟ'],
		'g': ['q', 'ɢ', 'ɡ', 'ġ', 'ğ', 'ǵ', 'ģ', 'ĝ', 'ǧ', 'ǥ'],
		'h': ['lh', 'ĥ', 'ȟ', 'ħ', 'ɦ', 'ḧ', 'ḩ', 'ⱨ', 'ḣ', 'ḥ', 'ḫ', 'ẖ'],
		'i': ['1', 'l', 'í', 'ì', 'ï', 'ı', 'ɩ', 'ǐ', 'ĭ', 'ỉ', 'ị', 'ɨ', 'ȋ', 'ī', 'ɪ'],
		'j': ['ʝ', 'ǰ', 'ɉ', 'ĵ'],
		'k': ['lk', 'ik', 'lc', 'ḳ', 'ḵ', 'ⱪ', 'ķ', 'ᴋ'],
		'l': ['1', 'i', 'ɫ', 'ł'],
		'm': ['n', 'nn', 'rn', 'rr', 'ṁ', 'ṃ', 'ᴍ', 'ɱ', 'ḿ'],
		'n': ['m', 'r', 'ń', 'ṅ', 'ṇ', 'ṉ', 'ñ', 'ņ', 'ǹ', 'ň', 'ꞑ'],
		'o': ['0', 'ȯ', 'ọ', 'ỏ', 'ơ', 'ó', 'ö', 'ᴏ'],
		'p': ['ƿ', 'ƥ', 'ṕ', 'ṗ'],
		'q': ['g', 'ʠ'],
		'r': ['ʀ', 'ɼ', 'ɽ', 'ŕ', 'ŗ', 'ř', 'ɍ', 'ɾ', 'ȓ', 'ȑ', 'ṙ', 'ṛ', 'ṟ'],
		's': ['ʂ', 'ś', 'ṣ', 'ṡ', 'ș', 'ŝ', 'š', 'ꜱ'],
		't': ['ţ', 'ŧ', 'ṫ', 'ṭ', 'ț', 'ƫ'],
		'u': ['ᴜ', 'ǔ', 'ŭ', 'ü', 'ʉ', 'ù', 'ú', 'û', 'ũ', 'ū', 'ų', 'ư', 'ů', 'ű', 'ȕ', 'ȗ', 'ụ'],
		'v': ['ṿ', 'ⱱ', 'ᶌ', 'ṽ', 'ⱴ', 'ᴠ'],
		'w': ['vv', 'ŵ', 'ẁ', 'ẃ', 'ẅ', 'ⱳ', 'ẇ', 'ẉ', 'ẘ', 'ᴡ'],
		'x': ['ẋ', 'ẍ'],
		'y': ['ʏ', 'ý', 'ÿ', 'ŷ', 'ƴ', 'ȳ', 'ɏ', 'ỿ', 'ẏ', 'ỵ'],
		'z': ['ʐ', 'ż', 'ź', 'ᴢ', 'ƶ', 'ẓ', 'ẕ', 'ⱬ']
		}

def homo_gen_1(domain):
	try:
		new = re.sub('[^a-z25 ]+', '', domain)
		new = "".join(dict.fromkeys(new))
		index_1 = random.randint(0, len(new)-1)
		char = new[index_1]
		index_2 = random.randint(0, len(glyphs[char])-1)
		char_replace = glyphs[char][index_2]
		result_1 = re.sub(f'[{char}]', char_replace, domain, 1)
		return result_1
	except:
		with open("errors.txt", "a+") as file_object:
			file_object.seek(0)
			data = file_object.read(100)
			if len(data) > 0 :
				file_object.write("\n")
			file_object.write(domain)
        
def homo_gen_2(domain):
	try:
		new = re.sub('[^a-z25 ]+', '', domain)
		new = "".join(dict.fromkeys(new))
		index_1 = random.randint(0, len(new)-1)
		char = new[index_1]
		index_2 = random.randint(0, len(glyphs[char])-1)
		char_replace = glyphs[char][index_2]
		result_1 = re.sub(f'[{char}]', char_replace, domain, 1)

		new = re.sub('[^a-z25 ]+', '', result_1)
		new = "".join(dict.fromkeys(new))
		index_1 = random.randint(0, len(new)-1)
		char = new[index_1]
		index_2 = random.randint(0, len(glyphs[char])-1)
		char_replace = glyphs[char][index_2]
		result_2 = re.sub(f'[{char}]', char_replace, result_1, 1)

		return result_2
	except:
		with open("errors.txt", "a+") as file_object:
			file_object.seek(0)
			data = file_object.read(100)
			if len(data) > 0 :
				file_object.write("\n")
			file_object.write(domain)

domain_file = os.path.join(path_arg, "domains_final.txt")

with open(domain_file, "r") as f:
	domains_1 = f.read().splitlines()[:1000000]

with open(domain_file, "r") as f:
	domains_2 = f.read().splitlines()[1000000:2000000]


homo = []

def multiprocessing_func_1(domains_lis):
	for domain in domains_lis:
		homo_1.append(homo_gen_1(domain))

def multiprocessing_func_2(domains_lis):
	for domain in domains_lis:
		homo_2.append(homo_gen_2(domain))

if __name__ == "__main__":
	starttime = time.time()
	with Manager() as manager:
		homo_1 = manager.list()
		homo_2 = manager.list()
		processes_1 = []
		processes_2 = []

		p = multiprocessing.Process(target=multiprocessing_func_1, args=(domains_1,))
		processes_1.append(p)
		p.start()

		for process in processes_1:
			process.join()
		
		p = multiprocessing.Process(target=multiprocessing_func_2, args=(domains_2,))
		processes_2.append(p)
		p.start()

		for process in processes_2:
			process.join()

		homo_1 = list(homo_1)
		homo_2 = list(homo_2)

		data_tuples_1 = list(zip(domains_1, homo_1))
		data_tuples_2 = list(zip(domains_2, homo_2))

		dataf_1 = pd.DataFrame(data_tuples_1, columns = ['domain', 'homoglyphs'])
		dataf_2 = pd.DataFrame(data_tuples_2, columns = ['domain', 'homoglyphs'])

		dataf = pd.concat([dataf_1, dataf_2])

		dataf.to_csv('dataset_final.csv', index = False)

		print()    
		print('Time taken = {} seconds'.format(time.time() - starttime))


