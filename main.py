import kenlm
model = kenlm.LanguageModel('mytestdata.arpa')
sent = 'justice officials quickly pulled the plug on the vandalised page'
result = model.score(sent)
print(result)
