
import treform as ptm

print('hello')
json_file = '../resources/korean_spelling_config.json'
corrector = ptm.spelling.DAESpellingCorrector(json_file=json_file)
sent = '아버지가 빵에 들어 가신다.'

corrected = corrector(sent)
print(corrected)
