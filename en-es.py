import argparse
from transformers import pipeline

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='English to Spanish translation')
	parser.add_argument('-i', '--input', default = 'hello, how are you?')

	args = parser.parse_args()

	translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
	print(f"English: {args.input}\nSpanish: {translator(args.input)[0]['translation_text']}")

