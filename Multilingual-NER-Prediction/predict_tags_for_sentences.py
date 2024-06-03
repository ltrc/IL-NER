"""Predict tags for sentences using HuggingFace Indian Language NERs."""
from argparse import ArgumentParser
from datasets import Dataset
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
from string import punctuation


model_name = "Sankalp-Bahad/Multilingual-NER-Model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

label_list = ['B-NEL','B-NEO','B-NEP','I-NEL','I-NEO','I-NEP','O','B-NETI','I-NETI','B-NEN','I-NEN','B-NEAR','I-NEAR']
label_to_id = {label_list[i] : i for i in range(len(label_list))}


token_specification = [
    ('datemonth',
     r'^(0?[1-9]|1[012])[-\/\.](0?[1-9]|[12][0-9]|3[01])[-\/\.](1|2)\d\d\d$'),
    ('monthdate',
     r'^(0?[1-9]|[12][0-9]|3[01])[-\/\.](0?[1-9]|1[012])[-\/\.](1|2)\d\d\d$'),
    ('yearmonth',
     r'^((1|2)\d\d\d)[-\/\.](0?[1-9]|1[012])[-\/\.](0?[1-9]|[12][0-9]|3[01])'),
    ('EMAIL1', r'([\w\.])+@(\w)+\.(com|org|co\.in)$'),
    ('url1', r'(www\.)([-a-z0-9]+\.)*([-a-z0-9]+.*)(\/[-a-z0-9]+)*/i'),
    ('url', r'/((?:https?\:\/\/|www\.)(?:[-a-z0-9]+\.)*[-a-z0-9]+.*)/i'),
    ('BRACKET', r'[\(\)\[\]\{\}]'),       # Brackets
    ('urdu_year', r'^(ء)(\d{4,4})'),
    ('NUMBER', r'^(\d+)([,\.٫٬]\d+)*(\w)*'),  # Integer or decimal number
    ('ASSIGN', r'[~:]'),          # Assignment operator
    ('END', r'[;!_]'),           # Statement terminator
    ('EQUAL', r'='),   # Equals
    ('OP', r'[+*\/\-]'),    # Arithmetic operators
    ('QUOTES', r'[\"\'‘’“”]'),          # quotes
    ('Fullstop', r'(\.+)$'),
    ('ellips', r'\.(\.)+'),
    ('HYPHEN', r'[-+\|+]'),
    ('Slashes', r'[\\\/]'),
    ('COMMA12', r'[,%]'),
    ('hin_stop', r'।'),
    ('urdu_stop', r'۔'),
    ('urdu_comma', r'،'),
    ('urdu_semicolon', r'؛'),
    ('urdu_question_mark', r'؟'),
    ('urdu_percent', r'٪'),
    ('quotes_question', r'[”\?]'),
    ('hashtag', r'#'),
    ('join', r'–')
]
# the below code converts the above expression into a python regex
tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
get_token = re.compile(tok_regex, re.U)
punctuations = punctuation + '\"\'‘’“”'


def tokenize(list_s):
    """Tokenize a list of tokens."""
    tkns = []
    for wrds in list_s:
        wrds_len = len(wrds)
        initial_pos = 0
        end_pos = 0
        while initial_pos <= (wrds_len - 1):
            mo = get_token.match(wrds, initial_pos)
            if mo is not None and len(mo.group(0)) == wrds_len:
                if mo.lastgroup == 'urdu_year':
                    tkns.append(wrds[: -4])
                    tkns.append(wrds[-4:])
                else:
                    tkns.append(wrds)
                initial_pos = wrds_len
            else:
                match_out = get_token.search(wrds, initial_pos)
                if match_out is not None:
                    end_pos = match_out.end()
                    if match_out.lastgroup == "NUMBER":
                        aa = wrds[initial_pos:(end_pos)]
                    else:
                        aa = wrds[initial_pos:(end_pos - 1)]
                    if aa != '':
                        tkns.append(aa)
                    if match_out.lastgroup != "NUMBER":
                        tkns.append(match_out.group(0))
                    initial_pos = end_pos
                else:
                    tkns.append(wrds[initial_pos:])
                    initial_pos = wrds_len
    return tkns


def tokenize_text_into_sentences_and_words(text):
    """Tokenize text into sentences and words"""
    sentences = [text]
    end_markers = ['?', '.', '!', '|', '؟', '!', '|', '۔', '।']
    proper_sentences = []
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if sentence:
            list_tokens = tokenize(sentence.split())
            end_sentence_markers = [index + 1 for index, token in enumerate(list_tokens) if token in end_markers]
            if len(end_sentence_markers) > 0:
                if end_sentence_markers[-1] != len(list_tokens):
                    end_sentence_markers += [len(list_tokens)]
                end_sentence_markers_with_sentence_end_positions = [0] + end_sentence_markers
                sentence_boundaries = list(zip(end_sentence_markers_with_sentence_end_positions, end_sentence_markers_with_sentence_end_positions[1:]))
                for start, end in sentence_boundaries:
                    individual_sentence = list_tokens[start: end]
                    proper_sentences.append(' '.join(individual_sentence))
            else:
                proper_sentences.append(' '.join(list_tokens))
            if index < len(sentences) - 1:
                next_sentence = sentences[index + 1]
                next_tokens = tokenize(next_sentence.split())
                punct_flag = True
                for token in next_tokens:
                    punct_flag &= token in punctuations
                if punct_flag:
                    if proper_sentences:
                        proper_sentences[-1] += ' ' + ' '.join(next_tokens)
                        sentences[index + 1] = ''
    return proper_sentences

def predict_labels_for_sentences(model, tokenizer, sentences):
    """Predict labels for sentences through a loaded model and tokenizer."""
    predicted_true_labels_for_all_sents = []
    with torch.no_grad():
        for index, sentence in enumerate(sentences):
            input_tensors = tokenizer(sentence, truncation=True, return_tensors='pt', max_length=512)
            outputs = model(**input_tensors)
            logit_values = outputs.logits
            arg_max_torch = torch.argmax(logit_values, axis=-1)
            predicted_tokens_classes = [
                model.config.id2label[t.item()] for t in arg_max_torch[0]]
            word_ids = input_tensors.word_ids()
            previous_word_idx = 0
            label_ids = []
            for word_index in range(len(word_ids)):
                if word_ids[word_index] == None:
                    previous_word_idx = word_ids[word_index]
                elif word_ids[word_index] != previous_word_idx:
                    label_ids.append(predicted_tokens_classes[word_index])
                    previous_word_idx = word_ids[word_index]
                else:
                    previous_word_idx = word_ids[word_index]
            predicted_true_labels_for_all_sents.append(label_ids)
    return predicted_true_labels_for_all_sents


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser()
    parser.add_argument('--input', dest='inp', help='Enter the input file')
    parser.add_argument('--output', dest='out', help='Enter the output file')
    args = parser.parse_args()
    file_name = args.inp
    with open(file_name, "r") as inp_file:
        text = inp_file.read()
    sentences = tokenize_text_into_sentences_and_words(text)
    predicted_labels = predict_labels_for_sentences(model, tokenizer, sentences)

    with open(args.out, 'w') as file:
        for sentence, label_list in zip(sentences, predicted_labels):
            for word, label in zip(sentence.split(), label_list):
                file.write(f"{word}\t{label}\n")
            file.write("\n")


if __name__ == '__main__':
    main()
