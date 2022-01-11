import re

sentence_start_tag = '<s>'


class FirstOrderHMM:
    corpus = ''
    word_tag_pairs = {}
    tag_pairs = {}
    words = {}
    tags = {}

    def __init__(self, corpus):
        self.corpus = self.clean_whitespace(corpus)
        self.create_model_data()
        return

    def create_model_data(self):
        words = self.calculate_words(self.corpus)
        words = sorted(words.items(), key=lambda x: x[1], reverse=False)
        for i in range(10):
            self.corpus = self.corpus.replace(words[i][0], 'unk')

        data = re.split(r'\t|\n', self.corpus)
        for line in data:
            line = sentence_start_tag + '/' + sentence_start_tag + ' ' + line
            self.word_tag_pairs, self.tag_pairs, self.words, self.tags = self.process_line(line,
                                                                                           self.word_tag_pairs,
                                                                                           self.tag_pairs,
                                                                                           self.words,
                                                                                           self.tags)

        self.word_tag_pairs = dict(sorted(self.word_tag_pairs.items(), key=lambda x: x[1], reverse=True))
        self.tag_pairs = dict(sorted(self.tag_pairs.items(), key=lambda x: x[1], reverse=True))
        self.words = dict(sorted(self.words.items(), key=lambda x: x[1], reverse=True))
        self.tags = dict(sorted(self.tags.items(), key=lambda x: x[1], reverse=True))
        
        return

    def clean_whitespace(self, corpus):
        data = re.split(r'\t|\n', corpus)
        result = ''
        for line in data:
            line = ''.join(line.rstrip().lstrip())
            line = ' '.join(line.split())

            if line == '\t':
                continue
            if line.isspace() or not line:
                continue
            else:
                result += line + '\n'

        return result

    def calculate_words(self, corpus):
        data = re.split(r'\t|\n', corpus)

        words = {}
        for line in data:
            line = sentence_start_tag + '/' + sentence_start_tag + ' ' + line + ' /'
            tokens = re.split(r"\s+", line)

            for i in range(0, len(tokens) - 1):
                pair1 = tokens[i].rsplit("/", 1)

                if pair1[0] and pair1[1]:
                    if pair1[0] in words:
                        words[pair1[0]] += 1
                    else:
                        words[pair1[0]] = 1

        return words

    def process_line(self, line, word_tag_pairs, tag_pairs, words, tags):
        line = line + ' /'
        tokens = re.split(r"\s+", line)

        for i in range(0, len(tokens) - 1):
            pair_0 = tokens[i].rsplit("/", 1)
            if tokens[i + 1]:
                pair_1 = tokens[i + 1].rsplit("/", 1)

            if pair_0[0] and pair_0[1]:
                word_tag_pair = (str(pair_0[0]), str(pair_0[1]))

                if pair_0[0] in words:
                    words[pair_0[0]] += 1
                else:
                    words[pair_0[0]] = 1

                if word_tag_pair in word_tag_pairs:
                    word_tag_pairs[word_tag_pair] += 1
                else:
                    word_tag_pairs[word_tag_pair] = 1

                if pair_0[1] in tags:
                    tags[pair_0[1]] += 1
                else:
                    tags[pair_0[1]] = 1

            if pair_0[1] and pair_1[1]:
                tag_pair = (pair_0[1], pair_1[1])
                if tag_pair in tag_pairs:
                    tag_pairs[tag_pair] += 1
                else:
                    tag_pairs[tag_pair] = 1

        return word_tag_pairs, tag_pairs, words, tags

    def get_corpus(self):
        return self.corpus

    def get_word_tag_pairs(self):
        return self.word_tag_pairs

    def get_tag_pairs(self):
        return self.tag_pairs

    def get_words(self):
        return self.words

    def get_tags(self):
        return self.tags

    def get_transition_probabilities(self):
        result = {}
        for tag_pair, value in self.tag_pairs.items():
            result[tag_pair] = float(value) / float(self.tags[tag_pair[0]])

        result = dict(sorted(result.items(), key=lambda x: x[0], reverse=False))
        return result

    def get_emission_probabilities(self):
        result = {}
        for tag in self.tags:
            for word_tag_pair, value in self.word_tag_pairs.items():
                if tag == word_tag_pair[1]:
                    result[(tag, word_tag_pair[0])] = float(value) / float(self.tags[tag])

        result = dict(sorted(result.items(), key=lambda x: x[0], reverse=False))
        return result

    def get_initial_tags_probabilities(self):
        result = {}
        for tag_pair, value in self.tag_pairs.items():
            if tag_pair[0] == sentence_start_tag:
                result[tag_pair] = value

        return result

    def get_vocabulary(self):
        result = {}
        total_word_count, unique_word_count = self.get_word_counts_from_tagged_corpus(self.corpus)

        result['Total Word Count - Unique Word Count'] = (total_word_count, unique_word_count)

        for word in self.words:
            most_freq_tag = ''
            max_freq = 0
            for word_pair, value in self.word_tag_pairs.items():
                if word_pair[0] == word:
                    if value >= max_freq:
                        most_freq_tag = word_pair[1]
                    else:
                        continue

            result[word] = (self.words[word], most_freq_tag)

        result = dict(sorted(result.items(), key=lambda x: x[1][0], reverse=True))
        return result

    def get_word_tag_pairs_from_line(self, line):
        tokens = re.split(r'\s+', line)
        result = []
        for token in tokens:
            pair = re.split(r'/', token)
            if pair[0] and pair[1]:
                result.append((pair[0], pair[1]))

        return result

    def get_word_counts_from_tagged_corpus(self, corpus):
        data = re.split(r'\t|\n', corpus)
        total_word_count = 0
        unique_word_count = 0
        words_list = {}
        for line in data:
            tokens = re.split(r'\s+', line)
            for token in tokens:
                if token:
                #if token[0] and token[1]:
                    if token[0] in words_list:
                        words_list[token[0]] += 1
                    else:
                        words_list[token[0]] = 1

        for word in words_list:
            total_word_count += words_list[word]

        unique_word_count = len(words_list)

        return total_word_count, unique_word_count

    def tag_words(self, words):
        result = []
        for t in words:
            dic = {}
            for key, value in self.word_tag_pairs.items():
                if key[0] == t:
                    dic[key[1]] = value / self.tags.get(key[1])
                    
            if dic:
                best = max(dic, key=dic.get)
                print(str(t) + "/" + str(best))
                result.append(str(t) + "/" + str(best))
                
        return result

    def remove_tags_from_corpus(self, corpus):
        corpus = re.split(r'\t|\n', corpus)
        result = ''
        for line in corpus:
            line = self.remove_tags_from_line(line)
            result += line + '\n'

        return result

    def remove_tags_from_line(self, line):
        tokens = re.split(r'\s+', line)
        words = []
        for token in tokens:
            pair = re.split(r'/', token)
            words.append(pair[0] + ' ')
        result = ''

        for i in range(len(words)):
            result += words[i]

        return result

    def seperate_punctuations_in_predicted_text(self, text):
        text = text.replace('.', ' .')
        text = text.replace(',', ' ,')
        text = text.replace('!', ' !')
        text = text.replace(':', ' :')
        text = text.replace(';', ' ;')
        text = text.replace('--', ' --')
        text = text.replace('(', ' (')
        text = text.replace(')', ' )')
        text = text.replace('\"', ' \"')
        return text

    def predict(self, text):
        text = self.seperate_punctuations_in_predicted_text(text)
        text = self.clean_whitespace(text)

        result = ''
        data = re.split("\n", text)
        for line in data:
            newline = line + " /"
            tokens = re.split("\s+", newline)
            tagged_words = self.viterbi(tokens)
            newline = ''
            for tagged_word in tagged_words:
                newline += tagged_word + ' '
            newline += '\n'
            result += newline

        return result

    def test_tagged_corpus(self, text):
        text = self.clean_whitespace(text)
        result = {}

        total_word_count, unique_word_count = self.get_word_counts_from_tagged_corpus(text)

        # retag
        viterbi_result = ''
        clean_text = self.remove_tags_from_corpus(text)
        retagged_data = re.split("\n", clean_text)
        for line in retagged_data:
            newline = line + " /"
            tokens = re.split("\s+", newline)
            tagged_words = self.viterbi(tokens)
            newline = ''
            for tagged_word in tagged_words:
                newline += tagged_word + ' '
            newline += '\n'
            viterbi_result += newline

        # compare
        correct_predictions_count = 0
        wrong_predictions_count = 0
        initial_data = re.split("\n", text)
        viterbi_data = re.split("\n", viterbi_result)
        for i in range(len(initial_data)):
            initial_pairs = self.get_word_tag_pairs_from_line(initial_data[i])
            pred_pairs = self.get_word_tag_pairs_from_line(viterbi_data[i])

            for i in range(len(initial_pairs)):
                if initial_pairs[i][1] != pred_pairs[i][1]:
                    wrong_predictions_count += 1
                else:
                    correct_predictions_count += 1

        # add results
        result['viterbi_result'] = viterbi_result
        result['total_word_count'] = total_word_count
        result['unique_word_count'] = unique_word_count
        result['correct_predictions_count'] = correct_predictions_count
        result['wrong_predictions_count'] = wrong_predictions_count

        return result

    def viterbi(self, words):
        observations = []

        # observations
        for index, token in enumerate(words):
            if index < len(words) - 1:
                dic = {}
                for key, value in self.word_tag_pairs.items():
                    if key[0] == token:
                        pair = (token, key[1])
                        dic[pair] = value / self.tags.get(key[1])
                if dic:
                    observations.append(dic)
                else:
                    for key, value in self.word_tag_pairs.items():
                        if key[0] == 'unk':
                            pair = (token, key[1])
                            dic[pair] = value / self.tags.get(key[1])

                    observations.append(dic)

        x = observations.pop()
        last = max(x, key=x.get)
        pos_tags = [last[0] + "/" + last[1]]

        # transitions
        while observations:
            element = observations.pop()
            dic = {}
            for keyb, valueb in element.items():
                for key, value in self.tag_pairs.items():
                    if key[1] == keyb[1]:
                        dic[keyb] = (value / self.tags.get(key[0])) * valueb
                        
            best = max(dic, key=dic.get)
            pos_tags.append(best[0] + "/" + best[1])

        return reversed(pos_tags)
