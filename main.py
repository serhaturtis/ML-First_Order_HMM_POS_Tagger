import os
import re

import hmm

training_files_path = './input/brown_hw/Train/'
test_files_path = './input/brown_hw/Test/'
ca41_file_path = './input/brown_hw/Test/ca41'
untagged_file_path = './input/to_predict.txt'
output_files_path = './output/'


def read_file(filepath):
    text = ''
    with open(filepath, "r") as f:
        text = f.read()
        f.close()
    return text.lower()

def read_files_in_folder(folder_path):
    files_list = [file for file in os.listdir(folder_path) if os.path.isfile(folder_path + file)]
    text = ''
    for file in files_list:
        f = open(folder_path + file, "r")
        text += str(f.read()) + "\n"

    return text.lower()


def write_pairs_to_file(pairs, folder_path, filename):
    filepath = os.path.abspath(folder_path) + '/' + filename
    os.makedirs(folder_path, exist_ok=True)
    with open(filepath, 'w') as file:
        for key, value in pairs.items():
            file.write(str(key) + ' : ' + str(value) + '\n')
        file.close()
    return


def main():
    training_data = read_files_in_folder(training_files_path)
    test_data = read_files_in_folder(test_files_path)
    ca41_data = read_file(ca41_file_path)
    untagged_data = read_file(untagged_file_path)

    model = hmm.FirstOrderHMM(training_data)

    write_pairs_to_file(model.get_tags(), output_files_path, 'PosTags.txt')

    write_pairs_to_file(model.get_transition_probabilities(), output_files_path, 'TransitionProbs.txt')

    write_pairs_to_file(model.get_vocabulary(), output_files_path, 'Vocabulary.txt')

    write_pairs_to_file(model.get_emission_probabilities(), output_files_path, 'EmissionProbs.txt')

    write_pairs_to_file(model.get_initial_tags_probabilities(), output_files_path, 'InitialProbs.txt')

    test_results = model.test_tagged_corpus(test_data)

    ca41_results = model.test_tagged_corpus(ca41_data)

    results = {'Total word count in test corpora': test_results['total_word_count'],
               'Correct predictions count in test corpora': test_results['correct_predictions_count'],
               'Total word count in ca41 corpus': ca41_results['total_word_count'],
               'Correct predictions count in ca41 corpus': ca41_results['correct_predictions_count'],
               'ca41 results': ca41_results['viterbi_result']}

    write_pairs_to_file(results, output_files_path, 'Results.txt')

    prediction_result = model.predict(untagged_data)
    print(prediction_result)

    return


if __name__ == '__main__':
    main()
