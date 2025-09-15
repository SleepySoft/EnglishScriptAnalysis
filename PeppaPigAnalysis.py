from CommonProcess import *
from EnglishAnalysisTools import count_word_frequency, analyze_collocations


if __name__ == "__main__":
    directory = 'PeppaPig'

    # If pure_text.txt has been generated, we don't have to parse docx again.
    # common_process_eng_docs_to_pure_text(directory)

    full_text = load_pure_text(directory)

    sentences, frequency = count_word_frequency(full_text)

    save_sentences_and_word_frequency(sentences, frequency, directory)

    # ------------------------------------------------------------------------------------------------------------------

    collocations = analyze_collocations(full_text)

    dump_collocations(collocations)
    save_collocations(collocations, directory)
