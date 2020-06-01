from tqdm import tqdm
from pipetools import pipe, where

from lib.clean_string import (
    alphanumeric_period_only,
    dirty_remove_speaker_name,
    do_nothing,
    lower_string,
    regulate_punctuation,
    remove_paren,
)

#
# Preprocessing Tools
#


def parse_text_to_word_sentences(
    input_text, no_paren=True, no_speaker=True, sentence_min_length=3
):

    """a text parser for TED text data

    We make tokens/symbols/words to represent concept pointing to reality. Concepts
    have a natural inter-relationship determined by rules of reality. Those inter-relationship
    can be of a parent/children or peer form, and can include status, properties, actions,
    changes, etc. We construct phrases such as (adj. + n.) to raise specification of concept
    with properties, as well sentence to represent those inter-relationsihp.

    ([entity]-<relationship/interaction>-[entity]) -> ([concept]-<relationship>-[concept])
    -> (context: [token]-<structure>-[token])

    reality space - concept space - context space

    This partially indicates that our language (word + sentences = token + structure)
    with the correct effort can model those inter-relationsihp between concepts to
    reflect reality to an extent. Thus, co-occured tokens in those structures(sentences)
    with certain level of truthfulness can denote the properties of concept represented
    by those tokens.

    There can be various different dimensions of information stacked together in one
    text clip on top of the simple concept relationship:
    0) Conceptual Relationship - reflection of reality
    1) Errors - It can also happens that the truthfulness of text can be doubtful
    resulting from intentional or unintentional mistakes.
    2) Formatting/Perception structure Information - Capital letters, line breaks
    , puctuations. Paragraphs can be usedd for a particular meaning structure purpose.
    3) Supplement Information - Using parentheses to further specificy pointer of
    a phrase/word.
    4) Medium/Speaker/Emotional Information - There can be content representing
    information in an unrelated often in the super-content/medium space, e.g. text
    annotation of audience reaction to a speech.

    For a simple model, it can be difficult for it to distinguish information from
    different information space initially. Therefore, it can be helpful to preprocess
    the text to include data only from a particular information space.

    Embeddings are one way to represent the concept relationship in a particular information
    space by relative spatial relationship of tokens in dense vectors. Word2Vec is a
    model to structure/learn the positions of tokens in the space via contextual
    co-occurances, e.g. contextual co-occurance relationship is shaped into a
    spatial distances, thus words with similar distances to a group of contextual
    co-occured words can be positioned closely, which is interpretted as similar
    words by the model.

    We may want to reduce noise information presented in the text, such as annotations
    of reaction, e.g. (Applaud), or speaker name, as they are not natural contextual
    co-occurance denoting concept relationsihp in the same space.

    Arguments:
      input_text {string} -- the raw string of text data

    Keyword Arguments:
      no_paren {bool} -- option to remove content with parentheses (default: {True})
      no_speaker {bool} -- option to remove speaker name text (default: {True})

    Returns:
      [list] -- tokenised sentences, text cut into sentences represented by
      list of words, [[word, word], [word, word]]
    """

    text_cleaner = (
        pipe
        | lower_string
        | regulate_punctuation
        | (remove_paren if no_paren else do_nothing)
        | (dirty_remove_speaker_name if no_speaker else do_nothing)
        | alphanumeric_period_only
    )

    strip_word_in_sentence = (
        pipe
        | (map, lambda word: word.strip())
        | where(lambda stripped: len(stripped) > 0)
        | list
    )

    word_sentences = [
        strip_word_in_sentence(sentence.split(" "))
        for paragraph in input_text.split("\n")
        for sentence in text_cleaner(paragraph).split(".")
    ]

    del input_text

    return [
        word_sentence
        for word_sentence in word_sentences
        if len(word_sentence) > sentence_min_length
    ]


def flat_clip_in_words(clip):
    return [
        "_BOC_",
        *[
            word
            for sentence in parse_text_to_word_sentences(clip)
            for word in ["_BOS_", *sentence, "_EOS_"]
        ],
        "_EOC_",
    ]


def build_encoder_decoder_from_vocab(vocab, has_unknown=False):
    vocab.discard("_PAD_")
    vocab.discard("_BOS_")
    vocab.discard("_EOS_")
    vocab.discard("_BOC_")
    vocab.discard("_EOC_")

    vocab_len = len(vocab)
    encoder = {
        "_PAD_": 0,
        "_BOS_": vocab_len + 1,
        "_EOS_": vocab_len + 2,
        "_BOC_": vocab_len + 3,
        "_EOC_": vocab_len + 4,
    }
    decoder = {
        0: "*",
        vocab_len + 1: "",
        vocab_len + 2: ".",
        vocab_len + 3: "",
        vocab_len + 4: "",
    }

    for index, word in enumerate(vocab):
        # make the vocabulary index [1, len(vocab)]
        encoder[word] = index + 1
        decoder[index + 1] = word

    if has_unknown:
        encoder["_UNK_"] = vocab_len + 5
        decoder[vocab_len + 5] = "?"

    return encoder, decoder


#
# stats tool
#


def build_word_frequency_map(all_words, output_sorted=True):
    frequency_map = {}

    for word in tqdm(all_words):
        if word in frequency_map.keys():
            frequency_map[word] += 1
        else:
            frequency_map[word] = 1

    return frequency_map


def get_top_frequent_words(all_words, many=1000, include_frequency=False):
    frequency_map = build_word_frequency_map(all_words)
    sorted_frequency_map = sorted(
        frequency_map.items(), reverse=True, key=lambda x: x[1]
    )

    top_frequent_words = list(map(lambda x: x[0], sorted_frequency_map[:many]))

    if include_frequency:
        top_frequency = list(map(lambda x: x[1], sorted_frequency_map[:many]))
        return top_frequent_words, top_frequency

    return top_frequent_words
