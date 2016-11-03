import rdflib
import spacy
from nltk import Tree
from spacy.attrs import LEMMA


def merge_phrases(matcher, doc, i, matches):
    '''
    FROM: https://www.bountysource.com/issues/38301771-subject-object-extraction-within-spacy
    '''
    if i != len(matches) - 1:
        return None
    # Get Span objects
    spans = [(ent_id, label, doc[start: end]) for ent_id, label, start, end in matches]
    for ent_id, label, span in spans:
        span.merge(label=label, tag='NNP' if label else span.root.tag_)

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_ + ' ' + node.tag_ + ' ' + node.dep_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_ + ' ' + node.tag_ + ' ' + node.dep_


def add_matchers(spc, datafile=None):
    matcher = spacy.matcher.Matcher(spc.vocab)
    entity_id = 0
    with open(datafile, "r", encoding="utf-8") as f:
        models = f.readlines()
        for model in models:
            model = model.rstrip(' \r\n').lower()
            entity_id_str = "Aircraft" + str(entity_id)
            entity_id += 1
            matcher.add_entity(
                entity_id_str,
                {"ent_type": "PRODUCT"},
                acceptor=None,
                on_match=merge_phrases
            )
            model_tokens = model.split(sep=' ')
            tokens = []
            for tk in model_tokens:
                tokens.append({LEMMA: tk})
            matcher.add_pattern(
                entity_id_str,
                tokens,
                label=None
            )
    return matcher


def parse_sentence_to_rdf(spacy, sentence, matcher):
    """
    Attempts to parse a natural language sentence to RDF
    :param spacy: pass the spacy pipeline by reference
    :param sentence: the sentence to be converted
    :return: rdflib Graph object                                                        
    """
    rdf = rdflib.Graph()
    parsed = spacy(sentence)
    matcher(parsed)

    # create RDF from dependency tree
    # separate the parse tree into branches
    for token in tokens:
        if token.
    # for each branch, start from the bottom of the tree and find noun, verb, noun triples
    # if a branch is missing a noun, use the nsubj of the sentence as the other verb
    # the higher verb is the subject and the lower is the object
    for token in parsed:
        if token.dep == 'nsubj':
            print(token.orth_)

    [to_nltk_tree(sent.root).pretty_print() for sent in parsed.sents]
    print(parsed.ents)
    for word in parsed:
        print(word.text, word.tag_, word.ent_type_, word.ent_iob)

    # example RDF for sentence Bombardier CRJ700 belonging to Adria Airways is flying to Lisbon Portela Airport.
    # https://en.wikipedia.org/wiki/Bombardier_CRJ700_series, http://conceptnet5.media.mit.edu/web/c/en/owner, https://www.adria.si/en/
    # https://en.wikipedia.org/wiki/Bombardier_CRJ700_series, http://conceptnet5.media.mit.edu/web/c/en/fly, https://en.wikipedia.org/wiki/Lisbon_Airport
    rdf.add((rdflib.URIRef('https://en.wikipedia.org/wiki/Bombardier_CRJ700_series'),
             rdflib.URIRef('http://conceptnet5.media.mit.edu/web/c/en/owner'),
             rdflib.URIRef('https://www.adria.si/en/')))
    rdf.add((rdflib.URIRef('https://en.wikipedia.org/wiki/Bombardier_CRJ700_series'),
             rdflib.URIRef('http://conceptnet5.media.mit.edu/web/c/en/fly'),
             rdflib.URIRef('https://en.wikipedia.org/wiki/Lisbon_Airport')))
    return rdf


def main():
    # get the sentences to process. We consider eah sentence out of context, to keep things simple
    text = 'Bombardier CRJ-700 belonging to Adria Airways is flying to Lisbon Portela Airport.'
    # text = 'The competition between Airbus and Boeing has been characterised as a duopoly in the large jet airliner market since the 1990s.[1] This resulted from a series of mergers within the global aerospace industry, with Airbus beginning as a European consortium while the American Boeing absorbed its former arch-rival, McDonnell Douglas, in a 1997 merger. Other manufacturers, such as Lockheed Martin, Convair and Fairchild Aircraft in the United States, and British Aerospace and Fokker in Europe, were no longer in a position to compete effectively and withdrew from this market.'
    # load the spacy english pipeline
    spc = spacy.en.English()
    matcher = add_matchers(spc, datafile='Aircraftmodels20161028.txt')
    print(str(parse_sentence_to_rdf(spc, text, matcher).serialize(format='n3')).replace('\\n', '\n'))


if __name__ == '__main__':
    main()
