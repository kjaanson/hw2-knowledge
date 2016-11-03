import rdflib
import spacy
import wikipedia
from nltk import Tree, edit_distance
from spacy.attrs import LEMMA

def get_wiki(text):
    try:
        page = wikipedia.page(text)
        title = page.title
        distance = edit_distance(text, title)
        url = page.url
        return (url, distance/max(len(text), len(title)))
    except wikipedia.exceptions.PageError:
        return ('', 1.0)


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
    '''
    FROM: http://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
    '''
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_ + ' ' + node.tag_ + ' ' + node.dep_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_ + ' ' + node.tag_ + ' ' + node.dep_


def to_branches(node):
    branches = []
    to_branches_(node, branches)
    return branches


def to_branches_(node, branches, branch=[]):
    branch = branch[:]
    branch.append(node)
    if node.n_lefts + node.n_rights > 0:
        for child in node.children:
            to_branches_(child, branches, branch)
    else:
        branches.append(branch)


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
    for ent in parsed.ents:
        ent.merge()
    # create RDF from dependency tree
    # separate the parse tree into branches
    triples = []
    for sent in parsed.sents:
        for word in sent:
            if word.dep_ == 'nsubj':
                subject = word
        branches = to_branches(sent.root)
        for branch in branches:
            # for each branch, start from the bottom of the tree and find noun, verb, noun triples
            # the higher verb is the subject and the lower is the object
            branch.reverse()
            triple = [None, None, None]
            for token in branch:
                if 'NNP' in token.tag_:
                    if triple[2] is None:
                        triple[2] = token.text
                    else:
                        triple[0] = token.text
                elif 'VBG' in token.tag_:
                    if triple[1] is None:
                        triple[1] = token.text
            if triple[0] is None:
                # if a branch is missing a noun, use the nsubj of the sentence as the other verb
                triple[0] = subject
            if (triple[0] is not None) and (triple[1] is not None) and (triple[2] is not None):
                triples.append(triple)

    for triple in triples:
        rdf.add((rdflib.Literal(triple[0]), rdflib.Literal(triple[1]), rdflib.Literal(triple[2])))

    # [to_nltk_tree(sent.root).pretty_print() for sent in parsed.sents]
    # print(parsed.ents)
    # for word in parsed:
    #    print(word.text, word.tag_, word.ent_type_, word.ent_iob)

    # example RDF for sentence Bombardier CRJ700 belonging to Adria Airways is flying to Lisbon Portela Airport.
    # https://en.wikipedia.org/wiki/Bombardier_CRJ700_series, http://conceptnet5.media.mit.edu/web/c/en/owner, https://www.adria.si/en/
    # https://en.wikipedia.org/wiki/Bombardier_CRJ700_series, http://conceptnet5.media.mit.edu/web/c/en/fly, https://en.wikipedia.org/wiki/Lisbon_Airport
    # rdf.add((rdflib.URIRef('https://en.wikipedia.org/wiki/Bombardier_CRJ700_series'),
    #          rdflib.URIRef('http://conceptnet5.media.mit.edu/web/c/en/owner'),
    #          rdflib.URIRef('https://www.adria.si/en/')))
    # rdf.add((rdflib.URIRef('https://en.wikipedia.org/wiki/Bombardier_CRJ700_series'),
    #          rdflib.URIRef('http://conceptnet5.media.mit.edu/web/c/en/fly'),
    #          rdflib.URIRef('https://en.wikipedia.org/wiki/Lisbon_Airport')))
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
