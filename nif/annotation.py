import re
import uuid

import rdflib

nif_ns = rdflib.namespace.ClosedNamespace(
    uri='http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#',
    terms=[
        # Classes
        "Phrase", "NormalizedContextOccurrence", "Structure",
        "CollectionOccurrence", "NormalizedCollectionOccurrence",
        "CStringInst", "OffsetBasedString", "Word", "Annotation", "Title",
        "ContextCollection", "URIScheme", "ContextOccurrence",
        "ContextHashBasedString", "RFC5147String", "Paragraph", "Sentence",
        "String", "CString", "Context",

        # ObjectProperties
        "superStringTrans", "lang", "previousWordTrans", "broaderContext",
        "opinion", "firstWord", "inter", "previousWord",
        "previousSentenceTrans", "sentence", "word", "hasContext", "sourceUrl",
        "referenceContext", "previousSentence", "nextSentenceTrans",
        "wasConvertedFrom", "lastWord", "oliaLink", "dependency",
        "nextSentence", "subString", "predLang", "subStringTrans",
        "narrowerContext", "dependencyTrans", "superString", "annotation",
        "nextWord", "nextWordTrans", "oliaProv",

        # Datatype Properties
        "anchorOf", "stem", "after", "dependencyRelationType", "oliaConf",
        "head", "isString", "lemma", "literalAnnotation", "confidence",
        "keyword", "topic", "endIndex", "before", "posTag", "beginIndex",
        "sentimentValue",

	# new datatypes
	"summary",
    ]
)
itsrdf_ns = rdflib.Namespace(
    'https://www.w3.org/2005/11/its/rdf-content/its-rdf.html#')


def do_suffix_offset(uri, begin_index, end_index):
    uri_str = uri.toPython() if hasattr(uri, 'toPython') else str(uri)
    uri_str = uri_str.rstrip('/')
    chars_indicator = '#offset_'
    if chars_indicator in uri_str:
        splitted = uri_str.split(chars_indicator)
        splitted[-1] = '{}_{}'.format(begin_index, end_index)
        out = chars_indicator.join(splitted)
    else:
        out = uri_str + chars_indicator + '{}_{}'.format(begin_index,
                                                         end_index)
    return rdflib.URIRef(out)


def _parse_attr_name(name):
    prefix, suffix = name.split('__')
    if prefix.lower() == 'nif':
        ns = nif_ns
    elif prefix.lower() == 'itsrdf':
        ns = itsrdf_ns
    elif prefix.lower() == 'rdf':
        ns = rdflib.RDF
    elif prefix.lower() == 'rdfs':
        ns = rdflib.RDFS
    elif prefix.lower() == 'owl':
        ns = rdflib.OWL
    else:
        raise AttributeError(
            'rdf attributes should start either from "nif__" or'
            'from "itsrdf__", you started "{}__"'.format(prefix))
    splitted = suffix.split('_')
    for i in range(len(splitted)):
        if i > 0:
            x = splitted[i]
            x = x[0].upper() + x[1:]
            splitted[i] = x
    rdf_suffix = ''.join(splitted)
    predicate = ns[rdf_suffix]
    return predicate


class NIFAnnotation(rdflib.Graph):
    nif_classes = [nif_ns.String, nif_ns.OffsetBasedString]

    def __init__(self,
                 begin_end_index, is_string=None,
                 ta_ident_ref=None, reference_context=None,
                 uri_prefix=None, anchor_of=None,
                 **kwargs):
        """
        A class to store NIF annotation block for a single entity. Hence,
        a single subject URI is used.

        :param begin_end_index: tuple (begin_index, end_index). If `None` then
            begin_index = 0. see http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core/nif-core.html#d4e436
        :param is_string: see http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core/nif-core.html#d4e669
        :param ta_ident_ref: URI of the extracted entity.
        :param reference_context: see http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core/nif-core.html#d4e1047
        :param uri_prefix: the prefix of the resulting URI. For example,
            `http://example.doc` would produce a URI of the form
            `http://example.doc#char=0,100`.
            :note: Only used if reference context is not given.
        :param anchor_of: see http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core/nif-core.html#d4e395
        :param **kwargs: any additional (predicate, object) pairs
        """
        super().__init__()
        try:
            begin_end_index = tuple(map(int, begin_end_index))
        except ValueError:
            raise ValueError(
                'begin_end_index should be convertible to integers, '
                '{} provided'.format(begin_end_index))
        self.reference_context = None  # this holds a separate graph
        if reference_context is not None:  # this is a not a context
            uri_prefix = reference_context.uri_prefix
            self.reference_context = reference_context
        self.uri = do_suffix_offset(uri_prefix, *begin_end_index)
        # URI obtained, set the predicate, object pairs
        self.__setattr__('nif__begin_index', begin_end_index[0], validate=False)
        self.__setattr__('nif__end_index', begin_end_index[1], validate=False)
        if reference_context is not None:
            self.__setattr__('nif__reference_context', reference_context.uri,
                             False)
        if is_string is not None:
            self.__setattr__('nif__is_string', is_string, False)
        if ta_ident_ref is not None:
            self.__setattr__('itsrdf__ta_ident_ref', ta_ident_ref, False)
        if anchor_of is not None:
            self.__setattr__('nif__anchor_of', anchor_of, False)
        self.add_nif_classes()
        self.validate()
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    def validate(self):
        if self.reference_context is not None:
            if not NIFContext.is_context(self.reference_context):
                raise ValueError(
                    'The provided reference context is not compatible with '
                    'nif.Context class.')
        if self.nif__is_string is not None:
            if not isinstance(self.nif__is_string, str):
                raise TypeError('is_string value {} should be '
                                'a string'.format(self.nif__is_string))
            if int(self.nif__begin_index) != 0 or \
                    int(self.nif__end_index) != len(self.nif__is_string):
                raise ValueError(
                    'Begin and end indices are provided ({}), '
                    'but do not fit the provided string (length = {})'
                    '.'.format((self.nif__begin_index, self.nif__end_index),
                               len(self.nif__is_string)))
            if self.nif__anchor_of is not None or \
                    self.nif__reference_context is not None or \
                    self.itsrdf__ta_ident_ref is not None:
                raise ValueError(
                    'If is_string is provided then '
                    'ta_ident_ref, reference_context and anchor are not allowed'
                    '. You have reference context = {}, anchor = {}, '
                    'ta_ident_ref = {}.'.format(self.nif__reference_context,
                                                self.nif__anchor_of,
                                                self.itsrdf__ta_ident_ref))
        if self.itsrdf__ta_ident_ref is not None:
            if self.nif__is_string is not None or \
                    self.nif__reference_context is None or \
                    self.nif__anchor_of is None:
                raise ValueError(
                    'If identifier ta_iden_ref is provided then '
                    'reference context and anchor are required and'
                    'is_string is not allowed. You have'
                    'reference context = {}, anchor = {}, is_string = {}'
                    '.'.format(self.nif__reference_context,
                               self.nif__anchor_of, self.nif__is_string))
        if self.nif__reference_context is not None:
            if self.nif__anchor_of is None:
                raise ValueError('When reference context is provided, '
                                 'anchor_of is required.')

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattribute__(name)
        elif '__' in name:
            predicate = _parse_attr_name(name)
            rs = list(self.objects(subject=self.uri, predicate=predicate))
            if len(rs) == 1:
                return rs.pop()
            elif len(rs) == 0:
                return None
            else:
                return rs
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name, value, validate=True):
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif '__' in name:
            predicate = _parse_attr_name(name)
            if not isinstance(value, (rdflib.URIRef, rdflib.Literal,
                                      rdflib.BNode, rdflib.Variable)):
                value = rdflib.Literal(value)
            if isinstance(value, (list, tuple)):
                self.remove((self.uri, predicate, None))
                for val_item in value:
                    self.add((self.uri, predicate, val_item))
            else:
                #print(self.uri + "   " + predicate + "    " + value)
                self.set((self.uri, predicate, value))
            if validate:
                self.validate()
        else:
            super().__setattr__(name, value)

    def add_nif_classes(self):
        for cls in self.nif_classes:
            self.add((self.uri, rdflib.RDF.type, cls))
        return self

    @classmethod
    def from_triples(cls, rdf_graph, ref_cxt):
        raise NotImplementedError

    # @classmethod
    # def from_triples(cls, rdf):
    #     kwargs = dict()
    #     other_triples = rdflib.Graph()
    #     for s, p, o in rdf:
    #         if p == nif_ns.beginIndex:
    #             kwargs['begin_index'] = int(o.toPython())
    #         elif p == nif_ns.endIndex:
    #             kwargs['end_index'] = int(o.toPython())
    #         elif p == nif_ns.referenceContext:
    #             ref_cxt_uri = o
    #         elif p == nif_ns.isString:
    #             kwargs['is_string'] = o.toPython()
    #         elif p == nif_ns.anchorOf:
    #             kwargs['anchor_of'] = o.toPython()
    #         elif p == itsrdf_ns.taIdentRef:
    #             kwargs['ta_ident_ref'] = o
    #         else:
    #             other_triples.add((s, p, o))
    #     uri_prefix = s.toPython()
    #     kwargs['begin_end_index'] = kwargs['begin_index'], kwargs['end_index']
    #     del kwargs['begin_index']
    #     del kwargs['end_index']
    #     if cls == NIFContext:
    #         del kwargs['begin_end_index']
    #     out = cls(uri_prefix=uri_prefix, **kwargs)
    #     try:
    #         out.__setattr__('nif__reference_context', ref_cxt_uri, False)
    #     except UnboundLocalError:
    #         pass
    #     out += other_triples
    #     return out


class NIFContext(NIFAnnotation):
    nif_classes = [nif_ns.Context, nif_ns.OffsetBasedString]

    def __init__(self, is_string, uri_prefix, **kwargs):
        begin_end_index = (0, len(is_string))
        super().__init__(
            begin_end_index=begin_end_index,
            is_string=is_string,
            uri_prefix=uri_prefix,anchor_of=None,**kwargs)
        self.uri_prefix = uri_prefix

    @classmethod
    def get_is_string(self):
        return NIFAnnotation.is_string

    @classmethod
    def get_uri_prefix(self):
        return NIFAnnotation.uri_prefix


    @staticmethod
    def is_context(cxt):
        return isinstance(cxt, NIFContext)

    @classmethod
    def from_triples(cls, rdf_graph, ref_cxt=None):
        kwargs = dict()
        other_triples = rdflib.Graph()
        for s, p, o in rdf_graph:
            if p == nif_ns.isString:
                if 'is_string' in kwargs:
                    raise ValueError(f'{p} found twice.')
                kwargs['is_string'] = o.toPython()
            elif p == nif_ns.beginIndex:
                begin_index = int(o.toPython())
            elif p == nif_ns.endIndex:
                end_index = int(o.toPython())
            else:
                other_triples.add((s, p, o))
        uri_prefix = s.toPython()

        out = cls(uri_prefix=uri_prefix, **kwargs)
        if (int(out.nif__begin_index) != begin_index or
                int(out.nif__end_index) != end_index):
            raise ValueError(f'Check the provided begin and end indices!')
        out += other_triples
        return out

    #@classmethod
    def set_atts(self, **kwargs):
        for key, value in kwargs.items():
            print(key + "    " + value)
            self.__setattr__(key, value, True)


class NIFStructure(NIFAnnotation):
    nif_classes = [nif_ns.Structure, nif_ns.OffsetBasedString]

    def __init__(self, reference_context, begin_end_index, anchor_of, **kwargs):
        super().__init__(
            reference_context=reference_context,
            begin_end_index=begin_end_index, anchor_of=anchor_of,
            **kwargs)

    def validate(self):
        super().validate()
        if self.nif__anchor_of is not None:
            ref_substring = self.reference_context.nif__is_string[
                            int(self.nif__begin_index):int(self.nif__end_index)]
            # Extractor returns different capitalization in matches! 
            if self.nif__anchor_of.toPython().lower() != ref_substring.lower():
                raise ValueError(
                    'Anchor should be equal exactly to the subtring of '
                    'the reference context. You have anchor = "{}", '
                    'substring in ref context = "{}"'.format(
                        self.nif__anchor_of, ref_substring))

    @staticmethod
    def is_structure(struct):
        return isinstance(struct, NIFStructure)

    @classmethod
    def from_triples(cls, rdf_graph, ref_cxt):
        kwargs = dict()
        other_triples = rdflib.Graph()
        for s, p, o in rdf_graph:
            if p == nif_ns.beginIndex:
                kwargs['begin_index'] = int(o.toPython())
            elif p == nif_ns.endIndex:
                kwargs['end_index'] = int(o.toPython())
            elif p == nif_ns.referenceContext:
                ref_cxt_uriref = o
                assert ref_cxt_uriref == ref_cxt.uri, \
                    (ref_cxt_uriref, ref_cxt.uri)
            elif p == nif_ns.anchorOf:
                kwargs['anchor_of'] = o.toPython()
            elif p == itsrdf_ns.taIdentRef:
                kwargs['ta_ident_ref'] = o
            else:
                other_triples.add((s, p, o))
        uri_prefix = s.toPython()
        kwargs['begin_end_index'] = kwargs['begin_index'], kwargs['end_index']
        del kwargs['begin_index']
        del kwargs['end_index']
        out = cls(reference_context=ref_cxt, uri_prefix=uri_prefix, **kwargs)
        out += other_triples
        return out


class NIFPhrase(NIFStructure):
    nif_classes = [nif_ns.Phrase, nif_ns.OffsetBasedString]

    def __init__(self, reference_context, begin_end_index, anchor_of, **kwargs):
        super().__init__(reference_context, begin_end_index, anchor_of,
                         **kwargs)


class NIFExtractedEntity(NIFPhrase):
    def __init__(self, reference_context, begin_end_index, anchor_of,
                 entity_uri, **kwargs):
        super().__init__(
            reference_context=reference_context,
            begin_end_index=begin_end_index, anchor_of=anchor_of,
            ta_ident_ref=rdflib.URIRef(entity_uri), **kwargs)


class NIFDocument:
    def __init__(self, context, structures):
        if not NIFContext.is_context(context):
            raise TypeError('The provided context {} is not a NIFContext'
                            '.'.format(context))
        self.rdf = rdflib.Graph()
        self.context = context
        self.uri_prefix = context.uri_prefix
        self.structures = []
        if structures:
            for struct in structures:
                self.add_structure(struct)
        self.rdf += self.context
        self.validate()

    def validate(self):
        for struct in self.structures:
            if not NIFStructure.is_structure(struct):
                raise TypeError('The provided structure {} is not a '
                                'NIFStructure.'.format(struct))
            if struct.nif__reference_context != self.context.uri:
                raise ValueError('The reference context {} for the structure {}'
                                 ' is different from the context {} of the '
                                 'document.'.format(
                                     struct.nif__reference_context,
                                     struct.uri, self.context.uri))

    @classmethod
    def from_text(cls, text, uri="http://example.doc/" + str(uuid.uuid4())):
        cxt = NIFContext(is_string=text, uri_prefix=uri)
        return cls(context=cxt, structures=[])

    def add_structure(self, struct):
        self.structures.append(struct)
        try:
            self.validate()
        except (ValueError, TypeError) as e:
            self.structures.pop()
            raise e
        else:
            self.rdf += struct
        return self

    def add_phrase(self, phrase):
        self.add_structure(phrase)

    def add_extracted_entity(self, ee):
        self.add_phrase(ee)

    def add_extracted_cpt(self, cpt_dict):
        """
        :param cpt_dict: expected to have 'uri',
            'matchings'-> [{'text': value,
                            'positions': [(begin, end), ...]},
                           ...]
        :return: self
        """
        cpt_uri = cpt_dict['uri']
        for matches in cpt_dict['matchings']:
            surface_form = matches['text']
            for match in matches['positions']:
                ee = NIFExtractedEntity(
                    reference_context=self.context,
                    begin_end_index=(match[0], match[1]),
                    anchor_of=surface_form,
                    entity_uri=cpt_uri
                )
                self.add_extracted_entity(ee)
        return self

    def serialize(self, format="xml", uri_format=nif_ns.OffsetBasedString):
        if uri_format not in [nif_ns.OffsetBasedString, nif_ns.RFC5147String]:
            raise ValueError("Only RFC5147 and OffsetBased strings are "
                             "currently soported URI schemes")
        rdf_text = self.rdf.serialize(format=format)

        if uri_format == nif_ns.RFC5147String:
            RFC5147_str = br"#char\=\(\1,\2\)"
            offset_regex = br"#offset_(\d*)_(\d*)"
            rdf_text = re.sub(offset_regex, RFC5147_str, rdf_text)
        return rdf_text

    @classmethod
    def parse_rdf(cls, rdf_text, format="n3"):
        rdf_graph = rdflib.Graph()
        rdf_graph.parse(data=rdf_text, format=format)

        context_uri = rdf_graph.value(predicate=rdflib.RDF.type,
                                      object=nif_ns.Context)
        context_triples = rdf_graph.triples((context_uri, None, None))
        context = NIFContext.from_triples(context_triples)

        structures = []
        struct_uris = list(rdf_graph[:nif_ns.referenceContext:context.uri])
        for struct_uri in struct_uris:
            struct_triples = rdf_graph.triples((struct_uri, None, None))
            struct = NIFStructure.from_triples(struct_triples, ref_cxt=context)
            structures.append(struct)
        out = cls(context=context, structures=structures)

        return out

    # @staticmethod
    # def extract_context(rdf_graph):
    #     for context_uri in rdf_graph[:rdflib.RDF.type:nif_ns.Context]:
    #         base_uri = str(context_uri).split("#")[0]
    #         for text in rdf_graph[context_uri:nif_ns.isString:]:
    #             ctx = NIFContext(is_string=text, uri_prefix=base_uri)
    #             return ctx
    #     raise ValueError("Nif file contains no context")
    #
    # @staticmethod
    # def extract_phrases(rdf_graph, context):
    #     # TODO: refactor
    #     structures = []
    #     for phrase_uri in rdf_graph[:rdflib.RDF.type:nif_ns.Phrase]:
    #         phrase_rdf = extract_subgraph(rdf_graph, phrase_uri,
    #                                       whole_graph=rdflib.Graph())
    #         begin_index = [c for c in phrase_rdf[phrase_uri:nif_ns.beginIndex:]][0]
    #         end_index = [c for c in phrase_rdf[phrase_uri:nif_ns.endIndex:]][0]
    #         begin_end_index = (begin_index, end_index)
    #         anchor_of = None
    #         try:
    #             anchor_of = [c for c in phrase_rdf[phrase_uri:nif_ns.anchorOf:]][0]
    #         except:
    #             pass
    #
    #         # This library uses RFC5147 URI's, maybe the file had other URIs
    #         new_uri = do_suffix_offset(context.uri_prefix,
    #                                    begin_index, end_index)
    #         uris_are_RFC5147 = new_uri == phrase_uri
    #         if not uris_are_RFC5147:
    #             new_phrase_rdf = rdflib.Graph()
    #             for s, p, o in phrase_rdf[::]:
    #                 if s == phrase_uri:
    #                     s = do_suffix_offset(new_uri, begin_index, end_index)
    #                 new_phrase_rdf.add((s,p,o))
    #             phrase_rdf=new_phrase_rdf
    #
    #         # We remove all reference in phrase_rdf to the original reference
    #         # context, as these might conflict with
    #         # those created with the NIFStructure contructor
    #         for s,o in phrase_rdf[:nif_ns.referenceContext:]:
    #             phrase_rdf.remove((s,nif_ns.referenceContext,o))
    #
    #         structure = NIFStructure(context, begin_end_index, anchor_of)
    #         structure += phrase_rdf
    #         structures.append(structure)
    #     return structures

# def extract_subgraph(rdf, uri_starting_node, whole_graph,
#                      ignore_predicates=(nif_ns.referenceContext,)):
#     if type(uri_starting_node) != rdflib.URIRef:
#         uri_starting_node = rdflib.URIRef(uri_starting_node)
#     these_so = rdf.predicate_objects(uri_starting_node)
#     current_size = len(whole_graph)
#     objects = []
#     for p, o in these_so:
#         whole_graph.add((uri_starting_node, p, o))
#         new_size = len(whole_graph)
#         if new_size > current_size:
#             current_size = new_size
#             if type(o) == rdflib.URIRef and p not in ignore_predicates:
#                 objects.append(o)
#     for o in objects:
#         whole_graph = extract_subgraph(rdf, o, whole_graph=whole_graph)
#     return whole_graph

if __name__ == '__main__':
    rdf_to_parse = '''
@prefix dbo:   <http://dbpedia.org/ontology/> .
@prefix geo:   <http://www.w3.org/2003/01/geo/wgs84_pos/> .
@prefix dktnif: <http://dkt.dfki.de/ontologies/nif#> .
@prefix nif-ann: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-annotation#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix itsrdf: <http://www.w3.org/2005/11/its/rdf#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix nif:   <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#> .
<http://dkt.dfki.de/documents#offset_11_17>
        a                     nif:RFC5147String , nif:String , nif:Structure ;
        nif:anchorOf          "Berlin" ;
        nif:beginIndex        "11"^^xsd:nonNegativeInteger ;
        nif:endIndex          "17"^^xsd:nonNegativeInteger ;
        nif:referenceContext  <http://dkt.dfki.de/documents#offset_0_26> ;
        itsrdf:taClassRef     dbo:Location ;
        itsrdf:taIdentRef     <http://www.wikidata.org/entity/Q64> ;
        itsrdf:taIdentRef     []  .
<http://dkt.dfki.de/documents#offset_0_26>
        a               nif:RFC5147String , nif:String , nif:Context ;
        nif:beginIndex  "0"^^xsd:nonNegativeInteger ;
        nif:endIndex    "26"^^xsd:nonNegativeInteger ;
        nif:isString    "Welcome to Berlin in 2016." .
'''
    print(rdf_to_parse)
    d = NIFDocument.parse_rdf(rdf_to_parse, format='turtle')
    print(d.serialize(format="ttl"))
    text = d.context.nif__is_string
    print(text)
    summary = "This is the summary of the text."

    kwargs = {"nif__summary" : summary} 

    d.context.set_atts(**kwargs)


    nc = NIFContext(d.context.nif__is_string,d.context.uri_prefix,**kwargs)
    d2 = NIFDocument(nc, structures=d.structures)
    print(d2.serialize(format="turtle"))
    print(nc.nif__summary)
