import os
from collections import Counter

import nose

from nif.annotation import *


class TestAnnotation:
    def setUp(self):
        self.cxt = NIFContext(
            is_string='some larger context. this is a phrase in this context.',
            uri_prefix="http://some.doc/"+str(uuid.uuid4())
        )

    def test_context_created(self):
        text = 'some string. some other string.'
        ann = NIFAnnotation(
            begin_end_index=(0, len(text)), is_string=text,
            ta_ident_ref=None, reference_context=None,
            uri_prefix="http://example.doc/"+str(uuid.uuid4()),
            anchor_of=None)
        assert len(ann) > 0
        subject_uri = ann.value(predicate=nif_ns.isString,
                                object=rdflib.Literal(text))
        assert subject_uri, ann.serialize(format='n3')
        assert subject_uri == ann.uri

    def test_context_additional_attributes(self):
        text = 'some string. some other string.'
        ann = NIFAnnotation(
            begin_end_index=(0, len(text)), is_string=text,
            ta_ident_ref=None, reference_context=None,
            uri_prefix="http://example.doc/" + str(uuid.uuid4()),
            anchor_of=None, nif__keyword='keyword')
        kw = ann.value(predicate=nif_ns.keyword, subject=ann.uri)
        assert kw, kw
        assert kw.toPython() == 'keyword', kw

    def test_phrase(self):
        text = 'this is a phrase'
        ann = NIFAnnotation(
            begin_end_index=(21, 37), is_string=None,
            ta_ident_ref=None, reference_context=self.cxt,
            uri_prefix=None, anchor_of=text)
        assert len(ann) > 0
        assert ann.uri.startswith(self.cxt.uri.toPython()[:10])


class TestContext:
    def setUp(self):
        pass

    def test_context_created(self):
        cxt = NIFContext(
            is_string='some larger context. this is a phrase in this context.',
            uri_prefix="http://some.doc/"+str(uuid.uuid4())
        )

    def test_from_triples(self):
        triples = rdflib.Graph()
        triples.parse(data="""
@prefix dbo:   <http://dbpedia.org/ontology/> .
@prefix geo:   <http://www.w3.org/2003/01/geo/wgs84_pos/> .
@prefix dktnif: <http://dkt.dfki.de/ontologies/nif#> .
@prefix nif-ann: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-annotation#> .
@prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .
@prefix itsrdf: <http://www.w3.org/2005/11/its/rdf#> .
@prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
@prefix nif:   <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#> .

<http://dkt.dfki.de/documents#offset_0_26>
        a               nif:RFC5147String , nif:String , nif:Context ;
        nif:beginIndex  "0"^^xsd:nonNegativeInteger ;
        nif:endIndex    "26"^^xsd:nonNegativeInteger ;
        nif:isString    "Welcome to Berlin in 2016." .""", format='turtle')
        na = NIFContext.from_triples(triples)
        assert len(na) == 7, (len(na), list(na))


class TestExtractedEntity:
    def setUp(self):
        self.cxt = NIFContext(
            is_string='some larger context. this is a phrase in this context.',
            uri_prefix="http://some.doc/"+str(uuid.uuid4())
        )

    def test_phrase_created(self):
        ex_uri = 'http://example.com/index#some'
        ee = NIFExtractedEntity(
            reference_context=self.cxt,
            begin_end_index=(0, 4),
            anchor_of='some',
            entity_uri=ex_uri
        )
        ta_ident_ref = ee.itsrdf__ta_ident_ref
        assert str(ta_ident_ref) == ex_uri, ta_ident_ref

    def test_phrase_mutations_check(self):
        ex_uri = 'http://example.com/index#some'
        ee = NIFExtractedEntity(
            reference_context=self.cxt,
            begin_end_index=(0, 4),
            anchor_of='some',
            entity_uri=ex_uri
        )
        with nose.tools.assert_raises(ValueError):
            ee.nif__begin_index = 1


class TestDocument:
    def setUp(self):
        self.txt = 'some larger context. this is a phrase in this context.'
        self.uri_prefix = "http://some.doc/" + str(uuid.uuid4())
        self.cxt = NIFContext(
            is_string=self.txt,
            uri_prefix=self.uri_prefix)
        self.cxt2 = NIFContext(
            is_string=self.txt[:-1],
            uri_prefix="http://some.doc/" + str(uuid.uuid4()))
        ex_uri = 'http://example.com/index#some'
        self.ee = NIFExtractedEntity(
            reference_context=self.cxt,
            begin_end_index=(0, 4),
            anchor_of='some',
            entity_uri=ex_uri)
        self.ee2 = NIFExtractedEntity(
            reference_context=self.cxt2,
            begin_end_index=(0, 4),
            anchor_of='some',
            entity_uri=ex_uri)

    def test_create_doc(self):
        d = NIFDocument(context=self.cxt, structures=[])
        assert d.context == self.cxt

    def test_create_with_struct(self):
        d = NIFDocument(context=self.cxt, structures=[self.ee])
        assert d.structures == [self.ee]

    def test_create_from_ttl(self):
        d = NIFDocument(context=self.cxt, structures=[self.ee])
        ttl_text = d.serialize("ttl").decode("utf-8")
        d2 = NIFDocument.parse_rdf(ttl_text)
        parsed_triples = set(d2.rdf)
        assert len(parsed_triples) >= len(d.rdf), (len(parsed_triples),
                                                   len(d.rdf))
        for s,p,o in d.rdf:
            assert (s,p,o) in parsed_triples, (s,p,o)
        assert d.structures == [self.ee]

    def test_not_create_with_wrong_ref_cxt(self):
        with nose.tools.assert_raises(ValueError):
            NIFDocument(context=self.cxt, structures=[self.ee2])

    def test_create_from_text(self):
        d = NIFDocument.from_text(self.txt, self.uri_prefix)
        assert d.context.uri.startswith(self.uri_prefix), (d.uri_prefix,
                                                           self.uri_prefix)
        assert d.uri_prefix == self.uri_prefix
        assert d.context.uri.startswith(self.uri_prefix)

    def test_add_struct(self):
        d = NIFDocument(context=self.cxt, structures=[])
        d.add_extracted_entity(self.ee)

    def test_not_add_invalid_struct(self):
        d = NIFDocument(context=self.cxt, structures=[])
        with nose.tools.assert_raises(ValueError):
            d.add_extracted_entity(self.ee2)

    def test_add_cpt(self):
        cpt = {
            'uri': 'http://some.uri',
            'matchings': [
                {'text': 'larger', 'positions': [(5,11)]},
                {'text': 'this', 'positions': [(21, 25), (41, 45)]}
            ]
        }
        d = NIFDocument(context=self.cxt, structures=[])
        d.add_extracted_cpt(cpt)


class TestSuffix:
    def test_suffix(self):
        uri_str = 'http://dkt.dfki.de/documents/'
        r = do_suffix_offset(uri=uri_str,
                             begin_index=0,
                             end_index=26)
        assert str(r) == 'http://dkt.dfki.de/documents#offset_0_26', r


class TestParsing:
    def setUp(self):
        self.rdf_to_parse = '''
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

    def test_parse_correct(self):
        parsed = NIFDocument.parse_rdf(self.rdf_to_parse, format='turtle')
        serialized = parsed.serialize(format='turtle')
        g_original = rdflib.Graph()
        g_original.parse(data=self.rdf_to_parse, format='turtle')
        g_processed = rdflib.Graph()
        g_processed.parse(data=serialized, format='turtle')

        sp_pairs_original = [(s, p) for s, p, o in g_original]
        sp_orig_counter = Counter(sp_pairs_original)
        sp_pairs_processed = [(s, p) for s, p, o in g_processed]
        sp_processed_counter = Counter(sp_pairs_processed)
        for k, v in (sp_orig_counter - sp_processed_counter).items():
            assert v <= 0, (k,v)
