import pytest

from dfl.parallel import boundaries, char_classes, jieba_coarse, jieba_tagger, parallelism

A = [("梅姐", "nr"), ("灯下", "s"), ("排", "v"), ("座次", "n")]
B = [("何蔓", "nr"), ("水边", "s"), ("打", "v"), ("低分", "n")]
B_MISCUT = [("何蔓水", "nr"), ("边打", "v"), ("低分", "n")]

LINES = {"梅姐灯下排座次": A, "何蔓水边打低分": B}


def tagger(line):
    return LINES[line]


@pytest.mark.parametrize("tag,coarse", [
    ("n", "N"), ("nr", "N"), ("ns", "N"), ("nz", "N"), ("t", "N"), ("s", "N"),
    ("r", "R"), ("rr", "R"),
    ("v", "V"), ("vn", "V"), ("vd", "V"),
    ("a", "A"), ("ad", "A"), ("z", "A"), ("b", "A"),
    ("d", "D"), ("m", "M"), ("q", "Q"), ("f", "F"),
    ("p", "X"), ("c", "X"), ("uj", "X"), ("ul", "X"), ("y", "X"),
    ("x", "O"), ("eng", "O"), ("i", "O"), ("", "O"),
])
def test_jieba_coarse(tag, coarse):
    assert jieba_coarse(tag) == coarse


def test_char_classes_spread_each_word_class_over_its_characters():
    assert char_classes(A) == ["N", "N", "N", "N", "V", "N", "N"]


def test_boundaries_exclude_the_end_of_the_line():
    assert boundaries(A) == {2, 4, 5}
    assert boundaries([("梅姐灯下排座次", "n")]) == set()


def test_fully_parallel_pair():
    r = parallelism("梅姐灯下排座次", "何蔓水边打低分", tagger=tagger)
    assert r == {"pos_match": 1.0, "boundary_match": 1.0}


def test_different_cut_lowers_both_scores():
    def miscut(line):
        return B_MISCUT if line.startswith("何蔓") else A

    r = parallelism("梅姐灯下排座次", "何蔓水边打低分", tagger=miscut)
    # classes: N N N N V N N against N N N V V N N -> 6 of 7 agree
    assert r["pos_match"] == pytest.approx(6 / 7)
    # cuts {2, 4, 5} against {3, 5}: one shared out of four
    assert r["boundary_match"] == pytest.approx(1 / 4)


def test_unequal_or_empty_lines_have_no_score():
    assert parallelism("梅姐灯下排座次", "何蔓水边打低", tagger=tagger) == {"pos_match": None, "boundary_match": None}
    assert parallelism("", "", tagger=tagger) == {"pos_match": None, "boundary_match": None}


def test_tagger_that_loses_characters_gives_no_score():
    r = parallelism("梅姐灯下排座次", "何蔓水边打低分", tagger=lambda line: [("梅姐", "nr")])
    assert r == {"pos_match": None, "boundary_match": None}


def test_single_word_lines_agree_on_boundaries():
    one = lambda line: [(line, "n")]
    assert parallelism("甲乙丙", "丁戊己", tagger=one) == {"pos_match": 1.0, "boundary_match": 1.0}


# ---------------------------------------------------------------- real jieba

def test_jieba_tagger_keeps_every_character():
    line = "王鹏借壳给老邓洗澡"
    tagged = jieba_tagger(line)
    assert "".join(w for w, _ in tagged) == line
    assert all(isinstance(t, str) and t for _, t in tagged)


def test_a_line_is_perfectly_parallel_with_itself_under_jieba():
    line = "梅姐灯下排座次"
    assert parallelism(line, line) == {"pos_match": 1.0, "boundary_match": 1.0}


def test_jieba_is_deterministic():
    line = "外婆借身炸米饺"
    assert jieba_tagger(line) == jieba_tagger(line)
