import pytest

from dfl.checker import (
    FORWARD, REVERSE, _even_opposition, check_generated, check_pair,
    check_reference, reference_match, repeated_characters, same_position_count,
)
from dfl.tones import BOTH, PING, UNKNOWN, ZE, Pingshui

# A synthetic couplet that keeps every rule: 睡 is tone 4, 还 tone 2.
FIRST = "三更舟中悄悄睡"
SECOND = "数日街头缓缓还"

PS = Pingshui({
    "睡": "去", "还": "平", "白": "入", "去": "去", "锁": "上", "庄": "平",
    "温": "平", "饺": "上", "看": "平去",
})


def fake_tagger(line):
    """One character per word, all nouns: keeps jieba out of rule tests."""
    return [(ch, "n") for ch in line]


def run(given, answer, direction, **kw):
    kw.setdefault("pingshui", PS)
    kw.setdefault("tagger", fake_tagger)
    return check_generated(given, answer, direction, **kw)


# ---------------------------------------------------------------- rule 3 helper

def test_repeat_at_a_different_position_is_a_violation():
    assert repeated_characters("半扇门一更上锁", "一席棋两客连庄") == ["一"]
    assert repeated_characters("蒲二隔窗敲木鼓", "老蒲开伞折梅花") == ["蒲"]


def test_repeat_at_the_same_position_is_allowed():
    assert repeated_characters("方生借笔给老师修书", "小童搬箧替老陈剪纸") == []
    assert same_position_count("方生借笔给老师修书", "小童搬箧替老陈剪纸") == 1


def test_same_position_exception_needs_every_occurrence_to_line_up():
    # 一 sits at position 0 in both lines, but the first line has it again at 3.
    assert repeated_characters("一二三一", "一五六七") == ["一"]
    # Same positions in both lines, twice: allowed.
    assert repeated_characters("一二一四", "一五一七") == []


def test_no_shared_characters():
    assert repeated_characters("花影临窗窗正启", "竹声入夜夜微凉") == []


def test_several_violations_are_listed_in_order():
    assert repeated_characters("甲乙丙丁", "丁丙戊己") == ["丙", "丁"]


# ---------------------------------------------------------------- generated lines

def test_forward_all_rules_pass():
    r = run(FIRST, SECOND, FORWARD)
    assert r["judged"] == "second"
    assert r["rule_length"] and r["rule_tone_modern"] and r["rule_tone_classical"]
    assert r["rule_no_repeat"] and r["all_rules_modern"] and r["all_rules_classical"]
    assert r["end_char"] == "还" and r["end_modern_tone"] == 2
    assert not r["tone_classical_fallback"]
    assert r["len_first"] == 7 and r["len_second"] == 7


def test_reverse_all_rules_pass():
    r = run(SECOND, FIRST, REVERSE)
    assert r["judged"] == "first"
    assert r["end_char"] == "睡" and r["end_modern_class"] == ZE
    assert r["all_rules_modern"] and r["all_rules_classical"]


def test_wrong_end_tone_fails_only_the_tone_rule():
    # A second line that ends 仄.
    r = run(FIRST, "数日街头缓缓去", FORWARD)
    assert r["rule_length"] and r["rule_no_repeat"]
    assert not r["rule_tone_modern"] and not r["rule_tone_classical"]
    assert not r["all_rules_modern"]
    # A first line that ends 平.
    r = run(SECOND, "三更舟中悄悄温", REVERSE)
    assert not r["rule_tone_modern"] and not r["all_rules_modern"]


def test_entering_tone_splits_modern_from_classical():
    # 白 is tone 2 today (平) and 入声 classically (仄).
    r = run("门前流水东西去", "岭外浮云一片白", FORWARD)
    assert r["rule_tone_modern"] is True
    assert r["rule_tone_classical"] is False
    assert r["all_rules_modern"] and not r["all_rules_classical"]
    assert r["end_classical_tones"] == "入"
    # As a first line the same character is right classically and wrong today.
    r = run("门前流水东西还", "岭外浮云一片白", REVERSE)
    assert r["rule_tone_modern"] is False and r["rule_tone_classical"] is True


def test_character_in_both_classical_classes_passes_either_way():
    assert run(FIRST, "数日街头缓缓看", FORWARD)["rule_tone_classical"] is True
    assert run(SECOND, "三更舟中悄悄看", REVERSE)["rule_tone_classical"] is True
    assert run(FIRST, "数日街头缓缓看", FORWARD)["end_classical_class"] == BOTH


def test_character_missing_from_the_table_falls_back_to_modern():
    r = run(FIRST, "数日街头缓缓郎", FORWARD)      # 郎 is not in the fixture table
    assert r["tone_classical_fallback"] is True
    assert r["rule_tone_classical"] == r["rule_tone_modern"] is True
    assert r["end_classical_class"] == UNKNOWN


def test_without_a_table_classical_equals_modern():
    r = check_generated(FIRST, SECOND, FORWARD, pingshui=None, tagger=fake_tagger)
    assert r["tone_classical_fallback"] and r["rule_tone_classical"] == r["rule_tone_modern"]
    assert r["even_opposition_classical"] is None


def test_the_given_line_is_never_blamed_for_its_own_tone():
    # The given first line ends 平 (it breaks the rule); the answer is still judged fairly.
    r = run("三更舟中悄悄温", SECOND, FORWARD)
    assert r["rule_tone_modern"] and r["all_rules_modern"]


def test_too_short_and_too_long_fail_length_and_have_no_position_scores():
    for answer in ("数日街头缓还", "数日街头缓缓缓还"):
        r = run(FIRST, answer, FORWARD)
        assert not r["rule_length"] and not r["all_rules_modern"]
        assert r["rule_tone_modern"]                    # still told the tone was right
        assert r["pos_match"] is None and r["even_opposition_modern"] is None


def test_stray_punctuation_fails_length_but_tone_and_repeats_are_still_judged():
    r = run(FIRST, "数日街头，缓缓还", FORWARD)
    assert not r["pure"] and not r["rule_length"] and not r["all_rules_modern"]
    assert r["len_second"] == 7
    assert r["rule_tone_modern"] and r["rule_no_repeat"]


def test_empty_answer_fails_everything():
    r = run(FIRST, "", FORWARD)
    assert r["empty"]
    for key in ("rule_length", "rule_tone_modern", "rule_tone_classical",
                "rule_no_repeat", "all_rules_modern", "all_rules_classical"):
        assert r[key] is False
    assert r["pos_match"] is None


def test_answer_with_no_hanzi_counts_as_empty():
    r = run(FIRST, "abc, 123", FORWARD)
    assert r["empty"] and not r["all_rules_modern"]


def test_plain_copy_of_the_given_line_cannot_pass_all_rules():
    # Given a second line that happens to end 仄, a copy would otherwise pass as a first line.
    given = "数日街头缓缓去"
    r = run(given, given, REVERSE)
    assert r["echo"]
    assert r["rule_length"] and r["rule_tone_modern"] and r["rule_no_repeat"]
    assert not r["all_rules_modern"] and not r["all_rules_classical"]


def test_repeated_character_fails_rule_3_and_is_reported():
    r = run("半扇门一更上锁", "一席棋两客连庄", FORWARD)
    assert not r["rule_no_repeat"] and r["repeated"] == "一"
    assert r["rule_length"] and r["rule_tone_modern"]
    assert not r["all_rules_modern"]


def test_invalid_direction_and_judged_raise():
    with pytest.raises(ValueError):
        check_generated(FIRST, SECOND, "backward")
    with pytest.raises(ValueError):
        check_pair(FIRST, SECOND, "both")


# ---------------------------------------------------------------- reference couplets

def test_reference_is_scored_once_per_direction():
    r = check_reference(FIRST, SECOND, pingshui=PS, tagger=fake_tagger)
    assert set(r) == {FORWARD, REVERSE}
    assert r[FORWARD]["judged"] == "second" and r[REVERSE]["judged"] == "first"
    assert r[FORWARD]["all_rules_modern"] and r[REVERSE]["all_rules_modern"]


def test_reference_that_breaks_a_rule_is_reported_as_breaking_it():
    r = check_reference("半扇门一更上锁", "一席棋两客连庄", pingshui=PS, tagger=fake_tagger)
    assert not r[FORWARD]["all_rules_modern"] and not r[REVERSE]["all_rules_modern"]
    assert r[FORWARD]["rule_tone_modern"] and r[REVERSE]["rule_tone_modern"]


# ---------------------------------------------------------------- even-position opposition

def test_even_opposition_helper():
    assert _even_opposition([PING, ZE, PING, ZE], [ZE, PING, ZE, PING]) == 1.0
    assert _even_opposition([PING, ZE, PING, ZE], [PING, ZE, PING, ZE]) == 0.0
    assert _even_opposition([PING, ZE, PING, ZE], [PING, PING, PING, ZE]) == 0.5
    assert _even_opposition([PING, BOTH], [ZE, BOTH]) == 1.0
    assert _even_opposition([PING, UNKNOWN], [ZE, PING]) == 0.0
    assert _even_opposition([PING], [ZE]) is None


def test_even_opposition_uses_positions_2_4_6():
    # Tones by position, injected: only the even positions differ between the lines.
    tones = {"甲乙丙丁戊己庚": [1, 1, 1, 1, 1, 1, 4], "子丑寅卯辰巳午": [1, 4, 1, 4, 1, 4, 1]}
    r = check_pair("甲乙丙丁戊己庚", "子丑寅卯辰巳午", "second",
                   tagger=fake_tagger, modern_tone_fn=lambda line: tones[line])
    assert r["even_opposition_modern"] == 1.0
    tones["子丑寅卯辰巳午"] = [4, 1, 4, 1, 4, 1, 1]
    r = check_pair("甲乙丙丁戊己庚", "子丑寅卯辰巳午", "second",
                   tagger=fake_tagger, modern_tone_fn=lambda line: tones[line])
    assert r["even_opposition_modern"] == 0.0


# ---------------------------------------------------------------- parallelism hookup

def test_parallelism_scores_are_attached_for_equal_length_lines():
    r = run(FIRST, SECOND, FORWARD)
    assert r["pos_match"] == 1.0 and r["boundary_match"] == 1.0   # the fake tagger tags all alike


def test_parallelism_can_be_switched_off():
    r = run(FIRST, SECOND, FORWARD, with_parallelism=False)
    assert r["pos_match"] is None and r["all_rules_modern"]


# ---------------------------------------------------------------- match to the reference

def test_reference_match_exact():
    m = reference_match(SECOND, SECOND)
    assert m == {"exact": True, "same_position_share": 1.0, "char_jaccard": 1.0}


def test_reference_match_partial():
    m = reference_match("数日街头缓缓睡", SECOND)
    assert not m["exact"]
    assert m["same_position_share"] == pytest.approx(6 / 7)
    assert 0 < m["char_jaccard"] < 1


def test_reference_match_ignores_punctuation_and_handles_empty():
    assert reference_match(f"“{SECOND}”。", SECOND)["exact"]
    assert reference_match("", SECOND) == {"exact": False, "same_position_share": 0.0, "char_jaccard": 0.0}


def test_reference_match_different_lengths():
    m = reference_match("数日街头", SECOND)
    assert not m["exact"] and m["same_position_share"] == pytest.approx(4 / 7)
