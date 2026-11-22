import pytest

from dfl.extract import extract_answer
from dfl.text import count_hanzi, hanzi_only, is_hanzi, is_pure_hanzi


# ---------------------------------------------------------------- text

def test_count_hanzi_counts_only_hanzi():
    assert count_hanzi("嬢嬢热抱屋头那个") == 8
    assert count_hanzi("下句：渔翁，临水!abc 123") == 6
    assert count_hanzi("") == 0


def test_hanzi_only_drops_everything_else():
    assert hanzi_only("下句：渔翁，临水!") == "下句渔翁临水"
    assert hanzi_only("　 \n") == ""


def test_ideographic_zero_is_a_hanzi():
    assert is_hanzi("〇")
    assert is_pure_hanzi("二〇四八")


def test_is_pure_hanzi():
    assert is_pure_hanzi("渔翁临水听潮声")
    assert not is_pure_hanzi("")
    assert not is_pure_hanzi("渔翁 临水")
    assert not is_pure_hanzi("渔翁临水，听潮声")
    assert not is_pure_hanzi("渔翁临水听潮声A")


def test_is_hanzi_rejects_punctuation_and_strings():
    assert not is_hanzi("，")
    assert not is_hanzi("a")
    assert not is_hanzi("渔翁")


# ---------------------------------------------------------------- extraction

LINE = "渔翁临水听潮声"
GIVEN = "船客登楼望远山"


def test_plain_line():
    e = extract_answer(LINE, "second")
    assert e.answer == LINE and e.pure and e.n_hanzi == 7 and e.n_lines == 1


def test_label_and_final_punctuation_are_stripped():
    assert extract_answer(f"下句：{LINE}。", "second").answer == LINE
    assert extract_answer(f"下联:{LINE}", "second").answer == LINE
    assert extract_answer(f"上句：{GIVEN}！", "first").answer == GIVEN


@pytest.mark.parametrize("wrapped", [f"「{LINE}」", f"“{LINE}”", f"《{LINE}》", f"**{LINE}**", f"  {LINE}　"])
def test_quotes_brackets_and_markup_are_stripped(wrapped):
    e = extract_answer(wrapped, "second")
    assert e.answer == LINE and e.pure


def test_lead_in_before_colon_is_dropped_and_later_lines_ignored():
    raw = f"好的，下句是：{LINE}\n\n解释：这一句与上句对仗。"
    e = extract_answer(raw, "second")
    assert e.answer == LINE
    assert e.n_lines == 2


def test_both_lines_echoed_with_labels_picks_the_requested_one():
    raw = f"上句：{GIVEN}\n下句：{LINE}"
    assert extract_answer(raw, "second").answer == LINE
    assert extract_answer(raw, "first").answer == GIVEN


def test_given_line_echoed_first_then_unlabelled_answer():
    raw = f"上句：{GIVEN}\n{LINE}"
    assert extract_answer(raw, "second").answer == LINE


def test_label_alone_on_its_line_takes_the_next_line():
    assert extract_answer(f"下句：\n{LINE}", "second").answer == LINE
    assert extract_answer(f"好的，以下是下句：\n\n{LINE}", "second").answer == LINE


def test_only_the_given_label_present_falls_back_to_that_line():
    # The model answered with the wrong label; we still read the line it wrote.
    e = extract_answer(f"上句：{LINE}", "second")
    assert e.answer == LINE


def test_empty_and_blank_replies():
    for raw in ("", "   \n\n", None):
        e = extract_answer(raw, "second")
        assert e.answer == "" and not e.pure and e.n_hanzi == 0 and e.n_lines == 0


def test_only_punctuation_reply():
    e = extract_answer("……", "second")
    assert e.answer == "" and not e.pure


def test_internal_punctuation_is_kept_and_marks_the_answer_impure():
    e = extract_answer("渔翁临水，听潮声", "second")
    assert e.answer == "渔翁临水，听潮声"
    assert not e.pure and e.n_hanzi == 7


def test_commentary_on_the_same_line_is_not_cleaned_up():
    e = extract_answer(f"{LINE}（与上句对仗）", "second")
    assert not e.pure          # the bracketed note stays inside the answer
    assert e.n_hanzi > 7


def test_invalid_want_raises():
    with pytest.raises(ValueError):
        extract_answer(LINE, "下句")
