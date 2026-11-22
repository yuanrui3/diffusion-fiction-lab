import pytest

from dfl.generate import item_seed
from dfl.prompts import FORWARD, REVERSE, build_prompt, chinese_numeral, wanted_line


@pytest.mark.parametrize("n,text", [(4, "四"), (6, "六"), (7, "七"), (8, "八"), (9, "九"), (10, "十"),
                                    (11, "十一"), (20, "二十"), (21, "二十一"), (99, "九十九")])
def test_chinese_numeral(n, text):
    assert chinese_numeral(n) == text


@pytest.mark.parametrize("n", [0, -1, 100])
def test_chinese_numeral_out_of_range(n):
    with pytest.raises(ValueError):
        chinese_numeral(n)


def test_forward_prompt_with_summary_states_every_checked_rule():
    p = build_prompt("外婆借身炸米饺", FORWARD, 7, "除夕夜摆了年夜饭。", version="v1")
    assert "梗概：除夕夜摆了年夜饭。" in p
    assert "上句：外婆借身炸米饺" in p
    assert "下句必须正好七个字" in p            # length, in characters
    assert "对仗" in p                          # parallelism
    assert "下句须以平声字收尾" in p            # end tone of the line to be written
    assert "不要重复上句用过的字" in p          # no repeated characters
    assert "内容要切合梗概" in p                # relevance
    assert p.rstrip().endswith("只写出下句本身，不加标点，不作解释。")
    assert p.count("、") >= 5 and "五、" in p


def test_reverse_prompt_swaps_the_roles():
    p = build_prompt("师傅隔海讲油温", REVERSE, 7, "除夕夜摆了年夜饭。")
    assert "下句：师傅隔海讲油温" in p
    assert "请补出上句" in p
    assert "上句必须正好七个字" in p
    assert "上句须以仄声字收尾" in p
    assert "不要重复下句用过的字" in p
    assert "只写出上句本身" in p
    assert "下句须以平声字收尾" not in p


def test_bare_prompt_has_no_summary_and_no_relevance_rule():
    p = build_prompt("老王母西池赐芳筵", REVERSE, 8)
    assert "梗概" not in p
    assert "上句必须正好八个字" in p
    assert "五、" not in p and "四、" in p
    assert p.count("。") >= 3


def test_prompt_never_asserts_the_tone_of_the_given_line():
    # Some original lines break the convention, so the prompt may state the convention
    # but must not claim that the given line ends in a particular tone.
    for direction in (FORWARD, REVERSE):
        p = build_prompt("清汤面点两碗", direction, 6, "梗概。")
        assert "上句以仄声收尾" not in p and "下句以平声收尾" not in p


def test_prompt_is_the_same_text_for_the_same_item():
    a = build_prompt("外婆借身炸米饺", FORWARD, 7, "梗概。")
    assert a == build_prompt("外婆借身炸米饺", FORWARD, 7, "梗概。")
    assert a != build_prompt("外婆借身炸米饺", FORWARD, 8, "梗概。")


def test_invalid_direction():
    with pytest.raises(ValueError):
        build_prompt("外婆借身炸米饺", "sideways", 7)


def test_wanted_line():
    assert wanted_line(FORWARD) == "second" and wanted_line(REVERSE) == "first"


def test_item_seed_is_stable_and_separates_everything():
    base = item_seed(20261119, "qwen", "huohou-01", "forward", "summary")
    assert base == item_seed(20261119, "qwen", "huohou-01", "forward", "summary")
    assert 0 <= base < 2**31
    others = {
        item_seed(20261120, "qwen", "huohou-01", "forward", "summary"),
        item_seed(20261119, "dream", "huohou-01", "forward", "summary"),
        item_seed(20261119, "qwen", "huohou-02", "forward", "summary"),
        item_seed(20261119, "qwen", "huohou-01", "reverse", "summary"),
        item_seed(20261119, "qwen", "huohou-01", "forward", "bare"),
    }
    assert base not in others and len(others) == 5


def test_v2_has_the_same_rules_and_ends_with_the_empty_slot():
    v1 = build_prompt("外婆借身炸米饺", FORWARD, 7, "除夕夜摆了年夜饭。", version="v1")
    v2 = build_prompt("外婆借身炸米饺", FORWARD, 7, "除夕夜摆了年夜饭。", version="v2")
    assert v2.endswith("梗概：除夕夜摆了年夜饭。\n上句：外婆借身炸米饺\n下句：")
    for piece in ("下句必须正好七个字", "对仗", "下句须以平声字收尾", "不要重复上句用过的字", "内容要切合梗概",
                  "只写出下句本身，不加标点，不作解释。"):
        assert piece in v1 and piece in v2
    # Same wording, different order: v2 is v1's text rearranged plus the empty slot label.
    assert sorted(v1.replace("\n", "")) == sorted(v2.replace("\n", "").removesuffix("下句："))


def test_v2_reverse_and_bare():
    v2 = build_prompt("老王母西池赐芳筵", REVERSE, 8, version="v2")
    assert v2.endswith("下句：老王母西池赐芳筵\n上句：")
    assert "梗概" not in v2 and "上句必须正好八个字" in v2


def test_unknown_prompt_version():
    with pytest.raises(ValueError):
        build_prompt("外婆借身炸米饺", FORWARD, 7, version="v3")


def test_default_layout_is_the_frozen_one():
    from dfl.prompts import PROMPT_VERSION

    assert PROMPT_VERSION == "v2"
    assert build_prompt("外婆借身炸米饺", FORWARD, 7) == build_prompt("外婆借身炸米饺", FORWARD, 7, version="v2")
