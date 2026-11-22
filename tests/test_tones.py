from dfl.tones import (
    BOTH, PING, UNKNOWN, ZE, Pingshui, class_of_tone, modern_classes,
    modern_is_ambiguous, modern_tones, satisfies, tone_number,
)


# ---------------------------------------------------------------- basics

def test_tone_number():
    assert tone_number("zhong1") == 1
    assert tone_number("suo3") == 3
    assert tone_number("de5") == 5
    assert tone_number("de") == 5
    assert tone_number("") == 5


def test_class_of_tone():
    assert class_of_tone(1) == PING and class_of_tone(2) == PING
    assert class_of_tone(3) == ZE and class_of_tone(4) == ZE
    assert class_of_tone(5) == UNKNOWN


# ---------------------------------------------------------------- modern, with fake readers

def test_neutral_tone_falls_back_to_first_toned_reading():
    tones = modern_tones(
        "了",
        line_reader=lambda line: ["le5"],
        char_reader=lambda ch: ["le5", "liao3", "liao4"],
    )
    assert tones == [3]


def test_no_toned_reading_stays_unknown():
    tones = modern_tones("X", line_reader=lambda line: ["x5"], char_reader=lambda ch: ["x5"])
    assert tones == [5]
    assert class_of_tone(tones[0]) == UNKNOWN


def test_character_the_reader_does_not_know():
    # pypinyin hands back the character itself when it has no reading.
    tones = modern_tones("𬺈", line_reader=lambda line: ["𬺈"], char_reader=lambda ch: ["𬺈"])
    assert tones == [5]


def test_context_reading_is_used_over_other_readings():
    tones = modern_tones(
        "还",
        line_reader=lambda line: ["huan2"],
        char_reader=lambda ch: ["hai2", "huan2", "fu2"],
    )
    assert tones == [2]


def test_ambiguity_flag():
    assert modern_is_ambiguous("看", char_reader=lambda ch: ["kan4", "kan1"])
    assert not modern_is_ambiguous("还", char_reader=lambda ch: ["hai2", "huan2"])


# ---------------------------------------------------------------- modern, with real pypinyin

def test_pypinyin_on_unambiguous_characters():
    assert modern_tones("锁庄") == [3, 1]
    assert modern_classes("睡温") == [ZE, PING]
    assert modern_tones("色郎腕花") == [4, 2, 4, 1]


def test_pypinyin_alignment_one_tone_per_character():
    line = "王鹏借壳给老邓洗澡"
    assert len(modern_tones(line)) == len(line)


def test_pypinyin_entering_tone_characters_read_as_modern():
    # 白 and 竹 are 入声 classically but tone 2 in Mandarin.
    assert modern_classes("白竹") == [PING, PING]


# ---------------------------------------------------------------- classical

TABLE = {"东": "平", "白": "入", "看": "平去", "锁": "上", "去": "去上", "發": "入", "发": "入"}


def test_pingshui_classes():
    ps = Pingshui(TABLE)
    assert ps.klass("东") == PING
    assert ps.klass("白") == ZE
    assert ps.klass("锁") == ZE
    assert ps.klass("看") == BOTH
    assert ps.klass("去") == ZE          # two 仄 categories are still 仄
    assert ps.klass("鹏") == UNKNOWN
    assert ps.classes("东白鹏") == [PING, ZE, UNKNOWN]


def test_pingshui_entering_tone():
    ps = Pingshui(TABLE)
    assert ps.is_entering("白") and ps.is_entering("发")
    assert not ps.is_entering("东") and not ps.is_entering("鹏")


def test_satisfies():
    assert satisfies(PING, PING) is True
    assert satisfies(ZE, PING) is False
    assert satisfies(BOTH, PING) is True and satisfies(BOTH, ZE) is True
    assert satisfies(UNKNOWN, PING) is None
