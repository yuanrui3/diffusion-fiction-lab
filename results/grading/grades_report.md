# Human grades, unblinded

48 of 51 lines graded; not graded: e11, e36, e46.

Literary quality, 1 to 5, by the author. One line per model for each of the same chapters and directions. Brackets are 95% bootstrap intervals over those chapter-direction cells.

| Model | Lines | Mean score | Lines scored 4 or 5 | Drawn from rule-passing samples |
|---|---|---|---|---|
| qwen | 16 | 2.88 [2.25, 3.44] | 6 | 12 |
| dream | 15 | 1.60 [1.20, 2.07] | 1 | 3 |
| llada | 17 | 2.71 [2.29, 3.12] | 3 | 10 |

## Differences on the same cells

| Comparison | Cells | Difference in mean score |
|---|---|---|
| dream − qwen | 14 | -1.43 [-2.21, -0.57] * |
| llada − qwen | 16 | -0.25 [-1.00, +0.50] |
| llada − dream | 15 | +1.20 [+0.67, +1.67] * |

## By direction

| Model | Forward (writes the second line) | Reverse (writes the first line) |
|---|---|---|
| qwen | 2.75 [2.00, 3.62] | 3.00 [2.25, 3.75] |
| dream | 2.00 [1.38, 2.75] | 1.14 [1.00, 1.43] |
| llada | 2.78 [2.22, 3.33] | 2.62 [2.12, 3.12] |

## Every graded line

| Entry | Chapter | Direction | Model | Line | Passed all rules | Score | Comment |
|---|---|---|---|---|---|---|---|
| e16 | huohou-03 | reverse | dream | 文从垃圾找生门 | no | 2 |  |
| e39 | huohou-03 | reverse | llada | 文到厨下验孕棒 | yes | 2 |  |
| e50 | huohou-03 | reverse | qwen | 梅将筵席挡出路 | yes | 2 |  |
| e23 | huohou-05 | forward | dream | 王鹏群发等级图 | yes | 2 |  |
| e32 | huohou-05 | forward | llada | 王鹏群里论等级 | yes | 4 |  |
| e11 | huohou-05 | forward | qwen | 王鹏群里争高低 | yes | – |  |
| e37 | huohou-08 | reverse | dream | 秒倒数老浦 | no | 1 |  |
| e43 | huohou-08 | reverse | llada | 两百毫秒老蒲试炒 | yes | 2 |  |
| e44 | huohou-08 | reverse | qwen | 八分时辰蒲翁戴镜 | yes | 4 |  |
| e08 | huohou-09 | forward | dream | 约定单漏写西奥 | no | 2 |  |
| e12 | huohou-09 | forward | llada | 约定中缺失科琳 | yes | 3 |  |
| e13 | huohou-09 | forward | qwen | 规矩原出自科琳 | yes | 2 |  |
| e21 | huohou-12 | forward | dream | 橡树下前落花 | no | 1 |  |
| e17 | huohou-12 | forward | llada | 农场路口吻别情 | yes | 3 |  |
| e35 | huohou-12 | forward | qwen | 橡树之下吻别离 | yes | 3 |  |
| e01 | huohou-13 | forward | dream | 卫衣袖子有红迹 | no | 3 |  |
| e19 | huohou-13 | forward | llada | 何蔓衣袖抹红泥 | yes | 4 |  |
| e30 | huohou-13 | forward | qwen | 何蔓袖红印泥 | no | 1 |  |
| e34 | huohou-15 | reverse | dream | 床头夫妻不 sleep | no | 1 |  |
| e41 | huohou-15 | reverse | llada | 冷库姐妹争亲属 | yes | 2 |  |
| e07 | huohou-15 | reverse | qwen | 库内姐妹相争执 | no | 4 |  |
| e36 | huohou-19 | forward | dream | 路易斯用perm生孩子 | no | – |  |
| e40 | huohou-19 | forward | llada | 路易斯近灶赠精子 | no | 2 |  |
| e25 | huohou-19 | forward | qwen | 拉法厨房传笑声 | yes | 2 |  |
| e45 | huohou-21 | forward | dream | 水房中情侣定约 | yes | 4 |  |
| e42 | huohou-21 | forward | llada | 私信里朋友相约 | yes | 3 |  |
| e02 | huohou-21 | forward | qwen | 水房里凤侣传情 | yes | 5 |  |
| e14 | huohou-23 | forward | dream | 乐子等结果坐钥匙 | no | 1 |  |
| e38 | huohou-23 | forward | llada | 乐子攥钥匙等结果 | no | 2 |  |
| e15 | huohou-23 | forward | qwen | 邓生离别赠钥匙 | yes | 4 |  |
| e27 | huohou-27 | reverse | dream | 王鹏远程为老邓洗洗 | yes | 1 |  |
| e47 | huohou-27 | reverse | llada | 王鹏远程陪老邓聊天 | no | 3 |  |
| e49 | huohou-27 | reverse | qwen | 铁子初来学腰花切断 | yes | 1 |  |
| e22 | huohou-29 | reverse | dream | 铁子面里故事 | no | 1 |  |
| e24 | huohou-29 | reverse | llada | 铁子案底诉往事 | yes | 4 |  |
| e04 | huohou-29 | reverse | qwen | 铁子言中带刀口 | yes | 3 |  |
| e05 | huohou-31 | forward | dream | 露丝十小时唱歌 | no | 2 |  |
| e09 | huohou-31 | forward | llada | 以斯拉一路唱主音 | yes | 3 |  |
| e29 | huohou-31 | forward | qwen | 露丝十小时唱和应 | yes | 2 |  |
| e18 | huohou-33 | reverse | dream | 张一帆见人走 | no | 1 |  |
| e26 | huohou-33 | reverse | llada | 张父看孩子编辫子 | no | 3 |  |
| e06 | huohou-33 | reverse | qwen | 张帆看辫子忆先贤 | no | 4 |  |
| e51 | huohou-37 | reverse | dream | 铁子看工程师 | no | 1 |  |
| e10 | huohou-37 | reverse | llada | 老者端锅知延迟 | no | 3 |  |
| e31 | huohou-37 | reverse | qwen | 铁汉观技识滞后 | yes | 4 |  |
| e46 | huohou-40 | reverse | dream | superior 装饭，拉法叶寺墙坐 | no | – |  |
| e28 | huohou-40 | reverse | llada | 迦勒送饭入寺门 | no | 2 |  |
| e48 | huohou-40 | reverse | qwen | 迦勒静坐寺墙根 | no | 2 |  |
| e03 | huohou-42 | forward | dream | Dave子 lost | no | 1 |  |
| e20 | huohou-42 | forward | llada | 郫都二嬢寿后失孙女 | no | 1 |  |
| e33 | huohou-42 | forward | qwen | 二娃田头失孙花 | yes | 3 |  |
