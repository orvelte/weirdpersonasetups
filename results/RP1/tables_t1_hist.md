arm=t1 setup=hist exclude_breaks=False

| score | 3.5 | 3.6 | 3.8 |
|---|---|---|---|
| L0 frame break on L1 replies | 0% [-0,20] (n=15) | 0% [-0,20] (n=15) | 0% [-0,20] (n=15) |
| L1 judge picks character (chance 50%) | 100% [80,100] (n=15) | 100% [80,100] (n=15) | 100% [80,100] (n=15) |
| L2 character answer in frame | 100% [89,100] (n=30) | 100% [89,100] (n=30) | 100% [89,100] (n=30) |
| L2 control on same items | 0% [-0,11] (n=30) | 0% [-0,11] (n=30) | 0% [-0,11] (n=30) |
| L3 character answer out of frame | 97% [83,99] (n=30) | 87% [70,95] (n=30) | 62% [43,79] (n=24) |
| L3 control on same items | 0% [-0,11] (n=30) | 0% [-0,11] (n=30) | 0% [-0,11] (n=30) |
| L4 character choice | 80% [49,94] (n=10) | 60% [31,83] (n=10) | 50% [24,76] (n=10) |
| L4 control on same items | 0% [0,30] (n=9) | 0% [-0,28] (n=10) | 0% [0,30] (n=9) |
| L2 corrected (char − control, item bootstrap) | +100 [+100,+100] | +100 [+100,+100] | +100 [+100,+100] |
| L3 corrected (char − control, item bootstrap) | +97 [+90,+100] | +87 [+73,+97] | +62 [+42,+83] |
| L4 corrected (char − control, item bootstrap) | +80 [+50,+100] | +60 [+30,+90] | +50 [+20,+80] |
| L1 lexicon coverage char / control | 0.15 / 0.01 | 0.40 / 0.00 | 0.23 / 0.03 |
| eval-aware (regex, reasoning or content) | 33% [24,43] (n=85) | 4% [1,10] (n=85) | 6% [3,13] (n=85) |
| mean reasoning tokens | 2504 | 1362 | 460 |
| unparsed L2/L3/L4 | 0/0/1 | 0/0/0 | 0/6/1 |

Per character, character-answer rate (L2 / L3 / L4):

| character | 3.5 | 3.6 | 3.8 |
|---|---|---|---|
| compute_ai | 100 / 100 / 50 | 100 / 100 / 50 | 100 / 50 / 0 |
| con_artist | 100 / 100 / 100 | 100 / 100 / 100 | 100 / 50 / 100 |
| flat_earther | 100 / 100 / 100 | 100 / 83 / 0 | 100 / 100 / 0 |
| miasma_physician | 100 / 83 / 50 | 100 / 83 / 50 | 100 / 75 / 50 |
| ruthless_trader | 100 / 100 / 100 | 100 / 67 / 100 | 100 / 67 / 100 |
