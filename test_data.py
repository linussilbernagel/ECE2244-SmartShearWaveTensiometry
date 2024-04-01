from sw_load_notSensor import sw_load_notSensor
from sw_process import sw_process

p = '.'
# f = 'ADAT25_240219_R_heelrise1_achilles_2024-2-19_14626_-0.03--0.03Hz_01.mat'
f = 'ADAT02_230905_walk03_achilles_2023-9-5_123428_-0.03--0.03Hz_01.mat'
raw = sw_load_notSensor()

raw.load_data(p, f)

sw = sw_process(raw)

sw.tens['cp']['accdis'] = [20, 28]
sw.tens['pp']['mfw'] = 5
sw.tens['pp']['window'] = [0, 2]
sw.tens['pp']['winseed'] = 0
sw.tens['pp']['kalseed'] = 'fwdbkwd'

sw.plot['option'] = 0
sw.process(raw.ad)