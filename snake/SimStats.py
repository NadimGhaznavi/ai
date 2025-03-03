"""
SimStats.py
"""
import os
import sys
import yaml
from collections import deque

MAX_LENGTH = 50
class SimStats:
    def __init__(self, ini, log):
        self.ini = ini
        self.log = log
        self.stats = {}
        self.max_len = MAX_LENGTH
        self.log.log('SimStats initialization:    [OK]')
        summary_freq = ini.get('show_summary_freq')
        self.stats['recent'] = {}
        self.stats['recent']['loss'] = deque(maxlen=summary_freq)
        self.stats['recent']['score'] = deque(maxlen=summary_freq)

    def __del__(self):
        self.save()

    def append(self, category, key, value):
        if category not in self.stats:
            self.stats[category] = {}
        if key not in self.stats[category]:
            self.stats[category][key] = [] 
        if type(self.stats[category][key]) != list and \
            type(self.stats[category][key]) != deque:
            self.log.log("ERROR: Cannot append to non-list value (" + str(key) + ") for category (" + category + ")")
            sys.exit(1)
        self.stats[category][key].append(value)

    def exists(self, category, key):
        if category not in self.stats:
            return False
        if key not in self.stats[category]:
            return False
        return True

    def incr(self, category, key):
        if category not in self.stats:
            self.stats[category] = {}
        if key not in self.stats[category]:
            self.stats[category][key] = 0
        if type(self.stats[category][key]) != int:
            self.log.log("ERROR: Cannot increment non-integer value (" + str(key) + ") for category (" + category + ")")
            sys.exit(1)
        self.stats[category][key] += 1

    def set(self, category, key, value):
        if category not in self.stats:
            self.stats[category] = {}
        self.stats[category][key] = value

    def get(self, category, key):
        return self.stats[category][key]
    
    def save(self):
        # Generate simulation specific file
        DATA_DIR = self.ini.get('sim_data_dir')
        SIM_NUM = str(self.ini.get('sim_num'))
        STATS_BASENAME = self.ini.get('stats_basename')
        STATS_FILE = os.path.join(DATA_DIR, SIM_NUM + STATS_BASENAME)
        with open(STATS_FILE, 'w') as file:
            yaml.dump(self.stats, file)
    