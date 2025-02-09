"""
SimStats.py
"""
import os
import sys
import yaml

class SimStats:
    def __init__(self, ini, log):
        self.ini = ini
        self.log = log
        self.stats = {}
        self.log.log('SimStats initialization:    [OK]')

    def __del__(self):
        self.save()

    def incr(self, category, key):
        if category not in self.stats:
            self.log.log("ERROR: No such category (" + category + ")")
            sys.exit(1)
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
        print("DEBUG self.stats", self.stats)
        with open(STATS_FILE, 'w') as file:
            yaml.dump(self.stats, file)
    