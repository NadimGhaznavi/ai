"""
SimStats.py
"""
import sys

class SimStats:
    def __init__(self, ini, log):
        self.ini = ini
        self.log = log
        self.stats = {}
        self.log.log('SimStats initialization:    [OK]')

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
    