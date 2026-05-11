"""
Pay Ranges reference data (monthly, EUR).
Kept as raw data so the analyst can edit without touching code.
"""
from __future__ import annotations

import numpy as np

PAY_RANGES_ROWS: list[dict] = [
    # RefLevel, Location, PayZone, P25, P50, P75, RFL, GroupMin, GroupMax, Diff
    {"ref_level": "8",  "location": "ΚΙΛΚΙΣ", "pay_zone": "Support", "p25": 1050.00, "p50": 1178.29, "p75": 1404.36, "rfl": "8",  "group_min": 1050.00, "group_max": 1596.64, "diff_min_max": 546.86},
    {"ref_level": "8",  "location": "ΚΙΛΚΙΣ", "pay_zone": "Ops",     "p25": 1095.00, "p50": 1303.57, "p75": 1596.64, "rfl": "8",  "group_min": 1095.00, "group_max": 1558.93, "diff_min_max": 463.93},

    {"ref_level": "9",  "location": "ΚΙΛΚΙΣ",          "pay_zone": "Support", "p25": 1097.00, "p50": 1274.86, "p75": 1558.93, "rfl": "9",  "group_min": 1097.00, "group_max": 1779.93, "diff_min_max": 682.86},
    {"ref_level": "9",  "location": "ΚΙΛΚΙΣ / ΞΑΝΘΗ",  "pay_zone": "Ops",     "p25": 1150.00, "p50": 1349.86, "p75": 1779.93, "rfl": "9",  "group_min": 1150.00, "group_max": 1617.21, "diff_min_max": 467.21},

    {"ref_level": "10", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ",          "pay_zone": "Support", "p25": 1100.00, "p50": 1308.00, "p75": 1617.21, "rfl": "10", "group_min": 1100.00, "group_max": 1850.64, "diff_min_max": 750.36},
    {"ref_level": "10", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ / ΑΧΑΡΝΑΙ", "pay_zone": "Ops",     "p25": 1302.00, "p50": 1500.00, "p75": 1850.64, "rfl": "10", "group_min": 1302.00, "group_max": 1606.43, "diff_min_max": 304.00},

    {"ref_level": "11", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ / ΑΧΑΡΝΑΙ", "pay_zone": "Support", "p25": 1150.00, "p50": 1359.07, "p75": 1606.43, "rfl": "11", "group_min": 1150.00, "group_max": 1979.36, "diff_min_max": 829.79},
    {"ref_level": "11", "location": "ΚΙΛΚΙΣ",                     "pay_zone": "Ops",     "p25": 1360.00, "p50": 1583.00, "p75": 1979.36, "rfl": "11", "group_min": 1360.00, "group_max": 1728.00, "diff_min_max": 368.43},

    {"ref_level": "12", "location": "ΕΥΚΑΡΠΙΑ",                  "pay_zone": "Sales",   "p25": 1207.00, "p50": 1523.43, "p75": 1728.00, "rfl": "12", "group_min": 1207.00, "group_max": 2088.21, "diff_min_max": 881.71},
    {"ref_level": "12", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ",         "pay_zone": "Support", "p25": 1302.00, "p50": 1501.93, "p75": 1782.00, "rfl": "12", "group_min": 1302.00, "group_max": 2000.14, "diff_min_max": 698.50},
    {"ref_level": "12", "location": "ΚΙΛΚΙΣ / ΞΑΝΘΗ",            "pay_zone": "Ops",     "p25": 1383.00, "p50": 1622.36, "p75": 1953.86, "rfl": "12", "group_min": 1383.00, "group_max": 2163.93, "diff_min_max": 780.57},
    {"ref_level": "12", "location": "ΕΥΚΑΡΠΙΑ",                  "pay_zone": "R&D",     "p25": 1388.00, "p50": 1769.09, "p75": 1952.00, "rfl": "12", "group_min": 1388.00, "group_max": 2344.57, "diff_min_max": 956.93},
    {"ref_level": "12", "location": "ΑΧΑΡΝΑΙ",                   "pay_zone": "Sales",   "p25": 1439.00, "p50": 1706.79, "p75": 2088.21, "rfl": "12", "group_min": 1439.00, "group_max": 2353.07, "diff_min_max": 913.86},

    {"ref_level": "13", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ",                       "pay_zone": "Sales",   "p25": 1308.00, "p50": 1636.00, "p75": 2000.14, "rfl": "13", "group_min": 1308.00, "group_max": 2482.29, "diff_min_max": 1173.93},
    {"ref_level": "13", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ / ΑΧΑΡΝΑΙ",             "pay_zone": "Support", "p25": 1600.00, "p50": 1862.21, "p75": 2163.93, "rfl": "13", "group_min": 1600.00, "group_max": 2447.07, "diff_min_max": 847.07},
    {"ref_level": "13", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "R&D",     "p25": 1683.00, "p50": 2125.66, "p75": 2344.57, "rfl": "13", "group_min": 1683.00, "group_max": 2500.00, "diff_min_max": 817.50},
    {"ref_level": "13", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ / ΞΑΝΘΗ / ΑΧΑΡΝΑΙ",     "pay_zone": "Ops",     "p25": 1722.00, "p50": 2039.64, "p75": 2353.07, "rfl": "13", "group_min": 1722.00, "group_max": 2235.00, "diff_min_max": 512.86},
    {"ref_level": "13", "location": "ΑΧΑΡΝΑΙ",                                 "pay_zone": "Sales",   "p25": 1785.00, "p50": 2075.00, "p75": 2482.29, "rfl": "13", "group_min": 1785.00, "group_max": 2587.71, "diff_min_max": 803.00},

    {"ref_level": "14", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "R&D",     "p25": 1700.00, "p50": 2193.83, "p75": 2447.07, "rfl": "14", "group_min": 1700.00, "group_max": 2587.71, "diff_min_max": 887.71},
    {"ref_level": "14", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ / ΑΧΑΡΝΑΙ",             "pay_zone": "Support", "p25": 1763.00, "p50": 2082.86, "p75": 2500.00, "rfl": "14", "group_min": 1763.00, "group_max": 2945.07, "diff_min_max": 1182.57},
    {"ref_level": "S",  "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "Sales",   "p25": 1783.00, "p50": 2000.00, "p75": 2235.00, "rfl": "S",  "group_min": 1783.00, "group_max": 3026.14, "diff_min_max": 1243.14},
    {"ref_level": "14", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ / ΑΧΑΡΝΑΙ",             "pay_zone": "Ops",     "p25": 1850.00, "p50": 2236.36, "p75": 2587.71, "rfl": "14", "group_min": 1850.00, "group_max": 3051.36, "diff_min_max": 1201.36},

    {"ref_level": "15", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "Sales",   "p25": 2202.00, "p50": 2516.79, "p75": 2945.07, "rfl": "15", "group_min": 2202.00, "group_max": 3195.29, "diff_min_max": 993.79},
    {"ref_level": "15", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "R&D",     "p25": 2227.00, "p50": 2778.56, "p75": 3026.14, "rfl": "15", "group_min": 2227.00, "group_max": 3892.21, "diff_min_max": 1665.29},
    {"ref_level": "15", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ / ΑΧΑΡΝΑΙ",             "pay_zone": "Support", "p25": 2247.00, "p50": 2611.64, "p75": 3051.36, "rfl": "15", "group_min": 2247.00, "group_max": 3623.21, "diff_min_max": 1376.21},
    {"ref_level": "15", "location": "ΚΙΛΚΙΣ",                                  "pay_zone": "Ops",     "p25": 2300.00, "p50": 2758.43, "p75": 3195.29, "rfl": "15", "group_min": 2300.00, "group_max": 3884.50, "diff_min_max": 1584.50},

    {"ref_level": "16", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "R&D",     "p25": 2437.00, "p50": 3423.31, "p75": 3892.21, "rfl": "16", "group_min": 2437.00, "group_max": 4104.29, "diff_min_max": 1667.29},
    {"ref_level": "16", "location": "ΑΧΑΡΝΑΙ",                                 "pay_zone": "Sales",   "p25": 2608.00, "p50": 3055.64, "p75": 3623.21, "rfl": "16", "group_min": 2608.00, "group_max": 4816.57, "diff_min_max": 2209.07},
    {"ref_level": "16", "location": "ΚΙΛΚΙΣ / ΑΧΑΡΝΑΙ",                        "pay_zone": "Ops",     "p25": 2633.00, "p50": 3304.07, "p75": 3884.50, "rfl": "16", "group_min": 2633.00, "group_max": 4481.43, "diff_min_max": 1848.86},
    {"ref_level": "16", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "Sales",   "p25": 2637.00, "p50": 3100.00, "p75": 3732.43, "rfl": "16", "group_min": 2637.00, "group_max": 4708.93, "diff_min_max": 2072.29},
    {"ref_level": "16", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ",                       "pay_zone": "Support", "p25": 3000.00, "p50": 3500.00, "p75": 4104.29, "rfl": "16", "group_min": 3000.00, "group_max": 4296.43, "diff_min_max": 1296.43},

    {"ref_level": "17", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "Sales",   "p25": 3107.00, "p50": 3520.21, "p75": 4816.57, "rfl": "17", "group_min": 3107.00, "group_max": 4296.43, "diff_min_max": 1189.57},
    {"ref_level": "17", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "R&D",     "p25": 3240.00, "p50": 3936.37, "p75": 4481.43, "rfl": "17", "group_min": 3240.00, "group_max": 5635.71, "diff_min_max": 2395.86},
    {"ref_level": "17", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ",                       "pay_zone": "Support", "p25": 3277.00, "p50": 3925.43, "p75": 4708.93, "rfl": "17", "group_min": 3277.00, "group_max": 5207.29, "diff_min_max": 1930.00},
    {"ref_level": "17", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ / ΞΑΝΘΗ",               "pay_zone": "Ops",     "p25": 3313.00, "p50": 3892.14, "p75": 4296.43, "rfl": "17", "group_min": 3313.00, "group_max": 5186.43, "diff_min_max": 1873.93},

    {"ref_level": "18", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "Sales",   "p25": 3417.00, "p50": 3872.01, "p75": 5635.71, "rfl": "18", "group_min": 3417.00, "group_max": 5511.07, "diff_min_max": 2093.73},
    {"ref_level": "18", "location": "ΑΧΑΡΝΑΙ",                                 "pay_zone": "Sales",   "p25": 3658.00, "p50": 4671.50, "p75": 5207.29, "rfl": "18", "group_min": 3658.00, "group_max": 7697.57, "diff_min_max": 4039.22},
    {"ref_level": "18", "location": "ΚΙΛΚΙΣ",                                  "pay_zone": "Ops",     "p25": 3813.00, "p50": 4682.14, "p75": 5186.43, "rfl": "18", "group_min": 3813.00, "group_max": 9018.73, "diff_min_max": 5205.95},
    {"ref_level": "18", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "R&D",     "p25": 3853.00, "p50": 4877.91, "p75": 5168.36, "rfl": "18", "group_min": 3853.00, "group_max": 11533.09, "diff_min_max": 7679.95},
    {"ref_level": "18", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "Support", "p25": 4270.00, "p50": 4634.00, "p75": 5511.07, "rfl": "18", "group_min": 4270.00, "group_max": 13461.88, "diff_min_max": 9192.31},

    {"ref_level": "19", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ",                       "pay_zone": "Executives", "p25": 5741.00,  "p50": 6612.00,  "p75": 7697.57,  "rfl": "19", "group_min": 5741.00,  "group_max": 7697.57,  "diff_min_max": 1956.13},
    {"ref_level": "20", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ",                       "pay_zone": "Executives", "p25": 6834.00,  "p50": 7665.32,  "p75": 9018.73,  "rfl": "20", "group_min": 6834.00,  "group_max": 9018.73,  "diff_min_max": 2185.01},
    {"ref_level": "21", "location": "ΕΥΚΑΡΠΙΑ",                                "pay_zone": "Executives", "p25": 8727.00,  "p50": 10013.20, "p75": 11533.09, "rfl": "21", "group_min": 8727.00,  "group_max": 11533.09, "diff_min_max": 2806.43},
    {"ref_level": "22", "location": "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ",                       "pay_zone": "Executives", "p25": 10560.00, "p50": 11745.00, "p75": 13461.88, "rfl": "22", "group_min": 10560.00, "group_max": 13461.88, "diff_min_max": 2902.26},

    # P50-only rows: P25=P50=P75 to keep them plottable
    {"ref_level": "24", "location": "ΕΥΚΑΡΠΙΑ", "pay_zone": "Executives", "p25": 18483.21, "p50": 18483.21, "p75": 18483.21, "rfl": "24", "group_min": np.nan, "group_max": np.nan, "diff_min_max": np.nan},
    {"ref_level": "25", "location": "ΕΥΚΑΡΠΙΑ", "pay_zone": "Executives", "p25": 20489.68, "p50": 20489.68, "p75": 20489.68, "rfl": "25", "group_min": np.nan, "group_max": np.nan, "diff_min_max": np.nan},
]
