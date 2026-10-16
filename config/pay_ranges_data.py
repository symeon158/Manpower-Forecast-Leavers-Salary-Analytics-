"""
Pay Ranges reference data (monthly, EUR).

⚠️ SAMPLE DATA — the numbers below are generated for illustration only and are
NOT real pay ranges of any company. Replace PAY_RANGES_ROWS with your own
(same keys) to use the Pay Ranges tab and the pay-range overlay in the
Salary & Grade tab. Keep real compensation data out of public repositories.
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (kept for np.nan in rows with missing bounds)

PAY_RANGES_ROWS: list[dict] = [
    # RefLevel, Location, PayZone, P25, P50, P75, RFL, GroupMin, GroupMax, Diff
    {"ref_level": "8", "location": "Site A / Site B", "pay_zone": "Ops", "p25": 880.00, "p50": 1000.00, "p75": 1150.00, "rfl": "8", "group_min": 880.00, "group_max": 1300.00, "diff_min_max": 420.00},
    {"ref_level": "8", "location": "Site A", "pay_zone": "Support", "p25": 850.00, "p50": 970.00, "p75": 1120.00, "rfl": "8", "group_min": 850.00, "group_max": 1260.00, "diff_min_max": 410.00},
    {"ref_level": "8", "location": "Site B", "pay_zone": "Sales", "p25": 920.00, "p50": 1050.00, "p75": 1210.00, "rfl": "8", "group_min": 920.00, "group_max": 1360.00, "diff_min_max": 440.00},
    {"ref_level": "8", "location": "Site A", "pay_zone": "R&D", "p25": 950.00, "p50": 1080.00, "p75": 1240.00, "rfl": "8", "group_min": 950.00, "group_max": 1400.00, "diff_min_max": 450.00},

    {"ref_level": "9", "location": "Site A / Site B", "pay_zone": "Ops", "p25": 990.00, "p50": 1120.00, "p75": 1290.00, "rfl": "9", "group_min": 990.00, "group_max": 1460.00, "diff_min_max": 470.00},
    {"ref_level": "9", "location": "Site A", "pay_zone": "Support", "p25": 960.00, "p50": 1090.00, "p75": 1250.00, "rfl": "9", "group_min": 960.00, "group_max": 1420.00, "diff_min_max": 460.00},
    {"ref_level": "9", "location": "Site B", "pay_zone": "Sales", "p25": 1040.00, "p50": 1180.00, "p75": 1360.00, "rfl": "9", "group_min": 1040.00, "group_max": 1530.00, "diff_min_max": 490.00},
    {"ref_level": "9", "location": "Site A", "pay_zone": "R&D", "p25": 1060.00, "p50": 1210.00, "p75": 1390.00, "rfl": "9", "group_min": 1060.00, "group_max": 1570.00, "diff_min_max": 510.00},

    {"ref_level": "10", "location": "Site A / Site B", "pay_zone": "Ops", "p25": 1100.00, "p50": 1250.00, "p75": 1440.00, "rfl": "10", "group_min": 1100.00, "group_max": 1620.00, "diff_min_max": 520.00},
    {"ref_level": "10", "location": "Site A", "pay_zone": "Support", "p25": 1070.00, "p50": 1220.00, "p75": 1400.00, "rfl": "10", "group_min": 1070.00, "group_max": 1590.00, "diff_min_max": 520.00},
    {"ref_level": "10", "location": "Site B", "pay_zone": "Sales", "p25": 1160.00, "p50": 1320.00, "p75": 1520.00, "rfl": "10", "group_min": 1160.00, "group_max": 1720.00, "diff_min_max": 560.00},
    {"ref_level": "10", "location": "Site A", "pay_zone": "R&D", "p25": 1190.00, "p50": 1350.00, "p75": 1550.00, "rfl": "10", "group_min": 1190.00, "group_max": 1760.00, "diff_min_max": 570.00},

    {"ref_level": "11", "location": "Site A / Site B", "pay_zone": "Ops", "p25": 1230.00, "p50": 1400.00, "p75": 1610.00, "rfl": "11", "group_min": 1230.00, "group_max": 1820.00, "diff_min_max": 590.00},
    {"ref_level": "11", "location": "Site A", "pay_zone": "Support", "p25": 1200.00, "p50": 1360.00, "p75": 1560.00, "rfl": "11", "group_min": 1200.00, "group_max": 1770.00, "diff_min_max": 570.00},
    {"ref_level": "11", "location": "Site B", "pay_zone": "Sales", "p25": 1300.00, "p50": 1480.00, "p75": 1700.00, "rfl": "11", "group_min": 1300.00, "group_max": 1920.00, "diff_min_max": 620.00},
    {"ref_level": "11", "location": "Site A", "pay_zone": "R&D", "p25": 1340.00, "p50": 1520.00, "p75": 1750.00, "rfl": "11", "group_min": 1340.00, "group_max": 1980.00, "diff_min_max": 640.00},

    {"ref_level": "12", "location": "Site A / Site B", "pay_zone": "Ops", "p25": 1380.00, "p50": 1570.00, "p75": 1810.00, "rfl": "12", "group_min": 1380.00, "group_max": 2040.00, "diff_min_max": 660.00},
    {"ref_level": "12", "location": "Site A", "pay_zone": "Support", "p25": 1350.00, "p50": 1530.00, "p75": 1760.00, "rfl": "12", "group_min": 1350.00, "group_max": 1990.00, "diff_min_max": 640.00},
    {"ref_level": "12", "location": "Site B", "pay_zone": "Sales", "p25": 1450.00, "p50": 1650.00, "p75": 1900.00, "rfl": "12", "group_min": 1450.00, "group_max": 2140.00, "diff_min_max": 690.00},
    {"ref_level": "12", "location": "Site A", "pay_zone": "R&D", "p25": 1500.00, "p50": 1700.00, "p75": 1950.00, "rfl": "12", "group_min": 1500.00, "group_max": 2210.00, "diff_min_max": 710.00},

    {"ref_level": "13", "location": "Site A / Site B", "pay_zone": "Ops", "p25": 1550.00, "p50": 1760.00, "p75": 2020.00, "rfl": "13", "group_min": 1550.00, "group_max": 2290.00, "diff_min_max": 740.00},
    {"ref_level": "13", "location": "Site A", "pay_zone": "Support", "p25": 1500.00, "p50": 1710.00, "p75": 1970.00, "rfl": "13", "group_min": 1500.00, "group_max": 2220.00, "diff_min_max": 720.00},
    {"ref_level": "13", "location": "Site B", "pay_zone": "Sales", "p25": 1630.00, "p50": 1850.00, "p75": 2130.00, "rfl": "13", "group_min": 1630.00, "group_max": 2400.00, "diff_min_max": 770.00},
    {"ref_level": "13", "location": "Site A", "pay_zone": "R&D", "p25": 1670.00, "p50": 1900.00, "p75": 2180.00, "rfl": "13", "group_min": 1670.00, "group_max": 2470.00, "diff_min_max": 800.00},

    {"ref_level": "14", "location": "Site A / Site B", "pay_zone": "Ops", "p25": 1730.00, "p50": 1970.00, "p75": 2270.00, "rfl": "14", "group_min": 1730.00, "group_max": 2560.00, "diff_min_max": 830.00},
    {"ref_level": "14", "location": "Site A", "pay_zone": "Support", "p25": 1680.00, "p50": 1910.00, "p75": 2200.00, "rfl": "14", "group_min": 1680.00, "group_max": 2480.00, "diff_min_max": 800.00},
    {"ref_level": "14", "location": "Site B", "pay_zone": "Sales", "p25": 1820.00, "p50": 2070.00, "p75": 2380.00, "rfl": "14", "group_min": 1820.00, "group_max": 2690.00, "diff_min_max": 870.00},
    {"ref_level": "14", "location": "Site A", "pay_zone": "R&D", "p25": 1870.00, "p50": 2130.00, "p75": 2450.00, "rfl": "14", "group_min": 1870.00, "group_max": 2770.00, "diff_min_max": 900.00},

    {"ref_level": "15", "location": "Site A / Site B", "pay_zone": "Ops", "p25": 1940.00, "p50": 2210.00, "p75": 2540.00, "rfl": "15", "group_min": 1940.00, "group_max": 2870.00, "diff_min_max": 930.00},
    {"ref_level": "15", "location": "Site A", "pay_zone": "Support", "p25": 1880.00, "p50": 2140.00, "p75": 2460.00, "rfl": "15", "group_min": 1880.00, "group_max": 2780.00, "diff_min_max": 900.00},
    {"ref_level": "15", "location": "Site B", "pay_zone": "Sales", "p25": 2040.00, "p50": 2320.00, "p75": 2670.00, "rfl": "15", "group_min": 2040.00, "group_max": 3020.00, "diff_min_max": 980.00},
    {"ref_level": "15", "location": "Site A", "pay_zone": "R&D", "p25": 2100.00, "p50": 2390.00, "p75": 2750.00, "rfl": "15", "group_min": 2100.00, "group_max": 3110.00, "diff_min_max": 1010.00},

    {"ref_level": "16", "location": "Site A / Site B", "pay_zone": "Ops", "p25": 2180.00, "p50": 2480.00, "p75": 2850.00, "rfl": "16", "group_min": 2180.00, "group_max": 3220.00, "diff_min_max": 1040.00},
    {"ref_level": "16", "location": "Site A", "pay_zone": "Support", "p25": 2110.00, "p50": 2400.00, "p75": 2760.00, "rfl": "16", "group_min": 2110.00, "group_max": 3120.00, "diff_min_max": 1010.00},
    {"ref_level": "16", "location": "Site B", "pay_zone": "Sales", "p25": 2290.00, "p50": 2600.00, "p75": 2990.00, "rfl": "16", "group_min": 2290.00, "group_max": 3380.00, "diff_min_max": 1090.00},
    {"ref_level": "16", "location": "Site A", "pay_zone": "R&D", "p25": 2350.00, "p50": 2670.00, "p75": 3070.00, "rfl": "16", "group_min": 2350.00, "group_max": 3470.00, "diff_min_max": 1120.00},

    {"ref_level": "17", "location": "Site A / Site B", "pay_zone": "Ops", "p25": 2440.00, "p50": 2770.00, "p75": 3190.00, "rfl": "17", "group_min": 2440.00, "group_max": 3600.00, "diff_min_max": 1160.00},
    {"ref_level": "17", "location": "Site A", "pay_zone": "Support", "p25": 2370.00, "p50": 2690.00, "p75": 3090.00, "rfl": "17", "group_min": 2370.00, "group_max": 3500.00, "diff_min_max": 1130.00},
    {"ref_level": "17", "location": "Site B", "pay_zone": "Sales", "p25": 2560.00, "p50": 2910.00, "p75": 3350.00, "rfl": "17", "group_min": 2560.00, "group_max": 3780.00, "diff_min_max": 1220.00},
    {"ref_level": "17", "location": "Site A", "pay_zone": "R&D", "p25": 2630.00, "p50": 2990.00, "p75": 3440.00, "rfl": "17", "group_min": 2630.00, "group_max": 3890.00, "diff_min_max": 1260.00},

    {"ref_level": "18", "location": "Site A / Site B", "pay_zone": "Ops", "p25": 2740.00, "p50": 3110.00, "p75": 3580.00, "rfl": "18", "group_min": 2740.00, "group_max": 4040.00, "diff_min_max": 1300.00},
    {"ref_level": "18", "location": "Site A", "pay_zone": "Support", "p25": 2650.00, "p50": 3010.00, "p75": 3460.00, "rfl": "18", "group_min": 2650.00, "group_max": 3910.00, "diff_min_max": 1260.00},
    {"ref_level": "18", "location": "Site B", "pay_zone": "Sales", "p25": 2870.00, "p50": 3260.00, "p75": 3750.00, "rfl": "18", "group_min": 2870.00, "group_max": 4240.00, "diff_min_max": 1370.00},
    {"ref_level": "18", "location": "Site A", "pay_zone": "R&D", "p25": 2950.00, "p50": 3350.00, "p75": 3850.00, "rfl": "18", "group_min": 2950.00, "group_max": 4360.00, "diff_min_max": 1410.00},

    {"ref_level": "S", "location": "Site B", "pay_zone": "Sales", "p25": 1850.00, "p50": 2100.00, "p75": 2420.00, "rfl": "S", "group_min": 1850.00, "group_max": 2730.00, "diff_min_max": 880.00},
    {"ref_level": "19", "location": "Site A / Site B", "pay_zone": "Executives", "p25": 3410.00, "p50": 3880.00, "p75": 4460.00, "rfl": "19", "group_min": 3410.00, "group_max": 5040.00, "diff_min_max": 1630.00},
    {"ref_level": "20", "location": "Site A / Site B", "pay_zone": "Executives", "p25": 4270.00, "p50": 4850.00, "p75": 5580.00, "rfl": "20", "group_min": 4270.00, "group_max": 6300.00, "diff_min_max": 2030.00},
    {"ref_level": "21", "location": "Site A / Site B", "pay_zone": "Executives", "p25": 5340.00, "p50": 6070.00, "p75": 6980.00, "rfl": "21", "group_min": 5340.00, "group_max": 7890.00, "diff_min_max": 2550.00},
    {"ref_level": "22", "location": "Site A / Site B", "pay_zone": "Executives", "p25": 6670.00, "p50": 7580.00, "p75": 8720.00, "rfl": "22", "group_min": 6670.00, "group_max": 9850.00, "diff_min_max": 3180.00},
]
