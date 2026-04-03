#!/usr/bin/env python3
"""
Enterprise Inventory Simulator — Enhanced Deterministic MILP Production Planner
=========================================================
Enhanced version with overtime, rework, capacity expansion, subcontractors, wastage, and random parameters for variability.
Paste into Google Colab (!pip install pulp numpy) or run locally.
Set USE_GUROBI = True if you have Gurobi.

Every # comment explains the formula. Comments don't affect execution.
"""
import pulp
import math
import numpy as np

USE_GUROBI = False

# Random seed for reproducibility
np.random.seed(42)

# ============================================================
# 1. DATA — Auto-exported from simulator. Update via Export button.
# ============================================================
# This section defines all input parameters, including random values for variability.
# Random yields and costs simulate real-world uncertainty without full stochastic modeling.

# Capacity mode: 'single' or 'shared'
capacity_mode = 'shared'  # Set to 'shared' for parallel/shared capacity

# Products (SKUs) - for shared capacity, add multiple
products = [
    {
        'name': 'Product A',
        'demand': [8,8,9,8,8,8,8,9,8,8,8,8,9,8,8,8,8,9,8,8,8,8,8,8,9,8,8,8,8,9,8,8,8,8,9,8,8,8,8,9,8,8,9,9,8,9,9,9,9,8,9,9,9,9,8,9,9,9,8,9,8,9,9,9,9,10,9,10,9,10,9,10,9,10,9,10,9,10,9,9,10,9,10,9,10,9,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,10,11,10,11,10,10,11,10,11,10,10,10,11,10,10,10,11,10,11,10,11,10,9,9,10,9,9,9,10,9,10,9,9,9,10,9,9,9,9,10,9,9,9,10,9,10,10,9,10,10,10,10,9,10,10,10,10,9,10,10,10,10,9,10,10,10,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,13,12,13,12,12,13,12,13,12,12,13,12,13,12,12,13,12,13,12,13,12,10,10,9,10,10,10,10,9,10,10,10,10,9,10,10,10,10,9,10,10,10,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10],
        'yield_pct': np.random.uniform(0.90, 0.98),
        'material_cost': 27.9,
        'labor_cost': np.random.uniform(4.5, 5.5),
        'sell_price': 100,
        'parts': [
            {"name": "Cream Bun", "qty": 1, "cost": 5.0, "trans": 0.50, "lt": 3, "hold_pct": 24, "partYield": np.random.uniform(0.95, 0.99), "moq": 20, "max_order": 200, "rm_cap": 1000},
            {"name": "Vanilla", "qty": 2, "cost": 10.0, "trans": 1.20, "lt": 5, "hold_pct": 24, "partYield": np.random.uniform(0.90, 0.95), "moq": 50, "max_order": 500, "rm_cap": 2000},
        ]
    },
    {
        'name': 'Product B',
        'demand': [5,5,6,5,5,5,5,6,5,5,5,5,6,5,5,5,5,6,5,5,5,5,5,5,6,5,5,5,5,6,5,5,5,5,6,5,5,5,5,6,5,5,6,6,5,6,6,6,6,5,6,6,6,6,5,6,6,6,5,6,5,6,6,6,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,6,7,6,7,6,7,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,8,7,8,7,7,8,7,8,7,7,7,8,7,7,7,8,7,8,7,8,7,6,6,7,6,6,6,7,6,7,6,6,6,7,6,6,6,6,7,6,6,6,7,6,7,7,6,7,7,7,7,6,7,7,7,7,6,7,7,7,7,6,7,7,7,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,10,9,10,9,9,10,9,10,9,9,10,9,10,9,9,10,9,10,9,10,9,7,7,6,7,7,7,7,6,7,7,7,7,6,7,7,7,7,6,7,7,7,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7,8,7],
        'yield_pct': np.random.uniform(0.85, 0.95),
        'material_cost': 25.0,
        'labor_cost': np.random.uniform(4.0, 5.0),
        'sell_price': 90,
        'parts': [
            {"name": "Cream Bun", "qty": 1, "cost": 5.0, "trans": 0.50, "lt": 3, "hold_pct": 24, "partYield": np.random.uniform(0.95, 0.99), "moq": 20, "max_order": 200, "rm_cap": 1000},
            {"name": "Vanilla", "qty": 1, "cost": 10.0, "trans": 1.20, "lt": 5, "hold_pct": 24, "partYield": np.random.uniform(0.90, 0.95), "moq": 50, "max_order": 500, "rm_cap": 2000},
        ]
    }
    # Add more products for shared capacity
]

n_products = len(products)

# Month selection logic (see below for selected_months)

shelf = 3
# FG expires after 3 days. Forces frequent small batches.

switch_cost = 50
# $ per BATCH START (not per day). Fires when production resumes after idle.

fg_hold_per_day = 0.0216
# Holding cost per unit per day

waste_cost = 5.58   # expired unit cost
fixed_daily = 100    # rent/insurance/staff

backorder_on = False

subcontract_enabled = False  # Set to True to enable subcontracting

# If single SKU, use first product
if capacity_mode == 'single':
    selected_product = products[0]
    demand = selected_product['demand']
    yield_pct = selected_product['yield_pct']
    material_cost = selected_product['material_cost']
    labor_cost = selected_product['labor_cost']
    sell_price = selected_product['sell_price']
    parts = selected_product['parts']
    unit_prod_cost = (material_cost + labor_cost) / yield_pct
    bo_penalty = sell_price if not backorder_on else 30
    selected_product['unit_prod_cost'] = unit_prod_cost
    selected_product['bo_penalty'] = bo_penalty
else:
    # Shared capacity: per product
    demand = {k: products[k]['demand'] for k in range(n_products)}
    for prod in products:
        prod['unit_prod_cost'] = (prod['material_cost'] + prod['labor_cost']) / prod['yield_pct']
        prod['bo_penalty'] = prod['sell_price'] if not backorder_on else 30
    # Assume parts from first product
    parts = products[0]['parts']
    material_cost = products[0]['material_cost']  # For subcontract
    labor_cost = products[0]['labor_cost']  # For subcontract

on_hand_fg = 0
fg_ss = 6
ss_penalty = 0.065
wh_max = 500

# Month selection logic (for scenario testing)
month_days = {'Jan':22, 'Feb':20, 'Mar':23, 'Apr':22, 'May':23, 'Jun':20, 'Jul':23, 'Aug':22, 'Sep':23, 'Oct':22, 'Nov':21, 'Dec':20}
selected_months = ['Mar']  # Change this list to select months, e.g., ['Jun', 'Jul', 'Aug']

cum_days = 0
start_day = 0
for m in month_days:
    if m in selected_months:
        start_day = cum_days
        break
    cum_days += month_days[m]
end_day = start_day + sum(month_days[m] for m in selected_months)
T = sum(month_days[m] for m in selected_months)

# For shared capacity, slice each product's demand
if capacity_mode == 'shared':
    for k in range(n_products):
        products[k]['demand'] = products[k]['demand'][start_day:end_day]
    demand = {k: products[k]['demand'] for k in range(n_products)}
else:
    demand = products[0]['demand'][start_day:end_day]

capacity = [21]*261
capacity = capacity[start_day:end_day]

# Override planning horizon to exactly 2 weeks (14 days) as requested
T = 14
if capacity_mode == 'single':
    if len(demand) < T:
        raise ValueError(f"Not enough demand data for {T} days")
    demand = demand[:T]
else:
    for k in range(n_products):
        if len(demand[k]) < T:
            raise ValueError(f"Not enough demand data for product {k} for {T} days")
        demand[k] = demand[k][:T]
capacity = capacity[:T]

# Overtime parameters
base_labor_rate = 20  # $/hour
overtime_multiplier = 1.5  # 50% higher for OT hours
shift_hours = 8
break_minutes = 50  # Total breaks
effective_hours = shift_hours - break_minutes / 60
overtime_cost_per_hour = base_labor_rate * overtime_multiplier
productivity_per_hour = 5  # Units per hour

# Rework parameters
# (removed from model: yield is already captured in production and raw material requirements)
# rework_cost_per_unit = 2.0  # $ per reworked unit
# rework_yield_loss = 0.05  # 5% loss in rework

# Capacity expansion
expansion_cost = 1000  # $ per expansion
expansion_increment = 5  # Units added

# Subcontractor costs
# Subcontractor uses same material cost base but slightly higher transport/labor than in-house.
subcontract_enabled = False  # default: OFF (set True to enable subcontracting)
subcontract_material_cost = material_cost
subcontract_labor_cost = labor_cost * 1.08  # e.g., 8% premium on labor
subcontract_transport_cost = 1.50  # additional transport/premium per unit
subcontract_cost_per_unit = (subcontract_material_cost
                             + subcontract_labor_cost
                             + subcontract_transport_cost)

# Ordering costs
ordering_fixed_cost = 30  # fixed cost per PO placement
ordering_unit_cost_multiplier = 1.0  # multiplier for part unit cost in purchase cost (1x by default)

# Wastage
wastage_rate = np.random.uniform(0.01, 0.05)  # Random wastage 1-5%

parts = [
        {"name": "Cream Bun", "qty": 1, "cost": 5.0, "trans": 0.50,
             "lt": 3, "hold_pct": 24, "partYield": np.random.uniform(0.95, 0.99),
                  "moq": 20, "max_order": 200, "rm_cap": 1000},
                      {"name": "Vanilla", "qty": 2, "cost": 10.0, "trans": 1.20,
                           "lt": 5, "hold_pct": 24, "partYield": np.random.uniform(0.90, 0.95),
                                "moq": 50, "max_order": 500, "rm_cap": 2000},
]

n_products = len(products)

# Overtime parameters: Allows extra production at higher cost.
# Rework: Fixes defects, recovers some value.
# Capacity expansion: Increases production capacity permanently.
# Subcontractor: Outsourced production.
# Wastage: Unavoidable losses in production.

# ============================================================
# 2. DECISION VARIABLES
# ============================================================
# Defines all variables the model can control to minimize cost.
model = pulp.LpProblem("MPS", pulp.LpMinimize)
days = list(range(T))
n_parts = len(parts)
max_lt = max(pp["lt"] for pp in parts)

p = {(k, t): pulp.LpVariable(f"produce_{k}_{t}", 0, cat="Integer") for k in range(n_products) for t in days}
# p[k,t] = FG units of product k to produce on day t (integer — can't make half a cake)

if capacity_mode == 'parallel':
    y = {(k, t): pulp.LpVariable(f"active_{k}_{t}", cat="Binary") for k in range(n_products) for t in days}
    sw = {(k, t): pulp.LpVariable(f"switch_{k}_{t}", cat="Binary") for k in range(n_products) for t in days}
else:
    y = {t: pulp.LpVariable(f"active_{t}", cat="Binary") for t in days}
    sw = {t: pulp.LpVariable(f"switch_{t}", cat="Binary") for t in days}
# y[t] or y[k,t] = 1 if line runs on day t. Links to capacity: p[t] ≤ C[t]×y[t]

# sw[t] or sw[k,t] = 1 if STARTING a batch (idle yesterday, active today)
# sw[t] ≥ y[t] - y[t-1]: fires only on transition from 0→1

Inv = {(k, t): pulp.LpVariable(f"fg_inv_{k}_{t}", 0) for k in range(n_products) for t in range(T + 1)}
# Inv[k,t] = FG inventory of product k at END of day t. Inv[k,0] = starting inventory.

e = {(k, t): pulp.LpVariable(f"expired_{k}_{t}", 0) for k in range(n_products) for t in days}
# e[k,t] = units of product k expiring on day t (age > shelf). Forced by constraint.

s = {(k, t): pulp.LpVariable(f"shortage_{k}_{t}", 0) for k in range(n_products) for t in days}
# s[k,t] = unmet demand of product k. Backorder ON: carries to t+1. OFF: forced to 0.

ss_viol = {(k, t): pulp.LpVariable(f"ss_viol_{k}_{t}", 0) for k in range(n_products) for t in days}
# ss_viol[k,t] = how far below SS for product k. Penalized but not forbidden.

rng_ext = range(-max_lt, T)
r = {(i, t): pulp.LpVariable(f"order_{i}_{t}", 0, cat="Integer")
     for i in range(n_parts) for t in rng_ext}
# r[i,t] = RM units of subpart i ordered on day t. Arrives t + LT[i].

RI = {(i, t): pulp.LpVariable(f"rm_inv_{i}_{t}", 0)
     for i in range(n_parts) for t in range(T + 1)}
# RI[i,t] = RM inventory of subpart i at end of day t.

zo = {(i, t): pulp.LpVariable(f"po_flag_{i}_{t}", cat="Binary")
     for i in range(n_parts) for t in rng_ext}
# zo[i,t] = 1 if PO placed for subpart i on day t. Links to MOQ.

# New variables: Overtime
ot = {t: pulp.LpVariable(f"overtime_{t}", 0, cat="Integer") for t in days}
# ot[t] = overtime hours on day t

# Capacity expansion
exp = {t: pulp.LpVariable(f"expand_{t}", cat="Binary") for t in days}
# exp[t] = 1 if capacity expanded on day t

# Subcontractors
sub = {(k, t): pulp.LpVariable(f"subcontract_{k}_{t}", 0, cat="Integer") for k in range(n_products) for t in days}
# sub[k,t] = units of product k subcontracted on day t

# Wastage
wst = {(k, t): pulp.LpVariable(f"wastage_{k}_{t}", 0) for k in range(n_products) for t in days}
# wst[k,t] = wasted units of product k on day t

# ============================================================
# 3. INITIAL CONDITIONS
# ============================================================
# Sets starting inventory levels for FG and RM.

# ============================================================
# 3. INITIAL CONDITIONS
# ============================================================
model += pulp.lpSum(Inv[k, 0] for k in range(n_products)) == on_hand_fg, "init_fg"
for i in range(n_parts):
    model += RI[i, 0] == 0, f"init_rm_{i}"

# ============================================================
# 4. OBJECTIVE — Minimize total cost
# ============================================================
# The goal: Sum of all costs over the planning horizon.
obj = pulp.lpSum([
    switch_cost * (pulp.lpSum(sw[k, t] for k in range(n_products)) if capacity_mode == 'parallel' else sw[t])           # Setup on batch start only
    + fg_hold_per_day * pulp.lpSum(Inv[k, t+1] for k in range(n_products))  # FG holding
    + pulp.lpSum(products[k]['unit_prod_cost'] * p[k, t] for k in range(n_products))  # Production (yield-adjusted)
    + waste_cost * pulp.lpSum(e[k, t] for k in range(n_products))  # Expired units
    + pulp.lpSum(products[k]['bo_penalty'] * s[k, t] for k in range(n_products))  # Shortage (backorder or lost sale)
    + ss_penalty * pulp.lpSum(ss_viol[k, t] for k in range(n_products))  # Dipping below safety stock
    + fixed_daily                 # Fixed overhead
    + overtime_cost_per_hour * ot[t]  # Overtime cost
    + expansion_cost * exp[t]          # Expansion cost
    + (subcontract_cost_per_unit * pulp.lpSum(sub[k, t] for k in range(n_products)) if subcontract_enabled else 0)  # Subcontract cost
    + material_cost * pulp.lpSum(wst[k, t] for k in range(n_products))  # Wastage cost (material loss)
    for t in days
])
for i in range(n_parts):
    rm_h = parts[i]["cost"] * parts[i]["hold_pct"] / 100 / 365
    obj += pulp.lpSum([rm_h * RI[i, t+1] for t in days])  # RM holding

    # Purchase and ordering cost for RM
    obj += pulp.lpSum([
        parts[i]["cost"] * ordering_unit_cost_multiplier * r[i, t] + ordering_fixed_cost * zo[i, t]
        for t in rng_ext
    ])
model += obj, "Total_Cost"

# ============================================================
# 5. CONSTRAINTS
# ============================================================
# Rules the model must follow, ensuring feasibility.
for t in days:
    # Per product constraints
    for k in range(n_products):
        # C1: FG FLOW BALANCE (with wastage, subcontracting)
        if subcontract_enabled:
            if backorder_on and t > 0:
                model += Inv[k, t+1] == Inv[k, t] + p[k, t] * products[k]['yield_pct'] - demand[k][t] - s[k, t-1] - e[k, t] + s[k, t] - wst[k, t] + sub[k, t], f"flow_{k}_{t}"
            else:
                model += Inv[k, t+1] == Inv[k, t] + p[k, t] * products[k]['yield_pct'] - demand[k][t] - e[k, t] + s[k, t] - wst[k, t] + sub[k, t], f"flow_{k}_{t}"
        else:
            if backorder_on and t > 0:
                model += Inv[k, t+1] == Inv[k, t] + p[k, t] * products[k]['yield_pct'] - demand[k][t] - s[k, t-1] - e[k, t] + s[k, t] - wst[k, t], f"flow_{k}_{t}"
            else:
                model += Inv[k, t+1] == Inv[k, t] + p[k, t] * products[k]['yield_pct'] - demand[k][t] - e[k, t] + s[k, t] - wst[k, t], f"flow_{k}_{t}"

        # C5: SAFETY STOCK (soft)
        model += ss_viol[k, t] >= fg_ss - Inv[k, t+1], f"ss_{k}_{t}"

        # C6: NO BACKORDER (if disabled)
        if not backorder_on:
            model += s[k, t] == 0, f"nobo_{k}_{t}"

        # New: Wastage
        model += wst[k, t] >= wastage_rate * p[k, t], f"wst_min_{k}_{t}"
        model += wst[k, t] <= (wastage_rate + 0.05) * p[k, t], f"wst_max_{k}_{t}"

        # New: Subcontract limit (disabled by default)
        if subcontract_enabled:
            model += sub[k, t] <= 10, f"sub_limit_{k}_{t}"  # Max 10 units subcontract per day
        else:
            model += sub[k, t] == 0, f"sub_off_{k}_{t}"

    # Shared constraints
    # C2: CAPACITY (with overtime and expansion)
    if capacity_mode == 'parallel':
        for k in range(n_products):
            model += p[k, t] <= capacity[t] * y[k, t] + expansion_increment * exp[t] + ot[t] * productivity_per_hour / n_products, f"cap_{k}_{t}"
    else:
        model += pulp.lpSum(p[k, t] for k in range(n_products)) <= capacity[t] * y[t] + expansion_increment * exp[t] + ot[t] * productivity_per_hour, f"cap_{t}"

    # C3: BATCH SWITCH (setup fires on START only)
    if capacity_mode == 'parallel':
        for k in range(n_products):
            model += sw[k, t] >= y[k, t] - (y[k, t-1] if t > 0 else 0), f"sw_{k}_{t}"
    else:
        model += sw[t] >= y[t] - (y[t-1] if t > 0 else 0), f"sw_{t}"

    # C4: WAREHOUSE MAX
    model += pulp.lpSum(Inv[k, t+1] for k in range(n_products)) <= wh_max, f"wh_{t}"

    # New: Overtime limit
    model += ot[t] <= 4, f"ot_limit_{t}"  # Max 4 hours OT

    # Per product shelf life
    for k in range(n_products):
        if t >= shelf:
            model += (
                pulp.lpSum([e[k, tau] for tau in range(max(0, t-shelf+1), t+1)])
                >= pulp.lpSum([p[k, tau] * products[k]['yield_pct'] for tau in range(max(0, t-2*shelf+1), t-shelf+1)])
                - pulp.lpSum([demand[k][tau] for tau in range(max(0, t-2*shelf+1), t-shelf+1)])
            ), f"shelf_{k}_{t}"

    # C8-C10: PER-SUBPART
    for i in range(n_parts):
        lt = parts[i]["lt"]
        od = t - lt
        # C8: BOM EXPLOSION
        if -max_lt <= od < T:
            model += r[i, od] * parts[i]["partYield"] >= pulp.lpSum(p[k, t] * parts[i]["qty"] for k in range(n_products)), f"bom_{i}_{t}"
            # C9: RM FLOW BALANCE (with wastage)
            if t > 0:
                model += RI[i, t] == RI[i, t-1] + r[i, od] - pulp.lpSum(p[k, t] * parts[i]["qty"] for k in range(n_products)) - pulp.lpSum(wst[k, t] * parts[i]["qty"] / products[k]['yield_pct'] for k in range(n_products)), f"rm_{i}_{t}"
                # C10: RM WAREHOUSE
                model += RI[i, t] <= parts[i]["rm_cap"], f"rmcap_{i}_{t}"

# C11-C12: MOQ & SUPPLIER MAX
for i in range(n_parts):
    for t in rng_ext:
        model += r[i, t] >= parts[i]["moq"] * zo[i, t], f"moq_{i}_{t}"
        model += r[i, t] <= parts[i]["max_order"] * zo[i, t], f"mxo_{i}_{t}"

# ============================================================
# 6. SOLVE & OUTPUT
# ============================================================
# Runs the optimization and prints results.
print("Solving... (1-2 minutes)")
model.solve(pulp.PULP_CBC_CMD(msg=1, timeLimit=180))
print(f"\nStatus: {pulp.LpStatus[model.status]}")
print(f"Total Cost: ${pulp.value(model.objective):,.0f}\n")

print(f"{'Day':>4} {'Dem':>5} {'Prod':>5} {'Inv':>5} {'Exp':>5} {'Short':>5} {'SW':>3} {'OT':>3} {'RW':>3} {'SUB':>3} {'WST':>3}")
print("-" * 58)
tp = te = ts = nsw = tot = tsub = twst = 0
for t in days:
    dem_t = sum(demand[k][t] for k in range(n_products))
    pv = sum(int(p[k, t].varValue or 0) for k in range(n_products))
    iv = sum(Inv[k, t+1].varValue or 0 for k in range(n_products))
    ev = sum(e[k, t].varValue or 0 for k in range(n_products))
    sv = sum(s[k, t].varValue or 0 for k in range(n_products))
    if capacity_mode == 'parallel':
        swv = sum(1 if sw[k, t].varValue > 0.5 else 0 for k in range(n_products))
    else:
        swv = 1 if (sw[t].varValue or 0) > 0.5 else 0
    otv = int(ot[t].varValue or 0)
    subv = sum(int(sub[k, t].varValue or 0) for k in range(n_products))
    wstv = sum(wst[k, t].varValue or 0 for k in range(n_products))
    tp += pv; te += ev; ts += sv; nsw += swv; tot += otv; tsub += subv; twst += wstv
    flag = ""
    if ev > 0.5: flag = " EXP"
    if sv > 0.5: flag = " SHORT"
    if t < 60 or pv > 0 or ev > 0.5 or sv > 0.5 or otv > 0 or subv > 0 or wstv > 0.5:
        print(f"{t+1:>4} {dem_t:>5} {pv:>5} {iv:>5.0f} {ev:>5.0f} {sv:>5.0f} {'Y' if swv else '':>3} {otv:>3}    {subv:>3} {wstv:>3.0f}{flag}")

total_demand = sum(sum(demand[k]) for k in range(n_products))
fill = (total_demand - ts) / total_demand * 100
print(f"\nProduced:{tp} Expired:{te:.0f} Short:{ts:.0f} Switches:{nsw} OT Hours:{tot} Sub:{tsub} Wastage:{twst:.0f} Fill:{fill:.1f}%")

# RM order summary with cost
rm_purchase_cost = 0
rm_ordering_cost = 0
for i in range(n_parts):
    for t in rng_ext:
        rv = r[i, t].varValue or 0
        rm_purchase_cost += parts[i]["cost"] * ordering_unit_cost_multiplier * rv
        if (zo[i, t].varValue or 0) > 0.5:
            rm_ordering_cost += ordering_fixed_cost

print(f"  RM purchase cost: ${rm_purchase_cost:,.2f}")
print(f"  RM ordering (PO) cost: ${rm_ordering_cost:,.2f}")
print(f"  RM combined purchase+order: ${rm_purchase_cost + rm_ordering_cost:,.2f}\n")

print(f"\nRM ORDERS:")
for i in range(n_parts):
    nm = parts[i]["name"]
    print(f"  {nm}:")
    tq = npo = 0
    for t in rng_ext:
        rv = r[i, t].varValue
        if rv and rv > 0.5:
            lt2 = parts[i]["lt"]
            print(f"    PO day {t+1:>4} qty {int(rv):>5} arrives day {t+lt2+1:>4}")
            tq += int(rv); npo += 1
    print(f"    Total: {npo} POs, {tq} units")

import json
mps_plan = {products[k]['name']: [int(p[k, t].varValue or 0) for t in days] for k in range(n_products)}

# Collect RM procurement
procurement = {}
for i in range(n_parts):
    nm = parts[i]["name"]
    procurement[nm] = []
    for t in rng_ext:
        rv = r[i, t].varValue or 0
        if rv > 0.5:
            lt2 = parts[i]["lt"]
            procurement[nm].append({"day": t+1, "qty": int(rv), "arrives": t+lt2+1})

print(f"\n--- JSON for simulator ---")
print(json.dumps({"source":"MILP","cost":round(pulp.value(model.objective),2),"mps_plan":mps_plan,"procurement":procurement,"fill":round(fill,1)},indent=2))
