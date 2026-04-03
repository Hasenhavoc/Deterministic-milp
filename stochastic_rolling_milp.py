#!/usr/bin/env python3
"""
Stochastic Rolling Horizon MILP Production Planner
=========================================================
Uses rolling horizon (30-day planning window) with stochastic scenarios for demand, yields, etc.
Re-plans every period, incorporating new realizations.
Requires pulp, numpy.
"""

import pulp
import numpy as np
import random

# Parameters
T_total = 261
horizon = 14  # Changed to 14 days as in deterministic
n_scenarios = 5  # Scenarios per planning step

# Month selection for T_total
month_days = {'Jan':22, 'Feb':20, 'Mar':23, 'Apr':22, 'May':23, 'Jun':20, 'Jul':23, 'Aug':22, 'Sep':23, 'Oct':22, 'Nov':21, 'Dec':20}
selected_months = ['Mar']  # Changed to single month for 14 days

cum_days = 0
start_day = 0
for m in month_days:
    if m in selected_months:
        start_day = cum_days
        break
    cum_days += month_days[m]
end_day = start_day + sum(month_days[m] for m in selected_months)
T_total = sum(month_days[m] for m in selected_months)

# ============================================================
# BASE DATA
# ============================================================
# Static inputs, similar to deterministic model.
shelf = 3
switch_cost = 50
fg_hold_per_day = 0.0216
material_cost = 27.9
labor_cost = 5.0
waste_cost = 5.58
fixed_daily = 100
sell_price = 100
backorder_on = False
bo_penalty = 30 if backorder_on else sell_price
on_hand_fg = 0
fg_ss = 6
ss_penalty = 0.065
wh_max = 500
capacity = [21] * T_total

# Subcontractor costs
subcontract_enabled = False  # Default OFF
subcontract_material_cost = material_cost
subcontract_labor_cost = labor_cost * 1.08
subcontract_transport_cost = 1.50
subcontract_cost_per_unit = (subcontract_material_cost + subcontract_labor_cost + subcontract_transport_cost)

# Ordering costs
ordering_fixed_cost = 30
ordering_unit_cost_multiplier = 1.0

demand_base = [8,8,9,8,8,8,8,9,8,8,8,8,9,8,8,8,8,9,8,8,8,8,8,8,9,8,8,8,8,9,8,8,8,8,9,8,8,8,8,9,8,8,9,9,8,9,9,9,9,8,9,9,9,9,8,9,9,9,8,9,8,9,9,9,9,10,9,10,9,10,9,10,9,10,9,10,9,10,9,9,10,9,10,9,10,9,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,10,11,10,11,10,10,11,10,11,10,10,10,11,10,10,10,11,10,11,10,11,10,9,9,10,9,9,9,10,9,10,9,9,9,10,9,9,9,9,10,9,9,9,10,9,10,10,9,10,10,10,10,9,10,10,10,10,9,10,10,10,10,9,10,10,10,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,11,12,13,12,13,12,12,13,12,13,12,12,13,12,13,12,12,13,12,13,12,13,12,10,10,9,10,10,10,10,9,10,10,10,10,9,10,10,10,10,9,10,10,10,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10]
demand_base = demand_base[start_day:end_day]  # Slice to selected months
parts = [
    {"name": "Cream Bun", "qty": 1, "cost": 5.0, "trans": 0.50, "lt": 3, "hold_pct": 24, "partYield": 0.98, "moq": 20, "max_order": 200, "rm_cap": 1000},
    {"name": "Vanilla", "qty": 2, "cost": 10.0, "trans": 1.20, "lt": 5, "hold_pct": 24, "partYield": 0.92, "moq": 50, "max_order": 500, "rm_cap": 2000},
]

# Function to generate stochastic demand
def generate_demand(t, base_demand):
    return max(0, int(np.random.normal(base_demand[t], base_demand[t] * 0.1)))  # 10% CV

# Function to generate stochastic yield
def generate_yield(base_yield):
    return np.random.uniform(max(0.8, base_yield - 0.05), min(1.0, base_yield + 0.05))

# ============================================================
# ROLLING HORIZON SIMULATION
# ============================================================
# Loops over time, planning, solving, and rolling forward.
current_time = 0
total_cost = 0
production_plan = []
inventory_history = [on_hand_fg]

while current_time < T_total:
    # Plan for next horizon
    T_plan = min(horizon, T_total - current_time)
    days_plan = list(range(T_plan))
    
    # ============================================================
    # GENERATE SCENARIOS
    # ============================================================
    # Creates multiple random scenarios for uncertainty.
    scenarios = []
    for s in range(n_scenarios):
        dem_scen = [generate_demand(current_time + t, demand_base) for t in days_plan]
        yield_scen = generate_yield(0.95)
        part_yields_scen = [generate_yield(p["partYield"]) for p in parts]
        scenarios.append({"demand": dem_scen, "yield": yield_scen, "part_yields": part_yields_scen})
    
    # Average scenario for deterministic approx
    avg_demand = [sum(s["demand"][t] for s in scenarios) / n_scenarios for t in days_plan]
    avg_yield = sum(s["yield"] for s in scenarios) / n_scenarios
    avg_part_yields = [sum(s["part_yields"][i] for s in scenarios) / n_scenarios for i in range(len(parts))]
    
    # Build model for planning horizon
    model = pulp.LpProblem("Rolling_MILP", pulp.LpMinimize)
    n_parts = len(parts)
    max_lt = max(p["lt"] for p in parts)
    rng_ext = range(-max_lt, T_plan)
    
    # ============================================================
    # DECISION VARIABLES
    # ============================================================
    # Same as deterministic, but for shorter horizon.
    p = {t: pulp.LpVariable(f"produce_{t}", 0, cat="Integer") for t in days_plan}
    y = {t: pulp.LpVariable(f"active_{t}", cat="Binary") for t in days_plan}
    sw = {t: pulp.LpVariable(f"switch_{t}", cat="Binary") for t in days_plan}
    Inv = {t: pulp.LpVariable(f"fg_inv_{t}", 0) for t in range(T_plan + 1)}
    e = {t: pulp.LpVariable(f"expired_{t}", 0) for t in days_plan}
    s = {t: pulp.LpVariable(f"shortage_{t}", 0) for t in days_plan}
    ss_viol = {t: pulp.LpVariable(f"ss_viol_{t}", 0) for t in days_plan}
    r = {(i, t): pulp.LpVariable(f"order_{i}_{t}", 0, cat="Integer") for i in range(n_parts) for t in rng_ext}
    RI = {(i, t): pulp.LpVariable(f"rm_inv_{i}_{t}", 0) for i in range(n_parts) for t in range(T_plan + 1)}
    zo = {(i, t): pulp.LpVariable(f"po_flag_{i}_{t}", cat="Binary") for i in range(n_parts) for t in rng_ext}
    sub = {t: pulp.LpVariable(f"subcontract_{t}", 0, cat="Integer") for t in days_plan}  # Added subcontract
    
    # Initial conditions
    model += Inv[0] == inventory_history[-1]
    for i in range(n_parts):
        model += RI[i, 0] == 0  # Assume RM starts at 0 for simplicity
    
    # Objective
    unit_prod_cost = (material_cost + labor_cost) / avg_yield
    obj = pulp.lpSum([
        switch_cost * sw[t] + fg_hold_per_day * Inv[t+1] + unit_prod_cost * p[t] + waste_cost * e[t] + bo_penalty * s[t] + ss_penalty * ss_viol[t] + fixed_daily + (subcontract_cost_per_unit * sub[t] if subcontract_enabled else 0)
        for t in days_plan
    ])
    for i in range(n_parts):
        rm_h = parts[i]["cost"] * parts[i]["hold_pct"] / 100 / 365
        obj += pulp.lpSum([rm_h * RI[i, t+1] for t in days_plan])
        obj += pulp.lpSum([
            parts[i]["cost"] * ordering_unit_cost_multiplier * r[i, t] + ordering_fixed_cost * zo[i, t]
            for t in rng_ext
        ])
    model += obj
    
    # Constraints
    for t in days_plan:
        # Flow
        if subcontract_enabled:
            if backorder_on and t > 0:
                model += Inv[t+1] == Inv[t] + p[t]*avg_yield - avg_demand[t] - s[t-1] - e[t] + s[t] + sub[t]
            else:
                model += Inv[t+1] == Inv[t] + p[t]*avg_yield - avg_demand[t] - e[t] + s[t] + sub[t]
        else:
            if backorder_on and t > 0:
                model += Inv[t+1] == Inv[t] + p[t]*avg_yield - avg_demand[t] - s[t-1] - e[t] + s[t]
            else:
                model += Inv[t+1] == Inv[t] + p[t]*avg_yield - avg_demand[t] - e[t] + s[t]
        
        model += p[t] <= capacity[current_time + t] * y[t]
        model += sw[t] >= y[t] - (y[t-1] if t > 0 else 0)
        model += Inv[t+1] <= wh_max
        model += ss_viol[t] >= fg_ss - Inv[t+1]
        if not backorder_on:
            model += s[t] == 0
        
        # Subcontract limit
        if subcontract_enabled:
            model += sub[t] <= 10
        else:
            model += sub[t] == 0
        
        # Shelf life (simplified)
        if t >= shelf:
            model += e[t] >= p[max(0, t-shelf)] * avg_yield - avg_demand[t]
        
        for i in range(n_parts):
            lt = parts[i]["lt"]
            od = t - lt
            if -max_lt <= od < T_plan:
                model += r[i, od] * avg_part_yields[i] >= p[t] * parts[i]["qty"]
            if t > 0:
                incoming = r[i, od] if -max_lt <= od < T_plan else 0
                model += RI[i, t] == RI[i, t-1] + incoming - p[t] * parts[i]["qty"]
            model += RI[i, t] <= parts[i]["rm_cap"]
    
    for i in range(n_parts):
        for t in rng_ext:
            model += r[i, t] >= parts[i]["moq"] * zo[i, t]
            model += r[i, t] <= parts[i]["max_order"] * zo[i, t]
    
    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    if pulp.LpStatus[model.status] != "Optimal":
        print(f"Warning: No optimal solution at time {current_time}")
        break
    
    # ============================================================
    # IMPLEMENT AND ROLL
    # ============================================================
    # Use first day's plan, update state, repeat.
    prod_today = int(p[0].varValue or 0)
    inv_today = Inv[1].varValue or 0
    production_plan.append(prod_today)
    inventory_history.append(inv_today)
    total_cost += pulp.value(model.objective) / T_plan  # Approximate
    
    # Roll forward
    current_time += 1

print(f"Total Cost: ${total_cost:,.0f}")
print(f"Production Plan (first 50 days): {production_plan[:50]}")

# RM order summary (approximate, since rolling)
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

print("Stochastic Rolling MILP completed.")