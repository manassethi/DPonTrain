# DPonTrain: Dynamic Pricing for Indian Railways

**Airline-style revenue management applied to a long-distance passenger train.** Using about 736,000 real booking records for train 12138, the Punjab Mail (Firozpur → Mumbai CSMT), this project:

- forecasts demand per departure, before and during the booking window;
- builds a booking simulator calibrated to how passengers actually book;
- compares pricing methods on unseen departures;
- extends pricing to the train's network of origin–destination journeys with bid prices.

> **Headline:** classic **forecast + optimize** pricing (dynamic programming) earns **+2% to +14% more revenue per departure** than a fixed fare in simulation. That is 95–100% of the theoretical best. Reinforcement learning (Q-learning) captures only 14–76%. On the multi-stop train, **network bid prices add up to +8.5%** over pricing each journey separately. **Valuing each passenger carried** keeps ridership at today's level while still earning **+5.5% to +9.2%**.

---

## At a glance

| | |
|---|---|
| **Train** | 12138 Punjab Mail, Firozpur Cantt → Mumbai CSMT, 52 stops, 1,930 km |
| **Data** | About 736,000 tickets (1.9 million passengers) for 730 daily departures, Aug 2023 – Jul 2025; focus on 3A (AC 3-tier) |
| **Forecasting** | Baselines, Ridge regression, gradient boosting, calendar blend, pickup forecasting; time-based validation |
| **Pricing** | Fixed fare, pacing rule, Q-learning, dynamic-programming optimizer (with Bayesian re-forecasting), perfect-information ceiling |
| **Network** | 36 origin–destination journeys over 8 legs; LP and dynamic-programming bid prices |
| **Stack** | Python, pandas, NumPy, SciPy (Poisson, `linprog` / HiGHS), scikit-learn, Matplotlib, pytest |

---

## What the data shows (3A class)

- **Weekends are busiest.** Saturdays average about 1,050 passengers vs about 900 on Thursdays, roughly 16% more.
- **Early summer is peak season.** May–June average about 1,025 passengers per day, vs about 845 in October.
- **The pattern repeats.** Daily demand in year one and year two correlates at 0.60.
- **One journey dominates.** Firozpur → New Delhi carries about 13% of 3A passengers.
- **Passengers book late.** 45% book within 2 days of departure (much of it Tatkal, the day before) and 56% within a week. The advance booking window shrank from 120 to 60 days in November 2024.

---

## 1. Demand forecasting: [`notebooks/01_demand_forecasting.ipynb`](notebooks/01_demand_forecasting.ipynb)

- **Target:** total 3A passengers per departure.
- **Features:** only information known when booking opens, about 60 days ahead:
  - weekday and cyclical seasonality;
  - festival windows (Diwali, Holi, Raksha Bandhan, Eid, Christmas) and long weekends;
  - same weekday last year;
  - the level of departures 61–90 days earlier;
  - a trend term.
- **Evaluation:** time-based. Train on Aug 2023 – Jan 2025 and score on **Feb – Jul 2025, never seen**. The method was chosen on a separate validation window (Nov 2024 – Jan 2025).

| Forecast (before booking opens) | Error (MAPE) | R² |
|---|---|---|
| Training average | 8.0% | −0.02 |
| Weekday average | 7.2% | 0.20 |
| Same weekday last year | 7.2% | 0.33 |
| Ridge regression | 7.3% | 0.27 |
| Gradient boosting | 7.5% | 0.17 |
| **Blend: weekday average + last year** | **6.3%** | **0.48** |

**During the booking window**, a **pickup forecast** (a regression on bookings already on hand plus the calendar blend) sharpens the estimate:

| Days before departure | 60 | 30 | 14 | 7 | 2 | 1 |
|---|---|---|---|---|---|---|
| Passengers already booked | 5% | 21% | 34% | 46% | 58% | 85% |
| Forecast error (MAPE) | 6.4% | **4.2%** | 4.4% | 4.5% | **3.0%** | 3.0% |

- **Simple beats complex here.** Aggregated per departure, two years give only about 550 training points, and the main lag feature exists for only half of them, so gradient boosting overfits. Averaging two simple rules cancels part of each one's error.
- **Time-based testing matters.** A random 80/20 split would report R² 0.64 for gradient boosting, against 0.17 on genuinely future months.

---

## 2. Pricing engine: [`notebooks/02_pricing_engine.ipynb`](notebooks/02_pricing_engine.ipynb)

### The simulator
One 90-seat 3A coach, a ₹1,100 base fare, a 60-day booking window, and six fare levels from −15% to +30%. Each day:

> **expected bookings λ = D₀ × demand level × booking-curve share × (fare multiple)^(−elasticity)**,
> and actual bookings ~ **Poisson(λ)**, capped at the seats left.

- **From data:** the booking curve, and each departure's demand level.
- **Assumed** (fares never changed in the data, so these are tested as scenarios):
  - demand vs capacity, D₀: 75 or 95 bookings per 90 seats;
  - price elasticity: 0.8 or 1.2, and 0.6–2.0 in a sweep.
- **Testing:** all **365 unseen departures** (Aug 2024 – Jul 2025), 20 booking windows each, with **identical random customers for every policy** (common random numbers) and paired 95% confidence intervals (±0.1–0.2 points).

### The optimizer
The **dynamic program** works backwards from departure:

> V(d, s) = max over fares of E[ fare × bookings + V(d − 1, s − bookings) ]

It gives the best fare for every combination of days left (d) and seats left (s). Each day, the pricer updates its demand estimate from bookings so far (a Gamma–Poisson Bayesian update) and looks up the fare.

### Results: revenue gain over a fixed fare

| Scenario | Pacing rule | Q-learning | **Forecast + optimizer** | Perfect-information ceiling |
|---|---|---|---|---|
| Price-sensitive passengers, spare seats | +0.7% | +0.3% | **+2.1%** | +2.1% |
| Price-sensitive, oversubscribed | +2.9% | +4.2% | **+6.6%** | +6.8% |
| Less price-sensitive, spare seats | +0.8% | +4.4% | **+6.4%** | +6.4% |
| Less price-sensitive, oversubscribed | +3.0% | +10.6% | **+14.0%** | +14.0% |

"Price-sensitive" = elasticity 1.2: a 10% fare rise loses about 11% of bookings. "Less price-sensitive" = 0.8: the same rise loses about 7%.

- **Forecast + optimize captures 95–100%** of what perfect knowledge of demand would allow. Each fare traces back to a forecast and a seats-left table, so a revenue manager can inspect and override it.
- **Re-forecasting from bookings is insurance.** It adds little normally. Under ±25% demand surprises it recovers part of the gap to the ceiling: +10.2% vs +9.8% without it, against a ceiling of +10.9%.
- **Dynamic pricing pays in two ways** (a 36-scenario sensitivity map):
  - raise fares on **busy trains with less price-sensitive passengers**: +14% to +27%, but fewer passengers;
  - discount **quiet trains with price-sensitive passengers**: +2% to +13%, and **more** passengers.

### Why Q-learning falls short
Q-learning captures only 14–76% of the achievable gain:

1. **It learns by trial and error what the optimizer calculates.** The dynamic program uses the demand model directly and is exact. Model-free Q-learning has to estimate the value of each fare from simulated outcomes, so at best it can match the optimizer.
2. **Its view is coarse.** 441 states (buckets of days left, seats left, booking pace and forecast) can't tell 16 seats left from 29, or day 8 from day 13, which are exactly the distinctions that matter near departure.
3. **Rewards are noisy and delayed.** Bookings are random, and 45% arrive in the last two days, so an early fare's effect appears weeks later, mixed with luck. Even 60,000 simulated departures leave rare but important states poorly learned.
4. **Demand is only partly visible.** The agent sees a three-level forecast bucket, while true demand varies continuously.
5. **Real-world learning is costly.** Training on live departures would mean charging random fares to real passengers, and the result is a hard-to-explain lookup table.

Reinforcement learning pays off when demand *can't* be modelled reliably (competitor reactions, choice across many trains and classes, passengers who wait for lower fares). Here it can, so optimizing the model directly is both better and explainable.

### Fairness: revenue vs passengers carried

| Guard-rail | Revenue gain | Passengers vs today |
|---|---|---|
| Pure revenue maximization (less price-sensitive, oversubscribed) | +14.2% | −12% |
| **Value each passenger at ₹1,100 in the objective** | **+5.5% to +9.2%** | **+0.2% to +2.0%** |
| Hard fare cap at +10% | +4.4% to +6.2% | −0.7% to −3.5% |

Telling the optimizer that each passenger carried is worth about the base fare beats a fare cap on both revenue and ridership. For a public railway, that's the guard-rail to use.

---

## 3. Network (O&D) pricing: [`notebooks/03_network_pricing.ipynb`](notebooks/03_network_pricing.ipynb)

One 3A berth is often sold two or three times per run: Firozpur → Delhi, then Delhi → Bhopal, then Bhusaval → Mumbai.
- **The network:** the 52 stops are grouped into **9 key stations**, giving **8 legs** and **36 origin–destination journeys**.
- **Calibration:** each journey's demand share and booking curve come from the bookings, and its base fare from the published 3A fare table (July 2025).
- **Uneven load:** about 420 passengers on board near Delhi, 260 near Mathura, and 510 near Jalgaon.

**Bid prices:**
- Each leg gets a value for one more seat: from the dual (shadow) prices of a network LP, or from a dynamic program per leg updated after every booking.
- A journey's fare is set to maximize expected bookings × (fare − the sum of the bid prices on its legs).

| Scenario (demand on the busiest leg) | Best single-journey pricer | **Network bid prices** | Network adds |
|---|---|---|---|
| Price-sensitive, spare seats (75 per 90) | −0.6% | **+2.5%** | **+3.2%** |
| Price-sensitive, oversubscribed (95) | −3.2% | **+5.0%** | **+8.5%** |
| Less price-sensitive, spare seats | +6.1% | +6.1% | 0% |
| Less price-sensitive, oversubscribed | +11.2% | +11.2% | 0% |

- **Pricing each journey on its own can be worse than fixed fares.** Every journey's optimizer discounts into the same empty seats. Bid prices raise fares only on journeys that cross the scarce Bhopal–Nashik stretch.
- **Updating seat values continuously drives the gain:**
  - a single LP price list earns about a third of it;
  - weekly LP re-solves earn about two-thirds;
  - per-booking DP bid prices earn all of it, at 98–99% of the LP upper bound.
- **The result survives a wrong elasticity.** With policies planning with the wrong price sensitivity, bid prices still beat single-journey pricing in 3 of 4 cases. The costly mistake is assuming passengers are *less* price-sensitive than they are.

---

## Interactive demo: [`demo/index.html`](demo/index.html)

A single-page simulator that runs in the browser, with no install. Pick a real departure from the test year, then set:
- how price-sensitive passengers are;
- demand vs capacity;
- what the pricer knows (forecast only, daily re-forecast, or true demand);
- the value placed on each passenger carried, and a fare cap.

The page shows, against the fixed fare and on the same simulated customers:
- the fare set each day;
- seats left;
- the dynamic program's full fare table, with the path the departure took;
- who paid what.

It uses the measured booking curve and each test departure's demand level and forecast. Its dynamic program reproduces `pricing_sim.optimal_fares` exactly. Open the file in any browser, or download it and double-click.

---

## Limitations

- **Price elasticity is assumed.** Fares never changed in the data. A controlled fare test, or natural experiments such as Tatkal premiums and flexi-fare trains, would measure it.
- **Waitlisted demand is missing.** Bookings on sold-out departures understate true demand, so the gains are probably conservative.
- **Simplified behaviour.** Passengers don't switch class or boarding station, and demand follows a constant-elasticity Poisson model.
- **One train, one class, simulated revenue.** The results are evidence about method, not a forecast of real-world revenue.

---

## Repository structure

```
├── data/
│   └── route_12138.csv           # stops and distances (public timetable)
├── demo/
│   └── index.html                # interactive pricing simulator (runs in the browser)
├── notebooks/
│   ├── 01_demand_forecasting.ipynb
│   ├── 02_pricing_engine.ipynb
│   └── 03_network_pricing.ipynb
├── src/
│   ├── data_loader.py            # loads the booking records (or a synthetic stand-in)
│   ├── synthetic_data.py         # synthetic bookings with the same columns, for running without the data
│   ├── pricing_sim.py            # single-coach simulator, policies, dynamic programming, evaluation
│   └── network_sim.py            # O&D network, LP and bid prices, network policies
└── tests/                        # 31 tests: simulator maths, optimizer vs simulation, network logic, data
```

## Running it

```bash
pip install -r requirements.txt
pytest tests
jupyter notebook notebooks/
```

**Data:** the booking records are confidential and **not included**. The notebooks are saved with their outputs from the real data. To re-run them on the real data, place the records at `data/private/bookings_12138.pkl` (git-ignored). Without that file, `src/data_loader.py` falls back to synthetic bookings with the same columns and similar patterns, so everything runs, but the numbers differ from the saved outputs.
