# Project Brief: AI Infrastructure Industry — Competitive Landscape Study

## Engagement Overview

Apex Capital Partners is conducting a competitive landscape study of the AI infrastructure sector to evaluate a potential platform investment. The study will map the competitive dynamics, unit economics, customer concentration, capacity strategies, and defensibility of the leading GPU cloud and AI infrastructure providers.

## Scope

**N = 15 expert calls** across the AI infrastructure value chain, targeting former and current executives at the following companies:

### Tier 1 — Hyperscale GPU Cloud (4 calls)
1. **CoreWeave** — Pure-play GPU cloud; raised $12B+ in debt/equity; NVIDIA's preferred partner
2. **Lambda Labs** — GPU cloud + on-prem clusters; developer-focused; strong ML community presence
3. **Together AI** — Inference and training cloud; open-source model focus; competitive pricing
4. **Crusoe Energy** — GPU cloud powered by stranded/renewable energy; differentiated cost structure

### Tier 2 — Established Cloud AI Infrastructure (4 calls)
5. **AWS (Amazon)** — Largest cloud provider; custom silicon (Trainium, Inferentia); broadest service portfolio
6. **Google Cloud (GCP)** — TPU ecosystem; Vertex AI platform; strong in ML research customers
7. **Microsoft Azure** — OpenAI exclusive partnership; largest enterprise AI footprint
8. **Oracle Cloud (OCI)** — Aggressive GPU capacity buildout; competitive pricing; targeting AI-native startups

### Tier 3 — Specialized AI Infra / Emerging Players (4 calls)
9. **Voltage Park** — Large-scale GPU cluster operator; wholesale capacity model
10. **Nebius (ex-Yandex Cloud)** — European AI cloud; full-stack from hardware to MLOps
11. **Applied Digital** — Data center operator pivoting to AI/HPC hosting
12. **Cerebras Systems** — Wafer-scale AI chips; alternative to GPU paradigm

### Tier 4 — Adjacent / Infrastructure Layer (3 calls)
13. **NVIDIA** — GPU supplier; DGX Cloud; increasingly competing with customers
14. **Run:ai** — GPU orchestration and scheduling software (acquired by NVIDIA)
15. **Vast Data** — AI-optimized data platform and storage infrastructure

## Key Hypotheses to Test

1. **Market Structure:** The GPU cloud market is consolidating around 3-4 major players (CoreWeave, hyperscalers) with specialists surviving in niches (inference, on-prem, vertical-specific)
2. **Unit Economics:** Pure-play GPU cloud providers operate at 40-60% gross margins on committed capacity but face margin pressure as supply normalizes
3. **NVIDIA Dependency:** All players are critically dependent on NVIDIA GPU allocation; access to H100/B200 supply is the primary competitive moat today
4. **Customer Concentration:** Most pure-play providers have dangerous customer concentration — top 3 customers often represent 50%+ of revenue
5. **Capacity vs. Demand:** Current GPU supply/demand imbalance will normalize by mid-2027, fundamentally changing the competitive landscape
6. **Defensibility:** Long-term defensibility comes from (a) energy/power procurement, (b) networking/interconnect expertise, (c) software/platform lock-in, not just GPU access
7. **Pricing Dynamics:** On-demand GPU pricing has dropped 30-50% in the last 12 months; reserved/committed pricing is the new battleground
8. **Energy as Moat:** Power availability and cost (not GPU supply) will become the binding constraint within 18 months

## Data Points to Collect Per Company

For each of the 15 companies, the expert calls should aim to validate:

### Operational Metrics
- Total GPU count (by model: H100, A100, B200, etc.)
- Data center capacity (MW) — current and planned
- Rack density and cooling approach (air vs. liquid)
- Network topology (InfiniBand vs. Ethernet, bandwidth per GPU)

### Financial Metrics
- Revenue run-rate (ARR or annualized)
- Gross margin on GPU compute
- CapEx intensity ($ per MW deployed)
- Pricing model (on-demand, reserved, committed spend)
- Average contract length and prepayment terms

### Customer & Market
- Number of active customers
- Top 3 customer concentration (% of revenue)
- Customer mix: AI labs vs. enterprise vs. government vs. startups
- Win/loss dynamics vs. competitors
- Geographic footprint

### Technology & Differentiation
- Software platform maturity (Kubernetes, orchestration, MLOps)
- Custom silicon or hardware differentiation
- Networking and interconnect capabilities
- Multi-tenancy architecture
- Security and compliance certifications

### Strategic
- Relationship with NVIDIA (allocation priority, partnership depth)
- Energy strategy (source, long-term PPAs, renewable %)
- Expansion plans (new regions, capacity additions)
- M&A appetite or partnership strategy
- Key person dependencies

## Expert Profiles Needed

| Call # | Company | Ideal Expert Profile |
|--------|---------|---------------------|
| 1 | CoreWeave | VP/Director of Sales or Business Development, departed 6-12mo |
| 2 | CoreWeave | Infrastructure/Platform Engineer (Sr. Director+), current or recent |
| 3 | Lambda Labs | Head of Cloud/Product, or founding engineer |
| 4 | Together AI | VP Engineering or Head of Infrastructure |
| 5 | Crusoe Energy | Director of Operations or Data Center Engineering |
| 6 | AWS | Principal Engineer or GM in EC2/AI infrastructure team |
| 7 | Google Cloud | TPU/Vertex AI product lead or engineering manager |
| 8 | Microsoft Azure | Director+ in Azure AI infrastructure or OpenAI partnership team |
| 9 | Oracle Cloud | VP/Director of AI/GPU Cloud strategy |
| 10 | Voltage Park | Operations or Sales leadership |
| 11 | Nebius | VP Engineering or Country Manager |
| 12 | Applied Digital | CFO, COO, or Head of AI/HPC business |
| 13 | Cerebras | VP Sales or Head of Cloud, or customer-facing engineer |
| 14 | NVIDIA | DGX Cloud team member, or Cloud Partner ecosystem manager |
| 15 | Run:ai / Vast Data | CTO, VP Product, or Solutions Architect |

## Known Data Points (Public Sources)

### CoreWeave
- Valued at ~$35B (May 2026 IPO filing)
- Revenue estimated $1.5-2B annualized (FY2025)
- 14 data centers operational or under construction
- Primary GPU fleet: NVIDIA H100, B200
- Major customers: Microsoft (significant contract), AI labs
- $7.5B debt facility backed by GPU collateral
- CEO Michael Intrator, CTO Brian Venturo

### Lambda Labs
- ~$400-600M revenue run-rate
- 1-Click Clusters product for distributed training
- On-prem + cloud hybrid model
- Strong developer community (Lambda Stack)
- CEO Stephen Balaban

### Together AI
- ~$300M+ revenue run-rate
- Focus on open-source model inference (Llama, Mixtral)
- Competitive pricing (often 30-50% below hyperscalers for inference)
- CEO Vipul Ved Prakash

### Crusoe Energy
- Raised $600M+ equity
- Differentiation: stranded natural gas and renewable energy
- Lower PUE and energy costs vs. traditional data centers
- Expanding into geothermal and wind-powered facilities
- CEO Chase Lochmiller

## Gaps to Fill

1. **Real unit economics** — public filings show revenue but margins are opaque; what are true gross margins after power, depreciation, and networking costs?
2. **NVIDIA allocation mechanics** — how does GPU allocation actually work? Who gets priority and why?
3. **Customer churn** — are AI startups churning off GPU clouds as they build their own? What's the retention curve?
4. **Interconnect differentiation** — does InfiniBand vs. RoCE actually matter for customer workloads? How much are customers willing to pay for better networking?
5. **Power procurement** — what are the actual $/MWh rates these companies are securing? How long are the PPAs?
6. **Competitive displacement** — which companies are winning deals from whom? What are the switching costs?
7. **B200/Blackwell transition** — who has early access? How does the architecture transition affect existing fleet value?
8. **Enterprise vs. AI lab demand mix** — is enterprise AI adoption creating sustainable demand, or is it still primarily AI labs/startups?

## Engagement Timeline

- Expert calls: June 2-20, 2026 (3 weeks, ~1 call/day)
- Interim findings deck: June 13, 2026
- Final competitive landscape report: June 27, 2026
- Investment Committee presentation: July 2, 2026
