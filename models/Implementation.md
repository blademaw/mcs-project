# Implementation/Sprints

## General roadmap/milestones

* [x] **Create/define baseline model**
	* [x] **Structural baseline** — find way to map data to 
		* [x] Investigate Cambodia data for usability
		* [x] Investigate MAP for usability
		* [x] If above two don't work out, investigate other data sources for usability
	* [x] Movement model
		* [x] Work through movement model ⇒ assumptions, how to code, etc.
	* [x] Test/validate model
* [x] Preventive measure test drive
	* [x] Embed single preventive measure to test dynamics
	* [x] Examine impacts/effects
* [x] Embed single behavioural theory to test dynamics
* [ ] Embed other preventive measures
* [ ] Create all behavioural theory models
* [ ] Conduct analysis



## Scrapbooking

I need three essential components to each model:

1. A baseline VBD ABM that is extendible
2. Preventive measures agents can use
3. A behavioural theory encoded in agents (or none at all)



**Vague ideas**

- Decay parameter to introduce normalization of disease over time in society?
- **Preventive measures**
	- Mosquito repellent/coil ⇒ wards off mosquitoes (reduces chance of being bitten; dampens probability)
	- Long-sleeved clothing ⇒ reduces $\sigma_h$? Not sure if feasible
	- Staying indoors ⇒ reducing chance to venture to outdoor locations (or, locations with high exposure)
- **Risk perception implementation**
  - Need some opinion dynamics-based logic ⇒ social norms influence risk perception (cf. Lopes-Rafegas 2023)
  - *Environmental*: agents will "sense" their surrounding environment to incorporate risk perception, and the influential factors are those supported by VBD literature ⇒ mosquito bite frequency, number of mosquitoes observed, etc.
  - *Social*: agents update risk perception based on social factors, such as people conveying their risk perception, and media.
- **Preventive behaviours**:
  - Modifying behaviour for agents with high risk perception
  	- Spending less time outdoors ⇒ Duval, Valiente Moro, and Aschan-Leygonie, “How Do Attitudes Shape Protective Practices against the Asian Tiger Mosquito in Community Gardens in a Nonendemic Country?” ⇒ agents are less likely to travel to outdoor region/activity & more likely to wear long-sleeved clothing
- Agents should be "deliberative" according to De Mooij et al. (2023)
  - Take into account habits, fatigue, political preferences
  - Norm-aware (explicitly capable of taking into account existing norms such as behavioural interventions/laws when making decisions)
  - Socially aware (take into account behaviour/mentalities of other agents)
  - Design should be backed by theory + grounded in data



**Potential data sources**

* **Cambodia**
	* Links - https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0266460 & https://malariajournal.biomedcentral.com/articles/10.1186/s12936-020-03482-4
		* Paper 1
			* A lot of the male population are "forest goers" who are the most at-risk.
			* "Malaria vectors are found mostly in forest environments: forest activities are therefore the most important risk factor."
			* "Malaria vectors also inhabit forest fringes ..."
			* "... vectors are mostly nocturnal ... but they can also be active during dusk and dawn, outside the hours of bed net use"
			* According to the paper, "performing mobility analyses on the complete dataset, including the suboptimal data, would not be representative of the true population mobility" which is a problem if I want to infer typical agent behaviour of the area
		* Paper 2
			* Empirically confirms that recent travels to forest sites/work are associated with PCR positivity
			* Crucially, the authors note that participants who are not necessary located in the forest but engage in work in the forest have higher risk of infection—this is something I'd need to capture and incorporate into the mobility model.
	* Tracks PCR positivity for various pathogens in participants in Cambodian villages clustered around a forest area, segmented into three main regions: outer forest, fringe forest, inner forest.
	* **Idea**: (1) Map regions (outer/fringe/inner forest) to three patches (aggregate villages); (2) Use % positive PCR results for participants as proxy for malaria prevalence to set parameters for three patch models; (3) Use mobility data from previous study to inform movement model of agents.
	* **Patches**: can use outer / fringe / inner forest and aggregate villages, **OR** I can do villages (there are 17 villages)
		* ⇒ 3 patches (regions)
		* ⇒ 17 patches (villages)
	* **Locations (nodes) & activities**: could estimate geographically or rely on qualitative descriptions of regions? Not sure how to get this data.
	* **Other features**: number of agents is 4,200 participants, 10,053 total.
	* **Questions based on assumptions**:
		* ~~Can I aggregate all of these villages to three regions?~~
		* ~~Can I use PCR positivity as a proxy for exposure/prevalence of mosquitoes in participants' regions to derive patch model parameters?~~
		* When simulating, I will assume constant (time step) travel time between locations—I know this is unrealistic, but practical for simulation. Is this OK?
		* The real population size (10,053) might make the simulation times too long—would using the number of participants (4,200) or a scaled number of agents with the same patch densities be OK?
		* Should I include seasonality?
		* Should I model days of the week (allows for work days vs non-working days)?

### Movement model

* The movement model should be grounded in real human mobility (same data should be used as above for structure), but should make simplifying assumptions that make the model generalizable.
* The movement model may:
	* **Be based on the time step.** At each time step, agents either move or don't move based on some probability. How this is computed is up in the air.
	* **Be influenced by the day/night cycle.** Agents should go home at night and go out during the day. Therefore, they should spend around half of their overall time at their home node.
	* **Have preferential places for agents.** Agents have defined home nodes. Something I'm exploring is that agents could have dedicated work nodes that have high connectivity.
	* **Be inspired by real mobility data.** Noting down any additional trends from papers here: the first paper draws no conclusions/analysis from the mobility data, but it does highlight that "performing mobility analyses on the complete dataset, including the suboptimal data, would not be representative of the true population mobility ..." and I would have to do my own analysis, which would be a big task in itself. That being said, I could just look briefly at most time spent in places and kind of estimate things.



## Baseline model

The baseline model is from Manore et al.



## Extended model

The baseline from Manore et al. is too unrealistic to apply different behavioural theories gene these are ground in human psychology. I require a **realistic but simple** model and agent mobility.

Using data from three main studies conducted in the Mondulkiri Province in Cambodia for different reasons:

1. [Cross-sectional survey of risk behaviour and infection risk](https://malariajournal.biomedcentral.com/articles/10.1186/s12936-020-03482-4) → used to define geographical characteristics of patches and regions
2. [Mobility survey tracking of participants in the region](https://dx.plos.org/10.1371/journal.pone.0266460) → used to ground model in realistic agent mobility
3. [Entological/abundance survey of different sites in the province](Entological/abundance survey of different sites in the province) → used to derive parameters in the extended model

these are referred to as [1], [2], and [3] from now on.



### Extensions (and justifications)

- **Model architecture**

	- **Network architecture and geography**

		- Patches are inspired by [1] classifying of regions of surveyed land with households into outer-, fringe-, and inner-forest.

			- ⇒ 3 patches (regions)

			- ⇒ 2,351 locations (households) [+ 2 field nodes & 1 forest site node]
			- ⇒ 10,053 agents (hosts)
		- Infection risk profiles are ordered from most dangerous to least: inner-forest, fringe-forest, outer-forest.
		- In addition to households, the next key risk locations are fields/plantations ([2] basically considers them the same) and forest work site(s) mostly focused on in [1]. To simplify the architecture of the model graph, I create two field nodes, one each for outer- and fringe-forest patches. A plantation/field node is not created for the forest patch since the geographic properties of the area indicate there is little/no plantation/field in the forest. This assumes the risk profiles of plantations and fields are identical but preserves heterogeneous exposure between the outer and fringe forest patch.
		- Two plantation/field nodes are added
		- Exposure rates

			- In households, agents are assumed to be semi-exposed. From [1] supplementary data, 57% of participants have "windows not screened", which I assume protects agents from vectors. Therefore, the exposure for households is $\alpha_{\text{household}}=0.43$.
			- In fields and forest sites, assume agents are fully exposed, i.e. $\alpha_{\text{field}}=\alpha_{\text{forest}}=1$

	- **Day/night cycle**

		- Agents wake up and begin their daily activities at 8am. Agents go home and go to sleep at 8pm (Table S6 from [3] shows that most [calculate this] agents go to sleep by 8pm).
		- According to [3] pp6, only 20% of mosquito activity occurs during 6am–6pm, meaning there is 4x activity outside these hours, so during these hours in the model there is a 4x increase in mosquito aggressiveness. However, from Figure 2 in [3], looks like mosquitoes persist until ~10am. In the model, from 6pm–10am, mosquitoes have 4x biting rates ($\sigma_v$).
			- And under the parameterisation of two-hour time steps, there are two time steps during which agents are at work/awake and vectors have 4x activity, these are usually the most dangerous/infectious hours.
			- Can be backed by multiple sources, e.g. NATNAT report.

	- **Temperature**

		- Based on a socioecological [2009 study](https://www.researchgate.net/publication/268383258_The_socio-ecology_of_the_black-shanked_douc_Pygathrix_nigripes_in_Mondulkiri_Province_Cambodia) that collected hourly temperature measurements, I make temperature a random variable centered around the average yearly temperature according to the study (≈25 Cº):
		- $T_b\sim \mathcal{N}(\mu=25,\sigma=2)$
		- $T(t)=T_b+5\sin\left(\left(\frac{\pi}{11}\right)t-.8\pi\right)$​
		- ==NOTE==: To be scrapped, replaced with static "dry/wet" seasonality.

	- **Agent occupations**

		- Agents are distributed into three types: field workers, non-workers, and forest workers. They are randomly assigned roles according to a 71%–24%–5% split respectively. This is derived from survey questionnaires about occupations from supplementary data in [1] (check the Excel for my calculation—I omit "other" workers and scale-up the rest).
		- Importantly, not all forest-goers live in the forest.

	- **Protective measure: ITN**

		- The protective measure built into the model is the insecticide-treated bed net (ITN).
		- At 8pm when agents come home and sleep, they can choose to either "adopt" the preventive measure or not with some probability.
		- When used, ITNs are assumed to have 99% efficacy, implemented via
			- `r < (1 - np.exp(- self.model.timestep * lambda_hj))*(1-efficacy)`
			- I.e., the efficacy $\varepsilon$ dampens the probability $\mathbb{P}(E\to I)$ by $1-\varepsilon$
		- When agents wake up and go to work at 8am, the ITN 'wears off' and they stop gaining from protection.

- **Mobility**

	- Day/night cycle
		- Working agents teleport and begin work at 8am, teleport home at 8pm.
		- Non-working agents roam their own patch throughout different households during the day.
		- At 8pm, all agents come home and "sleep" i.e. are stationary throughout the night.
	- Agent occupations
		- Forest workers commute to the forest node in the inner-forest patch.
		- Field workers commute to their field node of their patch (agents who live in the forest commute to the fringe forest patch's field)



### Parameterisation

- [1]
	- $k=3$ since three regions
	- Number of agents is population, which is 10,053
	- Number of household locations is 2,351
	- Patch densities are 85%, 8%, and 7% for the outer-, fringe-, and inner-forest areas (calculated from number of households in different regions, from [1] supplementary data available online)
- [3]
	- Timestep is 2 hours since risk profile of infection varies throughout the day, as mentioned in this paper (note dusk/dawn/nighttime key words) + [1] and [2].
- All other parameters held constant. Refer to appendix for full table
- How to determine $K_v$?
	- From [3] and some extra analysis (refer to meeting 20240814), we know the densities (mosquitoes per patch per time step).
	- First step is to calibrate the current model with $K_v^{(2)}=2.94K_v^{(1)},K_v^{(3)}=3.24K_v^{(1)}$ to Manore et al. conditions (qualitative comparison)
	- Then, once reproduce Manore et al. scenario, conduct sensitivity analysis on ±10% of $K_v^{(1)}$ (and adjust accordingly) to find how sensitive the model is to changes in 
	- I use high movement conditions, so looking to recreate high movement situations ⇒ time of peak is at 100.



## Behavioural theory implementation

- Many models seem to use rational choice theory / decision-making (e.g., Mohammad and Karl-Erich) and incorporate a cost-benefit ratio
- Seem to need way to discuss malaria (could duplicate network of households for agents to make agents "discuss" at end of day / knowledge-share about malaria.)
	- Agents influenced by the decisions of their network
- Need to decide the contextual factors available in the simulation:
	- Severity, prevalence of the disease (news; word of mouth)
	- 
	- mosquito bite frequency, number of mosquitoes observed, etc.
	- *Social*: agents update risk perception based on social factors, such as people conveying their risk perception, and media.



### Health belief model (HBM)

#### Sources

- Durham and Casman (2011) propose a mathematical model for the HBM integrated in an ABM for the SARS outbreak in Hong Kong. This will be my main source of inspiration (can _try_ to lift and shift this).
- Ryan et al. (2024) describe the HBM as a mathematical model for a compartmental model. Can use this for inspiration and to compare/contrast modelling approaches.



#### Representation

One way of representing this is from Durham and Casman, 2011:
$$
p(\text{behaviour})=\frac{\text{OR}_0\cdot\prod{\text{OR}_i^{x_i}}}{1+\text{OR}_0\cdot\prod{\text{OR}_i^{x_i}}},i=1,\dots,4.
$$
where each $i$ denotes the HBM constructs, each $x_i$ is a binary variable representing the state of the HBM construct ($1$ indicates high, $0$ low), $\text{OR}_i$ is the odds ratio of adopting the preventive behaviour when the $i$-th HBM construct is 'high', and $\text{OR}_0$ is a calibration constant that defines the probability of the behaviour when all $x_i$ are $0$ (low). Here, $p(\text{behaviour})\in[0,1]$. 

Each indicator $x_i$ is in the form
$$
x_i = \begin{cases}
    1 & \text{if } \verb|condition| \\
    0 & \text{otherwise}
\end{cases}
$$
In this approach, agents adopt behaviour when $p(\text{behaviour})\ge0.5$, but I can adapt to make agents use the probability during each decision time step (e.g., at end of day for ITNs).

The authors also note:

- This **does not** include the cues to action and self-efficacy components—I could add these in (but this may require additional justification for correctness).
- Different $\text{OR}$​s can be defined stratifying across population segments or risk groups ⇒ could do something for forest workers / field workers / non-workers.
- Individual variability can be modelled via sampling $\text{OR}$s from distributions for agents.
- Each $\text{OR}$​​ must be responsive to changes in the model and receptive of contextual information derived from the model environment.

For example, take perceived susceptibility, $i=1$: if you have high perceived susceptibility to the disease, you have an odds ratio of $\text{OR}_1^1=\text{OR}_1$ (e.g. $1.5$​, meaning you are 50% more likely to engage in preventive behaviour than if you had a low perceived susceptibility to disease).

Baseline engagement in the preventive behaviour when all constructs are 'low' is:
$$
p(\text{behaviour})&=&\frac{\text{OR}_0\cdot \prod_i{\text{OR}_i^{0}}}{1+\text{OR}_0\cdot \prod_i{\text{OR}_i^{0}}}\\
&=&\frac{\text{OR}_0}{1+\text{OR}_0}\\
\implies\text{OR}_0&=&\frac{p(\text{behaviour})}{1-p(\text{behaviour})}
$$


#### Components

1. $\text{OR}_0$

	- Interpretation: when perceived susceptibility, severity, benefits, and barriers are all low, what is $p(\text{behaviour})$?
	- Need to request data to calculate this.

2. **Perceived susceptibility**

	- Durham and Casman (2011) provide a source for this: existence and prevalence of reported cases over time. ⇒ lots of malaria cases go un-reported, and using model information directly would assume agents have perfect information.

		- $s_t=\sum_{i=0}^{t-2}{\delta^ic_{t-i-1}}$.

		- where $s_t$ is the weighted sum of illnesses over the time from inception until $t$, $\lambda$ is a calibration threshold used such that:
			$$
			x_1=\begin{cases}
			1&\text{if } s_t\ge\lambda\\
			0&\text{otherwise}
			\end{cases}
			$$

		- note that $\lambda$ can be varied/sampled to incorporate heterogeneous perceptions of "high" number of infections.

	- I could do the same thing, but **want to be careful about giving agents perfect/global information**—could possibly experiment with just providing agents with number of infections in an agent’s household-connected network, and define λ to be some proportion of their total immediate contacts.

	- Other risk factors we know for VBDs include:

		- **perceived mosquito prevalence** (Raude et al. 2012 + Phok et al. 2022) ⇒ could model for each agent as number of mosquitoes in patch ($N_v$).
		- **amount of mosquito bites on an individual** (Lopes-Rafegas et al. 2023) ⇒ can't really model this.
		- **previous days with rainfall** (Constant et al. 2020) ⇒ can't model this.

3. **Perceived severity**

	- Durham and Casman (2011) provide this ⇒ I don't have this in my model—the disease is not fatal at the moment. Should I add this?
	- Ideas:
		- Can either **omit in model**
		- Use % of contacts for an agent that currently have the disease?
		- Knowledge of malaria (start with randomly distributed, can control this via sensitivity analyses)

4. **Perceived benefits of behaviour**

	- Durham and Casman (2011) omit this in their paper since benefits from environment were not found; and were declared context-specific ⇒ what are the benefits of sleeping under an ITN, and how does this change over time?
	- **Perceived community social norms** ⇒ if immediate contacts are using ITNs (i.e. used previous night), use.

5. **Perceived barriers to behaviour**

	- Durham and Casman (2011) model this 'low' if there are sufficiently many other people wearing facemasks, otherwise 'high'—this models the perceived social acceptance. ⇒ I need to research what barriers there are to ITN adoption: cost, availability, nuisance?
	- Barriers to ITNs are usually cost, availability, quality, heat/warmth, belief that they attract bed bugs.
		- Cost and availability would be big ones (e.g. can reference this paper "Barriers in access to insecticide-treated bednets for malaria prevention: An analysis of Cambodian DHS data", but in specific Cambodia data it is not an issue; reference the distribution rate. + too complex to add distribution model)
	- Ideas:
	  - Make temperature a randomly-sampled variable per day, if temperature higher than an agent's "critical temperature" then 'high' barrier to behaviour; otherwise low.
	  - **Actually, looks like a barrier to ITN use is knowledge** - "Reversing the community’s misconceptions through information, education and communication (IEC), and behavioural change communication (BCC) would enhance ITN utilization." from Yirsaw et al. 2021. ⇒ may need to rethink this



##### Table of (adjusted) odds ratios from data sources

| HBM Construct            | Odds Ratio          | Reference                                                    | Comments                                                     |
| ------------------------ | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| N/A (constant)           | ?                   | ?                                                            | ?                                                            |
| Perceived susceptibility | 0.998, 0.977, 0.927 | [Storey et al. (2018)](https://link.springer.com/article/10.1186/s12889-018-5372-2#Tab1) | High perceived susceptibility for caregivers.                |
| Perceived severity       | 0.971, 0.973, 0.885 | [Storey et al. (2018)](https://link.springer.com/article/10.1186/s12889-018-5372-2#Tab1) | High perceived severity of caregivers.                       |
|                          | 2.78                | [Kakaire and Christofides (2023)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0289097#sec017) | High perceived threat of malaria among pregnant women.       |
| Perceived benefits       | 2.94                | [Perkins et al. (2019)](https://link.springer.com/article/10.1186/s12936-019-2798-7/tables/5) | Perceived as normative behaviour in community                |
|                          | 1.391, 1.024        | [Storey et al. (2018)](https://link.springer.com/article/10.1186/s12889-018-5372-2#Tab1) | Perceived as normative behaviour in community; for caregivers. |
|                          | 2.69                | [Phok et al. (2022)](https://doi.org/10.1186/s12936-022-04390-5) | Perceived community social norms toward ITN use; for forest-goers. |
| Perceived barriers       | 0.53                | [Yirsaw et al. (2021)](https://malariajournal.biomedcentral.com/articles/10.1186/s12936-021-03666-6) | Perceived barrier for pregnant women in Ethiopia             |
| Self-efficacy            | 1.567, 1.068, 1.034 | [Storey et al. (2018)](https://link.springer.com/article/10.1186/s12889-018-5372-2#Tab1) | For caregivers; self-efficacy to prevent malaria.            |



##### Table of HBM constructs for implementation

| HBM Construct            | Value | Description (from Champion & Skinner)                        | Representation in model                                      | Decision process in model for 'high'                         | Odds ratio         |
| ------------------------ | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ |
| —                        | —     | —                                                            | Baseline odds ratio when all HBM constructs are 'low'        | —                                                            | $\text{OR}_0=1$    |
| Perceived susceptibility | $x_1$ | Belief about the chances of experiencing a risk or getting a condition or disease. | Existence and prevalence of cases over time.                 | If the weighted number of cases over time is $>\lambda$.     | $\text{OR}_1=.998$ |
| Perceived severity       | $x_2$ | Belief about how serious a condition and its sequelae are.   | Number of immediate contacts who have the VBD.               | If at least proportion $\chi$ of an agent's immediate network have the VBD. | $\text{OR}_2=2.78$ |
| Perceived benefits       | $x_3$ | Belief in the advised action to benefit in health and non-health-related forms. | Number of immediate contacts who used ITNs last night.       | If at least proportion $\omega$ of an agent's immediate network used ITNs last night. | $\text{OR}_3=2.69$ |
| Perceived barriers       | $x_4$ | Belief about the tangible and psychological costs of the advised action. | Whether the ITN will make the agent uncomfortably hot when sleeping. | If the temperature at the current time is $>T_c^{(i)}$ for agent $i$. | $\text{OR}_4=.53$  |



### Protection motivation theory (PMT)

#### References

- Fuli et al. (2024)
- Ghoreishi and Lindenschmidt (2024) integrate the PMT into an ABM. I can use this for inspiration.
- Kurchyna et al. (2024) integrate the PMT into an ABM



## Limitations/drawbacks of model

- **Baseline model, extension & parameterisation**
	- *Reminder that while these limitations are valid, they are often necessary to keep things simple (for scope) but also to focus on the dynamics we care about—the behavioural theories; not to mention any limitations can be inspiration for future work.*
	- Inherit all limitations of Manore et al.
	- Mobility is more complex:
		- travel between locations is constant speed,
		- does not take into account varying levels of exposure due to travel between locations (e.g., going from indoors to indoors is constant low exposure when in reality this would involve going outdoors; e.g., [overnight] trips into forest).
		- does not take into account how other risk groups travel into the forest (Sandfort et al 2020)
	- Grouping of households into patches does not take into account varying household-household malaria risk exposure, which is an observed form of local heterogeneity omitted in the model [Anaïs et al. 2022 pp 2 + reference 17].
	- Seasonality (e.g., rainy and dry seasons) are not included in the model, as per Manore et al. ⇒ however, timeframe (200 days) is small enough for this to likely not be too significant.
		- ITN use can also depend on seasonality, e.g. Winch et al. 1994.
	- The model of the VBD is of an outbreak rather than a setting with previous VBD presence ⇒ however, doesn't discredit dynamics, and such scenarios would actually arise [find historical situations where applicable].
	- Risk groups are likely not exhaustive; for example, in the data there are an excess of workers, but the data do not clarify which clarify which occupations these are.
	- Parameterisation might not be correct in all instances; these are qualitative patterns fit to the Manore et al. architecture, and naturally may lack some accuracy. For example, exposure rates are defined based on the proportion of households that lack windows.
- **Behaviour change theories**
	- ?





## Swift Catch-up and Rapid Action Mission (SCRAM) Sprint \#1 (Jul 21–27)

### Goals

- [ ] Writing
	- [ ] Create Overleaf
	- [ ] Rough bullet point version of ODD
- [ ] Code
	- [x] Implement new baseline model w/ Cambodia data changes
	- [ ] Movement model
		- [x] Designed and implemented
		- [ ] Incorporated first round of feedback from Nic/Cam
	- [ ] Preventive measure
		- [x] Designed and implemented
		- [ ] Incorporated first round of feedback from Nic/Cam
	- [x] Start thinking about behavioural change theory implementations
	- [x] Run experiments, impacts, create plots
	- [ ] Sensitivity analysis following meeting with Nic/Cam
- [ ] Research
	- [ ] Go through desktop backlog (papers in browser tabs open)
	- [ ] Go through email backlog (papers from notifications)

### TImeline

- [x] Tuesday 23rd
	- [x] Look at Cambodia data, past meeting notes, decide what should be included in:
		- [x] baseline model architecture;
		- [x] movement model; and
		- [x] preventive measure(s).
	- [x] Draft rough ODD protocol (bullet points) to track changes
	- [x] Implement:
		- [x] baseline model architecture; and
		- [x] movement model.
	- [x] Conduct validation with no preventive measures to observe disease spread
	- [x] Implement preventive measure
	- [ ] Conduct validation, sensitivity analysis
	- [x] Prepare slides:
		- [x] Recap
		- [x] Changes made / where I am now
		- [x] Validation results (no preventive measure)
		- [x] Experiment results (preventive measure)
		- [x] Next steps
	- [ ] ~~Send agenda + email to Nic/Cam



## Swift Catch-up and Rapid Action Mission (SCRAM) Sprint \#2 (Jul 27–Aug 3)

### Goals

- [ ] Writing
	- [ ] Create Overleaf
	- [ ] Plan thesis outline
	- [ ] Introduction first draft
	- [ ] Rough bullet point version of ODD
- [ ] Code
	- [x] Implement baseline changes from Nic/Cam first round feedback
	- [x] Get model code/environment ready for Rachel/Leanne
	- [ ] First behavioural change theory implementation
	- [x] Sensitivity analysis for:
		- [x] parameters in baseline model
		- [x] aggregate plots for ITN impacts
- [ ] Research
	- [ ] Go through desktop backlog (papers in browser tabs open)
	- [ ] Go through email backlog (papers from notifications)

### Timeline

Work to do in order.

* [x] Model design choices
	* [x] ==Action== Modify the movement model to be more realistic (i.e., conform more to the real-life scenario).
	* [x] ==Action== Extend the model to include varied locations (not just households), heterogeneous exposure parameters for locations (e.g., households have lower exposure to outdoors, forest work sites have higher).
* [x] Code
	* [x] Implement updates from design choices
	* [x] ==Action== Compare different values for mosquito carrying capacity across patches, document results, try to choose a value and empirically justify it.
* [x] Slides/writing
	* [x] ==Action== Begin writing model description, with clear mapping to demonstrate how Cambodia data results have influenced decisions in the model.
	* [x] ==Action== Strengthen justifications for parameter value choices.



## Swift Catch-up and Rapid Action Mission (SCRAM) Sprint \#3 (Aug 3 – Aug 9)

### Goals

- [ ] Writing
	- [ ] Create Overleaf
	- [ ] Plan thesis outline
	- [ ] Introduction first draft
	- [ ] Rough bullet point version of ODD
- [ ] Code
	- [x] Implement changes to model from Nic/Cam
	- [ ] First behavioural change theory implementation
	- [x] Sensitivity analysis for:
		- [x] new carrying capacity
		- [x] parameters in baseline model
		- [x] aggregate plots for ITN impacts
	- [x] Add risk group level tracing of agent SEIR states, plot aggregated data
	- [ ] Trace agents' movements as a sanity check
	- [x] [Optional] Optimise code to make run faster
	- [ ] [Optional] Update visualisation with:
		- [ ] Cambodia-like architecture (plantation & forest nodes)
		- [ ] Actual Cambodia scenario overlay (ask Nic/Cam if this is worth doing)
	- [ ] Think: how to scale up experiments? This will be _a lot_ of data.
- [ ] Research
	- [ ] Go through desktop backlog (papers in browser tabs open)
	- [ ] Go through email backlog (papers from notifications)



## Swift Catch-up and Rapid Action Mission (SCRAM) Sprint \#4 (Aug 10 – Aug 16)

### Goals

- [ ] Writing
	- [ ] Create Overleaf
	- [ ] Plan thesis outline
	- [ ] Introduction first draft
	- [ ] Rough bullet point version of ODD
		- [ ] Baseline
		- [ ] Extended
		- [ ] [Foresight] Behavioural theory overlay
- [ ] Code
	- [x] Implement changes to model from Nic/Cam
	- [ ] First behavioural change theory implementation
	- [x] Add risk group level tracing of agent SEIR states, plot aggregated data
	- [ ] Trace agents' movements as a sanity check
	- [x] [Optional] Optimise code to make run faster
	- [ ] [Optional] Update visualisation with:
		- [ ] Cambodia-like architecture (plantation & forest nodes)
		- [ ] Actual Cambodia scenario overlay (ask Nic/Cam if this is worth doing)
	- [ ] Think: how to scale up experiments? This will be _a lot_ of data.
- [x] Research
	- [x] Go through desktop backlog (papers in browser tabs open)
	- [x] Go through email backlog (papers from notifications)



## Catch-up and Rapid Action Mission (CRAM) Sprint \#5 (Aug 16 — 22)

### Goals

- [ ] Writing
	- [ ] Create Overleaf
	- [x] Plan and finalise thesis outline
		- [x] Decide if should dedicate chapters to HBM/PMT/COM-B
	- [ ] Introduction first draft
	- [ ] Rough bullet point version of ODD
		- [ ] Baseline
		- [ ] Extended
		- [ ] [Foresight] Behavioural theory overlay
- [ ] Code
	- [ ] Baseline model
		- [ ] Derive new values for $K_v$ via spatial characteristics
		- [ ] Redo experiments for lower $K_v$​ (first two slides from last week)
	- [ ] Visualisation(s)
		- [ ] Update visualisation to include forest/field patches; location discernment
		- [ ] Investigate where non-workers are infected most (own vs other household)
		- [ ] Redo graphs from last week:
			- [ ] Lambda v
			- [ ] Infection location — forest/field/non workers
	- [ ] HBM implementation
		- [x] Come up with toy $\text{OR}_0$
		- [ ] Decide $\text{OR}$s
		- [ ] Implement
			- [ ] Implement temperature
		- [ ] Conduct experiments
	- [x] Optimise code - make run faster
	- [ ] Trace agents' movements as a sanity check
	- [ ] [Optional] How to scale up experiments? This will be _a lot_ of data.



